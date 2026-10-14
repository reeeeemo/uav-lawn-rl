from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces
from drone.drone import Drone
import numpy as np
import torch
from environment.grid import Grid
from drone.camera import Camera

class DroneEnv(VecEnv):
    """Custom gymnasium environment that simulates a drone.

    Allows for multiple environments.
    """
    def __init__(self, init_polygon: np.ndarray, num_envs: int = 10, num_objs: int = 40):
        self.wrapper = Drone(num_envs)

        ### PPO OBS / ACTION VARS ###
        self.obs_size = 22 + 5*3 + 8*3  # 61
        self.acts_size = 4
        self.num_envs = num_envs
        self.ep_limit = 1500
        self.num_objs = num_objs

        self._obs = np.zeros((self.num_envs, self.obs_size), dtype=np.float32)
        self._drone_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        
        # k=8: (dx, dy, 1-P (confidence)). k=5: (dx, dy, radius)
        self._k_obstacles = np.zeros((self.num_envs, 5, 3), dtype=np.float32)
        self._k_confidence_cells = np.zeros((self.num_envs, 8, 3), dtype=np.float32)

        # [x_near_cell, y_near_cell, z_near_cell]
        # 9d rot mat for [roll pitch yaw]
        # [x_vel, y_vel, z_vel]
        # [roll_vel, pitch_vel, yaw_vel]
        # [prev_thrust, prev_roll, prev_pitch, prev_yaw]
        self.observation_space = spaces.Box(
            np.ones(self.obs_size) * -np.inf,
            np.ones(self.obs_size) * np.inf, dtype=np.float32
        )

        # [continous thrust, roll, pitch, yaw body rates]
        self.action_space = spaces.Box(
            low=np.ones(self.acts_size) * -1.,
            high=np.ones(self.acts_size) * 1.,
            dtype=np.float32
        )

        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._rewards = [[] for _ in range(self.num_envs)]

        ### STATE VARIABLES ###
        self._done = np.zeros((self.num_envs,), dtype=bool)
        self._done_reason = ["" for _ in range(self.num_envs)]
        self._prev_action = np.zeros((self.num_envs, self.acts_size), dtype=np.float32)
        self._prev_dist = np.zeros((self.num_envs,), dtype=np.float32)
        self._cur_episodes = np.zeros((self.num_envs,), dtype=np.int16)
        self._info_gain = np.zeros(self.num_envs, dtype=np.float32)
        self._confs = [None] * self.num_envs
        self._visible_objs = np.zeros((self.num_envs, self.num_objs), dtype=bool)

        # setup grid environment
        self.survey_alt = 5  # altitude to stay at when surveying
        self._grids = [Grid(polygon=init_polygon) for _ in range(self.num_envs)]
        self._cur_cell = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._field_center = init_polygon.mean(axis=0).astype(np.float32)
        for i, g in enumerate(self._grids):
            g.generate_objects("tree", self.num_objs//2)
            g.generate_objects("rock", self.num_objs//2)
            self._cur_cell[i] = np.append(g.polygon.min(axis=0), -self.survey_alt)

        self._cameras = [Camera() for _ in range(self.num_envs)]
        self.sim_dt = self.wrapper.dt
        self.gamma = 0.999

        self.reset()

    def step(self, action: np.ndarray):
        """Computes step of quadrotor in the environment.

        Args:
            action: [normalized_thrust, roll, pitch, yaw body rates]
        Returns:
            observation, reward, done, env information
        """
        # xyz, lin_vel, ang_vel, R
        drone_state = self.wrapper.step(torch.as_tensor(action))

        self._update_detections_grid(drone_state)
        self._compute_observations(action, drone_state)
        self._compute_reward(action)

        self._prev_action = action.copy()

        info = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            self._rewards[i].append(self._reward[i])

        ### set info data to return ###
        for k in np.where(self._done)[0]:
            # terminal observations -- first 2 are for PPO algo to use.
            info[k]["terminal_observation"] = self._obs[k].copy()
            info[k]["terminal_global_observation"] = self._drone_pos[k].copy()
            info[k]["TimeLimit.truncated"] = self._done_reason[k] == "TimeLimit.truncated"
            info[k]["termination_reason"] = self._done_reason[k]
            info[k]["episode"] = {
                "r": float(np.sum(self._rewards[k])),
                "l": len(self._rewards[k])
            }
            info[k]["terminal_confidence"] = self._grids[k].confidence.copy()

        done_arr = self._done.copy()
        self.reset(np.where(self._done)[0])

        return self._obs.copy(), self._reward.copy(), \
            done_arr, info

    def reset(self, i: list[int] = None):
        """Resets drone. Takes in a list of indices of envs to reset."""
        if i is None:
            i = np.arange(0, self.num_envs)
        elif len(i) == 0:
            return
        drone_state = self.wrapper.reset(i, randomize=True)

        self._prev_action[i] = 0
        self._done[i] = np.zeros_like(self._done[i])
        for k in i:
            self._rewards[k] = []
            self._cur_cell[k] = np.append(self._grids[k].polygon.min(axis=0), -self.survey_alt)
            self._grids[k].reset()
            self._grids[k].generate_objects("tree", 25)
            self._grids[k].generate_objects("rock", 15)
        
        self._k_obstacles[i] = 0
        self._k_confidence_cells[i] = 0
        self._update_detections_grid(drone_state, i)
        self._compute_observations(np.zeros((self.num_envs, 4), dtype=np.float32), drone_state, i)

        return self._obs.copy()

    def _compute_reward(self, action: np.ndarray):
        ### termination clauses ###
        # episode limit reached
        ep_lens = np.array([len(self._rewards[i]) for i in range(self.num_envs)], dtype=np.int16)
        timeout = ep_lens+1 == self.ep_limit
        self._done[timeout] = True
        for k in np.where(timeout)[0]:
            self._done_reason[k] = "TimeLimit.truncated"

        # if drone hits ground (z > 0 since ned)
        crashed = self.drone_pos[:, 2] > 0.1
        self._done[crashed] = True
        for k in np.where(crashed)[0]:
            self._done_reason[k] = "terminated"
            
        # if drone flies outside from geofenced area
        # specifically for stopping pbrs reward from flipping if it goes too far determinstically
        d_field = np.linalg.norm(self.drone_pos[:, :2] - self._field_center, axis=1)
        lost = (d_field > 80.0) | (self.drone_pos[:, 2] < -100.0)
        self._done[lost] = True
        for k in np.where(lost)[0]:
            self._done_reason[k] = "lost"
            
        # if all grid cells have a confidence level over 95%
        complete = np.array([(g.confidence[g.inside_cells] > 0.95).all() for g in self._grids])
        self._done[complete] = True
        for k in np.where(complete)[0]:
            self._done_reason[k] = "complete"
        
        # temp reward for done and not terminated at episode length, since for hover its gonna time out.
        self._reward[self._done & (crashed | lost)] = -50.0
        self._reward[self._done & timeout & ~complete] = 0.0
        self._reward[self._done & complete] = 100.0

        # per-timestep reward, distance uses potential-based reward shaping
        # distance towards closest cell is merely a guideline for navigating cells
        # note that the main reward stems from maximizing certainty in a cell.
        # https://arxiv.org/pdf/2402.07411
        dist = np.linalg.norm(self._obs[:, 0:3], axis=1)
        self._reward[~self._done] = (
            self._info_gain[~self._done] +
            0.1 * (self._prev_dist[~self._done] - self.gamma * dist[~self._done]) +
            #-0.1 * np.linalg.norm(self._obs[~self._done, 0:3], axis=1)
            -0.001 * np.linalg.norm(self._obs[~self._done, 12:15], axis=1) +
            -0.001 * np.linalg.norm(self._obs[~self._done, 15:18], axis=1) +
            -0.01 * np.linalg.norm(action[~self._done] - self._prev_action[~self._done], axis=1)
        )

    def _compute_observations(self, action: np.ndarray, drone_state: tuple, i: list[int] = None):
        """TODO: setup observations for n amount of waypoints.

        Remember it should be vision, IMU estimates that could happen too.

        For a first start tho its fine.
        """
        if i is None:
            i = np.arange(0, self.num_envs)

        ### update all drone state observations ###
        self._prev_dist[i] = np.linalg.norm(self._obs[i, 0:3], axis=1)
        xyz, lin_vel, ang_vel, R  = drone_state
        self._drone_pos[i] = xyz.numpy()[i]
        self._obs[i, 0:3] = self._cur_cell[i] - self._drone_pos[i]
        self._obs[i, 3:12] = R.reshape(-1, 9).numpy()[i]
        self._obs[i, 12:15] = lin_vel.numpy()[i]
        self._obs[i, 15:18] = ang_vel.numpy()[i]
        self._obs[i, 18:22] = action[i]
                
        ### update all grid-based observations
        for j in i:
            c, g = self._cameras[j], self._grids[j]
            xyz, R = self._drone_pos[j], self._obs[j, 3:12].reshape(3,3)
            
            # get top 8 obstacles based on distance
            seen = g.obj_xy[self._visible_objs[j]]
            order = np.argsort(np.linalg.norm(seen - xyz[:2], axis=1))[:5]
            rel = np.column_stack([seen[order] - xyz[:2], np.zeros(len(order))]) @ R
            n = len(order)
            self._k_obstacles[j, :n] = np.column_stack([rel[:, :2], np.full(n, 0.5)])
            self._k_obstacles[j, n:] = 0
            
            # get top 5 lowest confidence cells, tiebreaker is dist
            conf = np.where(g.inside_cells, self._confs[j], np.inf)
            dist = np.hypot(g.cell_cx - xyz[0], g.cell_cy - xyz[1])
            order = np.lexsort((dist.ravel(), conf.ravel()))[:8]
            rows, cols = np.unravel_index(order, conf.shape)

            # stack sorted cell rows/cols and put them into drone-relative frames (forward, right)
            cx, cy = g.cell_cx[rows, cols], g.cell_cy[rows, cols]
            rel = np.column_stack([cx - xyz[0], cy - xyz[1], np.zeros(8)])
            rel = rel @ R
            self._k_confidence_cells[j] = np.column_stack([rel[:, :2], self._confs[j][rows, cols]])
            
            cur_r, cur_c = g.get_cur_cell(*self._cur_cell[j][:2])[:2]
            done_with_it = self._confs[j][cur_r, cur_c] > 0.95
            arrived = np.linalg.norm(self._cur_cell[j][:2] - xyz[:2]) < g.cell_size
            if done_with_it or arrived:
                self._cur_cell[j] = np.array([cx[0], cy[0], -self.survey_alt], dtype=np.float32)
        
        self._obs[i, 22:37] = self._k_obstacles[i].reshape(len(i), -1)
        self._obs[i, 37:61] = self._k_confidence_cells[i].reshape(len(i), -1)
        
    def _update_detections_grid(self, drone_state, i: list[int] = None):
        """Update grid certainty levels with any detected objects in camera's frustum."""
        if i is None:
            i = np.arange(0, self.num_envs)

        ### step through each grid/camera, find visible objects and update certainties
        xyz_t, _, _, R_t = drone_state
        for j in i:
            c, g = self._cameras[j], self._grids[j]
            xyz, R = xyz_t[j].numpy(), R_t[j].numpy()
            self._visible_objs[j] = c.visible(xyz, R, g.obj_xy)
            est, _ = c.measure(xyz, R, g.obj_xy[self._visible_objs[j]])
            # get all detected cells
            rows, cols, valid = g.get_cur_cell(*est.T)
            rr, cc = rows[valid], cols[valid]

            # get all non-detected cells
            seen = c.visible(xyz, R, g.cell_centers)
            seen[rr * g.n_cols + cc] = True  # center falls outside frustum
            sr, sc = np.unravel_index(np.nonzero(seen)[0], (g.n_rows, g.n_cols))
            
            # set z = det or not, then get distance of obj from cam and update certainties
            det = np.zeros((g.n_rows, g.n_cols), dtype=bool)
            det[rr, cc] = True
            z = det[sr, sc].astype(np.float32)
            rng_cells = np.hypot(g.cell_cx[sr, sc] - xyz[0], g.cell_cy[sr, sc] - xyz[1])
            R_meas = 1 + 3 * (rng_cells / 20.0) ** 2
            
            c_before = g.confidence[sr, sc].copy()
            g.update_certainties(sr, sc, detected=z, noise=R_meas)
            self._confs[j] = g.confidence
            self._info_gain[j] = (self._confs[j][sr, sc] - c_before).sum()

    def step_async(self, actions):
        self._async_actions = actions

    def step_wait(self):
        return self.step(self._async_actions)

    def get_attr(self, attr_name, indices=None):
        num = len(indices) if indices is not None else self.num_envs
        if hasattr(self, attr_name):
            return [getattr(self, attr_name)] * num
        return [None] * num

    def set_attr(self, attr_name, val, indices=None):
        raise RuntimeError('This method is not implemented')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise RuntimeError('This method is not implemented')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def close(self):
        print("closed")

    @property
    def render_mode(self):
        return None

    @property
    def drone_pos(self):
        return self._drone_pos.copy()
    
if __name__ == "__main__":
    env = DroneEnv(
        init_polygon=np.array([[0,0],[30,5],[25,20],[5,15]], dtype=np.float32),
        num_envs=1
    )
