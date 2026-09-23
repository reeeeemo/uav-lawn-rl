from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces
from drone.drone import Drone
import numpy as np
import torch

class DroneEnv(VecEnv):
    """Custom gymnasium environment that simulates a drone.

    Allows for multiple environments.
    """
    def __init__(self, num_envs: int = 10):
        self.wrapper = Drone(num_envs)

        ### PPO OBS / ACTION VARS ###
        self.obs_size = 22
        self.acts_size = 4
        self.num_envs = num_envs
        self.ep_limit = 500

        self._obs = np.zeros((self.num_envs, self.obs_size), dtype=np.float32)

        # [x_near_post, y_near_post, z_near_post]
        # 9d rot mat for [roll pitch yaw]
        # [x_vel, y_vel, z_vel]
        # [roll_vel, pitch_vel, yaw_vel]
        # [prev_thrust, prev_pitch, prev_yaw, prev_roll]
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
        self._cur_waypoint = np.zeros((self.num_envs,), dtype=np.int16)
        self._waypoints = np.zeros((self.num_envs, 3), dtype=np.float32) # change later to be dynamic to add whatever waypoints
        self._prev_action = np.zeros((self.num_envs, self.acts_size), dtype=np.float32)
        self._cur_episodes = np.zeros((self.num_envs,), dtype=np.int16)

        self.sim_dt = self.wrapper.dt

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
            info[k]["TimeLimit.truncated"] = self._done_reason[k] == "TimeLimit.truncated"
            info[k]["termination_reason"] = self._done_reason[k]

        done_arr = self._done.copy()
        self.reset(np.where(self._done)[0])

        return self._obs.copy(), self._reward.copy(), \
            done_arr, info

    def reset(self, i: list[int] = None):
        """Resets drone. Takes in a list of indices of envs to reset."""
        if i is None:
            i = np.arange(0, self.num_envs)
        drone_state = self.wrapper.reset(i)

        self._prev_action[i] = 0
        self._done[i] = np.zeros_like(self._done[i])
        for k in i:
            self._rewards[k] = []

        # temp for hover tests
        self._cur_waypoint[i] = np.zeros_like(self._cur_waypoint[i])
        self._waypoints[i] = np.full_like(self._waypoints[i], [0, 0, -5.0])

        self._compute_observations(np.zeros((len(i), 4), dtype=np.float32), drone_state)

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
        crashed = self._obs[:, 2] > 0
        self._done[crashed] = True
        for k in np.where(crashed)[0]:
            self._done_reason[k] = "terminated"

        # temp reward for done and not terminated at episode length, since for hover its gonna time out.
        self._reward[self._done & ~timeout] = -10
        self._reward[self._done & timeout] = 10
        self._reward[~self._done] = (
            -np.linalg.norm(self._waypoints[~self._done] - self._obs[~self._done, 0:3], axis=1) +
            -0.1 * np.linalg.norm(self._obs[~self._done, 12:15], axis=1) +
            -0.05 * np.linalg.norm(self._obs[~self._done, 15:18], axis=1) +
            -0.01 * np.linalg.norm(action[~self._done] - self._prev_action[~self._done], axis=1) +
            1.0
        )

    def _compute_observations(self, action: np.ndarray, drone_state: tuple, i: list[int] = None):
        """TODO: setup observations for n amount of waypoints.

        Remember it should be vision, IMU estimates that could happen too.

        For a first start tho its fine.
        """
        if i is None:
            i = np.arange(0, self.num_envs)
        xyz, lin_vel, ang_vel, R  = drone_state
        # dunno what im gonna do here yet....
        self._obs[i, 0:3] = xyz.numpy()[i]  # temp for now since this is global and bad
        self._obs[i, 3:12] = R.reshape(-1, 9).numpy()[i]
        self._obs[i, 12:15] = lin_vel.numpy()[i]
        self._obs[i, 15:18] = ang_vel.numpy()[i]
        self._obs[i, 18:22] = self._prev_action[i]


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
        return self._obs[:, 0:3].copy()