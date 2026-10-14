from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import argparse
from pathlib import Path
import torch
import uuid
from glob import glob

from environment.env import DroneEnv
from drone.drone_logger import DroneLogger

import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0,
                        help="random seed")
    parser.add_argument('--w', type=str, required=True,
                        help="PPO weights")
    parser.add_argument('--wn', type=str, required=True,
                        help="VecNormalize weights")
    parser.add_argument('--rollouts', type=int, default=1,
                        help="number of rollouts to perform")
    return parser.parse_args()

def run_rollouts(model, env, save_path: Path, n_rollouts: int = 10):
    """Run n amount of rollouts of the given environment and policy."""
    max_ep_len = env.ep_limit
    for n_roll in range(n_rollouts):
        g = env.venv._grids[0]
        traj = np.zeros((max_ep_len, 3), dtype=np.float32)
        conf = np.zeros((max_ep_len, g.n_rows, g.n_cols), dtype=np.float32)
        obs, done, ep_len = env.reset(), False, 0
        total_rew = 0.0

        ### run a rollout unti termination ###
        while not (done or (ep_len >= max_ep_len)):
            act, _ = model.predict(obs, deterministic=False)
            obs, rew ,done, infos = env.step(act)

            traj[ep_len] = infos[0]["terminal_global_observation"] if done[0] else env.drone_pos[0]
            conf[ep_len] = infos[0]["terminal_confidence"] if done[0] else g.confidence
            ep_len += 1
            total_rew += rew

        xe, ye = [], []
        if done[0]:
            xe = g.xmin + np.arange(g.n_cols + 1) * g.cell_size
            ye = g.ymin + np.arange(g.n_rows + 1) * g.cell_size

        n_save_path = str(save_path / f"rollout_{n_roll:03d}.npz")
        np.savez_compressed(n_save_path, traj = traj[:ep_len], conf = conf[:ep_len], xe=xe, ye=ye)
        print(f"Episode ended: step={ep_len}. rew={total_rew.item()}\n\n")
        print(f"Reason for termination: {infos[0]['termination_reason']}")
        print(f"Final obs: {infos[0]['terminal_observation']}")
        

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ### load weights / normalized weights / save_path ###
    w_path = Path(__file__).resolve().parent / args.w
    wn_path = Path(__file__).resolve().parent / args.wn
    save_path = w_path.with_suffix("") / str(uuid.uuid4())
    save_path.mkdir(parents=True, exist_ok=True)

    ### run inference using loaded env, model and gather stats ###
    init_poly = np.array([[0,0],[58,8],[52,42],[6,34]], dtype=np.float32)
    env = VecNormalize.load(
        wn_path, 
        DroneEnv(
            init_polygon=init_poly, 
            num_envs=1, 
            num_objs=40
        )
    )
    env.training = False
    env.norm_reward = False
    model = PPO.load(w_path, env=env)
    run_rollouts(model, env, save_path, n_rollouts=args.rollouts)

    ### plot all logged data to save path ###
    data_keys = ["traj", "conf", "xe", "ye"]
    data = [fn for fn in glob(f"{str(save_path)}/rollout_*")]
    logger = DroneLogger(sim_dt=env.sim_dt, save_path=save_path)
    logger.load_data(data, data_keys)
    logger.log_trajectories()
    logger.log_conf_map(polygon=init_poly)
    logger.log_video(polygon=init_poly)
    
if __name__ == "__main__":
    main()