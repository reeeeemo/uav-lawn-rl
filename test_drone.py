from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import argparse
from pathlib import Path
import torch

from env import DroneEnv

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

def run_rollouts(model, env, n_rollouts: int = 10):
    max_ep_len = env.ep_limit
    for _ in range(n_rollouts):
        obs, done, ep_len = env.reset(), False, 0
        total_rew = 0.0
        while not (done or (ep_len >= max_ep_len)):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew ,done, infos = env.step(act)
            ep_len += 1
            total_rew += rew
        print(f"Episode ended: step={ep_len}. rew={total_rew.item()}\n\n")
        print(f"Reason for termination: {infos[0]['termination_reason']}")
        print(f"Final obs: {infos[0]['terminal_observation']}")

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ### load weights / normalized weights ###
    w_path = Path(__file__).resolve().parent / args.w
    wn_path = Path(__file__).resolve().parent / args.wn

    env = VecNormalize.load(wn_path, DroneEnv(1))
    env.training = False
    env.norm_reward = False
    model = PPO.load(w_path, env=env)

    run_rollouts(model, env, n_rollouts=args.rollouts)

if __name__ == "__main__":
    main()