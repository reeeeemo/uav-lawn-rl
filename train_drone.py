from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import argparse
from pathlib import Path
import torch
import uuid

from environment.env import DroneEnv

TOTAL_TIMESTEPS = int(6e7)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0,
                        help="random seed")
    parser.add_argument('--save_dir', type=str, default="./saved",
                        help="directory to save models to")
    parser.add_argument('--envs', type=int, default=100,
                        help="num of environments to run vectorized")

    return parser.parse_args()

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dir_path = Path(__file__).resolve().parent / args.save_dir / str(uuid.uuid4())
    dir_path.mkdir(parents=True, exist_ok=True)

    env = VecNormalize(DroneEnv(np.array([[0,0],[58,8],[52,42],[6,34]], dtype=np.float32), args.envs))

    model = PPO(
        tensorboard_log=str(dir_path),
        policy="MlpPolicy",
        policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                           net_arch=dict(pi=[64,64], vf=[64,64])),
        env=env,
        gae_lambda=0.95,
        gamma=0.999,
        n_steps=2048,
        ent_coef=0.003,
        learning_rate=1e-4,
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=512,
        n_epochs=10,
        clip_range=0.2,
        verbose=1,
        device="cpu",
        seed=args.seed     
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    model.save(str(dir_path))
    env.save(str(dir_path / "vec_normalize.pkl"))

if __name__ == "__main__":
    main()