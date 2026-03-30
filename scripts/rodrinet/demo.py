"""
Demo: watch the trained RodriNet policy run on HalfCheetah-v5.

Usage:
    # Live rendering (requires a display):
    python demo.py --model checkpoints/best_model

    # Record a video (no display needed):
    python demo.py --model checkpoints/best_model --record

    # Use the final checkpoint instead of best:
    python demo.py --model checkpoints/rodrinet_halfcheetah_final
"""

import argparse
import os

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from rodrigues_network.envs import HALFCHEETAH_TREE, UprightWrapper


def run_demo(model_path: str, n_episodes: int = 5, record: bool = False):
    env_id = HALFCHEETAH_TREE['env_id']

    if record:
        os.makedirs('videos', exist_ok=True)
        env = gym.make(env_id, render_mode='rgb_array')
        env = UprightWrapper(env)
        env = gym.wrappers.RecordVideo(
            env, 'videos',
            episode_trigger=lambda _: True,
            name_prefix='rodrinet_demo',
        )
        print("Recording to videos/")
    else:
        env = gym.make(env_id, render_mode='human')
        env = UprightWrapper(env)
        print("Live render mode")

    model = PPO.load(model_path, env=env)
    print(f"Loaded model: {model_path}")

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        print(f"  Episode {ep + 1:2d}: reward = {ep_reward:8.1f}  ({steps} steps)")

    env.close()
    print(f"\nMean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    if record:
        print("Videos saved to: videos/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, default='checkpoints/best_model',
                   help='Path to saved model (without .zip)')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--record',   action='store_true',
                   help='Record video instead of live render')
    args = p.parse_args()

    if not os.path.exists(args.model + '.zip'):
        # Try to find the best or final model
        candidates = [
            'checkpoints/best_model',
            'checkpoints/rodrinet_halfcheetah_final',
        ]
        for c in candidates:
            if os.path.exists(c + '.zip'):
                args.model = c
                print(f"Auto-detected model: {c}")
                break
        else:
            print("No trained model found. Run `python train.py` first.")
            return

    run_demo(args.model, args.episodes, args.record)


if __name__ == '__main__':
    main()
