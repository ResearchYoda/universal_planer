"""
Train a PPO agent with a Rodrigues Network (RodriNet) actor on HalfCheetah-v5.

Usage:
    python train.py [--timesteps 2000000] [--n-envs 8] [--device auto]
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from rodrigues_network.envs import HALFCHEETAH_TREE, UprightWrapper
from rodrigues_network.policy import RodriNetExtractor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=2_000_000)
    p.add_argument('--n-envs',    type=int, default=8)
    p.add_argument('--device',    type=str, default='auto')
    p.add_argument('--save-dir',  type=str, default='checkpoints')
    p.add_argument('--log-dir',   type=str, default='logs')
    # RodriNet hyperparameters
    p.add_argument('--C-L',      type=int, default=4,  help='Link feature channels')
    p.add_argument('--C-J',      type=int, default=8,  help='Joint feature dim')
    p.add_argument('--d-model',  type=int, default=64, help='Attention model dim')
    p.add_argument('--n-heads',  type=int, default=4,  help='Attention heads')
    p.add_argument('--n-blocks', type=int, default=3,  help='RodriNet depth')
    return p.parse_args()


def main():
    args = parse_args()
    env_id = HALFCHEETAH_TREE['env_id']

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    train_env = make_vec_env(env_id, n_envs=args.n_envs,
                             wrapper_class=UprightWrapper)
    eval_env  = make_vec_env(env_id, n_envs=4,
                             wrapper_class=UprightWrapper)

    policy_kwargs = dict(
        features_extractor_class=RodriNetExtractor,
        features_extractor_kwargs=dict(
            C_L=args.C_L,
            C_J=args.C_J,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            env_tree=HALFCHEETAH_TREE,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        'MlpPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=args.log_dir,
        device=args.device,
        verbose=1,
    )

    n_params = sum(p.numel() for p in model.policy.parameters())
    n_rodri  = sum(p.numel() for p in model.policy.features_extractor.parameters())
    print(f"\n{'='*60}")
    print(f"  Environment : {env_id}")
    print(f"  Total params: {n_params:,}")
    print(f"  RodriNet    : {n_rodri:,} params")
    print(f"  Training for {args.timesteps:,} steps on {args.n_envs} envs")
    print(f"{'='*60}\n")

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=args.save_dir,
            log_path=args.log_dir,
            eval_freq=max(50_000 // args.n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(200_000 // args.n_envs, 1),
            save_path=args.save_dir,
            name_prefix='rodrinet_halfcheetah',
            verbose=0,
        ),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    final_path = os.path.join(args.save_dir, 'rodrinet_halfcheetah_upright_final')
    model.save(final_path)
    print(f"\nTraining complete. Model saved to: {final_path}.zip")


if __name__ == '__main__':
    main()
