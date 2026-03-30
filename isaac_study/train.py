"""Script to train UR5 reach task with RSL-RL PPO.

Usage:
    # Headless training with 4096 parallel envs (default)
    python train.py --task Isaac-Reach-UR5-v0 --headless --num_envs 4096

    # Training with GUI for visualization
    python train.py --task Isaac-Reach-UR5-v0 --num_envs 2048

    # Record video during training
    python train.py --task Isaac-Reach-UR5-v0 --headless --num_envs 4096 --video

    # Custom iterations
    python train.py --task Isaac-Reach-UR5-v0 --headless --max_iterations 3000
"""

import argparse
import sys
import os

# ── 1. Parse arguments & launch Isaac Sim ─────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train UR5 reach task with RSL-RL PPO.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR5-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Max training iterations (overrides agent cfg).")

# Append AppLauncher CLI args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras if recording video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Imports AFTER simulator launch ─────────────────────────────────
import gymnasium as gym
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register our custom gym environments
# This import triggers the gym.register() calls in ur5_reach/__init__.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ur5_reach  # noqa: F401

# Import the agent config
from ur5_reach.agents.rsl_rl_ppo_cfg import UR5ReachPPORunnerCfg


def main():
    """Train UR5 reach with RSL-RL PPO."""

    # ── 3. Configure environment and agent ────────────────────────────
    # Create environment through gym (this uses our registered entry point)
    env_cfg_entry = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]

    # Import the config class dynamically
    module_name, class_name = env_cfg_entry.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_name)
    EnvCfgClass = getattr(module, class_name)
    env_cfg = EnvCfgClass()

    # Override num_envs from CLI
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Agent configuration
    agent_cfg = UR5ReachPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # ── 4. Setup logging ──────────────────────────────────────────────
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # ── 5. Create environment ─────────────────────────────────────────
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ── 6. Create RSL-RL runner and train ─────────────────────────────
    # Wrap with RSL-RL vectorized env wrapper
    env = RslRlVecEnvWrapper(env)

    # Create PPO runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Dump configs for reproducibility
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ── 7. Run training! ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  UR5 Reach — RSL-RL PPO Training")
    print(f"  Envs: {args_cli.num_envs} | Iterations: {agent_cfg.max_iterations}")
    print(f"  Device: {agent_cfg.device}")
    print(f"{'='*60}\n")

    import time
    start_time = time.time()

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Training completed in {elapsed:.1f} seconds ({elapsed/60:.1f} min)")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
