"""Script to play/evaluate a trained UR5 reach policy.

Usage:
    # Play with default checkpoint (latest)
    python play.py --task Isaac-Reach-UR5-Play-v0 --num_envs 50

    # Play a specific checkpoint
    python play.py --task Isaac-Reach-UR5-Play-v0 --checkpoint logs/rsl_rl/ur5_reach/<run>/model_1500.pt
"""

import argparse
import sys
import os

# ── 1. Parse arguments & launch Isaac Sim ─────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained UR5 reach policy.")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments to visualize.")
parser.add_argument("--task", type=str, default="Isaac-Reach-UR5-Play-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator (NOT headless — we want to see it!)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Imports AFTER simulator launch ─────────────────────────────────
import gymnasium as gym
import torch
import glob

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register our custom gym environments
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ur5_reach  # noqa: F401

from ur5_reach.agents.rsl_rl_ppo_cfg import UR5ReachPPORunnerCfg


def find_latest_checkpoint(log_root: str) -> str:
    """Find the latest checkpoint from the most recent training run."""
    # Find all run directories
    run_dirs = sorted(glob.glob(os.path.join(log_root, "*")))
    if not run_dirs:
        raise FileNotFoundError(f"No training runs found in {log_root}")

    # Get the latest run
    latest_run = run_dirs[-1]

    # Find the latest model checkpoint
    checkpoints = sorted(glob.glob(os.path.join(latest_run, "model_*.pt")))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {latest_run}")

    return checkpoints[-1]


def main():
    """Play trained UR5 reach policy."""

    # ── 3. Configure environment ──────────────────────────────────────
    import importlib
    env_cfg_entry = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    module_name, class_name = env_cfg_entry.rsplit(":", 1)
    module = importlib.import_module(module_name)
    EnvCfgClass = getattr(module, class_name)
    env_cfg = EnvCfgClass()

    env_cfg.scene.num_envs = args_cli.num_envs

    # Agent config
    agent_cfg = UR5ReachPPORunnerCfg()

    # ── 4. Find checkpoint ────────────────────────────────────────────
    if args_cli.checkpoint is not None:
        resume_path = args_cli.checkpoint
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = find_latest_checkpoint(log_root_path)

    print(f"[INFO] Loading checkpoint from: {resume_path}")

    # ── 5. Create environment ─────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # ── 6. Create runner and load policy ──────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # Get the trained policy
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # ── 7. Run evaluation loop ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  UR5 Reach — Policy Evaluation")
    print(f"  Envs: {args_cli.num_envs}")
    print(f"  Checkpoint: {os.path.basename(resume_path)}")
    print(f"{'='*60}\n")

    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get actions from the policy (pass full TensorDict — policy indexes it internally)
            actions = policy(obs)
            # Step the environment
            obs, _, _, _ = env.step(actions)

    # Close
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
