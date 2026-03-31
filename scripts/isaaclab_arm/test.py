"""Demo / evaluation script for the trained pGraph arm reach policy.

Usage:
    # Interactive demo (renders Isaac Sim window)
    conda run -n env_isaaclab python scripts/isaaclab_arm/test.py \\
        --robot franka --checkpoint logs/pgraph_arm/franka/<run>/model_XXXX.pt

    # Headless evaluation (prints success rate)
    conda run -n env_isaaclab python scripts/isaaclab_arm/test.py \\
        --robot franka --headless --episodes 50

AppLauncher MUST be invoked before any Isaac Lab imports.
"""

import argparse
import os
import sys

# Ensure project root is on sys.path so 'scripts.*' imports work after AppLauncher
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demo / eval pGraph arm reach policy.")
parser.add_argument("--robot", type=str, default="franka",
                    choices=["franka", "ur10", "kinova_gen3"])
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to .pt checkpoint. If None, uses latest in logs/.")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# headless=False → renders Isaac Sim window for demo
# headless=True  → fast headless eval

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-AppLauncher imports ───────────────────────────────────────────────────
import os
import glob
import time
import torch
import gymnasium as gym
import numpy as np
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

import scripts.isaaclab_arm  # noqa: F401  (registers gym envs)
from scripts.isaaclab_arm.policy import PGraphTransformerActorCritic

ROBOT_ENV_MAP = {
    "franka": "Isaac-PGraph-Reach-Franka-Play-v0",
    "ur10": "Isaac-PGraph-Reach-UR10-Play-v0",
    "kinova_gen3": "Isaac-PGraph-Reach-KinovaGen3-Play-v0",
}


def find_latest_checkpoint(robot: str) -> str | None:
    """Find the most recent checkpoint in logs/pgraph_arm/<robot>/."""
    pattern = os.path.join("logs", "pgraph_arm", robot, "*", "model_*.pt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_policy(checkpoint: str, num_actions: int, device: str) -> PGraphTransformerActorCritic:
    """Load policy from checkpoint."""
    obs_dummy = TensorDict(
        {"policy": torch.zeros(1, 78)},  # OBS_DIM=78
        batch_size=[1],
    )
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = PGraphTransformerActorCritic(obs_dummy, obs_groups, num_actions)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    # rsl_rl saves {"model_state_dict": ...} or raw state dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # Extract actor_critic weights if wrapped
    ac_state = {k.replace("actor_critic.", ""): v for k, v in state.items()
                if k.startswith("actor_critic.")} or state
    policy.load_state_dict(ac_state, strict=False)
    policy.to(device).eval()
    return policy


def run_demo(robot: str, checkpoint: str | None, num_envs: int, n_episodes: int, device: str):
    env_id = ROBOT_ENV_MAP[robot]
    env_cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device

    render_mode = None if args_cli.headless else "human"
    env = gym.make(env_id, cfg=env_cfg, render_mode=render_mode)
    vec_env = RslRlVecEnvWrapper(env)

    num_actions = vec_env.num_actions
    print(f"[INFO] robot={robot}  num_actions={num_actions}  num_envs={num_envs}")

    if checkpoint is None:
        checkpoint = find_latest_checkpoint(robot)
    if checkpoint is None:
        print("[WARN] No checkpoint found — running with random policy.")
        policy = None
    else:
        print(f"[INFO] Loading checkpoint: {checkpoint}")
        policy = load_policy(checkpoint, num_actions, device)

    # ── Evaluation loop ────────────────────────────────────────────────────────
    pos_errors, ori_errors, successes = [], [], []
    ep = 0

    obs_td, _ = vec_env.get_observations(), None
    dones = torch.zeros(num_envs, dtype=torch.bool, device=device)

    while ep < n_episodes:
        if policy is not None:
            with torch.no_grad():
                actions = policy.act_inference(obs_td)
        else:
            actions = torch.randn(num_envs, num_actions, device=device) * 0.1

        # RslRlVecEnvWrapper.step returns (obs, rew, dones, infos)
        obs_td, rewards, dones, infos = vec_env.step(actions)

        # Log per-episode stats when any env resets
        if dones.any():
            ep += int(dones.sum())
            print(f"  Episodes completed: {ep}/{n_episodes}", end="\r")
            # rsl_rl packs episode metrics into extras["log"]
            log = infos.get("log", {})
            for k, v in log.items():
                if "position_error" in k and v is not None:
                    pos_errors.append(float(v))
                elif "orientation_error" in k and v is not None:
                    ori_errors.append(float(v))

    print()
    print(f"\n{'='*50}")
    print(f"Robot: {robot.upper()}")
    print(f"Checkpoint: {checkpoint or 'random'}")
    print(f"Episodes: {ep}")
    if pos_errors:
        print(f"Pos error  (mean): {np.mean(pos_errors):.4f} m")
    if ori_errors:
        print(f"Ori error  (mean): {np.mean(ori_errors)*180/np.pi:.2f} deg")
    if successes:
        print(f"Success rate:      {np.mean(successes)*100:.1f}%")
    print(f"{'='*50}")

    vec_env.close()


if __name__ == "__main__":
    run_demo(
        robot=args_cli.robot,
        checkpoint=args_cli.checkpoint,
        num_envs=args_cli.num_envs,
        n_episodes=args_cli.episodes,
        device=args_cli.device if hasattr(args_cli, "device") else "cuda:0",
    )
    simulation_app.close()
