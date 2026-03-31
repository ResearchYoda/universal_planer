"""Train pGraph universal arm reach policy with rsl_rl PPO.

Usage (single robot):
    cd /home/berk/VS_Projects/ICHORA_26/universal_planer
    conda run -n env_isaaclab python scripts/isaaclab_arm/train.py \\
        --robot franka --num_envs 1024 --iterations 1500

Usage (multi-robot sequential — trains one universal policy on all robots):
    conda run -n env_isaaclab python scripts/isaaclab_arm/train.py \\
        --robot franka ur10 --num_envs 512 --iterations 1500

AppLauncher MUST be launched before any Isaac Lab imports.
"""

import argparse
import os
import sys

# Ensure project root is on sys.path so 'scripts.*' imports work after AppLauncher
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 1. AppLauncher (must be first) ─────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train pGraph arm reach policy.")
parser.add_argument(
    "--robot",
    nargs="+",
    default=["franka"],
    choices=["franka", "ur10", "kinova_gen3"],
    help="Robot(s) to train on. Multiple robots → universal sequential training.",
)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=1500)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# Force headless for training
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Remaining imports (after AppLauncher) ───────────────────────────────────
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register our gym envs
import scripts.isaaclab_arm  # noqa: F401

# Inject our custom policy class into rsl_rl's eval namespace so
# OnPolicyRunner._construct_algorithm can find it via eval("PGraphTransformerActorCritic")
import rsl_rl.runners.on_policy_runner as _opr_module
from scripts.isaaclab_arm.policy import PGraphTransformerActorCritic
_opr_module.PGraphTransformerActorCritic = PGraphTransformerActorCritic  # type: ignore[attr-defined]

from scripts.isaaclab_arm.agent_cfg import UniversalArmPPORunnerCfg


# ── Robot → gym env id mapping ─────────────────────────────────────────────────
ROBOT_ENV_MAP = {
    "franka": "Isaac-PGraph-Reach-Franka-v0",
    "ur10": "Isaac-PGraph-Reach-UR10-v0",
    "kinova_gen3": "Isaac-PGraph-Reach-KinovaGen3-v0",
}


def make_env(robot: str, num_envs: int, device: str):
    """Create a vectorized Isaac Lab env for the given robot."""
    env_id = ROBOT_ENV_MAP[robot]
    from isaaclab_tasks.utils import load_cfg_from_registry
    env_cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    env = gym.make(env_id, cfg=env_cfg)
    return RslRlVecEnvWrapper(env)


def train_single(robot: str):
    """Train pGraph policy on a single robot."""
    log_dir = os.path.join(
        "logs", "pgraph_arm", robot,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")

    env = make_env(robot, args_cli.num_envs, args_cli.device)

    agent_cfg = UniversalArmPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.iterations
    agent_cfg.device = args_cli.device
    agent_cfg_dict = agent_cfg.to_dict()

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=args_cli.device)
    t0 = time.time()
    runner.learn(num_learning_iterations=args_cli.iterations, init_at_random_ep_len=True)
    print(f"[INFO] Training done in {time.time()-t0:.1f}s")

    env.close()
    return runner


def train_multi(robots: list[str]):
    """Universal training: round-robin across multiple robots, shared policy.

    Strategy:
        Repeat N times:
            for each robot:
                collect args_cli.iterations // (N * len(robots)) steps
                update shared policy
    """
    ROUNDS = 3
    iters_per_robot = max(1, args_cli.iterations // (ROUNDS * len(robots)))
    print(f"[INFO] Multi-robot training: {robots}")
    print(f"[INFO] {ROUNDS} rounds × {len(robots)} robots × {iters_per_robot} iters each")

    log_dir = os.path.join(
        "logs", "pgraph_arm", "_".join(robots),
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create all envs
    envs = {r: make_env(r, args_cli.num_envs, args_cli.device) for r in robots}

    # Build runner on first robot
    agent_cfg = UniversalArmPPORunnerCfg()
    agent_cfg.max_iterations = iters_per_robot
    agent_cfg.device = args_cli.device
    agent_cfg_dict = agent_cfg.to_dict()

    first_env = envs[robots[0]]
    runner = OnPolicyRunner(first_env, agent_cfg_dict, log_dir=log_dir, device=args_cli.device)

    t0 = time.time()
    for rnd in range(ROUNDS):
        for robot in robots:
            print(f"[INFO] Round {rnd+1}/{ROUNDS} — robot: {robot}")
            # Swap env in runner
            runner.env = envs[robot]
            runner.env.env.reset()
            runner.alg.init_storage(
                "rl",
                envs[robot].num_envs,
                runner.num_steps_per_env,
                envs[robot].get_observations(),
                [envs[robot].num_actions],
            )
            runner.learn(num_learning_iterations=iters_per_robot, init_at_random_ep_len=True)

    print(f"[INFO] Multi-robot training done in {time.time()-t0:.1f}s")
    for env in envs.values():
        env.close()
    return runner


def main():
    if len(args_cli.robot) == 1:
        train_single(args_cli.robot[0])
    else:
        train_multi(args_cli.robot)


if __name__ == "__main__":
    main()
    simulation_app.close()
