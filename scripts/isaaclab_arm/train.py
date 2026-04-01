"""Train pGraph universal arm reach policy with rsl_rl PPO.

Usage (single robot):
    cd /home/berk/VS_Projects/ICHORA_26/universal_planer
    conda run -n env_isaaclab python scripts/isaaclab_arm/train.py \\
        --robot franka --num_envs 1024 --iterations 1500

Usage (multi-robot curriculum — round-robin with shared policy, one robot at a time):
    conda run -n env_isaaclab python scripts/isaaclab_arm/train.py \\
        --robot franka ur10 --num_envs 1024 --iterations 1500

    Isaac Lab allows only one SimulationContext per process, so robots are trained
    sequentially. Policy weights are preserved between env swaps; every mini-batch
    uses experience from the current robot only.  Over multiple rounds the policy
    learns to generalise across morphologies.

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
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    metavar="CHECKPOINT",
    help="Load policy weights from this .pt file before training (for curriculum fine-tuning).",
)
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
from scripts.isaaclab_arm.pgraph import MAX_JOINTS


# ── Robot → gym env id mapping ─────────────────────────────────────────────────
ROBOT_ENV_MAP = {
    "franka": "Isaac-PGraph-Reach-Franka-v0",
    "ur10": "Isaac-PGraph-Reach-UR10-v0",
    "kinova_gen3": "Isaac-PGraph-Reach-KinovaGen3-v0",
}


def make_env(robot: str, num_envs: int, device: str) -> RslRlVecEnvWrapper:
    """Create a vectorized Isaac Lab env for the given robot."""
    env_id = ROBOT_ENV_MAP[robot]
    from isaaclab_tasks.utils import load_cfg_from_registry
    env_cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    env = gym.make(env_id, cfg=env_cfg)
    return RslRlVecEnvWrapper(env)


class _UniversalEnvWrapper:
    """Thin wrapper that overrides num_actions → MAX_JOINTS.

    Isaac Lab only allows one SimulationContext per process, so multi-robot
    training must be sequential (one env at a time).  This wrapper ensures
    every env always reports num_actions=MAX_JOINTS so the OnPolicyRunner
    creates one shared policy with a consistent action space.

    The actual robot DOF slice is applied inside step() before forwarding
    actions to the real env.
    """

    def __init__(self, env: RslRlVecEnvWrapper):
        self._env = env
        self._real_num_actions = env.num_actions
        # Expose unified action dim to OnPolicyRunner
        self.num_actions = MAX_JOINTS
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length

    @property
    def cfg(self):
        return self._env.cfg

    @property
    def episode_length_buf(self):
        return self._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self._env.episode_length_buf = value

    def get_observations(self):
        return self._env.get_observations()

    def step(self, actions):
        return self._env.step(actions[:, : self._real_num_actions])

    def reset(self):
        return self._env.reset()

    def close(self):
        self._env.close()


def train_single(robot: str):
    """Train pGraph policy on a single robot.

    If --resume is given, load policy weights from that checkpoint first.
    This enables curriculum fine-tuning:
        Step 1:  train.py --robot franka --iterations 1500
        Step 2:  train.py --robot ur10   --iterations 1500 --resume <franka_ckpt>
        Step 3:  train.py --robot franka --iterations 750  --resume <ur10_ckpt>
        ...
    """
    log_dir = os.path.join(
        "logs", "pgraph_arm", robot,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")

    env = _UniversalEnvWrapper(make_env(robot, args_cli.num_envs, args_cli.device))
    print(f"[INFO] env.num_actions={env.num_actions}  robot_real_dof={env._real_num_actions}")

    agent_cfg = UniversalArmPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.iterations
    agent_cfg.device = args_cli.device
    agent_cfg_dict = agent_cfg.to_dict()

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=args_cli.device)

    if args_cli.resume:
        print(f"[INFO] Resuming from: {args_cli.resume}")
        state = torch.load(args_cli.resume, map_location=args_cli.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        ac_state = {k.replace("actor_critic.", ""): v for k, v in state.items()
                    if k.startswith("actor_critic.")} or state
        # Resize _log_std to MAX_JOINTS if checkpoint used a different DOF
        if "_log_std" in ac_state:
            ckpt_dim = ac_state["_log_std"].shape[0]
            if ckpt_dim != MAX_JOINTS:
                new_std = ac_state["_log_std"].mean().expand(MAX_JOINTS).clone()
                new_std[:ckpt_dim] = ac_state["_log_std"]
                ac_state["_log_std"] = new_std
                print(f"[INFO] _log_std resized {ckpt_dim}→{MAX_JOINTS}")
        runner.alg.policy.load_state_dict(ac_state, strict=False)
        print("[INFO] Weights loaded.")

    t0 = time.time()
    runner.learn(num_learning_iterations=args_cli.iterations, init_at_random_ep_len=True)
    print(f"[INFO] Training done in {time.time()-t0:.1f}s")

    env.close()
    return runner


def train_multi(robots: list[str]):
    """Universal training: round-robin curriculum with one env at a time.

    Isaac Lab allows only one SimulationContext per process, so envs are
    created, trained, and closed one at a time.  Policy weights survive
    across env swaps via state_dict save/load.

    All envs are wrapped with _UniversalEnvWrapper so the runner always
    sees num_actions=MAX_JOINTS regardless of robot DOF.

    Strategy (ROUNDS=3):
        for rnd in 0..ROUNDS-1:
            for robot in robots:
                create env → (optionally load saved weights) → train N iters
                save policy weights → close env
    """
    ROUNDS = 3
    iters_per_slot = max(1, args_cli.iterations // (ROUNDS * len(robots)))
    print(f"[INFO] Multi-robot curriculum: {robots}")
    print(f"[INFO] {ROUNDS} rounds × {len(robots)} robots × {iters_per_slot} iters/slot")

    log_dir = os.path.join(
        "logs", "pgraph_arm", "_".join(robots),
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)

    saved_weights: dict | None = None   # policy state_dict carried between slots
    saved_opt: dict | None = None       # optimizer state (optional)
    t0 = time.time()
    runner = None

    for rnd in range(ROUNDS):
        for robot in robots:
            print(f"\n[INFO] ── Round {rnd+1}/{ROUNDS}  robot={robot} ──")

            env = _UniversalEnvWrapper(make_env(robot, args_cli.num_envs, args_cli.device))
            print(f"[INFO] env.num_actions={env.num_actions}  (robot real DOF={env._real_num_actions})")

            agent_cfg = UniversalArmPPORunnerCfg()
            agent_cfg.max_iterations = iters_per_slot
            agent_cfg.device = args_cli.device
            agent_cfg_dict = agent_cfg.to_dict()

            runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=args_cli.device)

            # Restore weights from previous slot
            if saved_weights is not None:
                runner.alg.policy.load_state_dict(saved_weights, strict=False)
                if saved_opt is not None:
                    try:
                        runner.alg.optimizer.load_state_dict(saved_opt)
                    except Exception:
                        pass  # shape mismatch between robots is OK — fresh optimizer
                print("[INFO] Loaded policy weights from previous slot.")

            runner.learn(num_learning_iterations=iters_per_slot, init_at_random_ep_len=True)

            # Preserve weights before tearing down this env
            saved_weights = {k: v.cpu().clone() for k, v in runner.alg.policy.state_dict().items()}
            saved_opt = runner.alg.optimizer.state_dict()

            env.close()

    print(f"\n[INFO] Multi-robot curriculum done in {time.time()-t0:.1f}s")
    return runner


def main():
    if len(args_cli.robot) == 1:
        train_single(args_cli.robot[0])
    else:
        print(
            "[ERROR] Isaac Lab does not support multiple SimulationContexts in one process.\n"
            "        Use --resume for curriculum training across robots:\n\n"
            "  Step 1 — Train Franka:\n"
            "    python train.py --robot franka --num_envs 1024 --iterations 1500\n\n"
            "  Step 2 — Fine-tune on UR10 (loads Franka weights):\n"
            "    python train.py --robot ur10 --num_envs 1024 --iterations 1500 \\\n"
            "        --resume logs/pgraph_arm/franka/<run>/model_1499.pt\n\n"
            "  Step 3 — Optional: repeat alternating for more rounds.\n"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
