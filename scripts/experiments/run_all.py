"""
scripts/experiments/run_all.py
===============================
Run all 3 experiment types sequentially:
  1. Specialist policies   — 4 robots × 1 policy each (10M steps)
  2. No-Pgraph universal   — universal policy with pgraph zeroed (10M steps)
  3. Zero-shot transfer    — train on 3 robots, test on ant (10M steps)

Usage:
    python scripts/experiments/run_all.py
    python scripts/experiments/run_all.py --steps 5000000   # quick run
"""

import os, sys, time, argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.experiments.env_v2 import UniversalEnvV2, OBS_DIM, MAX_DOF, ROBOT_CONFIGS
from scripts.experiments.policy import MLPPolicy, PPOTrainer, RolloutBuffer, RunningNorm

CKPT_BASE = os.path.join(os.path.dirname(__file__), 'checkpoints')
RESULTS_BASE = os.path.join(os.path.dirname(__file__), 'results')

ALL_ROBOTS = list(ROBOT_CONFIGS.keys())   # hopper, halfcheetah, walker2d, ant


# ── Vectorized env helpers ─────────────────────────────────────────────────────
class VecEnv:
    """Synchronous vectorized env for one or more robots."""

    def __init__(self, robot_list, n_per_robot=1, zero_pgraph=False):
        self.envs = []
        self.robot_ids = []
        for robot in robot_list:
            for _ in range(n_per_robot):
                self.envs.append(UniversalEnvV2(robot, zero_pgraph=zero_pgraph))
                self.robot_ids.append(robot)
        self.n_envs = len(self.envs)

    def reset(self):
        return np.stack([env.reset()[0] for env in self.envs])

    def step(self, actions):
        obs_l, rew_l, done_l, info_l = [], [], [], []
        for env, act in zip(self.envs, actions):
            obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
            if done:
                obs, _ = env.reset()
            obs_l.append(obs); rew_l.append(rew)
            done_l.append(float(done)); info_l.append(info)
        return (np.stack(obs_l), np.array(rew_l, np.float32),
                np.array(done_l, np.float32), info_l)

    def close(self):
        for env in self.envs:
            env.close()


# ── Core training function ─────────────────────────────────────────────────────
def run_experiment(tag: str,
                   robot_list: list,
                   total_steps: int,
                   n_per_robot: int = 4,
                   n_steps: int = 2048,
                   zero_pgraph: bool = False,
                   device: str = 'auto'):
    """
    Train one experiment and save checkpoints + log to CKPT_BASE/<tag>/.
    Returns dict of per-robot episode rewards list (raw).
    """
    save_dir = os.path.join(CKPT_BASE, tag)
    os.makedirs(save_dir, exist_ok=True)

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)
    use_amp = (dev.type == 'cuda' and torch.cuda.is_bf16_supported())

    print(f"\n{'='*60}")
    print(f"  Experiment : {tag}")
    print(f"  Robots     : {robot_list}")
    print(f"  zero_pgraph: {zero_pgraph}")
    print(f"  total_steps: {total_steps:,}   device: {device}")
    print(f"{'='*60}\n")

    vec_env = VecEnv(robot_list, n_per_robot, zero_pgraph=zero_pgraph)
    n_envs  = vec_env.n_envs

    policy  = MLPPolicy(obs_dim=OBS_DIM, action_dim=MAX_DOF).to(dev)
    trainer = PPOTrainer(policy, lr=3e-4, n_epochs=10, batch_size=512,
                         clip=0.2, ent_coef=0.02, vf_coef=0.25,
                         max_grad=0.3, device=device)
    buffer  = RolloutBuffer(n_steps, n_envs, OBS_DIM, MAX_DOF, dev)

    obs_norm  = RunningNorm((OBS_DIM,))
    rew_norms = {r: RunningNorm((1,)) for r in robot_list}

    total_iters = total_steps // (n_steps * n_envs)
    # LR schedule: linear decay
    lr_init, lr_end = 3e-4, 3e-5
    def _lr_fn(t):
        prog = t / max(total_iters, 1)
        return max(1.0 - prog * (1 - lr_end / lr_init), lr_end / lr_init)
    scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, _lr_fn)
    trainer.set_scheduler(scheduler)

    log_ep_rew   = {r: [] for r in robot_list}
    log_pl, log_vl, log_ent, log_steps = [], [], [], []
    ep_rewards = np.zeros(n_envs)

    obs_np = vec_env.reset()
    obs_norm.update(obs_np)
    obs_t  = torch.tensor(obs_norm.normalize(obs_np), device=dev)

    global_step = 0
    t0 = time.time()

    for iteration in range(1, total_iters + 1):
        # Anneal entropy
        progress = iteration / total_iters
        trainer.ent_coef = 0.02 + (0.005 - 0.02) * progress

        policy.eval()
        with torch.no_grad():
            for _ in range(n_steps):
                with torch.autocast('cuda', torch.bfloat16, enabled=use_amp):
                    action_t, lp_t, val_t = policy.get_action(obs_t)
                action_t = action_t.float()
                lp_t     = lp_t.float()
                val_t    = val_t.float()

                actions_np = action_t.cpu().numpy()
                next_obs_np, rew_np, done_np, infos = vec_env.step(actions_np)

                # Per-robot reward normalisation
                rew_norm_np = np.zeros_like(rew_np)
                for i, robot in enumerate(vec_env.robot_ids):
                    rew_norms[robot].update(rew_np[i:i+1].reshape(-1, 1))
                    rew_norm_np[i] = np.clip(
                        rew_np[i] / np.sqrt(rew_norms[robot].var[0] + 1e-8),
                        -10.0, 10.0)

                buffer.add(obs_t, action_t, lp_t,
                           torch.tensor(rew_norm_np, device=dev),
                           val_t,
                           torch.tensor(done_np, device=dev))

                ep_rewards += rew_np
                for i, (done, info) in enumerate(zip(done_np, infos)):
                    if done:
                        robot = info.get('robot', robot_list[0])
                        log_ep_rew[robot].append(float(ep_rewards[i]))
                        ep_rewards[i] = 0.0

                obs_norm.update(next_obs_np)
                obs_t = torch.tensor(obs_norm.normalize(next_obs_np), device=dev)
                global_step += n_envs

        # Returns
        with torch.no_grad():
            with torch.autocast('cuda', torch.bfloat16, enabled=use_amp):
                _, _, last_val = policy.get_action(obs_t)
            last_val = last_val.float()
        buffer.compute_returns(last_val)

        policy.train()
        stats = trainer.update(buffer)
        log_pl.append(stats['pl']); log_vl.append(stats['vl'])
        log_ent.append(stats['ent']); log_steps.append(global_step)

        if iteration % 20 == 0 or iteration == 1:
            elapsed = time.time() - t0
            fps = global_step / elapsed
            per_robot = " | ".join(
                f"{r[:4]}={np.mean(log_ep_rew[r][-10:]):.0f}" if log_ep_rew[r] else f"{r[:4]}=--"
                for r in robot_list)
            print(f"  [{tag}] iter {iteration:5d}/{total_iters}  "
                  f"step {global_step/1e6:.2f}M  fps={fps:.0f}  "
                  f"pl={stats['pl']:+.3f}  vl={stats['vl']:.3f}  "
                  f"cf={stats['cf']:.2f}  |  {per_robot}")

    # Save
    torch.save(policy.state_dict(), os.path.join(save_dir, 'policy_final.pt'))
    obs_norm.save(os.path.join(save_dir, 'obs_norm_final.npz'))
    np.savez(os.path.join(save_dir, 'log_final.npz'),
             steps=log_steps, policy_loss=log_pl,
             value_loss=log_vl, entropy=log_ent,
             **{f'ep_rew_{r}': np.array(log_ep_rew[r]) for r in robot_list})

    elapsed = time.time() - t0
    print(f"\n  [{tag}] Done in {elapsed/60:.1f} min  →  {save_dir}")

    vec_env.close()
    return log_ep_rew


# ── Experiment suite ───────────────────────────────────────────────────────────
def main(total_steps: int = 10_000_000, n_per_robot: int = 4, device: str = 'auto'):
    os.makedirs(RESULTS_BASE, exist_ok=True)
    t_start = time.time()

    # ── 1. Specialist policies (one per robot) ─────────────────────────────
    print("\n\n>>> Phase 1: Specialist policies")
    specialist_results = {}
    for robot in ALL_ROBOTS:
        specialist_results[robot] = run_experiment(
            tag=f'specialist_{robot}',
            robot_list=[robot],
            total_steps=total_steps,
            n_per_robot=n_per_robot,
            device=device
        )

    # ── 2. No-Pgraph universal ─────────────────────────────────────────────
    print("\n\n>>> Phase 2: No-Pgraph universal policy")
    run_experiment(
        tag='no_pgraph',
        robot_list=ALL_ROBOTS,
        total_steps=total_steps,
        n_per_robot=n_per_robot,
        zero_pgraph=True,
        device=device
    )

    # ── 3. Zero-shot transfer: train on 3 robots, hold out ant ────────────
    print("\n\n>>> Phase 3: Zero-shot transfer (train without Ant)")
    train_robots = [r for r in ALL_ROBOTS if r != 'ant']
    run_experiment(
        tag='zeroshot_no_ant',
        robot_list=train_robots,
        total_steps=total_steps,
        n_per_robot=n_per_robot,
        device=device
    )

    total_elapsed = time.time() - t_start
    print(f"\n\n{'='*60}")
    print(f"All experiments complete in {total_elapsed/3600:.1f} h")
    print(f"Checkpoints → {CKPT_BASE}")
    print("Run compare.py to generate figures.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',  type=int, default=10_000_000,
                        help='Training steps per experiment (default 10M)')
    parser.add_argument('--envs',   type=int, default=4,
                        help='Parallel envs per robot (default 4)')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    main(total_steps=args.steps, n_per_robot=args.envs, device=args.device)
