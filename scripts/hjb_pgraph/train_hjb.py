"""
scripts/hjb_pgraph/train_hjb.py
================================
Training script for the pGraph + PINN/HJB policy.

Usage:
    python scripts/hjb_pgraph/train_hjb.py
    python scripts/hjb_pgraph/train_hjb.py --steps 5000000 --lambda_hjb 0.1
    python scripts/hjb_pgraph/train_hjb.py --resume
"""

import os, sys, time, argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.universal_locomotion.universal_env import (
    UniversalLocomotionEnv, ROBOT_CONFIGS,
)
from scripts.hjb_pgraph.pinn_policy import (
    MorphHJBPolicy, HJBRolloutBuffer, HJBPPOTrainer,
    OBS_DIM, N_JOINT,
)
from scripts.universal_locomotion.train import RunningNorm, _OBS_PIN

ROBOTS   = list(ROBOT_CONFIGS.keys())
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


# ── Vectorised multi-robot env ────────────────────────────────────────────────
class MultiRobotVecEnv:
    def __init__(self, robots: list, n_per_robot: int = 4):
        self.envs: list = []
        self.robot_ids: list = []
        for robot in robots:
            for _ in range(n_per_robot):
                self.envs.append(UniversalLocomotionEnv(robot))
                self.robot_ids.append(robot)
        self.n_envs = len(self.envs)

    def reset(self):
        return np.stack([env.reset()[0] for env in self.envs])

    def step(self, actions: np.ndarray):
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


# ── Training loop ──────────────────────────────────────────────────────────────
def train(total_steps:  int   = 20_000_000,
          n_per_robot:  int   = 4,
          n_steps:      int   = 4096,
          lambda_hjb:   float = 0.05,
          lambda_mc:    float = 0.01,
          lr:           float = 1e-4,
          device:       str   = 'auto',
          resume:       bool  = False,
          compile_mode: bool  = False):

    os.makedirs(SAVE_DIR, exist_ok=True)

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)
    use_amp = (dev.type == 'cuda' and torch.cuda.is_bf16_supported())
    print(f"Device: {device}  |  BF16 AMP: {use_amp}")
    print(f"λ_hjb={lambda_hjb}  λ_mc={lambda_mc}  lr={lr}")

    # ── Environments ──────────────────────────────────────────────────
    print(f"Building {len(ROBOTS) * n_per_robot} environments…")
    vec_env = MultiRobotVecEnv(ROBOTS, n_per_robot)
    n_envs  = vec_env.n_envs

    # Observation normaliser (binary mask indices pinned)
    obs_norm  = RunningNorm((OBS_DIM,), pin_indices=_OBS_PIN)
    rew_norms = {r: RunningNorm((1,)) for r in ROBOTS}

    # ── Policy & trainer ──────────────────────────────────────────────
    policy  = MorphHJBPolicy(obs_dim=OBS_DIM, action_dim=N_JOINT, gamma=0.99)
    trainer = HJBPPOTrainer(
        policy,
        lr=lr, n_epochs=5, batch_size=512,
        clip=0.2, ent_coef=0.02, vf_coef=0.25,
        lambda_hjb=lambda_hjb, lambda_mc=lambda_mc,
        max_grad=0.5, device=device,
    )
    buffer = HJBRolloutBuffer(n_steps, n_envs, OBS_DIM, N_JOINT, dev)

    total_iters = total_steps // (n_steps * n_envs)
    start_iter  = 1

    # ── Optional resume ───────────────────────────────────────────────
    if resume:
        pp = os.path.join(SAVE_DIR, 'policy_final.pt')
        np_ = os.path.join(SAVE_DIR, 'obs_norm_final.npz')
        if os.path.exists(pp):
            print(f"Resuming from {pp}")
            policy.load_state_dict(torch.load(pp, map_location=device))
            obs_norm = RunningNorm.load(np_, (OBS_DIM,), pin_indices=_OBS_PIN)
        else:
            print("No checkpoint found, starting from scratch.")

    policy.to(dev)

    # ── torch.compile (optional) ──────────────────────────────────────
    if compile_mode:
        torch._dynamo.config.suppress_errors = True
        policy = torch.compile(policy, mode='reduce-overhead')
        trainer.policy = policy
        print("torch.compile enabled")

    # ── LR schedule: linear warmup (20 iter) then cosine decay ───────
    warmup = 20
    lr_min = lr * 0.1
    def _lr_fn(t):
        if t < warmup:
            return (t + 1) / warmup
        prog = (t - warmup) / max(total_iters - warmup, 1)
        return max(1.0 - prog * (1 - lr_min / lr), lr_min / lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, _lr_fn)
    trainer.set_scheduler(scheduler)

    # ── Entropy schedule: 0.02 → 0.005 ───────────────────────────────
    ent_start, ent_end = 0.02, 0.005

    # ── Logging ───────────────────────────────────────────────────────
    log = dict(
        steps=[], policy_loss=[], value_loss=[], entropy=[],
        hjb_loss=[], mc_loss=[],
        grad_morph_norm=[], grad_joint_norm=[], grad_root_norm=[],
        hjb_residual=[],
        **{f'ep_rew_{r}': [] for r in ROBOTS},
    )
    ep_rewards = np.zeros(n_envs)

    obs_np = vec_env.reset()
    obs_norm.update(obs_np)
    obs_t  = torch.tensor(obs_norm.normalize(obs_np), device=dev)

    global_step = 0
    t0 = time.time()

    print(f"\nRobots: {ROBOTS}")
    print(f"n_envs={n_envs} | n_steps={n_steps} | "
          f"batch/iter={n_steps * n_envs:,} | total_iters={total_iters:,}\n")

    # ── Main loop ─────────────────────────────────────────────────────
    for iteration in range(start_iter, total_iters + 1):

        progress = iteration / total_iters
        trainer.ent_coef = ent_start + (ent_end - ent_start) * progress

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
                rew_norm = np.zeros_like(rew_np)
                for i, robot in enumerate(vec_env.robot_ids):
                    rew_norms[robot].update(rew_np[i:i+1].reshape(-1, 1))
                    rew_norm[i] = np.clip(
                        rew_np[i] / np.sqrt(rew_norms[robot].var[0] + 1e-8),
                        -10.0, 10.0)

                # next_obs before normalisation (raw for HJB finite diff)
                next_obs_t_raw = torch.tensor(next_obs_np, device=dev,
                                              dtype=torch.float32)

                buffer.add(
                    obs_t, next_obs_t_raw,
                    action_t, lp_t,
                    torch.tensor(rew_norm, device=dev),
                    val_t,
                    torch.tensor(done_np, device=dev),
                    list(vec_env.robot_ids),
                )

                ep_rewards += rew_np
                for i, (done, info) in enumerate(zip(done_np, infos)):
                    if done:
                        r = info.get('robot', 'unknown')
                        if r in log:
                            log[f'ep_rew_{r}'].append(float(ep_rewards[i]))
                        ep_rewards[i] = 0.0

                obs_norm.update(next_obs_np)
                obs_t = torch.tensor(obs_norm.normalize(next_obs_np), device=dev)
                global_step += n_envs

        # ── Returns ───────────────────────────────────────────────────
        with torch.no_grad():
            with torch.autocast('cuda', torch.bfloat16, enabled=use_amp):
                _, _, last_val = policy.get_action(obs_t)
            last_val = last_val.float()
        buffer.compute_returns(last_val)

        # ── PPO + HJB update ──────────────────────────────────────────
        policy.train()
        stats = trainer.update(buffer)

        for k in ['policy_loss', 'value_loss', 'entropy',
                  'hjb_loss', 'mc_loss',
                  'grad_morph_norm', 'grad_joint_norm', 'grad_root_norm',
                  'hjb_residual']:
            log[k].append(stats.get(k, 0.0))
        log['steps'].append(global_step)

        # ── Console logging ───────────────────────────────────────────
        if iteration % 10 == 0 or iteration == 1:
            elapsed = time.time() - t0
            fps     = global_step / elapsed
            lr_now  = trainer.optimizer.param_groups[0]['lr']
            per_robot = " | ".join(
                f"{r[:4]}={np.mean(log[f'ep_rew_{r}'][-10:]):.0f}"
                if log[f'ep_rew_{r}'] else f"{r[:4]}=--"
                for r in ROBOTS)
            print(
                f"iter {iteration:5d}/{total_iters}  "
                f"step {global_step/1e6:.2f}M  fps={fps:.0f}  "
                f"lr={lr_now:.2e}  "
                f"pl={stats['policy_loss']:+.3f}  "
                f"vl={stats['value_loss']:.3f}  "
                f"hjb={stats['hjb_loss']:.4f}  "
                f"mc={stats['mc_loss']:.4f}  "
                f"∇m={stats['grad_morph_norm']:.3f}  "
                f"∇j={stats['grad_joint_norm']:.3f}  "
                f"cf={stats['clip_frac']:.2f}  |  {per_robot}"
            )

        # ── Checkpoint ────────────────────────────────────────────────
        if iteration % 100 == 0:
            _save(policy, obs_norm, rew_norms, log, tag=f'ckpt_{iteration}')

    # ── Final save ────────────────────────────────────────────────────
    _save(policy, obs_norm, rew_norms, log, tag='final')
    vec_env.close()
    print(f"\nTraining done in {(time.time()-t0)/60:.1f} min")
    print(f"Checkpoints → {SAVE_DIR}")


def _save(policy, obs_norm, rew_norms, log, tag='final'):
    sd = (policy._orig_mod.state_dict()
          if hasattr(policy, '_orig_mod') else policy.state_dict())
    torch.save(sd, os.path.join(SAVE_DIR, f'policy_{tag}.pt'))
    obs_norm.save(os.path.join(SAVE_DIR, f'obs_norm_{tag}.npz'))
    for robot, rn in rew_norms.items():
        rn.save(os.path.join(SAVE_DIR, f'rew_norm_{robot}_{tag}.npz'))
    save_log = {k: np.array(v) for k, v in log.items() if isinstance(v, list)}
    np.savez(os.path.join(SAVE_DIR, f'log_{tag}.npz'), **save_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',       type=int,   default=20_000_000)
    parser.add_argument('--envs',        type=int,   default=4,
                        help='Parallel envs per robot')
    parser.add_argument('--nsteps',      type=int,   default=4096)
    parser.add_argument('--lambda_hjb',  type=float, default=0.05,
                        help='HJB residual loss weight (paper: 0.05–0.15)')
    parser.add_argument('--lambda_mc',   type=float, default=0.01,
                        help='Morphology consistency loss weight')
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--device',      type=str,   default='auto')
    parser.add_argument('--resume',      action='store_true')
    parser.add_argument('--compile',     action='store_true')
    args = parser.parse_args()

    train(
        total_steps  = args.steps,
        n_per_robot  = args.envs,
        n_steps      = args.nsteps,
        lambda_hjb   = args.lambda_hjb,
        lambda_mc    = args.lambda_mc,
        lr           = args.lr,
        device       = args.device,
        resume       = args.resume,
        compile_mode = args.compile,
    )
