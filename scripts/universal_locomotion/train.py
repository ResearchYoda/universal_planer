"""
scripts/universal_locomotion/train.py
======================================
Train a universal locomotion policy across Hopper, HalfCheetah, Walker2d, Ant
using PPO with the Pgraph morphology-aware encoder-decoder architecture.

Improvements over v1
--------------------
- Per-robot reward normalisation (each robot has its own RunningNorm)
- Linear learning-rate schedule (3e-4 → 3e-5 over training)
- Default: 10M steps, 4 envs per robot
- Resume from checkpoint with --resume flag
- Entropy coefficient annealing (0.02 → 0.005)
- Larger batch size matching the bigger network in ppo.py

Usage:
    python scripts/universal_locomotion/train.py
    python scripts/universal_locomotion/train.py --steps 10000000 --envs 4
    python scripts/universal_locomotion/train.py --resume          # continue final ckpt
"""

import os, sys, time, argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.universal_locomotion.universal_env import (
    UniversalLocomotionEnv, OBS_DIM, MAX_DOF, ROBOT_CONFIGS,
    N_MORPH, NODE_FEAT, N_JOINT, JOINT_FEAT
)
from scripts.universal_locomotion.ppo import (
    UniversalActorCritic, PPOTrainer, RolloutBuffer
)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
ROBOTS   = list(ROBOT_CONFIGS.keys())
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')

# Binary mask features in obs — must NOT be renormalized by RunningNorm
# body_mask: index 4 of each 5-feature morph token  → slots 4,9,14,...,79
# dof_mask:  index 2 of each 6-feature joint token  → slots 82,88,...,124
_OBS_PIN = (
    [i * NODE_FEAT + (NODE_FEAT - 1) for i in range(N_MORPH)] +     # body_mask
    [N_MORPH * NODE_FEAT + i * JOINT_FEAT + 2 for i in range(N_JOINT)]  # dof_mask
)


# ── Running normalizer (online mean/std, with optional pinned indices) ────────
class RunningNorm:
    """Welford online normalizer. pin_indices: obs dims passed through unchanged."""

    def __init__(self, shape, clip=10.0, pin_indices=None):
        self.mean  = np.zeros(shape, np.float64)
        self.var   = np.ones(shape,  np.float64)
        self.count = 1e-4
        self.clip  = clip
        self._pin  = np.array(pin_indices, dtype=int) if pin_indices is not None else None

    def update(self, x: np.ndarray):
        batch  = x.reshape(-1, x.shape[-1])
        b_mean = batch.mean(0)
        b_var  = batch.var(0)
        n      = len(batch)
        total  = self.count + n
        delta  = b_mean - self.mean
        self.mean  = self.mean  + delta * n / total
        m_a = self.var   * self.count
        m_b = b_var      * n
        self.var   = (m_a + m_b + delta**2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        result = np.clip((x - self.mean) / np.sqrt(self.var + 1e-8),
                         -self.clip, self.clip).astype(np.float32)
        if self._pin is not None:
            result[..., self._pin] = x[..., self._pin].astype(np.float32)
        return result

    def save(self, path):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    @classmethod
    def load(cls, path, shape, clip=10.0, pin_indices=None):
        d = np.load(path)
        obj = cls(shape, clip, pin_indices=pin_indices)
        obj.mean, obj.var, obj.count = d['mean'], d['var'], float(d['count'])
        return obj


# ── Multi-robot vectorized collection ────────────────────────────────────────
class MultiRobotVecEnv:
    """
    Synchronous vectorized env with n_per_robot envs per robot.
    All environments share the same fixed obs/action space.
    """

    def __init__(self, robots: list, n_per_robot: int = 4):
        self.envs: list = []
        self.robot_ids: list = []
        for robot in robots:
            for _ in range(n_per_robot):
                self.envs.append(UniversalLocomotionEnv(robot))
                self.robot_ids.append(robot)
        self.n_envs = len(self.envs)

    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.stack(obs_list)

    def step(self, actions: np.ndarray):
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for i, (env, act) in enumerate(zip(self.envs, actions)):
            obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(float(done))
            info_list.append(info)
        return (np.stack(obs_list), np.array(rew_list, np.float32),
                np.array(done_list, np.float32), info_list)

    def close(self):
        for env in self.envs:
            env.close()


# ── Training loop ─────────────────────────────────────────────────────────────
def train(total_steps: int = 10_000_000,
          n_per_robot: int = 4,
          n_steps:     int = 2048,
          device:      str = 'auto',
          resume:      bool = False):

    os.makedirs(SAVE_DIR, exist_ok=True)

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)
    print(f"Device: {device}")

    # ── Environments ─────────────────────────────────────────────────────
    print(f"Building {len(ROBOTS) * n_per_robot} environments "
          f"({n_per_robot} per robot)…")
    vec_env = MultiRobotVecEnv(ROBOTS, n_per_robot)
    n_envs  = vec_env.n_envs

    # Shared observation norm; per-robot reward norms
    obs_norm  = RunningNorm((OBS_DIM,), pin_indices=_OBS_PIN)
    rew_norms = {r: RunningNorm((1,)) for r in ROBOTS}

    # ── Policy & trainer ─────────────────────────────────────────────────
    policy  = UniversalActorCritic(obs_dim=OBS_DIM, action_dim=MAX_DOF)
    # Transformer needs lower lr + fewer epochs than MLP to stay stable
    trainer = PPOTrainer(policy, n_steps=n_steps, device=device,
                         batch_size=512, ent_coef=0.02, lr=1e-4,
                         n_epochs=5, vf_coef=0.25, max_grad=0.3)
    buffer  = RolloutBuffer(n_steps, n_envs, OBS_DIM, MAX_DOF, dev)

    total_iters  = total_steps // (n_steps * n_envs)
    start_iter   = 1

    # ── Optional resume ───────────────────────────────────────────────────
    if resume:
        policy_path   = os.path.join(SAVE_DIR, 'policy_final.pt')
        obs_norm_path = os.path.join(SAVE_DIR, 'obs_norm_final.npz')
        if os.path.exists(policy_path):
            print(f"Resuming from {policy_path}")
            policy.load_state_dict(torch.load(policy_path, map_location=device))
            obs_norm = RunningNorm.load(obs_norm_path, shape=(OBS_DIM,), pin_indices=_OBS_PIN)
        else:
            print("No checkpoint found, starting from scratch.")

    policy.to(dev)

    # ── LR schedule: warmup 20 iter then linear decay 1e-4 → 1e-5 ───────
    lr_init   = 1e-4
    lr_end    = 1e-5
    warmup    = 20
    def _lr_fn(t):
        if t < warmup:
            return (t + 1) / warmup       # linear warmup
        prog = (t - warmup) / max(total_iters - warmup, 1)
        return max(1.0 - prog * (1 - lr_end / lr_init), lr_end / lr_init)
    scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, _lr_fn)
    trainer.set_scheduler(scheduler)

    # ── Entropy coefficient schedule: 0.02 → 0.005 ───────────────────────
    ent_start = 0.02
    ent_end   = 0.005

    print(f"Robots: {ROBOTS}")
    print(f"n_envs={n_envs} | n_steps={n_steps} | "
          f"batch/iter={n_steps * n_envs:,} | iters={total_iters:,}")
    print()

    # ── Logging ───────────────────────────────────────────────────────────
    log_steps:       list = []
    log_ep_rew:      dict = {r: [] for r in ROBOTS}
    log_policy_loss: list = []
    log_value_loss:  list = []
    log_entropy:     list = []

    ep_rewards = np.zeros(n_envs)

    obs_np = vec_env.reset()
    obs_norm.update(obs_np)
    obs_t  = torch.tensor(obs_norm.normalize(obs_np), device=dev)

    global_step = 0
    t0 = time.time()

    # ── Collect + update loop ─────────────────────────────────────────────
    for iteration in range(start_iter, total_iters + 1):

        # Anneal entropy coefficient
        progress = iteration / total_iters
        trainer.ent_coef = ent_start + (ent_end - ent_start) * progress

        policy.eval()
        with torch.no_grad():
            for step in range(n_steps):
                action_t, log_prob_t, value_t = policy.get_action(obs_t)

                actions_np = action_t.cpu().numpy()
                next_obs_np, rew_np, done_np, infos = vec_env.step(actions_np)

                # Per-robot reward normalisation
                rew_norm_np = np.zeros_like(rew_np)
                for i, robot in enumerate(vec_env.robot_ids):
                    rew_norms[robot].update(rew_np[i:i+1].reshape(-1, 1))
                    rew_norm_np[i] = np.clip(
                        rew_np[i] / np.sqrt(rew_norms[robot].var[0] + 1e-8),
                        -10.0, 10.0
                    )

                buffer.add(
                    obs_t,
                    action_t,
                    log_prob_t,
                    torch.tensor(rew_norm_np, device=dev),
                    value_t,
                    torch.tensor(done_np, device=dev),
                )

                # Episode reward tracking (raw, unnormalised)
                ep_rewards += rew_np
                for i, (done, info) in enumerate(zip(done_np, infos)):
                    if done:
                        robot = info.get('robot', 'unknown')
                        log_ep_rew[robot].append(float(ep_rewards[i]))
                        ep_rewards[i] = 0.0

                obs_norm.update(next_obs_np)
                obs_t = torch.tensor(
                    obs_norm.normalize(next_obs_np), device=dev
                )
                global_step += n_envs

        # Compute returns
        with torch.no_grad():
            _, _, last_val = policy.get_action(obs_t)
        buffer.compute_returns(last_val)

        # PPO update
        policy.train()
        stats = trainer.update(buffer)

        log_policy_loss.append(stats['policy_loss'])
        log_value_loss.append(stats['value_loss'])
        log_entropy.append(stats['entropy'])
        log_steps.append(global_step)

        # Logging
        if iteration % 10 == 0 or iteration == 1:
            elapsed = time.time() - t0
            fps     = global_step / elapsed
            lr_now  = trainer.optimizer.param_groups[0]['lr']
            per_robot = " | ".join(
                f"{r[:4]}={np.mean(log_ep_rew[r][-10:]):.0f}"
                if log_ep_rew[r] else f"{r[:4]}=--"
                for r in ROBOTS
            )
            print(f"iter {iteration:5d}/{total_iters}  "
                  f"step {global_step/1e6:.2f}M  fps={fps:.0f}  "
                  f"lr={lr_now:.2e}  ent={trainer.ent_coef:.4f}  "
                  f"pl={stats['policy_loss']:+.3f}  vl={stats['value_loss']:.3f}  "
                  f"cf={stats['clip_frac']:.2f}  |  {per_robot}")

        # Checkpoint every 100 iterations
        if iteration % 100 == 0:
            _save(policy, obs_norm, rew_norms,
                  log_steps, log_ep_rew, log_policy_loss, log_value_loss,
                  log_entropy, tag=f'ckpt_{iteration}')

    # ── Final save ────────────────────────────────────────────────────────
    _save(policy, obs_norm, rew_norms,
          log_steps, log_ep_rew, log_policy_loss, log_value_loss,
          log_entropy, tag='final')

    vec_env.close()
    print(f"\nTraining done in {(time.time()-t0)/60:.1f} min")
    print(f"Checkpoints → {SAVE_DIR}")


def _save(policy, obs_norm, rew_norms,
          log_steps, log_ep_rew, log_pl, log_vl, log_ent, tag='final'):
    torch.save(policy.state_dict(),
               os.path.join(SAVE_DIR, f'policy_{tag}.pt'))
    obs_norm.save(os.path.join(SAVE_DIR, f'obs_norm_{tag}.npz'))
    # Save per-robot reward norms
    for robot, rn in rew_norms.items():
        rn.save(os.path.join(SAVE_DIR, f'rew_norm_{robot}_{tag}.npz'))
    np.savez(os.path.join(SAVE_DIR, f'log_{tag}.npz'),
             steps=log_steps,
             policy_loss=log_pl,
             value_loss=log_vl,
             entropy=log_ent,
             **{f'ep_rew_{r}': np.array(log_ep_rew[r]) for r in log_ep_rew})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',  type=int, default=10_000_000)
    parser.add_argument('--envs',   type=int, default=4,
                        help='Parallel envs per robot')
    parser.add_argument('--nsteps', type=int, default=2048,
                        help='Rollout steps per update')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from final checkpoint')
    args = parser.parse_args()

    train(total_steps=args.steps,
          n_per_robot=args.envs,
          n_steps=args.nsteps,
          device=args.device,
          resume=args.resume)
