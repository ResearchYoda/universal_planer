"""
scripts/universal_locomotion/test.py
======================================
Evaluate the trained universal policy on all robots + generate result figures.

Usage:
    python scripts/universal_locomotion/test.py              # evaluate + plots
    python scripts/universal_locomotion/test.py --render ant  # MuJoCo viewer
    python scripts/universal_locomotion/test.py --ckpt ckpt_100  # specific ckpt
"""

import os, sys, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore', category=DeprecationWarning)

import matplotlib
for _b in ('TkAgg', 'Qt5Agg', 'Agg'):
    try:
        matplotlib.use(_b)
        import matplotlib.pyplot as _t; break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.universal_locomotion.universal_env import (
    UniversalLocomotionEnv, OBS_DIM, MAX_DOF, ROBOT_CONFIGS
)
from scripts.universal_locomotion.ppo import UniversalActorCritic
from scripts.universal_locomotion.train import RunningNorm, SAVE_DIR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
N_EVAL_EPS  = 5
MAX_STEPS   = 1000

ROBOT_COLORS = {
    'hopper':      'steelblue',
    'halfcheetah': 'darkorange',
    'walker2d':    'seagreen',
    'ant':         'tomato',
}


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(policy, obs_norm: RunningNorm, robot_name: str,
             n_eps: int, render: bool = False, device='cpu'):
    """Run policy for n_eps episodes. Returns per-episode metrics."""

    render_mode = 'human' if render else None
    env = UniversalLocomotionEnv(robot_name, render_mode=render_mode)

    ep_rewards, ep_lengths = [], []
    x_trajs,    z_trajs    = [], []
    joint_trajs            = []   # actuated joint positions over time

    policy.eval()
    dev = torch.device(device)

    for ep in range(n_eps):
        obs, _ = env.reset()
        total_r, steps = 0.0, 0
        x_traj, z_traj, j_traj = [], [], []

        # Set camera to track the root body after first reset
        if render:
            try:
                import mujoco
                viewer = env._env.unwrapped.mujoco_renderer.viewer
                if viewer is not None:
                    viewer.cam.trackbodyid = 1
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.distance = 3.5
                    viewer.cam.elevation = -15.0
            except Exception:
                pass

        done = False
        while not done and steps < MAX_STEPS:
            obs_n = obs_norm.normalize(obs[None])[0]
            obs_t = torch.tensor(obs_n, dtype=torch.float32, device=dev)

            with torch.no_grad():
                act_t, _, _ = policy.get_action(obs_t.unsqueeze(0))
            action = act_t[0].cpu().numpy()

            obs, rew, term, trunc, _ = env.step(action)
            done = term or trunc
            total_r += rew
            steps   += 1

            d = env._data
            x_traj.append(float(d.xpos[1][0]))
            z_traj.append(float(d.xpos[1][2]))

            nu = env.n_actuators
            j_traj.append(
                np.array([d.qpos[i] for i in env._qpos_ids[:nu]], np.float32)
            )

        ep_rewards.append(total_r)
        ep_lengths.append(steps)
        x_trajs.append(x_traj)
        z_trajs.append(z_traj)
        joint_trajs.append(np.stack(j_traj))

        print(f"  [{robot_name}] ep {ep+1}/{n_eps}  "
              f"reward={total_r:7.1f}  steps={steps}")

    env.close()
    return dict(
        rewards=ep_rewards,
        lengths=ep_lengths,
        x_trajs=x_trajs,
        z_trajs=z_trajs,
        joint_trajs=joint_trajs,
    )


# ── Training curve plots ───────────────────────────────────────────────────────
def plot_training_curves(log_path: str, out_dir: str):
    if not os.path.exists(log_path):
        print(f"  [skip] training log not found: {log_path}")
        return

    d = np.load(log_path, allow_pickle=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Universal Policy – Training Curves', fontsize=13, fontweight='bold')

    steps = np.array(d['steps'])

    # 1. Policy loss
    axes[0,0].plot(steps, d['policy_loss'], color='steelblue', linewidth=1.2)
    axes[0,0].set_title('Policy Loss'); axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True, alpha=0.3)

    # 2. Value loss
    axes[0,1].plot(steps, d['value_loss'], color='darkorange', linewidth=1.2)
    axes[0,1].set_title('Value Loss'); axes[0,1].set_ylabel('MSE')
    axes[0,1].grid(True, alpha=0.3)

    # 3. Entropy
    axes[1,0].plot(steps, d['entropy'], color='seagreen', linewidth=1.2)
    axes[1,0].set_title('Policy Entropy'); axes[1,0].set_ylabel('Nats')
    axes[1,0].grid(True, alpha=0.3)

    # 4. Episode reward per robot (smoothed)
    ax = axes[1,1]
    for robot, color in ROBOT_COLORS.items():
        key = f'ep_rew_{robot}'
        if key in d and len(d[key]) > 0:
            rews = np.array(d[key])
            # smooth
            w = min(20, len(rews))
            smooth = np.convolve(rews, np.ones(w)/w, mode='valid')
            ax.plot(smooth, label=robot, color=color, linewidth=1.4)
    ax.set_title('Episode Reward (smoothed)'); ax.set_ylabel('Return')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('PPO Iteration' if ax != axes[1,1] else 'Episode')

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


# ── Evaluation result plots ───────────────────────────────────────────────────
def plot_eval_results(results: dict, out_dir: str):
    robots = list(results.keys())
    colors = [ROBOT_COLORS[r] for r in robots]
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle('Universal Locomotion Policy — Evaluation Results',
                 fontsize=14, fontweight='bold')

    # ── Row 0: episode rewards bar chart ─────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    means  = [np.mean(results[r]['rewards']) for r in robots]
    stds   = [np.std(results[r]['rewards'])  for r in robots]
    bars   = ax0.bar(robots, means, color=colors, alpha=0.8,
                     yerr=stds, capsize=6, error_kw=dict(linewidth=1.5))
    for bar, m in zip(bars, means):
        ax0.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(stds, default=1)*0.05,
                 f'{m:.0f}', ha='center', va='bottom', fontsize=9)
    ax0.set_title('Mean Episode Reward ± Std')
    ax0.set_ylabel('Total Reward'); ax0.grid(True, alpha=0.3, axis='y')

    # ── Row 0: episode lengths ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 2:])
    for r, c in zip(robots, colors):
        ax1.plot(results[r]['lengths'], 'o-', label=r, color=c,
                 linewidth=1.4, markersize=5)
    ax1.set_title('Episode Length (survival)'); ax1.set_ylabel('Steps')
    ax1.set_xlabel('Episode'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ── Row 1: X-position trajectories (best ep) ─────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    for r, c in zip(robots, colors):
        best = np.argmax(results[r]['rewards'])
        ax2.plot(results[r]['x_trajs'][best], label=r, color=c, linewidth=1.4)
    ax2.set_title('X-Position — Best Episode')
    ax2.set_xlabel('Step'); ax2.set_ylabel('X (m)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # ── Row 1: Z-position (height) trajectories (best ep) ────────────────
    ax3 = fig.add_subplot(gs[1, 2:])
    for r, c in zip(robots, colors):
        best = np.argmax(results[r]['rewards'])
        ax3.plot(results[r]['z_trajs'][best], label=r, color=c, linewidth=1.4)
    ax3.set_title('Z-Position (height) — Best Episode')
    ax3.set_xlabel('Step'); ax3.set_ylabel('Z (m)')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # ── Row 2: Joint angle trajectories per robot (best ep) ──────────────
    for col, robot in enumerate(robots):
        ax = fig.add_subplot(gs[2, col])
        best  = np.argmax(results[robot]['rewards'])
        jtraj = results[robot]['joint_trajs'][best]   # (T, n_act)
        for j in range(jtraj.shape[1]):
            ax.plot(np.degrees(jtraj[:, j]),
                    linewidth=1.0, alpha=0.8, label=f'j{j+1}')
        ax.set_title(f'{robot} joints (best ep)', fontsize=9)
        ax.set_xlabel('Step'); ax.set_ylabel('Angle (°)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'eval_results.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main(ckpt_tag: str = 'final', render_robot: str = None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    policy_path  = os.path.join(SAVE_DIR, f'policy_{ckpt_tag}.pt')
    obs_norm_path = os.path.join(SAVE_DIR, f'obs_norm_{ckpt_tag}.npz')
    log_path     = os.path.join(SAVE_DIR, f'log_{ckpt_tag}.npz')

    if not os.path.exists(policy_path):
        print(f"No checkpoint found at {policy_path}")
        print("Run train.py first:")
        print("  python scripts/universal_locomotion/train.py")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading policy from {policy_path}")
    policy = UniversalActorCritic(obs_dim=OBS_DIM, action_dim=MAX_DOF)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.to(device).eval()

    obs_norm = RunningNorm.load(obs_norm_path, shape=(OBS_DIM,))

    # Training curves
    print("\nPlotting training curves…")
    plot_training_curves(log_path, RESULTS_DIR)

    # Evaluate each robot
    results = {}
    for robot in ROBOT_CONFIGS:
        print(f"\nEvaluating {robot}…")
        render = (robot == render_robot)
        results[robot] = evaluate(
            policy, obs_norm, robot, N_EVAL_EPS, render=render, device=device
        )

    # Summary
    print("\n" + "="*52)
    print(f"{'Robot':15s} {'Mean':>8s} {'Std':>7s} {'Min':>7s} {'Max':>7s}")
    print("-"*52)
    for robot in ROBOT_CONFIGS:
        r = results[robot]['rewards']
        print(f"{robot:15s} {np.mean(r):8.1f} {np.std(r):7.1f} "
              f"{np.min(r):7.1f} {np.max(r):7.1f}")
    print("="*52)

    # Plot evaluation results
    print("\nPlotting evaluation results…")
    plot_eval_results(results, RESULTS_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',   default='final',
                        help='Checkpoint tag (default: final)')
    parser.add_argument('--render', default=None,
                        help='Robot to show in MuJoCo viewer (e.g. ant)')
    args = parser.parse_args()
    main(ckpt_tag=args.ckpt, render_robot=args.render)
