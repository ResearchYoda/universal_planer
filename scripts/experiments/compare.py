"""
scripts/experiments/compare.py
================================
Load all experiment results and generate a comprehensive comparison figure:

  Panel layout (3 rows × 4 cols):
    Row 0: Mean episode reward bar chart — Specialist vs Universal vs No-Pgraph
    Row 1: Training curves (smoothed) per robot (4 subplots)
    Row 2: Zero-shot ant comparison + radar chart + clip-fraction/entropy table

Usage:
    python scripts/experiments/compare.py
"""

import os, sys, warnings
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

from scripts.experiments.env_v2 import UniversalEnvV2, OBS_DIM, MAX_DOF, ROBOT_CONFIGS
from scripts.experiments.policy import MLPPolicy, RunningNorm

CKPT_BASE    = os.path.join(os.path.dirname(__file__), 'checkpoints')
RESULTS_BASE = os.path.join(os.path.dirname(__file__), 'results')
UNIVERSAL_CKPT = os.path.join(os.path.dirname(__file__), '..', 'universal_locomotion', 'checkpoints')

ALL_ROBOTS = list(ROBOT_CONFIGS.keys())
N_EVAL_EPS = 5
MAX_STEPS  = 1000

ROBOT_COLORS = {
    'hopper':      'steelblue',
    'halfcheetah': 'darkorange',
    'walker2d':    'seagreen',
    'ant':         'tomato',
}


# ── Policy evaluation ──────────────────────────────────────────────────────────
def evaluate_policy(policy, obs_norm, robot, n_eps=N_EVAL_EPS,
                    zero_pgraph=False, device='cpu'):
    """Return list of episode returns."""
    env = UniversalEnvV2(robot, zero_pgraph=zero_pgraph)
    dev = torch.device(device)
    policy.eval()
    returns = []
    for _ in range(n_eps):
        obs, _ = env.reset()
        total_r, done, steps = 0.0, False, 0
        while not done and steps < MAX_STEPS:
            obs_n = obs_norm.normalize(obs[None])[0]
            obs_t = torch.tensor(obs_n, dtype=torch.float32, device=dev)
            with torch.no_grad():
                act_t, _, _ = policy.get_action(obs_t.unsqueeze(0))
            obs, rew, term, trunc, _ = env.step(act_t[0].cpu().numpy())
            done = term or trunc
            total_r += rew
            steps += 1
        returns.append(total_r)
    env.close()
    return returns


def load_policy(ckpt_dir, device='cpu'):
    policy_path   = os.path.join(ckpt_dir, 'policy_final.pt')
    obs_norm_path = os.path.join(ckpt_dir, 'obs_norm_final.npz')
    if not os.path.exists(policy_path):
        return None, None
    policy = MLPPolicy(obs_dim=OBS_DIM, action_dim=MAX_DOF)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.to(device).eval()
    obs_norm = RunningNorm.load(obs_norm_path, (OBS_DIM,))
    return policy, obs_norm


def load_log(ckpt_dir):
    path = os.path.join(ckpt_dir, 'log_final.npz')
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


# ── Main comparison ────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_BASE, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading and evaluating all policies…")

    # ── 1. Specialist policies ─────────────────────────────────────────────
    specialist = {}
    for robot in ALL_ROBOTS:
        ckpt_dir = os.path.join(CKPT_BASE, f'specialist_{robot}')
        policy, obs_norm = load_policy(ckpt_dir, device)
        if policy is None:
            print(f"  [skip] specialist_{robot}: no checkpoint")
            specialist[robot] = None
            continue
        print(f"  Evaluating specialist_{robot}…")
        specialist[robot] = evaluate_policy(policy, obs_norm, robot, device=device)
        print(f"    → mean={np.mean(specialist[robot]):.1f}")

    # ── 2. Universal Pgraph policy (from universal_locomotion) ────────────
    # Load the v3 Transformer policy if available; fall back gracefully
    universal = {}
    ul_ckpt = os.path.join(UNIVERSAL_CKPT, 'policy_final.pt')
    ul_norm  = os.path.join(UNIVERSAL_CKPT, 'obs_norm_final.npz')
    if os.path.exists(ul_ckpt):
        try:
            from scripts.universal_locomotion.ppo import UniversalActorCritic
            from scripts.universal_locomotion.train import RunningNorm as ULRunningNorm, _OBS_PIN
            from scripts.universal_locomotion.universal_env import (
                UniversalLocomotionEnv, OBS_DIM as UL_OBS_DIM, MAX_DOF as UL_MAX_DOF
            )
            ul_policy = UniversalActorCritic(obs_dim=UL_OBS_DIM, action_dim=UL_MAX_DOF)
            ul_policy.load_state_dict(torch.load(ul_ckpt, map_location=device))
            ul_policy.to(device).eval()
            ul_obs_norm = ULRunningNorm.load(ul_norm, shape=(UL_OBS_DIM,), pin_indices=_OBS_PIN)

            for robot in ALL_ROBOTS:
                print(f"  Evaluating universal (Pgraph) on {robot}…")
                env = UniversalLocomotionEnv(robot)
                returns = []
                for _ in range(N_EVAL_EPS):
                    obs, _ = env.reset()
                    total_r, done, steps = 0.0, False, 0
                    while not done and steps < MAX_STEPS:
                        obs_n = ul_obs_norm.normalize(obs[None])[0]
                        obs_t = torch.tensor(obs_n, dtype=torch.float32, device=device)
                        with torch.no_grad():
                            act_t, _, _ = ul_policy.get_action(obs_t.unsqueeze(0))
                        obs, rew, term, trunc, _ = env.step(act_t[0].cpu().numpy())
                        done = term or trunc
                        total_r += rew
                        steps += 1
                    returns.append(total_r)
                env.close()
                universal[robot] = returns
                print(f"    → mean={np.mean(returns):.1f}")
        except Exception as e:
            print(f"  [warn] Could not load universal Transformer policy: {e}")
    else:
        print(f"  [skip] universal Transformer policy not found at {ul_ckpt}")

    # ── 3. No-Pgraph universal ─────────────────────────────────────────────
    no_pgraph = {}
    ckpt_dir = os.path.join(CKPT_BASE, 'no_pgraph')
    policy, obs_norm = load_policy(ckpt_dir, device)
    if policy is not None:
        for robot in ALL_ROBOTS:
            print(f"  Evaluating no_pgraph on {robot}…")
            no_pgraph[robot] = evaluate_policy(policy, obs_norm, robot,
                                               zero_pgraph=True, device=device)
            print(f"    → mean={np.mean(no_pgraph[robot]):.1f}")
    else:
        print("  [skip] no_pgraph: no checkpoint")

    # ── 4. Zero-shot on Ant ────────────────────────────────────────────────
    ckpt_dir = os.path.join(CKPT_BASE, 'zeroshot_no_ant')
    zs_policy, zs_obs_norm = load_policy(ckpt_dir, device)
    zs_ant = None
    if zs_policy is not None:
        print("  Evaluating zero-shot policy on Ant (never seen during training)…")
        zs_ant = evaluate_policy(zs_policy, zs_obs_norm, 'ant', device=device)
        print(f"    → mean={np.mean(zs_ant):.1f}")
    else:
        print("  [skip] zeroshot_no_ant: no checkpoint")

    # ── Build figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)
    fig.suptitle('Universal Locomotion — Experimental Comparison',
                 fontsize=15, fontweight='bold', y=0.98)

    # ── Row 0: grouped bar chart (all robots × all conditions) ────────────
    ax0 = fig.add_subplot(gs[0, :])

    conditions = ['Specialist', 'Universal\n(Pgraph)', 'No-Pgraph', 'Zero-shot\n(Pgraph)']
    n_cond = len(conditions)
    x = np.arange(len(ALL_ROBOTS))
    width = 0.18
    offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * width

    def _mean_std(d, robot):
        if d and robot in d and d[robot]:
            v = np.array(d[robot])
            return v.mean(), v.std()
        return 0.0, 0.0

    for ci, (cond, data) in enumerate(zip(conditions,
                                           [specialist, universal, no_pgraph,
                                            {r: (zs_ant if r == 'ant' else None) for r in ALL_ROBOTS}])):
        means, stds = [], []
        for robot in ALL_ROBOTS:
            m, s = _mean_std(data, robot)
            means.append(m); stds.append(s)
        bars = ax0.bar(x + offsets[ci], means, width,
                       label=conditions[ci], alpha=0.85,
                       yerr=stds, capsize=4, error_kw=dict(linewidth=1.2))

    ax0.set_xticks(x); ax0.set_xticklabels([r.capitalize() for r in ALL_ROBOTS])
    ax0.set_title('Mean Episode Reward ± Std by Condition', fontsize=11)
    ax0.set_ylabel('Total Reward')
    ax0.legend(loc='upper left', fontsize=8, ncol=n_cond)
    ax0.grid(True, alpha=0.3, axis='y')
    ax0.axhline(0, color='black', linewidth=0.5)

    # ── Row 1: training curves per robot (smoothed ep reward) ─────────────
    for col, robot in enumerate(ALL_ROBOTS):
        ax = fig.add_subplot(gs[1, col])
        color = ROBOT_COLORS[robot]

        # Specialist
        log = load_log(os.path.join(CKPT_BASE, f'specialist_{robot}'))
        if log is not None and f'ep_rew_{robot}' in log:
            rews = np.array(log[f'ep_rew_{robot}'])
            if len(rews) > 0:
                w = min(20, len(rews))
                ax.plot(np.convolve(rews, np.ones(w)/w, 'valid'),
                        color=color, linewidth=1.4, label='Specialist')

        # No-pgraph universal
        log2 = load_log(os.path.join(CKPT_BASE, 'no_pgraph'))
        if log2 is not None and f'ep_rew_{robot}' in log2:
            rews2 = np.array(log2[f'ep_rew_{robot}'])
            if len(rews2) > 0:
                w = min(20, len(rews2))
                ax.plot(np.convolve(rews2, np.ones(w)/w, 'valid'),
                        color='gray', linewidth=1.2, linestyle='--', label='No-Pgraph')

        ax.set_title(f'{robot.capitalize()} training curve', fontsize=9)
        ax.set_xlabel('Episode'); ax.set_ylabel('Return')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── Row 2 left: Zero-shot ant comparison bar ───────────────────────────
    ax2 = fig.add_subplot(gs[2, :2])
    zs_label  = ['Specialist\n(Ant)', 'Universal\n(Pgraph)', 'No-Pgraph', 'Zero-shot\n(Pgraph)']
    zs_vals   = []
    zs_errs   = []

    def _ms(lst):
        if lst:
            a = np.array(lst)
            return a.mean(), a.std()
        return 0.0, 0.0

    m, s = _ms(specialist.get('ant'))
    zs_vals.append(m); zs_errs.append(s)
    m, s = _ms(universal.get('ant'))
    zs_vals.append(m); zs_errs.append(s)
    m, s = _ms(no_pgraph.get('ant'))
    zs_vals.append(m); zs_errs.append(s)
    m, s = _ms(zs_ant)
    zs_vals.append(m); zs_errs.append(s)

    colors_zs = ['steelblue', 'gold', 'gray', 'tomato']
    bars2 = ax2.bar(zs_label, zs_vals, color=colors_zs, alpha=0.85,
                    yerr=zs_errs, capsize=6, error_kw=dict(linewidth=1.5))
    for bar, m in zip(bars2, zs_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(zs_errs, default=1)*0.05,
                 f'{m:.0f}', ha='center', va='bottom', fontsize=9)
    ax2.set_title('Ant Performance: Conditions Comparison\n(Zero-shot = never seen Ant during training)',
                  fontsize=9)
    ax2.set_ylabel('Mean Episode Reward')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(0, color='black', linewidth=0.5)

    # ── Row 2 right: Summary table ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 2:])
    ax3.axis('off')

    rows = [['Robot', 'Specialist', 'Universal\n(Pgraph)', 'No-Pgraph', 'Zero-shot']]
    for robot in ALL_ROBOTS:
        def _fmt(d, r=robot):
            if d and r in d and d[r]:
                return f"{np.mean(d[r]):.0f}"
            return '—'
        zs_val = _fmt({'ant': zs_ant} if zs_ant else {}, robot) if robot == 'ant' else '—'
        rows.append([
            robot.capitalize(),
            _fmt(specialist),
            _fmt(universal),
            _fmt(no_pgraph),
            zs_val,
        ])

    tbl = ax3.table(cellText=rows[1:], colLabels=rows[0],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Header row styling
    for j in range(5):
        tbl[(0, j)].set_facecolor('#2c3e50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)):
        for j in range(5):
            tbl[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

    ax3.set_title('Summary: Mean Episode Rewards', fontsize=9, pad=10)

    path = os.path.join(RESULTS_BASE, 'comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved → {path}")
    plt.show()


if __name__ == '__main__':
    main()
