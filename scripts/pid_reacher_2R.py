"""
2R Planar Robot – PID Reacher (Position Task)
=============================================
Analytical IK  →  joint-space PID  →  MuJoCo simulation

Robot:   2R_robotic_arm.xml   (L1=0.75 m, L2=0.50 m)
Control: Proportional-Integral-Derivative on joint angles
         Desired angles are computed via closed-form 2R inverse kinematics.

Usage:
    python scripts/pid_reacher_2R.py
    python scripts/pid_reacher_2R.py --target 0.8 0.3
    python scripts/pid_reacher_2R.py --time 8 --seed 0
"""

import argparse
import os
import time
from typing import Optional

import matplotlib
# PySide2/shiboken2 built against NumPy 1.x crashes with NumPy 2.x.
# Try lightweight backends before falling back to file-only Agg.
for _backend in ('TkAgg', 'Qt5Agg', 'Agg'):
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as _plt_test  # noqa: F401
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

# ── Robot geometry ────────────────────────────────────────────────────────────
L1 = 0.75  # link-1 length [m]
L2 = 0.50  # link-2 length [m]

# ── PID gains (joint-space) ───────────────────────────────────────────────────
KP = np.array([60.0, 45.0])   # proportional
KI = np.array([ 1.0,  0.8])   # integral
KD = np.array([10.0,  7.0])   # derivative

CTRL_LIMIT = 50.0  # motor saturation [Nm]
DT         = 0.01  # simulation timestep – must match XML
SUCCESS_DIST = 0.02  # 2 cm end-effector threshold


# ─────────────────────────────────────────────────────────────────────────────
class PIDController:
    """Multi-axis PID with anti-windup (integral clamping)."""

    def __init__(self, kp, ki, kd, dt, ctrl_limit=None, integral_limit=None):
        self.kp  = np.asarray(kp, dtype=float)
        self.ki  = np.asarray(ki, dtype=float)
        self.kd  = np.asarray(kd, dtype=float)
        self.dt  = dt
        self.ctrl_limit     = ctrl_limit
        self.integral_limit = integral_limit if integral_limit is not None \
                              else (ctrl_limit / (ki + 1e-12) if ctrl_limit else None)
        self.reset()

    def reset(self):
        self.integral   = np.zeros_like(self.kp)
        self.prev_error = np.zeros_like(self.kp)
        self._first     = True

    def compute(self, error: np.ndarray) -> np.ndarray:
        # Derivative: zero on first call to avoid spike
        if self._first:
            derivative    = np.zeros_like(error)
            self._first   = False
        else:
            derivative = (error - self.prev_error) / self.dt

        self.integral  += error * self.dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral,
                                    -self.integral_limit, self.integral_limit)

        self.prev_error = error.copy()

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.ctrl_limit is not None:
            output = np.clip(output, -self.ctrl_limit, self.ctrl_limit)
        return output


# ─────────────────────────────────────────────────────────────────────────────
def analytical_ik_2r(target_xy: np.ndarray,
                     elbow_up: bool = False) -> np.ndarray:
    """
    Closed-form IK for a 2R planar arm.

    Returns (q1, q2) [rad].  If the target is out of reach it is clamped to
    the workspace boundary.
    """
    dx, dy = target_xy[0], target_xy[1]
    r2 = dx ** 2 + dy ** 2
    r  = np.sqrt(r2)

    # Clamp to reachable workspace
    max_reach = L1 + L2 - 1e-4
    min_reach = abs(L1 - L2) + 1e-4
    if r > max_reach:
        s = max_reach / r
        dx, dy = dx * s, dy * s
        r2, r  = dx**2 + dy**2, max_reach
    elif r < min_reach:
        s = min_reach / r
        dx, dy = dx * s, dy * s
        r2, r  = dx**2 + dy**2, min_reach

    cos_q2 = (r2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sin_q2 = np.sqrt(max(0.0, 1.0 - cos_q2**2))
    if elbow_up:
        sin_q2 = -sin_q2
    q2 = np.arctan2(sin_q2, cos_q2)

    k1 = L1 + L2 * cos_q2
    k2 = L2 * sin_q2
    q1 = np.arctan2(dy, dx) - np.arctan2(k2, k1)

    return np.array([q1, q2])


def fk_2r(q: np.ndarray) -> np.ndarray:
    """Forward kinematics – returns (x, y) of end-effector."""
    q1, q2 = q
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([x, y])


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap to [-π, π]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def random_target(rng: np.random.Generator, margin: float = 0.15) -> np.ndarray:
    r_min = abs(L1 - L2) + margin
    r_max = L1 + L2 - margin
    r     = rng.uniform(r_min, r_max)
    theta = rng.uniform(-np.pi, np.pi)
    return np.array([r * np.cos(theta), r * np.sin(theta)])


# ─────────────────────────────────────────────────────────────────────────────
def run(target_xy: Optional[np.ndarray] = None,
        sim_time:  float = 5.0,
        seed:      int   = 42,
        elbow_up:  bool  = False) -> None:

    rng = np.random.default_rng(seed)

    # ── Load model ────────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path   = os.path.join(script_dir, '..', 'robots', '2R_robotic_arm.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # ── Body / joint IDs ──────────────────────────────────────────────────────
    ee_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  'ee_tip')
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  'target')

    # ── Set target ────────────────────────────────────────────────────────────
    if target_xy is None:
        target_xy = random_target(rng)
    target_xy = np.asarray(target_xy, dtype=float)

    model.body_pos[target_id][0] = target_xy[0]
    model.body_pos[target_id][1] = target_xy[1]
    model.body_pos[target_id][2] = 0.05  # same height as base
    mujoco.mj_forward(model, data)

    # ── IK → desired joint angles ─────────────────────────────────────────────
    q_des = analytical_ik_2r(target_xy, elbow_up=elbow_up)
    ee_ik = fk_2r(q_des)
    print("=" * 52)
    print(f"  Target          : ({target_xy[0]:+.4f}, {target_xy[1]:+.4f}) m")
    print(f"  IK solution     : q1={np.degrees(q_des[0]):+.1f}°, "
          f"q2={np.degrees(q_des[1]):+.1f}°")
    print(f"  IK FK check     : ({ee_ik[0]:+.4f}, {ee_ik[1]:+.4f}) m  "
          f"err={np.linalg.norm(ee_ik - target_xy)*1e3:.2f} mm")
    print("=" * 52)

    # ── Controller ────────────────────────────────────────────────────────────
    pid = PIDController(KP, KI, KD, DT, ctrl_limit=CTRL_LIMIT)
    pid.reset()

    # ── Logging ───────────────────────────────────────────────────────────────
    n_steps    = int(sim_time / DT)
    log_t      = np.zeros(n_steps)
    log_q      = np.zeros((n_steps, 2))
    log_q_des  = np.tile(q_des, (n_steps, 1))
    log_ee     = np.zeros((n_steps, 2))
    log_err    = np.zeros(n_steps)
    log_torque = np.zeros((n_steps, 2))
    first_success = None

    # ── Simulation ────────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 3.5
        viewer.cam.elevation = -35.0
        viewer.cam.azimuth   = 90.0

        for step in range(n_steps):
            if not viewer.is_running():
                break

            q  = data.qpos[:2].copy()
            dq = data.qvel[:2].copy()

            error  = wrap_angle(q_des - q)
            torque = pid.compute(error)
            data.ctrl[:2] = torque

            mujoco.mj_step(model, data)
            viewer.sync()

            ee_pos   = data.xpos[ee_id][:2].copy()
            dist_err = np.linalg.norm(ee_pos - target_xy)

            log_t[step]      = data.time
            log_q[step]      = q
            log_ee[step]     = ee_pos
            log_err[step]    = dist_err
            log_torque[step] = torque

            if first_success is None and dist_err < SUCCESS_DIST:
                first_success = data.time
                print(f"  [SUCCESS] EE within {SUCCESS_DIST*100:.0f} cm "
                      f"at t={first_success:.3f} s")

            time.sleep(DT * 0.3)  # slow-down for real-time viewing

    if first_success is None:
        print(f"  [INFO] Did not converge within {sim_time:.1f} s")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle('2R PID Reacher', fontsize=14, fontweight='bold')

    # 1. End-effector position error
    axes[0].plot(log_t, log_err * 100, 'tomato', linewidth=1.8, label='EE error')
    axes[0].axhline(SUCCESS_DIST * 100, color='green', linestyle='--',
                    linewidth=1.2, label=f'{SUCCESS_DIST*100:.0f} cm threshold')
    if first_success:
        axes[0].axvline(first_success, color='gray', linestyle=':', linewidth=1.2,
                        label=f'success @ {first_success:.2f}s')
    axes[0].set_ylabel('Error (cm)')
    axes[0].set_title('End-Effector Position Error')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.4)

    # 2. Joint angle tracking
    colors = ['steelblue', 'darkorange']
    for i in range(2):
        axes[1].plot(log_t, np.degrees(log_q[:, i]),
                     color=colors[i], linewidth=1.5, label=f'q{i+1} actual')
        axes[1].plot(log_t, np.degrees(log_q_des[:, i]),
                     color=colors[i], linewidth=1.2, linestyle='--',
                     label=f'q{i+1} desired')
    axes[1].set_ylabel('Angle (°)')
    axes[1].set_title('Joint Angle Tracking')
    axes[1].legend(ncol=2, loc='upper right')
    axes[1].grid(True, alpha=0.4)

    # 3. EE trajectory
    axes[2].plot(log_ee[:, 0], log_ee[:, 1], 'steelblue',
                 linewidth=1.5, label='EE path')
    axes[2].scatter(*target_xy, color='green', s=120, zorder=5,
                    marker='*', label='Target')
    axes[2].scatter(*log_ee[0], color='black', s=60, zorder=5,
                    marker='o', label='Start')
    # Workspace boundary
    theta_w = np.linspace(-np.pi, np.pi, 200)
    for r, ls in [(L1 + L2, '--'), (abs(L1 - L2), ':')]:
        axes[2].plot(r * np.cos(theta_w), r * np.sin(theta_w),
                     'gray', linestyle=ls, linewidth=0.8, alpha=0.6)
    axes[2].set_xlabel('x (m)')
    axes[2].set_ylabel('y (m)')
    axes[2].set_title('End-Effector Trajectory')
    axes[2].legend()
    axes[2].grid(True, alpha=0.4)
    axes[2].set_aspect('equal')

    plt.tight_layout()
    out_path = os.path.join(script_dir, 'pid_reacher_2R_results.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2R PID Reacher (MuJoCo)')
    parser.add_argument('--target', nargs=2, type=float, metavar=('X', 'Y'),
                        default=None, help='Target position in metres')
    parser.add_argument('--time',   type=float, default=5.0,
                        help='Simulation duration [s]  (default: 5)')
    parser.add_argument('--seed',   type=int,   default=42,
                        help='Random seed for target generation')
    parser.add_argument('--elbow-up', action='store_true',
                        help='Use elbow-up IK solution (default: elbow-down)')
    args = parser.parse_args()

    target = np.array(args.target) if args.target else None
    run(target_xy=target,
        sim_time=args.time,
        seed=args.seed,
        elbow_up=args.elbow_up)
