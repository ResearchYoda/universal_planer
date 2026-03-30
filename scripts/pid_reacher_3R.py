"""
3R Planar Robot – PID Reacher (Position Task)
=============================================
Damped-Least-Squares iterative IK  →  joint-space PID  →  MuJoCo simulation

Robot:   3R_robotic_arm.xml   (L1=0.60 m, L2=0.50 m, L3=0.40 m)
Control: Proportional-Integral-Derivative on joint angles.
         The robot is kinematically redundant (3 DOF, 2D task).
         Null-space projection is used to keep joints near a neutral posture.

Usage:
    python scripts/pid_reacher_3R.py
    python scripts/pid_reacher_3R.py --target 0.9 0.4
    python scripts/pid_reacher_3R.py --time 8 --seed 3
"""

import argparse
import os
import time
from typing import Optional

import matplotlib
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
L1 = 0.60  # link-1 length [m]
L2 = 0.50  # link-2 length [m]
L3 = 0.40  # link-3 length [m]

# ── PID gains (joint-space) ───────────────────────────────────────────────────
KP = np.array([55.0, 45.0, 30.0])
KI = np.array([ 0.8,  0.6,  0.5])
KD = np.array([ 9.0,  7.0,  5.0])

CTRL_LIMIT   = 50.0   # motor saturation [Nm]
DT           = 0.01   # simulation timestep – must match XML
SUCCESS_DIST = 0.02   # 2 cm end-effector threshold

# ── Iterative IK hyper-parameters ────────────────────────────────────────────
IK_ALPHA      = 0.8    # step size
IK_ITERS      = 500    # maximum iterations
IK_TOL        = 5e-5   # convergence tolerance [m]
IK_LAMBDA     = 0.04   # damping factor for DLS
IK_NULL_GAIN  = 0.15   # null-space posture attraction gain


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
        if self._first:
            derivative  = np.zeros_like(error)
            self._first = False
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
def fk_3r(q: np.ndarray) -> np.ndarray:
    """Forward kinematics – returns (x, y) of end-effector."""
    q1, q2, q3 = q
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2) + L3 * np.cos(q1 + q2 + q3)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2) + L3 * np.sin(q1 + q2 + q3)
    return np.array([x, y])


def jacobian_3r(q: np.ndarray) -> np.ndarray:
    """Analytical 2×3 position Jacobian for the 3R planar arm."""
    q1, q2, q3 = q
    s1,   c1   = np.sin(q1),          np.cos(q1)
    s12,  c12  = np.sin(q1 + q2),     np.cos(q1 + q2)
    s123, c123 = np.sin(q1 + q2 + q3), np.cos(q1 + q2 + q3)

    J = np.array([
        [-L1*s1 - L2*s12 - L3*s123,  -L2*s12 - L3*s123,  -L3*s123],
        [ L1*c1 + L2*c12 + L3*c123,   L2*c12 + L3*c123,   L3*c123],
    ])
    return J   # shape (2, 3)


def iterative_ik_3r(target_xy: np.ndarray,
                    q_init:    Optional[np.ndarray] = None,
                    q_neutral: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Damped Least Squares (DLS) iterative IK with null-space posture control.

    The redundancy is resolved by attracting joint-3 toward a "neutral" posture
    q_neutral (default: all zeros) via the null-space projector N = I - J†J.

    Returns the joint configuration q [rad] that places the EE at target_xy.
    """
    q       = np.zeros(3) if q_init    is None else q_init.copy()
    q_neut  = np.zeros(3) if q_neutral is None else q_neutral.copy()

    for _ in range(IK_ITERS):
        pos = fk_3r(q)
        err = target_xy - pos
        if np.linalg.norm(err) < IK_TOL:
            break

        J    = jacobian_3r(q)                                    # (2,3)
        JJT  = J @ J.T                                           # (2,2)
        A    = JJT + IK_LAMBDA**2 * np.eye(2)
        J_ps = J.T @ np.linalg.inv(A)                           # (3,2) DLS pseudo-inverse

        # Null-space projector  N = I - J†J
        N    = np.eye(3) - J_ps @ J                              # (3,3)
        dq_ns = N @ (q_neut - q)                                 # posture gradient

        dq = IK_ALPHA * (J_ps @ err + IK_NULL_GAIN * dq_ns)
        q  = np.clip(q + dq, -np.pi, np.pi)

    return q


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def random_target(rng: np.random.Generator, margin: float = 0.15) -> np.ndarray:
    r_max = L1 + L2 + L3 - margin
    r_min = 0.25
    r     = rng.uniform(r_min, r_max)
    theta = rng.uniform(-np.pi, np.pi)
    return np.array([r * np.cos(theta), r * np.sin(theta)])


# ─────────────────────────────────────────────────────────────────────────────
def run(target_xy: Optional[np.ndarray] = None,
        sim_time:  float = 5.0,
        seed:      int   = 7,
        q_neutral: Optional[np.ndarray] = None) -> None:

    rng = np.random.default_rng(seed)

    # ── Load model ────────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path   = os.path.join(script_dir, '..', 'robots', '3R_robotic_arm.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # ── Body IDs ──────────────────────────────────────────────────────────────
    ee_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ee_tip')
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')

    # ── Set target ────────────────────────────────────────────────────────────
    if target_xy is None:
        target_xy = random_target(rng)
    target_xy = np.asarray(target_xy, dtype=float)

    model.body_pos[target_id][0] = target_xy[0]
    model.body_pos[target_id][1] = target_xy[1]
    model.body_pos[target_id][2] = 0.05
    mujoco.mj_forward(model, data)

    # ── IK → desired joint angles ─────────────────────────────────────────────
    if q_neutral is None:
        q_neutral = np.array([0.0, np.pi / 4, -np.pi / 4])  # slightly folded

    q_des   = iterative_ik_3r(target_xy, q_neutral=q_neutral)
    ee_ik   = fk_3r(q_des)
    ik_err  = np.linalg.norm(ee_ik - target_xy)

    print("=" * 56)
    print(f"  Target          : ({target_xy[0]:+.4f}, {target_xy[1]:+.4f}) m")
    print(f"  IK solution     : q1={np.degrees(q_des[0]):+.1f}°, "
          f"q2={np.degrees(q_des[1]):+.1f}°, "
          f"q3={np.degrees(q_des[2]):+.1f}°")
    print(f"  IK FK check     : ({ee_ik[0]:+.4f}, {ee_ik[1]:+.4f}) m  "
          f"err={ik_err*1e3:.2f} mm")
    if ik_err > SUCCESS_DIST:
        print("  [WARN] IK did not fully converge – target may be near workspace "
              "boundary or a singularity.")
    print("=" * 56)

    # ── Controller ────────────────────────────────────────────────────────────
    pid = PIDController(KP, KI, KD, DT, ctrl_limit=CTRL_LIMIT)
    pid.reset()

    # ── Logging ───────────────────────────────────────────────────────────────
    n_steps    = int(sim_time / DT)
    log_t      = np.zeros(n_steps)
    log_q      = np.zeros((n_steps, 3))
    log_q_des  = np.tile(q_des, (n_steps, 1))
    log_ee     = np.zeros((n_steps, 2))
    log_err    = np.zeros(n_steps)
    log_torque = np.zeros((n_steps, 3))
    first_success = None

    # ── Simulation ────────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 4.0
        viewer.cam.elevation = -35.0
        viewer.cam.azimuth   = 90.0

        for step in range(n_steps):
            if not viewer.is_running():
                break

            q  = data.qpos[:3].copy()

            error  = wrap_angle(q_des - q)
            torque = pid.compute(error)
            data.ctrl[:3] = torque

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

            time.sleep(DT * 0.3)

    if first_success is None:
        print(f"  [INFO] Did not converge within {sim_time:.1f} s")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(11, 13))
    fig.suptitle('3R PID Reacher', fontsize=14, fontweight='bold')

    # 1. Position error
    axes[0].plot(log_t, log_err * 100, 'tomato', linewidth=1.8, label='EE error')
    axes[0].axhline(SUCCESS_DIST * 100, color='green', linestyle='--',
                    linewidth=1.2, label=f'{SUCCESS_DIST*100:.0f} cm threshold')
    if first_success:
        axes[0].axvline(first_success, color='gray', linestyle=':',
                        linewidth=1.2, label=f'success @ {first_success:.2f}s')
    axes[0].set_ylabel('Error (cm)')
    axes[0].set_title('End-Effector Position Error')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.4)

    # 2. Joint angle tracking
    colors = ['steelblue', 'darkorange', 'seagreen']
    for i in range(3):
        axes[1].plot(log_t, np.degrees(log_q[:, i]),
                     color=colors[i], linewidth=1.5, label=f'q{i+1} actual')
        axes[1].plot(log_t, np.degrees(log_q_des[:, i]),
                     color=colors[i], linewidth=1.2, linestyle='--',
                     label=f'q{i+1} desired')
    axes[1].set_ylabel('Angle (°)')
    axes[1].set_title('Joint Angle Tracking')
    axes[1].legend(ncol=3, loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.4)

    # 3. Control torques
    for i in range(3):
        axes[2].plot(log_t, log_torque[:, i],
                     color=colors[i], linewidth=1.2, label=f'τ{i+1}')
    axes[2].axhline( CTRL_LIMIT, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[2].axhline(-CTRL_LIMIT, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel('Torque (Nm)')
    axes[2].set_title('Control Torques')
    axes[2].legend(ncol=3, loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.4)

    # 4. EE trajectory
    axes[3].plot(log_ee[:, 0], log_ee[:, 1], 'steelblue',
                 linewidth=1.5, label='EE path')
    axes[3].scatter(*target_xy, color='green', s=120, zorder=5,
                    marker='*', label='Target')
    axes[3].scatter(*log_ee[0], color='black', s=60, zorder=5,
                    marker='o', label='Start')
    # Outer workspace boundary
    theta_w = np.linspace(-np.pi, np.pi, 300)
    r_max   = L1 + L2 + L3
    axes[3].plot(r_max * np.cos(theta_w), r_max * np.sin(theta_w),
                 'gray', linestyle='--', linewidth=0.8, alpha=0.6,
                 label='Max reach')
    axes[3].set_xlabel('x (m)')
    axes[3].set_ylabel('y (m)')
    axes[3].set_title('End-Effector Trajectory')
    axes[3].legend()
    axes[3].grid(True, alpha=0.4)
    axes[3].set_aspect('equal')

    plt.tight_layout()
    out_path = os.path.join(script_dir, 'pid_reacher_3R_results.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3R PID Reacher (MuJoCo)')
    parser.add_argument('--target', nargs=2, type=float, metavar=('X', 'Y'),
                        default=None, help='Target position in metres')
    parser.add_argument('--time',   type=float, default=5.0,
                        help='Simulation duration [s]  (default: 5)')
    parser.add_argument('--seed',   type=int,   default=7,
                        help='Random seed for target generation')
    args = parser.parse_args()

    target = np.array(args.target) if args.target else None
    run(target_xy=target, sim_time=args.time, seed=args.seed)
