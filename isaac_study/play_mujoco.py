from __future__ import annotations

"""Deploy a trained Isaac Lab UR5 reach policy in MuJoCo.

This script loads an RSL-RL checkpoint trained in Isaac Lab and runs it
in a MuJoCo simulation with the UR5e model from mujoco_menagerie.

The observation space is reconstructed to match Isaac Lab's:
    [0:6]   joint_pos_rel   (current - default joint positions)
    [6:12]  joint_vel       (joint velocities)
    [12:19] pose_command    (target pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
                            in the robot BASE frame
    [19:25] last_action     (previous action sent to the robot)

Actions are joint position deltas scaled by 0.5, added to default positions.

Usage:
    python play_mujoco.py
    python play_mujoco.py --checkpoint logs/rsl_rl/ur5_reach/<run>/model_950.pt
    python play_mujoco.py --target 0.4 0.1 0.3
"""

import argparse
import glob
import os

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn


# ─── Policy Network (must match RSL-RL architecture) ──────────────────

class ActorNetwork(nn.Module):
    """Standalone actor network matching the RSL-RL MLP architecture.

    RSL-RL's MLP is an nn.Sequential with layers indexed as:
        0: Linear(25, 256)
        1: ELU
        2: Linear(256, 128)
        3: ELU
        4: Linear(128, 64)
        5: ELU
        6: Linear(64, 6)

    In the checkpoint state_dict, these are stored as:
        actor.0.weight, actor.0.bias, actor.2.weight, etc.
    """

    def __init__(self, obs_dim: int = 25, act_dim: int = 6, hidden_dims: tuple = (256, 128, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, act_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def load_policy_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> ActorNetwork:
    """Load the actor network weights from an RSL-RL checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Extract the model state dict
    if "model_state_dict" in checkpoint:
        full_state_dict = checkpoint["model_state_dict"]
    else:
        full_state_dict = checkpoint

    # Debug: print ALL keys so we can verify the naming
    print("\n[DEBUG] Checkpoint keys:")
    for key, value in full_state_dict.items():
        print(f"  {key:50s} shape={list(value.shape)}")

    # Filter actor weights — keep the 'actor.' prefix since our module also uses 'actor'
    actor_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith("actor."):
            actor_state_dict[key] = value

    print(f"\n[DEBUG] Actor state_dict keys being loaded:")
    for key, value in actor_state_dict.items():
        print(f"  {key:50s} shape={list(value.shape)}")

    # Create and load the actor network
    policy = ActorNetwork()

    # Print what our model expects
    print(f"\n[DEBUG] Model expects these keys:")
    for key, value in policy.state_dict().items():
        print(f"  {key:50s} shape={list(value.shape)}")

    policy.load_state_dict(actor_state_dict, strict=False)
    policy.eval()
    policy.to(device)

    # Verify weights were loaded (not all zeros)
    first_weight = next(iter(policy.actor.parameters()))
    print(f"\n[DEBUG] First layer weight stats: mean={first_weight.mean():.4f}, std={first_weight.std():.4f}")
    print(f"[INFO] Loaded policy with {sum(p.numel() for p in policy.parameters()):,} parameters")
    return policy


def find_latest_checkpoint(experiment_name: str = "ur5_reach") -> str:
    """Find the latest checkpoint from the most recent training run."""
    log_root = os.path.join("logs", "rsl_rl", experiment_name)
    log_root = os.path.abspath(log_root)

    run_dirs = sorted(glob.glob(os.path.join(log_root, "*")))
    if not run_dirs:
        raise FileNotFoundError(f"No training runs found in {log_root}")

    latest_run = run_dirs[-1]
    checkpoints = sorted(glob.glob(os.path.join(latest_run, "model_*.pt")))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {latest_run}")

    return checkpoints[-1]


# ─── Quaternion utilities ─────────────────────────────────────────────

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def rotate_vector_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a quaternion [w, x, y, z]."""
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return result[1:4]


def world_to_base_frame(pos_w: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
    """Transform a world-frame position to the robot base frame.

    Args:
        pos_w: Position in world frame [x, y, z]
        base_pos: Robot base position in world frame
        base_quat: Robot base orientation as quaternion [w, x, y, z]
    """
    # Translate to base origin
    pos_rel = pos_w - base_pos
    # Rotate by inverse of base orientation
    base_quat_inv = quat_conjugate(base_quat)
    return rotate_vector_by_quat(pos_rel, base_quat_inv)


# ─── MuJoCo UR5 Environment ──────────────────────────────────────────

class MuJoCoUR5Env:
    """MuJoCo UR5e environment with Isaac Lab-compatible observation/action spaces."""

    # Default joint positions (must match Isaac Lab ur5_cfg.py init_state)
    DEFAULT_QPOS = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

    # Joint names in order
    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # Action scale (must match Isaac Lab JointPositionActionCfg scale=0.5)
    ACTION_SCALE = 0.5

    def __init__(self, xml_path: str, target_pos_world: np.ndarray | None = None):
        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ── Fix 1: Disable self-collision to match Isaac Lab training ──
        # Isaac Lab trained with enabled_self_collisions=False
        for i in range(self.model.ngeom):
            self.model.geom_contype[i] = 0
            self.model.geom_conaffinity[i] = 0

        # ── Fix 2: Remove the 180° Z base rotation ──
        # MuJoCo UR5e XML has base quat="0 0 0 -1" (180° Z rotation)
        # but Isaac Lab uses identity rotation (1, 0, 0, 0).
        base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.model.body_quat[base_body_id] = [1.0, 0.0, 0.0, 0.0]

        # ── Fix 3: Disable gravity to match Isaac Lab (disable_gravity=True) ──
        self.model.opt.gravity[:] = [0.0, 0.0, 0.0]

        # ── Fix 4: Match actuator PD gains to Isaac Lab ──
        # Isaac Lab implicit actuator: torque = Kp*(target - current) - Kd*vel
        # MuJoCo general actuator:     force  = gain*ctrl + bias[0] + bias[1]*qpos + bias[2]*qvel
        # Mapping: gain=Kp, biasprm=[0, -Kp, -Kd]
        #
        # Isaac Lab UR5 config:
        #   shoulder (0,1): Kp=800,  Kd=40
        #   elbow    (2):   Kp=400,  Kd=20
        #   wrist    (3-5): Kp=200,  Kd=10
        actuator_gains = {
            0: (800.0, 40.0),   # shoulder_pan
            1: (800.0, 40.0),   # shoulder_lift
            2: (400.0, 20.0),   # elbow
            3: (200.0, 10.0),   # wrist_1
            4: (200.0, 10.0),   # wrist_2
            5: (200.0, 10.0),   # wrist_3
        }
        for act_id, (kp, kd) in actuator_gains.items():
            self.model.actuator_gainprm[act_id, 0] = kp
            self.model.actuator_biasprm[act_id, 0] = 0.0
            self.model.actuator_biasprm[act_id, 1] = -kp
            self.model.actuator_biasprm[act_id, 2] = -kd

        # Simulation timestep
        # Isaac Lab uses dt=1/60 with decimation=2, but MuJoCo needs more
        # sub-steps for the PD controller to settle between policy steps.
        self.model.opt.timestep = 1.0 / 120.0
        self.decimation = 10  # More sub-steps for smoother response
        self.control_dt = self.model.opt.timestep * self.decimation

        # Get joint and body indices
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.JOINT_NAMES
        ]
        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link"
        )
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )

        # Target position in WORLD frame
        if target_pos_world is None:
            self.target_pos_world = np.array([0.4, 0.0, 0.3])
        else:
            self.target_pos_world = np.array(target_pos_world)

        # Target orientation in base frame (quaternion [w, x, y, z])
        # For a reach task with pitch=pi/2, this is a 90° rotation
        self.target_quat_b = np.array([0.707, 0.707, 0.0, 0.0])  # 90° pitch

        # State tracking
        self.last_action = np.zeros(6)
        self.smoothed_action = np.zeros(6)

        # Reset to initial state
        self.reset()

    def reset(self):
        """Reset the robot to default joint positions."""
        mujoco.mj_resetData(self.model, self.data)

        # Set default joint positions
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = self.DEFAULT_QPOS[i]

        # Set actuator controls to default positions
        for i in range(6):
            self.data.ctrl[i] = self.DEFAULT_QPOS[i]

        self.last_action = np.zeros(6)
        self.smoothed_action = np.zeros(6)

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

    def get_base_pos(self) -> np.ndarray:
        """Get robot base position in world frame."""
        return self.data.xpos[self.base_body_id].copy()

    def get_base_quat(self) -> np.ndarray:
        """Get robot base orientation as [w, x, y, z]."""
        mat = self.data.xmat[self.base_body_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return quat

    def get_ee_pos(self) -> np.ndarray:
        """Get end-effector (wrist_3_link) position in world frame."""
        return self.data.xpos[self.ee_body_id].copy()

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        return np.array([self.data.qpos[jid] for jid in self.joint_ids])

    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        return np.array([self.data.qvel[jid] for jid in self.joint_ids])

    def compute_observation(self) -> np.ndarray:
        """Compute the 25-dim observation matching Isaac Lab's format.

        [0:6]   joint_pos_rel = current_qpos - default_qpos
        [6:12]  joint_vel = current joint velocities
        [12:19] pose_command = [pos_x, pos_y, pos_z, qw, qx, qy, qz] in BASE frame
        [19:25] last_action = previous action
        """
        joint_pos = self.get_joint_positions()
        joint_vel = self.get_joint_velocities()
        joint_pos_rel = joint_pos - self.DEFAULT_QPOS

        # Transform target position from WORLD frame to ROBOT BASE frame
        # This is critical — Isaac Lab commands are in the base frame!
        base_pos = self.get_base_pos()
        base_quat = self.get_base_quat()
        target_pos_b = world_to_base_frame(self.target_pos_world, base_pos, base_quat)

        # Pose command in base frame: [pos_x, pos_y, pos_z, qw, qx, qy, qz]
        pose_command = np.concatenate([target_pos_b, self.target_quat_b])

        # Build full observation
        obs = np.concatenate([
            joint_pos_rel,    # [0:6]
            joint_vel,        # [6:12]
            pose_command,     # [12:19]
            self.last_action, # [19:25]
        ])

        return obs

    def step(self, action: np.ndarray, smoothing_alpha: float = 0.3):
        """Apply action and step the simulation.

        Action: 6-dim joint position deltas from the policy.
        Target = default_qpos + ACTION_SCALE * smoothed_action

        Args:
            action: Raw policy output.
            smoothing_alpha: EMA alpha (0=ignore new, 1=no smoothing). Default 0.3.
        """
        # Clamp raw action to prevent extreme commands
        action = np.clip(action, -2.0, 2.0)

        # Exponential moving average for smooth transitions
        self.smoothed_action = smoothing_alpha * action + (1.0 - smoothing_alpha) * self.smoothed_action

        target_qpos = self.DEFAULT_QPOS + self.ACTION_SCALE * self.smoothed_action

        for i in range(6):
            self.data.ctrl[i] = target_qpos[i]

        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        self.last_action = action.copy()

        obs = self.compute_observation()
        ee_pos = self.get_ee_pos()
        distance = np.linalg.norm(ee_pos - self.target_pos_world)

        return obs, ee_pos, distance


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deploy trained UR5 reach policy in MuJoCo.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--xml", type=str,
        default=os.path.expanduser("~/VS_Projects/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"),
    )
    parser.add_argument("--target", type=float, nargs=3, default=[0.4, 0.0, 0.3])
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument(
        "--smoothing", type=float, default=0.3,
        help="Action smoothing alpha (0=very smooth, 1=no smoothing). Default 0.3.",
    )
    args = parser.parse_args()

    # Load
    if args.checkpoint is None:
        checkpoint_path = find_latest_checkpoint()
    else:
        checkpoint_path = args.checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    policy = load_policy_from_checkpoint(checkpoint_path)

    # Create env
    print(f"[INFO] Loading MuJoCo model: {args.xml}")
    env = MuJoCoUR5Env(xml_path=args.xml, target_pos_world=args.target)

    print(f"\n{'='*60}")
    print(f"  UR5 Reach — MuJoCo Deployment")
    print(f"  Target (world): {args.target}")
    print(f"  Threshold: {args.threshold} m")
    print(f"  Action smoothing: {args.smoothing}")
    print(f"{'='*60}\n")

    # Debug: print initial observations
    obs = env.compute_observation()
    print(f"[DEBUG] Initial observation (25-dim):")
    print(f"  joint_pos_rel [0:6]:  {obs[0:6]}")
    print(f"  joint_vel    [6:12]:  {obs[6:12]}")
    print(f"  pose_cmd    [12:19]:  {obs[12:19]}")
    print(f"  last_action [19:25]:  {obs[19:25]}")
    print(f"  Base pos:  {env.get_base_pos()}")
    print(f"  Base quat: {env.get_base_quat()}")
    print(f"  EE pos:    {env.get_ee_pos()}")
    print()

    step_count = 0
    reached = False
    target_pos = args.target

    def render_callback(viewer):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0, 0],
            pos=np.array(target_pos),
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32),
        )
        viewer.user_scn.ngeom = 1

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        render_callback(viewer)

        while viewer.is_running():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                action = policy(obs_tensor).squeeze(0).numpy()

            # Debug: print first few actions
            if step_count < 5:
                print(f"[DEBUG] Step {step_count}: action = {action}")

            obs, ee_pos, distance = env.step(action, smoothing_alpha=args.smoothing)
            step_count += 1

            if step_count % 100 == 0:
                print(
                    f"  Step {step_count:5d} | "
                    f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] | "
                    f"Dist: {distance:.4f} m"
                )

            if not reached and distance < args.threshold:
                reached = True
                elapsed = step_count * env.control_dt
                print(f"\n  ✓ TARGET REACHED in {step_count} steps ({elapsed:.2f}s)!")
                print(f"    EE: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
                print(f"    Distance: {distance:.4f} m")
                print(f"    (Close viewer to exit)\n")

            render_callback(viewer)
            viewer.sync()

    ee_pos = env.get_ee_pos()
    distance = np.linalg.norm(ee_pos - np.array(args.target))
    print(f"\n[RESULT] Final EE:    [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
    print(f"[RESULT] Target:      [{args.target[0]:.4f}, {args.target[1]:.4f}, {args.target[2]:.4f}]")
    print(f"[RESULT] Distance:    {distance:.4f} m")
    print(f"[RESULT] Reached:     {'YES ✓' if reached else 'NO ✗'}")


if __name__ == "__main__":
    main()
