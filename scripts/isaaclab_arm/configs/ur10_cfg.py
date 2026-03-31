"""UR10 (6-DOF) reach environment with pGraph observations."""

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_assets import UR10_CFG  # isort: skip

from scripts.isaaclab_arm.env_cfg import UniversalArmReachEnvCfg


@configclass
class UR10PGraphReachEnvCfg(UniversalArmReachEnvCfg):
    """UR10 with pGraph universal obs. Action: 6 joint position targets."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot ─────────────────────────────────────────────────────────────
        self.scene.robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)

        # ── Asset cfg helpers ─────────────────────────────────────────────────
        # UR10 joint pattern ".*" selects all 6 arm joints
        arm_cfg = SceneEntityCfg("robot", joint_names=[".*"])
        ee_cfg = SceneEntityCfg("robot", body_names=["ee_link"])

        # ── Observations ──────────────────────────────────────────────────────
        self.observations.policy.pgraph.params["robot_name"] = "ur10"
        self.observations.policy.padded_joint_pos.params["asset_cfg"] = arm_cfg
        self.observations.policy.padded_joint_vel.params["asset_cfg"] = arm_cfg
        self.observations.policy.ee_pose.params["asset_cfg"] = ee_cfg

        # ── Actions (6-DOF joint position control) ────────────────────────────
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )

        # ── Rewards ───────────────────────────────────────────────────────────
        for rw in [
            self.rewards.end_effector_position_tracking,
            self.rewards.end_effector_position_tracking_fine_grained,
            self.rewards.end_effector_orientation_tracking,
        ]:
            rw.params["asset_cfg"].body_names = ["ee_link"]

        # ── Command goal pose ──────────────────────────────────────────────────
        self.commands.ee_pose.body_name = "ee_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class UR10PGraphReachEnvCfg_PLAY(UR10PGraphReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
