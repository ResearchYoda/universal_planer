"""Franka Panda (7-DOF) reach environment with pGraph observations."""

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip

from scripts.isaaclab_arm.env_cfg import UniversalArmReachEnvCfg


@configclass
class FrankaPGraphReachEnvCfg(UniversalArmReachEnvCfg):
    """Franka Panda with pGraph universal obs. Action: 7 joint position targets."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot ────────────────────────────────────────────────────────────
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── Asset cfg helpers ────────────────────────────────────────────────
        arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
        ee_cfg = SceneEntityCfg("robot", body_names=["panda_hand"])

        # ── Observations ─────────────────────────────────────────────────────
        self.observations.policy.pgraph.params["robot_name"] = "franka"
        self.observations.policy.padded_joint_pos.params["asset_cfg"] = arm_cfg
        self.observations.policy.padded_joint_vel.params["asset_cfg"] = arm_cfg
        self.observations.policy.ee_pose.params["asset_cfg"] = ee_cfg

        # ── Actions (7-DOF joint position control) ────────────────────────────
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )

        # ── Rewards ───────────────────────────────────────────────────────────
        for rw in [
            self.rewards.end_effector_position_tracking,
            self.rewards.end_effector_position_tracking_fine_grained,
            self.rewards.end_effector_orientation_tracking,
        ]:
            rw.params["asset_cfg"].body_names = ["panda_hand"]

        # ── Command goal pose ─────────────────────────────────────────────────
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class FrankaPGraphReachEnvCfg_PLAY(FrankaPGraphReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
