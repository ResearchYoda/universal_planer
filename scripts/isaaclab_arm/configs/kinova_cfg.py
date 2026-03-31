"""Kinova Gen3 (7-DOF, no gripper) reach environment with pGraph observations."""

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_assets import KINOVA_GEN3_N7_CFG  # isort: skip

from scripts.isaaclab_arm.env_cfg import UniversalArmReachEnvCfg

# NOTE: Verify "end_effector_link" body name by running:
#   python -c "... print(robot.data.body_names)"
# after spawning the Kinova Gen3 USD in Isaac Sim.
KINOVA_GEN3_EE_BODY = "end_effector_link"


@configclass
class KinovaGen3PGraphReachEnvCfg(UniversalArmReachEnvCfg):
    """Kinova Gen3 with pGraph universal obs. Action: 7 joint position targets."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot ─────────────────────────────────────────────────────────────
        self.scene.robot = KINOVA_GEN3_N7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── Asset cfg helpers — each obs term gets its own instance ──────────
        ee_cfg = SceneEntityCfg("robot", body_names=[KINOVA_GEN3_EE_BODY])

        # ── Observations ──────────────────────────────────────────────────────
        self.observations.policy.pgraph.params["robot_name"] = "kinova_gen3"
        self.observations.policy.padded_joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["joint_[1-7]"]
        )
        self.observations.policy.padded_joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["joint_[1-7]"]
        )
        self.observations.policy.ee_pose.params["asset_cfg"] = ee_cfg

        # ── Actions (7-DOF joint position control) ────────────────────────────
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-7]"],
            scale=0.5,
            use_default_offset=True,
        )

        # ── Rewards ───────────────────────────────────────────────────────────
        for rw in [
            self.rewards.end_effector_position_tracking,
            self.rewards.end_effector_position_tracking_fine_grained,
            self.rewards.end_effector_orientation_tracking,
        ]:
            rw.params["asset_cfg"].body_names = [KINOVA_GEN3_EE_BODY]

        # ── Command goal pose ─────────────────────────────────────────────────
        self.commands.ee_pose.body_name = KINOVA_GEN3_EE_BODY
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class KinovaGen3PGraphReachEnvCfg_PLAY(KinovaGen3PGraphReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
