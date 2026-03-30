# Copyright (c) 2024, Custom Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""UR5 Reach environment configuration.

Inherits from the built-in Isaac Lab reach task and swaps the robot
to a UR5 configuration. The task is simple: move the end-effector
to a randomly sampled target pose in 3D space.
"""

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

# Import our UR5 config
from .ur5_cfg import UR5_CFG


@configclass
class UR5ReachEnvCfg(ReachEnvCfg):
    """Configuration for the UR5 reach end-effector pose tracking environment.

    Uses 4096 parallel environments by default for fast GPU-accelerated training.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # -- Scene: switch robot to UR5
        self.scene.robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5

        # -- Events: reset joint randomization (narrower range for UR5)
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)

        # -- Domain Randomization: for sim-to-sim transfer robustness
        # Randomize actuator stiffness/damping by ±30%
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": (0.7, 1.3),
                "damping_distribution_params": (0.7, 1.3),
                "operation": "scale",
            },
        )

        # Randomize joint friction
        self.events.randomize_joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "friction_distribution_params": (0.5, 1.5),
                "operation": "scale",
            },
        )

        # -- Observation noise is already enabled via the parent config (Unoise +-0.01)

        # -- Rewards: set end-effector body name for tracking
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["wrist_3_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["wrist_3_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["wrist_3_link"]

        # -- Increase position tracking reward weight for more precise reaching
        self.rewards.end_effector_position_tracking.weight = -0.5
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.3

        # -- Actions: joint position control with scaling
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )

        # -- Commands: end-effector target pose ranges (tuned for UR5 reach ~850mm)
        self.commands.ee_pose.body_name = "wrist_3_link"
        self.commands.ee_pose.ranges.pos_x = (0.25, 0.55)
        self.commands.ee_pose.ranges.pos_y = (-0.25, 0.25)
        self.commands.ee_pose.ranges.pos_z = (0.15, 0.45)
        # end-effector is along x-direction
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class UR5ReachEnvCfg_PLAY(UR5ReachEnvCfg):
    """Configuration for playing/evaluating the trained UR5 reach policy."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # smaller scene for visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable observation noise for cleaner evaluation
        self.observations.policy.enable_corruption = False
