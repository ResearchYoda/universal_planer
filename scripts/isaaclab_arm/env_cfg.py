"""Universal Arm Reach environment configuration with pGraph observations.

Observation layout (OBS_DIM = 78):
    pgraph_features   : MAX_JOINTS * 5 = 40  [depth, type, lim_lo, lim_hi, mask]
    padded_joint_pos  : MAX_JOINTS     =  8
    padded_joint_vel  : MAX_JOINTS     =  8
    ee_pose           : 7              [pos(3) + quat_wxyz(4)]
    pose_command      : 7              [pos(3) + quat_wxyz(4)]  (goal)
    padded_last_action: MAX_JOINTS     =  8
                        ---
    Total                              78
"""

from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import scripts.isaaclab_arm.mdp as mdp_arm
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from scripts.isaaclab_arm.pgraph import MAX_JOINTS

OBS_DIM = MAX_JOINTS * 5 + MAX_JOINTS + MAX_JOINTS + 7 + 7 + MAX_JOINTS  # 78


@configclass
class UniversalArmObsCfg:
    """Observation groups for the universal arm policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Fixed-size obs used by the pGraph Transformer policy."""

        # pGraph topology encoding — static per robot, broadcast to all envs
        pgraph = ObsTerm(
            func=mdp_arm.pgraph_features,
            params={"robot_name": MISSING, "max_joints": MAX_JOINTS},
        )
        # Joint states, padded to MAX_JOINTS
        padded_joint_pos = ObsTerm(
            func=mdp_arm.padded_joint_pos,
            params={"asset_cfg": MISSING, "max_joints": MAX_JOINTS},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        padded_joint_vel = ObsTerm(
            func=mdp_arm.padded_joint_vel,
            params={"asset_cfg": MISSING, "max_joints": MAX_JOINTS},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # Current EE pose and goal pose (both env-relative, 7D each)
        ee_pose = ObsTerm(
            func=mdp_arm.ee_pose_in_env,
            params={"asset_cfg": MISSING},
        )
        pose_command = ObsTerm(
            func=mdp_arm.generated_commands,
            params={"command_name": "ee_pose"},
        )
        # Last action, padded to MAX_JOINTS
        padded_last_action = ObsTerm(
            func=mdp_arm.padded_last_action,
            params={"max_joints": MAX_JOINTS},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class UniversalArmReachEnvCfg(ReachEnvCfg):
    """Base class: inherits scene, rewards, terminations from ReachEnvCfg.
    Replaces observations with fixed-size pGraph-enhanced obs.

    Subclasses must call super().__post_init__() and then fill in the MISSING
    values: robot, arm_asset_cfg, ee_asset_cfg, robot_name, ee_body_name.
    """

    observations: UniversalArmObsCfg = UniversalArmObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # scale down envs for faster iteration (override in subclass if needed)
        self.scene.num_envs = 1024
