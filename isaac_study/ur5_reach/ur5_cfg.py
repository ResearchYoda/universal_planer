# Copyright (c) 2024, Custom Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots UR5 arm.

The UR5 is a 6-DOF collaborative robot arm with ~850mm reach.
Since Isaac Lab does not ship a UR5 USD by default, we use the UR10e USD
model and adjust the actuator gains and effort limits to approximate UR5 dynamics.

Joint structure (same across UR family):
    - shoulder_pan_joint
    - shoulder_lift_joint
    - elbow_joint
    - wrist_1_joint
    - wrist_2_joint
    - wrist_3_joint

UR5 effort limits (from datasheet):
    - Shoulder: 150 Nm
    - Elbow: 150 Nm
    - Wrist: 28 Nm
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

UR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5708,
            "elbow_joint": 1.5708,
            "wrist_1_joint": -1.5708,
            "wrist_2_joint": -1.5708,
            "wrist_3_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # Shoulder joints: UR5 rated at 150 Nm
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            effort_limit_sim=150.0,
            velocity_limit_sim=3.14,
            stiffness=800.0,
            damping=40.0,
            friction=0.0,
            armature=0.0,
        ),
        # Elbow joint: UR5 rated at 150 Nm
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=3.14,
            stiffness=400.0,
            damping=20.0,
            friction=0.0,
            armature=0.0,
        ),
        # Wrist joints: UR5 rated at 28 Nm
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            effort_limit_sim=28.0,
            velocity_limit_sim=6.28,
            stiffness=200.0,
            damping=10.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""Configuration of UR5 arm using implicit actuator models.

Uses the UR10e USD model with adjusted PD gains and effort limits
to approximate UR5 dynamics.
"""
