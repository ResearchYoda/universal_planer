"""pGraph (morphology graph) encoding for serial robot arm topologies.

Each joint is a node in the kinematic chain. Features per joint:
    [0] depth_norm   : normalized position in chain (0=base, 1=EE-side)
    [1] joint_type   : 0=revolute, 1=prismatic
    [2] lim_lo_norm  : lower limit / π
    [3] lim_hi_norm  : upper limit / π
    [4] mask         : 1 if real joint, 0 if padding slot
"""

import math
import torch

MAX_JOINTS = 8  # supports up to 8-DOF arms

# ── per-robot topology config ──────────────────────────────────────────────────
ROBOT_PGRAPH_CFG: dict[str, dict] = {
    "franka": {
        "n_joints": 7,
        "ee_body": "panda_hand",
        "joint_pattern": "panda_joint.*",
        "joint_limits": [
            (-2.8973,  2.8973),
            (-1.7628,  1.7628),
            (-2.8973,  2.8973),
            (-3.0718, -0.0698),
            (-2.8973,  2.8973),
            (-0.0175,  3.7525),
            (-2.8973,  2.8973),
        ],
        "joint_types": [0] * 7,
    },
    "ur10": {
        "n_joints": 6,
        "ee_body": "ee_link",
        "joint_pattern": ".*",
        "joint_limits": [
            (-6.2832, 6.2832),
            (-6.2832, 6.2832),
            (-3.1416, 3.1416),
            (-6.2832, 6.2832),
            (-6.2832, 6.2832),
            (-6.2832, 6.2832),
        ],
        "joint_types": [0] * 6,
    },
    "kinova_gen3": {
        "n_joints": 7,
        "ee_body": "end_effector_link",
        "joint_pattern": "joint_[1-7]",
        "joint_limits": [
            (-6.2832,  6.2832),
            (-2.4100,  2.4100),
            (-6.2832,  6.2832),
            (-2.6600,  2.6600),
            (-6.2832,  6.2832),
            (-2.2300,  2.2300),
            (-6.2832,  6.2832),
        ],
        "joint_types": [0] * 7,
    },
}


def compute_pgraph_features(robot_name: str, max_joints: int = MAX_JOINTS) -> torch.Tensor:
    """Return a fixed (max_joints, 5) pGraph feature matrix for the given robot.

    Padded joints have mask=0 and all other features zero.
    """
    cfg = ROBOT_PGRAPH_CFG[robot_name]
    n = cfg["n_joints"]
    limits = cfg["joint_limits"]
    jtypes = cfg["joint_types"]

    feats = torch.zeros(max_joints, 5)
    for i in range(n):
        depth = i / (n - 1) if n > 1 else 0.0
        lo = limits[i][0] / math.pi
        hi = limits[i][1] / math.pi
        feats[i] = torch.tensor([depth, float(jtypes[i]), lo, hi, 1.0])
    return feats  # (max_joints, 5)
