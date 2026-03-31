"""Custom observation terms for the pGraph universal arm environment."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from scripts.isaaclab_arm.pgraph import MAX_JOINTS, compute_pgraph_features


def pgraph_features(
    env: ManagerBasedEnv,
    robot_name: str,
    max_joints: int = MAX_JOINTS,
) -> torch.Tensor:
    """Broadcast pre-computed pGraph features to all parallel envs.

    Returns: (num_envs, max_joints * 5)  — static per episode
    """
    feats = compute_pgraph_features(robot_name, max_joints).to(env.device)
    return feats.flatten().unsqueeze(0).expand(env.num_envs, -1).clone()


def padded_joint_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_joints: int = MAX_JOINTS,
) -> torch.Tensor:
    """Arm joint positions zero-padded to max_joints.

    Returns: (num_envs, max_joints)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (N, n_active)
    out = torch.zeros(env.num_envs, max_joints, device=env.device, dtype=pos.dtype)
    out[:, : pos.shape[1]] = pos
    return out


def padded_joint_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_joints: int = MAX_JOINTS,
) -> torch.Tensor:
    """Arm joint velocities zero-padded to max_joints.

    Returns: (num_envs, max_joints)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    out = torch.zeros(env.num_envs, max_joints, device=env.device, dtype=vel.dtype)
    out[:, : vel.shape[1]] = vel
    return out


def ee_pose_in_env(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """End-effector pose relative to each env's origin.

    Returns: (num_envs, 7)  — [pos(3), quat_wxyz(4)]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # body_pose_w: (N, n_bodies, 7) with layout [x,y,z,qw,qx,qy,qz]
    pose = asset.data.body_pose_w[:, asset_cfg.body_ids, :7].reshape(env.num_envs, -1).clone()
    # make position env-relative
    pose[:, :3] -= env.scene.env_origins
    return pose  # (N, 7)


def padded_last_action(
    env: ManagerBasedEnv,
    max_joints: int = MAX_JOINTS,
) -> torch.Tensor:
    """Last applied action zero-padded to max_joints.

    Returns: (num_envs, max_joints)
    """
    act = env.action_manager.action  # (N, n_actions)
    out = torch.zeros(env.num_envs, max_joints, device=env.device, dtype=act.dtype)
    out[:, : act.shape[1]] = act
    return out
