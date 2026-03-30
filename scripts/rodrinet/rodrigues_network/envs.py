"""
Kinematic tree definitions for MuJoCo environments.

HalfCheetah kinematic tree:
    torso (root)
    ├── bthigh  [joint 0]
    │   └── bshin   [joint 1]
    │       └── bfoot   [joint 2]
    └── fthigh  [joint 3]
        └── fshin   [joint 4]
            └── ffoot   [joint 5]

Links:  0=torso, 1=bthigh, 2=bshin, 3=bfoot, 4=fthigh, 5=fshin, 6=ffoot
Joints: 0=bthigh, 1=bshin, 2=bfoot, 3=fthigh, 4=fshin, 5=ffoot
"""

import numpy as np
import gymnasium as gym


class UprightWrapper(gym.Wrapper):
    """
    Prevents the common HalfCheetah local optimum of running upside-down.

    Two mechanisms:
      1. Upright bonus:  +upright_coef * cos(theta)  — rewarded for staying level
      2. Early termination: episode ends if |theta| > max_tilt_deg
    """

    def __init__(self, env, upright_coef: float = 2.0, max_tilt_deg: float = 70.0):
        super().__init__(env)
        self.upright_coef = upright_coef
        self.max_tilt     = np.radians(max_tilt_deg)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # obs[1] = torso pitch angle; normalise to [-π, π]
        theta = (obs[1] + np.pi) % (2 * np.pi) - np.pi

        # Smooth upright bonus: 0 when flat, -2·coef when upside-down
        reward += self.upright_coef * (np.cos(theta) - 1.0)

        # Terminate if too tilted (prevents wasted rollout time)
        if abs(theta) > self.max_tilt:
            terminated = True

        return obs, reward, terminated, truncated, info

HALFCHEETAH_TREE = {
    'n_links': 7,
    'n_joints': 6,
    # (parent_link_idx, child_link_idx) per joint
    'joint_edges': [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)],
    # HalfCheetah-v5 observation layout (17-dim):
    #   obs[0]    = z_torso
    #   obs[1]    = theta_torso
    #   obs[2:8]  = joint angles  (bthigh, bshin, bfoot, fthigh, fshin, ffoot)
    #   obs[8:11] = root velocities (x, z, theta)
    #   obs[11:17]= joint velocities
    'joint_angle_obs_idx': [2, 3, 4, 5, 6, 7],
    'joint_vel_obs_idx': [11, 12, 13, 14, 15, 16],
    'obs_dim': 17,
    'action_dim': 6,
    'env_id': 'HalfCheetah-v5',
}
