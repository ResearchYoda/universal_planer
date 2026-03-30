"""
Stable-Baselines3 compatible feature extractor wrapping RodriNet.
"""

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .network import RodriNet
from .envs import HALFCHEETAH_TREE


class RodriNetExtractor(BaseFeaturesExtractor):
    """
    RodriNet wrapped as an SB3 BaseFeaturesExtractor.

    The extractor encodes the flat observation into structured link/joint
    features, processes them through N RodriguesBlocks, and returns a flat
    feature vector for the actor/critic MLP heads.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        C_L: int = 4,
        C_J: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 3,
        env_tree: dict = None,
    ):
        tree = env_tree or HALFCHEETAH_TREE

        net = RodriNet(
            n_links=tree['n_links'],
            n_joints=tree['n_joints'],
            joint_edges=tree['joint_edges'],
            obs_dim=observation_space.shape[0],
            C_L=C_L,
            C_J=C_J,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            joint_angle_obs_idx=tree.get('joint_angle_obs_idx'),
            joint_vel_obs_idx=tree.get('joint_vel_obs_idx'),
        )

        super().__init__(observation_space, features_dim=net.output_dim)
        self.net = net

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
