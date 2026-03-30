"""
Rodrigues Network (RodriNet) — full architecture.

Building blocks (Sec. 4 of the paper):

  RodriguesLayer  — updates link features from joints  (Eq. 9-10)
  JointLayer      — updates joint features from links  (Eq. 11)
  SelfAttentionLayer — global information exchange among links (Sec. 4.3)
  RodriguesBlock  — one stage: RodriguesLayer → JointLayer → SelfAttentionLayer
  RodriNet        — N stacked RodriguesBlocks with obs encoder/decoder
"""

import torch
import torch.nn as nn
from .operator import NeuralRodriguesOperator


# ---------------------------------------------------------------------------
# Rodrigues Layer
# ---------------------------------------------------------------------------

class RodriguesLayer(nn.Module):
    """
    Updates link features using the Neural Rodrigues Operator (Eq. 9-10).

    For each joint j connecting parent p_j → child c_j:
        F^trans_j   = Rodrigues(F^in_{p_j}, W*_j, Θ^in_j)
        F^out_{c_j} = LayerNorm(F^in_{c_j} + F^trans_j)

    Root link (no parent):
        F^out_0 = LayerNorm(F^in_0)
    """

    def __init__(self, C_L: int, C_J: int, joint_edges: list, n_links: int):
        super().__init__()
        self.C_L         = C_L
        self.joint_edges = joint_edges
        self.n_links     = n_links

        # One Rodrigues Kernel per joint (W*_j)
        self.kernels = nn.ModuleList([
            NeuralRodriguesOperator(C_L, C_L, C_J)
            for _ in joint_edges
        ])

        # One LayerNorm per link (operates on flattened C_L × 16 features)
        self.norms = nn.ModuleList([
            nn.LayerNorm(C_L * 16)
            for _ in range(n_links)
        ])

    def forward(
        self,
        link_feats: torch.Tensor,   # (B, n_links, C_L, 4, 4)
        joint_feats: torch.Tensor,  # (B, n_joints, C_J)
    ) -> torch.Tensor:
        B = link_feats.shape[0]

        # Build output list without in-place ops to keep autograd happy
        out_list = [link_feats[:, i] for i in range(self.n_links)]

        for j, (p_idx, c_idx) in enumerate(self.joint_edges):
            F_parent = link_feats[:, p_idx]  # (B, C_L, 4, 4)
            F_child  = link_feats[:, c_idx]  # (B, C_L, 4, 4)
            theta_j  = joint_feats[:, j]     # (B, C_J)

            F_trans = self.kernels[j](F_parent, theta_j)  # (B, C_L, 4, 4)

            combined = (F_child + F_trans).reshape(B, self.C_L * 16)
            out_list[c_idx] = self.norms[c_idx](combined).reshape(B, self.C_L, 4, 4)

        # Normalize root link (no parent → identity message)
        out_list[0] = self.norms[0](
            out_list[0].reshape(B, self.C_L * 16)
        ).reshape(B, self.C_L, 4, 4)

        return torch.stack(out_list, dim=1)


# ---------------------------------------------------------------------------
# Joint Layer
# ---------------------------------------------------------------------------

class JointLayer(nn.Module):
    """
    Updates joint features from child link features (Eq. 11).

        Θ^out_j = Linear_j(Flatten(F^in_{c_j})) + Θ^in_j
    """

    def __init__(self, C_L: int, C_J: int, joint_edges: list):
        super().__init__()
        self.joint_edges = joint_edges

        # Independent linear layer per joint
        self.linears = nn.ModuleList([
            nn.Linear(C_L * 16, C_J)
            for _ in joint_edges
        ])

    def forward(
        self,
        link_feats: torch.Tensor,   # (B, n_links, C_L, 4, 4)
        joint_feats: torch.Tensor,  # (B, n_joints, C_J)
    ) -> torch.Tensor:
        B = link_feats.shape[0]

        out_list = [
            self.linears[j](link_feats[:, c_idx].reshape(B, -1)) + joint_feats[:, j]
            for j, (_, c_idx) in enumerate(self.joint_edges)
        ]
        return torch.stack(out_list, dim=1)


# ---------------------------------------------------------------------------
# Self-Attention Layer
# ---------------------------------------------------------------------------

class SelfAttentionLayer(nn.Module):
    """
    Global self-attention across all link tokens (Sec. 4.3).

    Flow:
      link_feats  →  flatten  →  proj_in  →  MultiheadAttention
                  →  proj_out  →  residual add  →  LayerNorm
    """

    def __init__(self, C_L: int, d_model: int, n_heads: int):
        super().__init__()
        self.C_L = C_L

        self.proj_in  = nn.Linear(C_L * 16, d_model)
        self.attn     = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj_out = nn.Linear(d_model, C_L * 16)
        self.norm     = nn.LayerNorm(C_L * 16)

    def forward(self, link_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:  link_feats: (B, n_links, C_L, 4, 4)
        Returns:           (B, n_links, C_L, 4, 4)
        """
        B, n_links = link_feats.shape[:2]

        tokens = link_feats.reshape(B, n_links, self.C_L * 16)  # (B, n_links, C_L*16)
        tokens_proj = self.proj_in(tokens)                        # (B, n_links, d_model)

        attn_out, _ = self.attn(tokens_proj, tokens_proj, tokens_proj)

        out = self.norm(tokens + self.proj_out(attn_out))        # residual + norm
        return out.reshape(B, n_links, self.C_L, 4, 4)


# ---------------------------------------------------------------------------
# Rodrigues Block
# ---------------------------------------------------------------------------

class RodriguesBlock(nn.Module):
    """
    One Rodrigues Block = RodriguesLayer + JointLayer + SelfAttentionLayer.
    """

    def __init__(
        self,
        C_L: int,
        C_J: int,
        d_model: int,
        n_heads: int,
        joint_edges: list,
        n_links: int,
    ):
        super().__init__()
        self.rodrigues_layer = RodriguesLayer(C_L, C_J, joint_edges, n_links)
        self.joint_layer     = JointLayer(C_L, C_J, joint_edges)
        self.attn_layer      = SelfAttentionLayer(C_L, d_model, n_heads)

    def forward(self, link_feats, joint_feats):
        link_feats  = self.rodrigues_layer(link_feats, joint_feats)
        joint_feats = self.joint_layer(link_feats, joint_feats)
        link_feats  = self.attn_layer(link_feats)
        return link_feats, joint_feats


# ---------------------------------------------------------------------------
# RodriNet
# ---------------------------------------------------------------------------

class RodriNet(nn.Module):
    """
    Rodrigues Network: N stacked RodriguesBlocks.

    Encoding: observation → per-link features + per-joint features
    Processing: N × RodriguesBlock
    Decoding: flatten(link_feats, joint_feats) → output feature vector
    """

    def __init__(
        self,
        n_links: int,
        n_joints: int,
        joint_edges: list,
        obs_dim: int,
        C_L: int = 4,
        C_J: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 3,
        joint_angle_obs_idx: list = None,
        joint_vel_obs_idx: list = None,
    ):
        super().__init__()
        self.n_links   = n_links
        self.n_joints  = n_joints
        self.C_L       = C_L
        self.C_J       = C_J
        self.joint_edges          = joint_edges
        self.joint_angle_obs_idx  = joint_angle_obs_idx
        self.joint_vel_obs_idx    = joint_vel_obs_idx

        # ---- Observation Encoders ----------------------------------------
        # One MLP per link: obs → (C_L, 4, 4)
        self.link_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ELU(),
                nn.Linear(64, C_L * 16),
            )
            for _ in range(n_links)
        ])

        # One linear per joint: [angle, vel] → C_J
        joint_in = 2 if (joint_angle_obs_idx and joint_vel_obs_idx) else obs_dim
        self.joint_encoders = nn.ModuleList([
            nn.Linear(joint_in, C_J)
            for _ in range(n_joints)
        ])

        # ---- Rodrigues Blocks --------------------------------------------
        self.blocks = nn.ModuleList([
            RodriguesBlock(C_L, C_J, d_model, n_heads, joint_edges, n_links)
            for _ in range(n_blocks)
        ])

        # ---- Output dimension --------------------------------------------
        self.output_dim = n_links * C_L * 16 + n_joints * C_J

    def encode(self, obs: torch.Tensor):
        """Encode obs into initial link and joint feature tensors."""
        B = obs.shape[0]

        link_feats = torch.stack([
            enc(obs).reshape(B, self.C_L, 4, 4)
            for enc in self.link_encoders
        ], dim=1)  # (B, n_links, C_L, 4, 4)

        if self.joint_angle_obs_idx and self.joint_vel_obs_idx:
            joint_inputs = [
                torch.stack([
                    obs[:, self.joint_angle_obs_idx[j]],
                    obs[:, self.joint_vel_obs_idx[j]],
                ], dim=-1)
                for j in range(self.n_joints)
            ]
        else:
            joint_inputs = [obs] * self.n_joints

        joint_feats = torch.stack([
            self.joint_encoders[j](joint_inputs[j])
            for j in range(self.n_joints)
        ], dim=1)  # (B, n_joints, C_J)

        return link_feats, joint_feats

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim)
        Returns:
            features: (B, output_dim)
        """
        B = obs.shape[0]
        link_feats, joint_feats = self.encode(obs)

        for block in self.blocks:
            link_feats, joint_feats = block(link_feats, joint_feats)

        link_out  = link_feats.reshape(B, -1)   # (B, n_links·C_L·16)
        joint_out = joint_feats.reshape(B, -1)  # (B, n_joints·C_J)
        return torch.cat([link_out, joint_out], dim=-1)
