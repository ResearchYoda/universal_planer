"""pGraph Transformer Actor-Critic for universal arm reach.

Observation layout assumed (OBS_DIM=78):
    [  0: 40]  pgraph features   — (MAX_JOINTS, 5) flattened
    [ 40: 48]  padded_joint_pos  — (MAX_JOINTS,)
    [ 48: 56]  padded_joint_vel  — (MAX_JOINTS,)
    [ 56: 63]  ee_pose           — (7,)  [pos(3) + quat(4)]
    [ 63: 70]  pose_command      — (7,)  goal pose
    [ 70: 78]  padded_last_action— (MAX_JOINTS,)

Token construction (10 tokens × d_model):
    Joint tokens 0..7 : cat(pgraph[i] (5), pos[i] (1), vel[i] (1), act[i] (1))  → 8 dims
    EE token       8  : cat(ee_pose (7), 0 (1))                                   → 8 dims
    Goal token     9  : cat(goal_pose (7), 0 (1))                                 → 8 dims

    Padding mask: pgraph[:, 4] == 0  →  those joint tokens are masked in Transformer.

Actor head : joint tokens 0..num_actions-1 → Linear(d_model, 1) each → (num_actions,)
Critic head: mean-pool valid tokens → Linear(d_model, 1) → scalar V(s)

Implements the rsl_rl ActorCritic interface so it can be plugged into OnPolicyRunner.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDict

from scripts.isaaclab_arm.pgraph import MAX_JOINTS

# ── observation slice constants ────────────────────────────────────────────────
_PG_END = MAX_JOINTS * 5           # 40
_JP_END = _PG_END + MAX_JOINTS     # 48
_JV_END = _JP_END + MAX_JOINTS     # 56
_EE_END = _JV_END + 7              # 63
_GO_END = _EE_END + 7              # 70
_AC_END = _GO_END + MAX_JOINTS     # 78
OBS_DIM = _AC_END                  # 78
TOKEN_DIM = 8                      # input dim of each token (before projection)
N_TOKENS = MAX_JOINTS + 2          # joint tokens + EE token + goal token


class PGraphTransformerActorCritic(nn.Module):
    """Universal arm actor-critic using pGraph Transformer.

    Compatible with rsl_rl OnPolicyRunner:
        actor_critic = PGraphTransformerActorCritic(obs, obs_groups, num_actions, **policy_cfg)
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        init_noise_std: float = 1.0,
        max_joints: int = MAX_JOINTS,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"[PGraphTransformerActorCritic] ignoring unknown kwargs: {list(kwargs)}")

        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.max_joints = max_joints
        self.d_model = d_model

        # ── Token projector (shared for all token types) ───────────────────────
        self.token_proj = nn.Linear(TOKEN_DIM, d_model)

        # ── Transformer encoder (shared actor + critic backbone) ───────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # avoid warning with norm_first
        )

        # ── Actor head: maps each joint token → scalar action mean ─────────────
        self.actor_head = nn.Linear(d_model, 1)

        # ── Critic head: maps pooled tokens → V(s) ────────────────────────────
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ELU(),
            nn.Linear(d_model, 1),
        )

        # ── Learnable log-std (per action) ─────────────────────────────────────
        self._log_std = nn.Parameter(torch.full((num_actions,), torch.log(torch.tensor(init_noise_std))))

        # Will be set by _update_distribution
        self.distribution: Normal | None = None

    # ── rsl_rl interface ───────────────────────────────────────────────────────

    def reset(self, dones=None):
        """No recurrent state to reset."""
        pass

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        obs_vec = self.get_actor_obs(obs)
        self._update_distribution(obs_vec)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs_vec = self.get_actor_obs(obs)
        self._update_distribution(obs_vec)
        return self.distribution.mean

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        obs_vec = self.get_critic_obs(obs)
        return self._critic_forward(obs_vec)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[k] for k in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[k] for k in self.obs_groups["critic"]], dim=-1)

    def update_normalization(self, obs: TensorDict):
        """No external normalizer (handled via Transformer's robustness)."""
        pass

    # ── Properties required by rsl_rl PPO ────────────────────────────────────

    @property
    def is_recurrent(self) -> bool:
        return False

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(-1)

    @property
    def std(self) -> torch.Tensor:
        return self._log_std.exp()

    # ── Internal forward passes ───────────────────────────────────────────────

    def _update_distribution(self, obs_vec: torch.Tensor):
        mean = self._actor_forward(obs_vec)
        std = self._log_std.exp().unsqueeze(0).expand_as(mean)
        self.distribution = Normal(mean, std)

    def _build_tokens_and_mask(self, obs_vec: torch.Tensor):
        """Unpack flat obs → token matrix and boolean padding mask.

        Returns:
            tokens: (B, N_TOKENS, TOKEN_DIM)
            mask:   (B, N_TOKENS) bool — True = VALID (attend), False = masked out
        """
        B = obs_vec.shape[0]
        pg = obs_vec[:, :_PG_END].reshape(B, self.max_joints, 5)    # (B,8,5)
        jp = obs_vec[:, _PG_END:_JP_END]                              # (B,8)
        jv = obs_vec[:, _JP_END:_JV_END]                              # (B,8)
        ee = obs_vec[:, _JV_END:_EE_END]                              # (B,7)
        go = obs_vec[:, _EE_END:_GO_END]                              # (B,7)
        ac = obs_vec[:, _GO_END:_AC_END]                              # (B,8)

        # Joint tokens: (B, max_joints, 8)
        joint_tokens = torch.cat([
            pg,
            jp.unsqueeze(-1),
            jv.unsqueeze(-1),
            ac.unsqueeze(-1),
        ], dim=-1)  # (B, 8, 8)

        # EE and goal tokens: pad to TOKEN_DIM=8
        pad = torch.zeros(B, 1, device=obs_vec.device)
        ee_token = torch.cat([ee, pad], dim=-1).unsqueeze(1)   # (B, 1, 8)
        go_token = torch.cat([go, pad], dim=-1).unsqueeze(1)   # (B, 1, 8)

        tokens = torch.cat([joint_tokens, ee_token, go_token], dim=1)  # (B, 10, 8)

        # Mask: joint slots where pgraph mask (dim 4) == 0 are padding
        joint_mask = pg[:, :, 4].bool()                                 # (B, 8)
        ego_mask = torch.ones(B, 2, dtype=torch.bool, device=obs_vec.device)
        mask = torch.cat([joint_mask, ego_mask], dim=1)                 # (B, 10)

        return tokens, mask

    def _encode(self, obs_vec: torch.Tensor):
        """Project tokens and run Transformer. Returns (B, N_TOKENS, d_model)."""
        tokens, mask = self._build_tokens_and_mask(obs_vec)
        tokens = self.token_proj(tokens)  # (B, 10, d_model)
        # src_key_padding_mask: True where tokens should be IGNORED
        tokens = self.transformer(tokens, src_key_padding_mask=~mask)
        return tokens, mask

    def _actor_forward(self, obs_vec: torch.Tensor) -> torch.Tensor:
        """Returns action means (B, num_actions)."""
        tokens, _ = self._encode(obs_vec)
        # Use first num_actions joint token outputs
        joint_out = tokens[:, : self.num_actions, :]      # (B, num_actions, d_model)
        return self.actor_head(joint_out).squeeze(-1)      # (B, num_actions)

    def _critic_forward(self, obs_vec: torch.Tensor) -> torch.Tensor:
        """Returns value estimates (B, 1)."""
        tokens, mask = self._encode(obs_vec)
        # Mean-pool over VALID tokens
        valid = tokens * mask.unsqueeze(-1).float()
        n_valid = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = valid.sum(dim=1) / n_valid               # (B, d_model)
        return self.critic_head(pooled)                   # (B, 1)

    def forward(self, obs_vec: torch.Tensor) -> torch.Tensor:
        """Convenience forward — returns action means."""
        return self._actor_forward(obs_vec)
