"""
scripts/universal_locomotion/ppo.py
====================================
Self-contained PPO with a Transformer-based morphology-aware Actor-Critic.

Architecture (v3 — Graph-aware Transformer)
--------------------------------------------
obs (139-dim) is parsed into three groups of tokens:

  Morph tokens  (16 × 5)  – one token per Pgraph slot
                             [pgraph_norm, jdof_norm, jtype_norm, body_mass_norm, body_mask]
  Joint tokens  ( 8 × 6)  – one token per actuated DOF slot
                             [joint_pos, joint_vel, dof_mask, lim_lo_norm, lim_hi_norm, gear_norm]
  Root token    ( 1 × 11) – root body state

Each token is projected to d_model=256, then all 25 tokens pass through a
2-layer, 4-head Transformer Encoder (pre-LN) with a key-padding mask that
blanks out Pgraph-padding and unused-DOF tokens.

Actor : joint token outputs → per-joint Gaussian (mean + log_std from shared
        Linear(256→2)), masked by dof_mask so padding joints output action=0
        and contribute 0 to log_prob / entropy.

Critic: mean-pool over all valid tokens → Linear(256→1).

Why Transformer over GNN?
  Self-attention is generalised message-passing with learned adjacency weights.
  The Transformer attends across all Pgraph nodes in every layer, learning
  which body–body relationships matter for each robot topology without needing
  hand-crafted edges. Padding masks enforce that invalid (padded) tokens are
  never attended to.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ── Observation layout constants (must match universal_env.py) ────────────────
N_MORPH    = 16
N_JOINT    = 8
NODE_FEAT  = 5
JOINT_FEAT = 6
ROOT_FEAT  = 11
N_TOKENS   = N_MORPH + N_JOINT + 1     # 25
OBS_DIM    = N_MORPH * NODE_FEAT + N_JOINT * JOINT_FEAT + ROOT_FEAT  # 139

# ── Transformer hyper-parameters ──────────────────────────────────────────────
D_MODEL  = 256
N_HEADS  = 4
N_LAYERS = 2
D_FF     = D_MODEL * 4   # 1024

LOG_STD_MIN = -5.0
LOG_STD_MAX =  2.0


# ── Actor-Critic ─────────────────────────────────────────────────────────────
class UniversalActorCritic(nn.Module):

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = N_JOINT,
                 d_model: int = D_MODEL):
        super().__init__()

        # ── Token input projections ───────────────────────────────────
        self.morph_proj = nn.Linear(NODE_FEAT,  d_model)
        self.joint_proj = nn.Linear(JOINT_FEAT, d_model)
        self.root_proj  = nn.Linear(ROOT_FEAT,  d_model)

        # ── Transformer encoder (pre-LN for stability) ────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=0.0,
            batch_first=True,
            norm_first=True,      # pre-LN: more stable than post-LN
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)

        # ── Actor: per-joint [mean_raw, log_std] ──────────────────────
        self.actor_proj  = nn.Linear(d_model, 2)

        # ── Critic: scalar value from pooled features ─────────────────
        self.critic_proj = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small init for actor output (keeps initial actions near zero)
        nn.init.orthogonal_(self.actor_proj.weight, gain=0.01)
        nn.init.zeros_(self.actor_proj.bias)
        nn.init.orthogonal_(self.critic_proj.weight, gain=1.0)
        nn.init.zeros_(self.critic_proj.bias)

    # ── Core encoder ─────────────────────────────────────────────────────────
    def _encode(self, obs: torch.Tensor):
        """
        Parse obs → tokenize → Transformer → outputs.

        Returns
        -------
        joint_out  : (B, 8, d_model)  per-joint token outputs
        pooled     : (B, d_model)     mean-pool of valid tokens for critic
        dof_mask_f : (B, 8)           float 1/0 for valid/padding joints
        """
        B = obs.shape[0]

        # ── Parse obs into token matrices ─────────────────────────────
        morph_t = obs[:, :80].reshape(B, N_MORPH, NODE_FEAT)    # (B, 16, 5)
        joint_t = obs[:, 80:128].reshape(B, N_JOINT, JOINT_FEAT) # (B,  8, 6)
        root_t  = obs[:, 128:]                                    # (B, 11)

        # ── Masks: True = token should be IGNORED (padding) ───────────
        # body_mask is feature index 4 in morph tokens; dof_mask is index 2
        body_pad  = (morph_t[:, :, 4] < 0.5)         # (B, 16) bool
        dof_pad   = (joint_t[:, :, 2] < 0.5)         # (B,  8) bool
        dof_mask_f = (~dof_pad).float()               # (B,  8) float 1=valid

        root_valid = torch.zeros(B, 1, dtype=torch.bool, device=obs.device)
        key_pad_mask = torch.cat([body_pad, dof_pad, root_valid], dim=1)  # (B, 25)

        # ── Embed tokens ──────────────────────────────────────────────
        t_morph = self.morph_proj(morph_t)                # (B, 16, d)
        t_joint = self.joint_proj(joint_t)                # (B,  8, d)
        t_root  = self.root_proj(root_t).unsqueeze(1)     # (B,  1, d)
        tokens  = torch.cat([t_morph, t_joint, t_root], dim=1)  # (B, 25, d)

        # ── Transformer (with padding mask) ───────────────────────────
        out = self.transformer(tokens, src_key_padding_mask=key_pad_mask)  # (B, 25, d)

        # ── Extract joint token outputs ───────────────────────────────
        joint_out = out[:, N_MORPH:N_MORPH + N_JOINT, :]  # (B, 8, d)

        # ── Critic: mean-pool over all valid tokens ───────────────────
        valid_f = (~key_pad_mask).float().unsqueeze(-1)    # (B, 25, 1)
        pooled  = (out * valid_f).sum(1) / valid_f.sum(1).clamp(min=1.0)  # (B, d)

        return joint_out, pooled, dof_mask_f

    # ── Action sampling ───────────────────────────────────────────────────────
    def get_action(self, obs: torch.Tensor):
        joint_out, pooled, dof_mask = self._encode(obs)

        actor_out = self.actor_proj(joint_out)                    # (B, 8, 2)
        mean      = torch.tanh(actor_out[..., 0]) * dof_mask     # (B, 8)
        log_std   = actor_out[..., 1].clamp(LOG_STD_MIN, LOG_STD_MAX)

        # For padding joints keep std=1 so Normal is never degenerate
        std = log_std.exp() * dof_mask + (1.0 - dof_mask)        # (B, 8)

        dist   = Normal(mean, std)
        action = dist.sample() * dof_mask                         # mask padding

        log_prob = (dist.log_prob(action) * dof_mask).sum(-1, keepdim=True)  # (B, 1)
        value    = self.critic_proj(pooled)                       # (B, 1)
        return action, log_prob, value

    # ── Log-prob / entropy for PPO update ────────────────────────────────────
    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        joint_out, pooled, dof_mask = self._encode(obs)

        actor_out = self.actor_proj(joint_out)
        mean      = torch.tanh(actor_out[..., 0]) * dof_mask
        log_std   = actor_out[..., 1].clamp(LOG_STD_MIN, LOG_STD_MAX)
        std       = log_std.exp() * dof_mask + (1.0 - dof_mask)

        dist     = Normal(mean, std)
        log_prob = (dist.log_prob(action) * dof_mask).sum(-1, keepdim=True)
        entropy  = (dist.entropy()        * dof_mask).sum(-1, keepdim=True)
        value    = self.critic_proj(pooled)
        return log_prob, entropy, value


# ── Rollout buffer ────────────────────────────────────────────────────────────
class RolloutBuffer:

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int,
                 act_dim: int, device: torch.device, gamma=0.99, gae_lambda=0.95):
        self.n_steps    = n_steps
        self.n_envs     = n_envs
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = device

        T, E = n_steps, n_envs
        self.obs       = torch.zeros(T, E, obs_dim,  device=device)
        self.actions   = torch.zeros(T, E, act_dim,  device=device)
        self.log_probs = torch.zeros(T, E, 1,        device=device)
        self.rewards   = torch.zeros(T, E, 1,        device=device)
        self.values    = torch.zeros(T, E, 1,        device=device)
        self.dones     = torch.zeros(T, E, 1,        device=device)
        self.ptr       = 0

    def add(self, obs, actions, log_probs, rewards, values, dones):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr]   = rewards.unsqueeze(-1)
        self.values[self.ptr]    = values
        self.dones[self.ptr]     = dones.unsqueeze(-1)
        self.ptr                += 1

    def compute_returns(self, last_values: torch.Tensor):
        T = self.n_steps
        advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.n_envs, 1, device=self.device)

        for t in reversed(range(T)):
            next_val  = last_values if t == T - 1 else self.values[t + 1]
            next_done = self.dones[t]
            delta     = (self.rewards[t]
                         + self.gamma * next_val * (1 - next_done)
                         - self.values[t])
            gae       = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae

        self.returns    = advantages + self.values
        self.advantages = advantages
        self.ptr = 0

    def get_batches(self, batch_size: int):
        T, E = self.n_steps, self.n_envs
        flat_obs = self.obs.view(T * E, -1)
        flat_act = self.actions.view(T * E, -1)
        flat_lp  = self.log_probs.view(T * E, -1)
        flat_ret = self.returns.view(T * E, -1)
        flat_adv = self.advantages.view(T * E, -1)
        flat_val = self.values.view(T * E, -1)

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        idx = torch.randperm(T * E, device=self.device)
        for start in range(0, T * E, batch_size):
            b = idx[start:start + batch_size]
            yield flat_obs[b], flat_act[b], flat_lp[b], flat_ret[b], flat_adv[b], flat_val[b]


# ── PPO trainer ───────────────────────────────────────────────────────────────
class PPOTrainer:

    def __init__(self,
                 policy:     UniversalActorCritic,
                 n_steps:    int   = 2048,
                 n_epochs:   int   = 5,
                 batch_size: int   = 512,
                 lr:         float = 1e-4,
                 clip_range: float = 0.2,
                 ent_coef:   float = 0.01,
                 vf_coef:    float = 0.25,
                 max_grad:   float = 0.3,
                 device:     str   = 'auto'):

        self.policy     = policy
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef   = ent_coef
        self.vf_coef    = vf_coef
        self.max_grad   = max_grad

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.scheduler = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def update(self, buffer: RolloutBuffer) -> dict:
        stats = dict(policy_loss=[], value_loss=[], entropy=[], clip_frac=[])

        for _ in range(self.n_epochs):
            for obs_b, act_b, lp_old, ret_b, adv_b, val_old in \
                    buffer.get_batches(self.batch_size):

                lp_new, ent, val = self.policy.evaluate(obs_b, act_b)

                # Policy loss (clipped surrogate)
                ratio = (lp_new - lp_old).exp()
                pg1   = -adv_b * ratio
                pg2   = -adv_b * ratio.clamp(1 - self.clip_range, 1 + self.clip_range)
                pl    = torch.max(pg1, pg2).mean()

                # Value loss with clipping
                v_clip = val_old + (val - val_old).clamp(-self.clip_range, self.clip_range)
                vl     = torch.max(F.mse_loss(val, ret_b), F.mse_loss(v_clip, ret_b))

                el   = ent.mean()
                loss = pl + self.vf_coef * vl - self.ent_coef * el

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
                self.optimizer.step()

                with torch.no_grad():
                    cf = ((ratio - 1).abs() > self.clip_range).float().mean()
                    stats['policy_loss'].append(pl.item())
                    stats['value_loss'].append(vl.item())
                    stats['entropy'].append(el.item())
                    stats['clip_frac'].append(cf.item())

        if self.scheduler is not None:
            self.scheduler.step()

        return {k: float(np.mean(v)) for k, v in stats.items()}
