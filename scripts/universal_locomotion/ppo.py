"""
scripts/universal_locomotion/ppo.py
====================================
Self-contained PPO implementation (no stable-baselines3 dependency).

Improvements over v1
--------------------
- log_std is predicted from the morphology encoder (per-robot action noise)
- Value function clipping (standard PPO stabilisation)
- Action clamping removed → log_prob is consistent with sampled actions
- RolloutBuffer yields old values for value clipping
- PPOTrainer accepts an external LR scheduler

Architecture
------------
Observation (99-dim)
       │
       ├─ Morphology stream  [pgraph, jdof, jtype, mask]  (64-dim)
       │         └─ MorphEncoder: Linear(64→128) → LN → ReLU → Linear(128→128) → ReLU
       │                                  │
       │                          log_std_head: Linear(128→8)   ← per-robot action noise
       │
       └─ State stream        [joint_pos, joint_vel, mask, root_state]  (35-dim)
                 └─ StateEncoder: Linear(35→128) → LN → ReLU → Linear(128→128) → ReLU
                        │
                  Fusion: Linear(256→512) → ReLU → Linear(512→512) → ReLU
                 ┌───────┴───────┐
             Actor head       Critic head
          Linear(512→512)   Linear(512→512)
          ReLU              ReLU
          Linear(512→8)     Linear(512→1)
          Tanh (mean)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

MORPH_DIM = 16 * 4     # 64  (pgraph + jdof + jtype + body_mask)
STATE_DIM  = 8 * 3 + 11  # 35  (joint_pos + joint_vel + dof_mask + root_state)
OBS_DIM    = MORPH_DIM + STATE_DIM   # = 99


# ── Actor-Critic network ──────────────────────────────────────────────────────
class UniversalActorCritic(nn.Module):
    """
    Encoder-decoder Actor-Critic with morphology-conditioned action noise.
    The log_std is predicted from the morphology stream so each robot topology
    can learn its own optimal exploration scale.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = 8,
                 hidden: int = 512):
        super().__init__()

        # ── Encoder: morphology stream ────────────────────────────────────
        self.morph_enc = nn.Sequential(
            nn.Linear(MORPH_DIM, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # ── Encoder: state stream ─────────────────────────────────────────
        self.state_enc = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # ── Fusion (deeper than v1) ───────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(256, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # ── Actor decoder ─────────────────────────────────────────────────
        self.actor_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

        # Per-robot log_std predicted from morphology features
        self.log_std_head = nn.Linear(128, action_dim)

        # ── Critic decoder ────────────────────────────────────────────────
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor_head[-1].bias)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
        # Initialise log_std_head to produce ~0 (std≈1)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def _encode(self, obs: torch.Tensor):
        morph = obs[:, :MORPH_DIM]
        state = obs[:, MORPH_DIM:]
        morph_feat = self.morph_enc(morph)
        state_feat = self.state_enc(state)
        fused = self.fusion(torch.cat([morph_feat, state_feat], dim=-1))
        return fused, morph_feat

    def get_action(self, obs: torch.Tensor):
        fused, morph_feat = self._encode(obs)
        mean    = torch.tanh(self.actor_head(fused))
        log_std = self.log_std_head(morph_feat).clamp(self.LOG_STD_MIN,
                                                       self.LOG_STD_MAX)
        std     = log_std.exp()
        dist    = Normal(mean, std)
        action  = dist.sample()           # no clamp → log_prob is consistent
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        value   = self.critic_head(fused)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        fused, morph_feat = self._encode(obs)
        mean    = torch.tanh(self.actor_head(fused))
        log_std = self.log_std_head(morph_feat).clamp(self.LOG_STD_MIN,
                                                       self.LOG_STD_MAX)
        std     = log_std.exp()
        dist    = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy  = dist.entropy().sum(-1, keepdim=True)
        value    = self.critic_head(fused)
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
        """GAE-Lambda advantage estimation."""
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

        self.returns    = advantages + self.values   # target values
        self.advantages = advantages
        self.ptr = 0

    def get_batches(self, batch_size: int):
        """Yield random mini-batches including old values for value clipping."""
        T, E = self.n_steps, self.n_envs
        flat_obs   = self.obs.view(T * E, -1)
        flat_act   = self.actions.view(T * E, -1)
        flat_lp    = self.log_probs.view(T * E, -1)
        flat_ret   = self.returns.view(T * E, -1)
        flat_adv   = self.advantages.view(T * E, -1)
        flat_val   = self.values.view(T * E, -1)   # old values for clipping

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        idx = torch.randperm(T * E, device=self.device)
        for start in range(0, T * E, batch_size):
            b = idx[start:start + batch_size]
            yield (flat_obs[b], flat_act[b], flat_lp[b],
                   flat_ret[b], flat_adv[b], flat_val[b])


# ── PPO trainer ───────────────────────────────────────────────────────────────
class PPOTrainer:

    def __init__(self,
                 policy:      UniversalActorCritic,
                 n_steps:     int   = 2048,
                 n_epochs:    int   = 10,
                 batch_size:  int   = 512,
                 lr:          float = 3e-4,
                 gamma:       float = 0.99,
                 gae_lambda:  float = 0.95,
                 clip_range:  float = 0.2,
                 ent_coef:    float = 0.01,
                 vf_coef:     float = 0.5,
                 max_grad:    float = 0.5,
                 device:      str   = 'auto'):

        self.policy     = policy
        self.n_steps    = n_steps
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
        self.scheduler = None   # set externally via set_scheduler()

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
                pg2   = -adv_b * ratio.clamp(1 - self.clip_range,
                                              1 + self.clip_range)
                pl    = torch.max(pg1, pg2).mean()

                # Value loss with clipping
                v_clip = val_old + (val - val_old).clamp(-self.clip_range,
                                                          self.clip_range)
                vl = torch.max(F.mse_loss(val, ret_b),
                               F.mse_loss(v_clip, ret_b))

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
