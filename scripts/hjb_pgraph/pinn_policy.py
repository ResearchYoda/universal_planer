"""
scripts/hjb_pgraph/pinn_policy.py
===================================
Morph-Aware Physics-Informed pGraph Policy
===========================================

Architecture: pGraph Transformer Actor-Critic  +  HJB Physics Loss

Observation is structured as 25 tokens:
  morph tokens  (16 × 5)  →  [pgraph_norm, jdof_norm, jtype_norm, body_mass_norm, body_mask]
  joint tokens  ( 8 × 6)  →  [joint_pos, joint_vel, dof_mask, lim_lo, lim_hi, gear_norm]
  root  token   ( 1 ×11)  →  [lin_vel(3), ang_vel(3), height(1), quat(4)]

Physics-Informed additions
--------------------------
  1. HJB Residual Loss
     r_hjb = V(x)·ln(γ) + R + ∇_x V(x)ᵀ · (xₜ₊₁ − xₜ)/Δt
     L_hjb = E[r_hjb² · (1 − done)]

  2. Gradient decomposition
     ∇_x V splits into three groups matching the token structure:
       ∇_morph V  (B, 80)  — sensitivity to robot topology / morphology
       ∇_joint V  (B, 48)  — sensitivity to joint kinematics
       ∇_root  V  (B, 11)  — sensitivity to root body state

  3. Morphology-gradient consistency loss  (the 'morph-aware' contribution)
     For a fixed robot, morphology tokens never change during an episode.
     Therefore ∇_morph V should be consistent across states of the same robot.
     We penalise the within-robot variance of ∇_morph V:
       L_mc = (1/R) Σ_r  Var_batch[∇_morph V | robot=r].mean()
     This encourages the policy to build a stable structural representation.

Total value loss
----------------
  L_value = 0.5 · L_bellman + λ_hjb · L_hjb + λ_mc · L_mc

  λ_hjb = 0.05 – 0.15   (tunable; start small — same range as original paper)
  λ_mc  = 0.01           (light regulariser)

Why this is novel vs. plain HJBPPO
------------------------------------
  * The Transformer already reads pGraph topology.  ∇_morph V flows through
    the full self-attention graph, so the physics gradient carries structural
    information about which body links matter for energy-optimal control.
  * L_mc explicitly enforces that the policy has a stable, robot-specific
    morphology embedding — improving zero-shot transfer.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ── Observation layout (must match universal_env.py) ─────────────────────────
N_MORPH    = 16
N_JOINT    = 8
NODE_FEAT  = 5
JOINT_FEAT = 6
ROOT_FEAT  = 11
N_TOKENS   = N_MORPH + N_JOINT + 1        # 25
OBS_DIM    = N_MORPH * NODE_FEAT + N_JOINT * JOINT_FEAT + ROOT_FEAT  # 139

# Slice boundaries in the flat obs vector
_S1 = N_MORPH * NODE_FEAT                 # 80  — end of morph block
_S2 = _S1 + N_JOINT * JOINT_FEAT          # 128 — end of joint block

# Transformer hyper-parameters
D_MODEL  = 256
N_HEADS  = 4
N_LAYERS = 2
D_FF     = D_MODEL * 4

LOG_STD_MIN = -5.0
LOG_STD_MAX =  2.0

# MuJoCo timestep: 0.002 s × frame_skip(5) = 0.01 s
ENV_DT = 0.01


# ── pGraph Transformer + HJB Critic ───────────────────────────────────────────
class MorphHJBPolicy(nn.Module):
    """
    pGraph Transformer Actor-Critic with Physics-Informed HJB value function.

    Parameters
    ----------
    obs_dim   : flat observation dimension (default 139)
    action_dim: number of actuated DOFs (default 8)
    d_model   : Transformer hidden size (default 256)
    gamma     : discount factor — needed for ln(γ) in HJB residual
    """

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = N_JOINT,
                 d_model: int = D_MODEL, gamma: float = 0.99):
        super().__init__()
        self.gamma     = gamma
        self.log_gamma = math.log(gamma)   # ln(γ) ≈ −0.01005 for γ=0.99
        self.d_model   = d_model

        # ── Token projections ──────────────────────────────────────────
        self.morph_proj = nn.Linear(NODE_FEAT,  d_model)
        self.joint_proj = nn.Linear(JOINT_FEAT, d_model)
        self.root_proj  = nn.Linear(ROOT_FEAT,  d_model)

        # ── Transformer encoder (pre-LN, no dropout) ──────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=N_HEADS,
            dim_feedforward=D_FF, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS,
                                                  enable_nested_tensor=False)

        # ── Actor: per-joint [mean_raw, log_std] ──────────────────────
        self.actor_proj  = nn.Linear(d_model, 2)

        # ── Critic: valid-token mean-pool → scalar value ───────────────
        # Important: uses only standard differentiable ops so that
        # torch.autograd.grad can propagate through V(x) w.r.t. obs.
        self.critic_proj = nn.Linear(d_model, 1)

        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_proj.weight,  gain=0.01)
        nn.init.zeros_(self.actor_proj.bias)
        nn.init.orthogonal_(self.critic_proj.weight, gain=1.0)
        nn.init.zeros_(self.critic_proj.bias)

    # ── Core Transformer encoder ──────────────────────────────────────────────
    def _encode(self, obs: torch.Tensor):
        """
        Parse flat obs → tokenize → Transformer.

        Returns
        -------
        joint_out  : (B, 8, d)  per-joint token outputs
        pooled     : (B, d)     mean-pool over valid tokens (for critic)
        dof_mask_f : (B, 8)     float 1/0 valid/pad joints
        key_pad    : (B, 25)    True = ignore this token (for diagnostics)
        """
        B = obs.shape[0]

        # Parse into token matrices
        morph_t = obs[:, :_S1].reshape(B, N_MORPH, NODE_FEAT)    # (B,16,5)
        joint_t = obs[:, _S1:_S2].reshape(B, N_JOINT, JOINT_FEAT) # (B, 8,6)
        root_t  = obs[:, _S2:]                                     # (B,11)

        # Padding masks (True = token should be IGNORED)
        body_pad  = (morph_t[:, :, 4] < 0.5)          # (B,16)
        dof_pad   = (joint_t[:, :, 2] < 0.5)          # (B, 8)
        dof_mask_f = (~dof_pad).float()                # (B, 8)
        root_valid = torch.zeros(B, 1, dtype=torch.bool, device=obs.device)
        key_pad    = torch.cat([body_pad, dof_pad, root_valid], dim=1)  # (B,25)

        # Embed
        t_morph = self.morph_proj(morph_t)              # (B,16,d)
        t_joint = self.joint_proj(joint_t)              # (B, 8,d)
        t_root  = self.root_proj(root_t).unsqueeze(1)   # (B, 1,d)
        tokens  = torch.cat([t_morph, t_joint, t_root], dim=1)  # (B,25,d)

        # Transformer
        out = self.transformer(tokens, src_key_padding_mask=key_pad)  # (B,25,d)

        # Extract joint outputs
        joint_out = out[:, N_MORPH:N_MORPH + N_JOINT, :]  # (B,8,d)

        # Critic pooling: mean over valid tokens
        valid_f = (~key_pad).float().unsqueeze(-1)         # (B,25,1)
        pooled  = (out * valid_f).sum(1) / valid_f.sum(1).clamp(min=1.0)  # (B,d)

        return joint_out, pooled, dof_mask_f, key_pad

    # ── Action sampling (used during rollout collection) ──────────────────────
    def get_action(self, obs: torch.Tensor):
        joint_out, pooled, dof_mask, _ = self._encode(obs)

        actor_out = self.actor_proj(joint_out)                    # (B,8,2)
        mean      = torch.tanh(actor_out[..., 0]) * dof_mask     # (B,8)
        log_std   = actor_out[..., 1].clamp(LOG_STD_MIN, LOG_STD_MAX)
        std       = log_std.exp() * dof_mask + (1.0 - dof_mask)  # pad→std=1

        dist     = Normal(mean, std)
        action   = dist.sample() * dof_mask

        log_prob = (dist.log_prob(action) * dof_mask).sum(-1, keepdim=True)
        value    = self.critic_proj(pooled)
        return action, log_prob, value

    # ── Evaluate (used in PPO update) ─────────────────────────────────────────
    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        joint_out, pooled, dof_mask, _ = self._encode(obs)

        actor_out = self.actor_proj(joint_out)
        mean      = torch.tanh(actor_out[..., 0]) * dof_mask
        log_std   = actor_out[..., 1].clamp(LOG_STD_MIN, LOG_STD_MAX)
        std       = log_std.exp() * dof_mask + (1.0 - dof_mask)

        dist     = Normal(mean, std)
        log_prob = (dist.log_prob(action) * dof_mask).sum(-1, keepdim=True)
        entropy  = (dist.entropy()        * dof_mask).sum(-1, keepdim=True)
        value    = self.critic_proj(pooled)
        return log_prob, entropy, value

    # ── Physics-informed: HJB residual + gradient decomposition ──────────────
    def compute_hjb_residual(self,
                              obs:      torch.Tensor,
                              next_obs: torch.Tensor,
                              rewards:  torch.Tensor,
                              dones:    torch.Tensor,
                              dt:       float = ENV_DT):
        """
        Compute the HJB residual and gradient decomposition.

        The HJB equation for the optimal value function V*(x) is:

            V(x)·ln(γ) + max_u [ R(x,u) + ∇_x V(x)ᵀ·f(x,u) ] = 0

        At the current policy π, we approximate:

            r_hjb = V(x)·ln(γ) + R + ∇_x V(x)ᵀ · (x_{t+1}−x_t)/Δt

        This should equal zero for the optimal policy.  We minimise r_hjb².

        Gradient decomposition
        ----------------------
        ∇_x V ∈ ℝ^139 is split into:
          ∇_morph V  [0:80]    — 16 pgraph nodes × 5 features
          ∇_joint V  [80:128]  —  8 joint tokens  × 6 features
          ∇_root  V  [128:139] —  root body state (11 features)

        Since morphology is FIXED during an episode:
          (x_morph_{t+1} − x_morph_t) / Δt  ≈  0
        so the morph-block contributes nothing to ṽ_dot.
        Instead, ∇_morph V encodes how sensitive the VALUE is to structure —
        used in the morphology-consistency regulariser (L_mc).

        Parameters
        ----------
        obs      : (B, 139) current state
        next_obs : (B, 139) next state (detached from grad)
        rewards  : (B, 1)   immediate rewards
        dones    : (B, 1)   episode terminal flags
        dt       : float    env timestep in seconds

        Returns
        -------
        hjb_loss      : scalar — mean squared HJB residual
        grad_morph    : (B, 80)  morph gradient (for L_mc)
        grad_joint    : (B, 48)  joint gradient
        grad_root     : (B, 11)  root gradient
        diagnostics   : dict of scalar tensors for logging
        """

        # Create a differentiable copy of obs (leaf tensor)
        obs_g = obs.detach().requires_grad_(True)

        # Forward pass — critic only.
        # Flash / mem-efficient attention does NOT support create_graph=True
        # (second-order gradients).  Force the math (unfused) backend.
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            _sdp_ctx = sdpa_kernel([SDPBackend.MATH])
        except ImportError:                              # PyTorch < 2.3 fallback
            _sdp_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False)
        with _sdp_ctx:
            _, pooled, _, _ = self._encode(obs_g)
            value = self.critic_proj(pooled)          # (B, 1)

            # ∇_x V(x) — exact via autograd through the full Transformer
            # create_graph=True so L_hjb can be backpropagated
            grad_v = torch.autograd.grad(
                value.sum(), obs_g,
                create_graph=True,
                retain_graph=True,
            )[0]                                      # (B, 139)

        # Gradient decomposition by token group
        grad_morph = grad_v[:, :_S1]                 # (B, 80)
        grad_joint = grad_v[:, _S1:_S2]              # (B, 48)
        grad_root  = grad_v[:, _S2:]                 # (B, 11)

        # Finite-difference dynamics: f(x,u) ≈ (x_{t+1}−x_t)/Δt
        # Detach next_obs — it is data, not a parameter
        delta      = (next_obs.detach() - obs.detach()) / dt   # (B, 139)
        delta_joint = delta[:, _S1:_S2]
        delta_root  = delta[:, _S2:]
        # delta_morph ≈ 0 (morphology fixed), skip to save compute

        # HJB inner product: ∇_state V · ẋ_state
        v_dot = ((grad_joint * delta_joint).sum(-1, keepdim=True) +
                 (grad_root  * delta_root ).sum(-1, keepdim=True))  # (B,1)

        # HJB residual
        hjb_res = value * self.log_gamma + rewards.detach() + v_dot

        # Mask terminal transitions (HJB is undefined at episode end)
        hjb_res = hjb_res * (1.0 - dones.detach())

        hjb_loss = (hjb_res ** 2).mean()

        # ── Diagnostics ───────────────────────────────────────────────
        with torch.no_grad():
            diag = dict(
                hjb_residual_mean = hjb_res.mean().item(),
                grad_morph_norm   = grad_morph.norm(dim=-1).mean().item(),
                grad_joint_norm   = grad_joint.norm(dim=-1).mean().item(),
                grad_root_norm    = grad_root.norm(dim=-1).mean().item(),
                v_dot_mean        = v_dot.mean().item(),
            )

        return hjb_loss, grad_morph, grad_joint, grad_root, diag

    # ── Morphology-gradient consistency loss ──────────────────────────────────
    @staticmethod
    def morph_consistency_loss(grad_morph: torch.Tensor,
                                robot_ids:  list) -> torch.Tensor:
        """
        Penalise within-robot variance of ∇_morph V.

        For a given robot, morphology tokens are constant across all timesteps
        of an episode.  If the value function truly understands morphology,
        ∇_morph V should be CONSISTENT across different states of the same
        robot — varying only with position in state space, not with identity.

        L_mc = (1/|R|) Σ_{r ∈ robots}  Var_{t: robot_id=r}[∇_morph V_t].mean()

        A perfectly structured representation would have near-zero L_mc.

        Parameters
        ----------
        grad_morph : (B, 80) — morph gradient for each sample in batch
        robot_ids  : list[str] of length B — robot name per sample

        Returns
        -------
        l_mc : scalar tensor (0 if only one robot in batch)
        """
        device    = grad_morph.device
        unique    = list(set(robot_ids))
        if len(unique) < 2:
            return torch.tensor(0.0, device=device)

        losses = []
        for robot in unique:
            mask = torch.tensor(
                [r == robot for r in robot_ids],
                dtype=torch.bool, device=device
            )
            if mask.sum() < 2:
                continue
            gm = grad_morph[mask]                         # (n_r, 80)
            # Unbiased variance over the n_r samples, then mean over 80 dims
            var_per_dim = gm.var(dim=0, unbiased=True)    # (80,)
            losses.append(var_per_dim.mean())

        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()


# ── HJB Rollout Buffer ────────────────────────────────────────────────────────
class HJBRolloutBuffer:
    """
    Extended RolloutBuffer that additionally stores next_obs and robot_ids.
    These are required for the HJB residual computation.
    """

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int,
                 act_dim: int, device: torch.device,
                 gamma: float = 0.99, gae_lambda: float = 0.95):
        self.n_steps    = n_steps
        self.n_envs     = n_envs
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = device

        T, E = n_steps, n_envs
        self.obs       = torch.zeros(T, E, obs_dim,  device=device)
        self.next_obs  = torch.zeros(T, E, obs_dim,  device=device)  # NEW
        self.actions   = torch.zeros(T, E, act_dim,  device=device)
        self.log_probs = torch.zeros(T, E, 1,        device=device)
        self.rewards   = torch.zeros(T, E, 1,        device=device)
        self.values    = torch.zeros(T, E, 1,        device=device)
        self.dones     = torch.zeros(T, E, 1,        device=device)
        # robot_ids[t][e] = robot name string
        self.robot_ids = [['' for _ in range(E)] for _ in range(T)]  # NEW
        self.ptr       = 0

    def add(self, obs, next_obs, actions, log_probs, rewards, values, dones, robot_ids):
        t = self.ptr
        self.obs[t]       = obs
        self.next_obs[t]  = next_obs                            # NEW
        self.actions[t]   = actions
        self.log_probs[t] = log_probs
        self.rewards[t]   = rewards.unsqueeze(-1)
        self.values[t]    = values
        self.dones[t]     = dones.unsqueeze(-1)
        self.robot_ids[t] = list(robot_ids)                     # NEW
        self.ptr         += 1

    def compute_returns(self, last_values: torch.Tensor):
        T          = self.n_steps
        advantages = torch.zeros_like(self.rewards)
        gae        = torch.zeros(self.n_envs, 1, device=self.device)

        for t in reversed(range(T)):
            nv    = last_values if t == T - 1 else self.values[t + 1]
            delta = (self.rewards[t]
                     + self.gamma * nv * (1 - self.dones[t])
                     - self.values[t])
            gae   = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae

        self.returns    = advantages + self.values
        self.advantages = advantages
        self.ptr = 0

    def get_batches(self, batch_size: int):
        """Yield random mini-batches including next_obs and robot_ids."""
        T, E = self.n_steps, self.n_envs
        N    = T * E

        flat_obs      = self.obs.view(N, -1)
        flat_nobs     = self.next_obs.view(N, -1)
        flat_act      = self.actions.view(N, -1)
        flat_lp       = self.log_probs.view(N, -1)
        flat_ret      = self.returns.view(N, -1)
        flat_adv      = self.advantages.view(N, -1)
        flat_val      = self.values.view(N, -1)
        flat_rew      = self.rewards.view(N, -1)
        flat_done     = self.dones.view(N, -1)
        flat_rids     = [self.robot_ids[t][e] for t in range(T) for e in range(E)]

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        idx = torch.randperm(N, device=self.device)
        for start in range(0, N, batch_size):
            b    = idx[start:start + batch_size]
            rids = [flat_rids[i.item()] for i in b]
            yield (flat_obs[b], flat_nobs[b], flat_act[b], flat_lp[b],
                   flat_ret[b], flat_adv[b], flat_val[b],
                   flat_rew[b], flat_done[b], rids)


# ── HJB PPO Trainer ───────────────────────────────────────────────────────────
class HJBPPOTrainer:
    """
    PPO trainer with Physics-Informed HJB regularisation.

    Loss breakdown
    --------------
    L_actor    = clipped PPO surrogate
    L_bellman  = clipped value MSE  (standard PPO critic loss)
    L_hjb      = mean squared HJB residual  (physics constraint)
    L_mc       = morphology gradient consistency  (structural regulariser)

    L_total = L_actor + vf_coef * L_bellman
                       + λ_hjb  * L_hjb
                       + λ_mc   * L_mc
                       − ent_coef * entropy
    """

    def __init__(self,
                 policy:     MorphHJBPolicy,
                 lr:         float = 1e-4,
                 n_epochs:   int   = 5,
                 batch_size: int   = 512,
                 clip:       float = 0.2,
                 ent_coef:   float = 0.02,
                 vf_coef:    float = 0.25,
                 lambda_hjb: float = 0.05,
                 lambda_mc:  float = 0.01,
                 max_grad:   float = 0.5,
                 device:     str   = 'auto'):

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.policy     = policy.to(torch.device(device))
        self.device     = torch.device(device)
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.clip       = clip
        self.ent_coef   = ent_coef
        self.vf_coef    = vf_coef
        self.lambda_hjb = lambda_hjb
        self.lambda_mc  = lambda_mc
        self.max_grad   = max_grad
        self._use_amp   = (self.device.type == 'cuda' and
                           torch.cuda.is_bf16_supported())

        self.optimizer  = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.scheduler  = None

    def set_scheduler(self, s):
        self.scheduler = s

    def update(self, buf: HJBRolloutBuffer) -> dict:
        stats = dict(
            policy_loss=[], value_loss=[], entropy=[], clip_frac=[],
            hjb_loss=[], mc_loss=[],
            grad_morph_norm=[], grad_joint_norm=[], grad_root_norm=[],
            hjb_residual=[], v_dot=[],
        )

        for _ in range(self.n_epochs):
            for (obs_b, nobs_b, act_b, lp_old,
                 ret_b, adv_b, val_old, rew_b, done_b, rids_b) \
                    in buf.get_batches(self.batch_size):

                # ── Standard PPO forward (BF16 when available) ─────────
                with torch.autocast('cuda', torch.bfloat16, enabled=self._use_amp):
                    lp_new, ent, val = self.policy.evaluate(obs_b, act_b)
                lp_new = lp_new.float()
                ent    = ent.float()
                val    = val.float()

                # Actor loss
                ratio = (lp_new - lp_old).exp()
                pg1   = -adv_b * ratio
                pg2   = -adv_b * ratio.clamp(1 - self.clip, 1 + self.clip)
                l_actor = torch.max(pg1, pg2).mean()

                # Value loss (clipped Bellman)
                v_clip   = val_old + (val - val_old).clamp(-self.clip, self.clip)
                l_bellman = torch.max(F.mse_loss(val, ret_b),
                                      F.mse_loss(v_clip, ret_b))

                # ── Physics-Informed losses ─────────────────────────────
                # HJB residual (always in FP32 — needs exact gradients)
                (l_hjb, grad_morph,
                 grad_joint, grad_root, diag) = self.policy.compute_hjb_residual(
                    obs_b, nobs_b, rew_b, done_b
                )

                # Morphology gradient consistency
                l_mc = MorphHJBPolicy.morph_consistency_loss(grad_morph, rids_b)

                # ── Combined loss ───────────────────────────────────────
                loss = (l_actor
                        + self.vf_coef    * l_bellman
                        + self.lambda_hjb * l_hjb
                        + self.lambda_mc  * l_mc
                        - self.ent_coef   * ent.mean())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
                self.optimizer.step()

                # ── Logging ─────────────────────────────────────────────
                with torch.no_grad():
                    cf = ((ratio - 1).abs() > self.clip).float().mean()
                    stats['policy_loss'].append(l_actor.item())
                    stats['value_loss'].append(l_bellman.item())
                    stats['entropy'].append(ent.mean().item())
                    stats['clip_frac'].append(cf.item())
                    stats['hjb_loss'].append(l_hjb.item())
                    stats['mc_loss'].append(l_mc.item())
                    stats['grad_morph_norm'].append(diag['grad_morph_norm'])
                    stats['grad_joint_norm'].append(diag['grad_joint_norm'])
                    stats['grad_root_norm'].append(diag['grad_root_norm'])
                    stats['hjb_residual'].append(diag['hjb_residual_mean'])
                    stats['v_dot'].append(diag['v_dot_mean'])

        if self.scheduler is not None:
            self.scheduler.step()

        return {k: float(np.mean(v)) for k, v in stats.items()}
