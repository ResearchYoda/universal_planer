"""
scripts/experiments/policy.py
==============================
V2 MLP Actor-Critic + PPO + RolloutBuffer + RunningNorm
Used consistently across all experiments for fair comparison.
"""

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from .env_v2 import MORPH_DIM, STATE_DIM, OBS_DIM, MAX_DOF

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


# ── V2 MLP Policy ─────────────────────────────────────────────────────────────
class MLPPolicy(nn.Module):
    """
    Encoder-decoder MLP Actor-Critic (v2 architecture).
    Per-robot action noise: log_std predicted from morphology encoder output.
    """
    def __init__(self, obs_dim=OBS_DIM, action_dim=MAX_DOF, hidden=512):
        super().__init__()
        self.morph_enc = nn.Sequential(
            nn.Linear(MORPH_DIM, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.state_enc = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor_head[-1].bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def _encode(self, obs):
        morph = obs[:, :MORPH_DIM]
        state = obs[:, MORPH_DIM:]
        mf = self.morph_enc(morph)
        sf = self.state_enc(state)
        return self.fusion(torch.cat([mf, sf], dim=-1)), mf

    def get_action(self, obs):
        feat, mf = self._encode(obs)
        mean    = torch.tanh(self.actor_head(feat))
        log_std = self.log_std_head(mf).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()
        dist    = Normal(mean, std)
        action  = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        value   = self.critic_head(feat)
        return action, log_prob, value

    def evaluate(self, obs, action):
        feat, mf = self._encode(obs)
        mean    = torch.tanh(self.actor_head(feat))
        log_std = self.log_std_head(mf).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()
        dist    = Normal(mean, std)
        return (dist.log_prob(action).sum(-1, keepdim=True),
                dist.entropy().sum(-1, keepdim=True),
                self.critic_head(feat))


# ── Rollout buffer ─────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_dim, act_dim, device,
                 gamma=0.99, gae_lambda=0.95):
        self.n_steps = n_steps; self.n_envs = n_envs
        self.gamma = gamma; self.gae_lambda = gae_lambda; self.device = device
        T, E = n_steps, n_envs
        self.obs       = torch.zeros(T, E, obs_dim, device=device)
        self.actions   = torch.zeros(T, E, act_dim, device=device)
        self.log_probs = torch.zeros(T, E, 1,       device=device)
        self.rewards   = torch.zeros(T, E, 1,       device=device)
        self.values    = torch.zeros(T, E, 1,       device=device)
        self.dones     = torch.zeros(T, E, 1,       device=device)
        self.ptr = 0

    def add(self, obs, actions, log_probs, rewards, values, dones):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr]   = rewards.unsqueeze(-1)
        self.values[self.ptr]    = values
        self.dones[self.ptr]     = dones.unsqueeze(-1)
        self.ptr += 1

    def compute_returns(self, last_val):
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.n_envs, 1, device=self.device)
        for t in reversed(range(self.n_steps)):
            nv  = last_val if t == self.n_steps-1 else self.values[t+1]
            d   = self.rewards[t] + self.gamma*nv*(1-self.dones[t]) - self.values[t]
            gae = d + self.gamma*self.gae_lambda*(1-self.dones[t])*gae
            adv[t] = gae
        self.returns = adv + self.values
        self.advantages = adv; self.ptr = 0

    def get_batches(self, batch_size):
        T, E = self.n_steps, self.n_envs
        fo = self.obs.view(T*E,-1); fa = self.actions.view(T*E,-1)
        fl = self.log_probs.view(T*E,-1); fr = self.returns.view(T*E,-1)
        fad = self.advantages.view(T*E,-1); fv = self.values.view(T*E,-1)
        fad = (fad - fad.mean())/(fad.std()+1e-8)
        idx = torch.randperm(T*E, device=self.device)
        for s in range(0, T*E, batch_size):
            b = idx[s:s+batch_size]
            yield fo[b], fa[b], fl[b], fr[b], fad[b], fv[b]


# ── PPO Trainer ────────────────────────────────────────────────────────────────
class PPOTrainer:
    def __init__(self, policy, lr=3e-4, n_epochs=10, batch_size=512,
                 clip=0.2, ent_coef=0.02, vf_coef=0.25, max_grad=0.3,
                 device='auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy    = policy.to(torch.device(device))
        self.device    = torch.device(device)
        self.n_epochs  = n_epochs
        self.batch_size = batch_size
        self.clip      = clip
        self.ent_coef  = ent_coef
        self.vf_coef   = vf_coef
        self.max_grad  = max_grad
        self.use_amp   = (self.device.type == 'cuda' and
                          torch.cuda.is_bf16_supported())
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.scheduler = None

    def set_scheduler(self, s): self.scheduler = s

    def update(self, buf):
        stats = dict(pl=[], vl=[], ent=[], cf=[])
        for _ in range(self.n_epochs):
            for ob, ac, lp_old, ret, adv, val_old in buf.get_batches(self.batch_size):
                with torch.autocast('cuda', torch.bfloat16, enabled=self.use_amp):
                    lp_new, ent, val = self.policy.evaluate(ob, ac)
                lp_new = lp_new.float(); ent = ent.float(); val = val.float()

                ratio = (lp_new - lp_old).exp()
                pl = torch.max(-adv*ratio,
                               -adv*ratio.clamp(1-self.clip, 1+self.clip)).mean()
                v_clip = val_old + (val-val_old).clamp(-self.clip, self.clip)
                vl = torch.max(F.mse_loss(val, ret), F.mse_loss(v_clip, ret))
                loss = pl + self.vf_coef*vl - self.ent_coef*ent.mean()

                self.optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
                self.optimizer.step()

                with torch.no_grad():
                    cf = ((ratio-1).abs() > self.clip).float().mean()
                    stats['pl'].append(pl.item()); stats['vl'].append(vl.item())
                    stats['ent'].append(ent.mean().item()); stats['cf'].append(cf.item())
        if self.scheduler: self.scheduler.step()
        return {k: float(np.mean(v)) for k,v in stats.items()}


# ── Running normalizer ─────────────────────────────────────────────────────────
class RunningNorm:
    def __init__(self, shape, clip=10.0):
        self.mean=np.zeros(shape,np.float64); self.var=np.ones(shape,np.float64)
        self.count=1e-4; self.clip=clip

    def update(self, x):
        b = x.reshape(-1, x.shape[-1])
        bm, bv, n = b.mean(0), b.var(0), len(b)
        tot = self.count + n; d = bm - self.mean
        self.mean += d*n/tot
        self.var = (self.var*self.count + bv*n + d**2*self.count*n/tot)/tot
        self.count = tot

    def normalize(self, x):
        return np.clip((x-self.mean)/np.sqrt(self.var+1e-8),
                       -self.clip, self.clip).astype(np.float32)

    def save(self, path):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    @classmethod
    def load(cls, path, shape):
        d = np.load(path); obj = cls(shape)
        obj.mean, obj.var, obj.count = d['mean'], d['var'], float(d['count'])
        return obj
