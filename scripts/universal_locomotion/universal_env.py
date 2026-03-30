"""
scripts/universal_locomotion/universal_env.py
=============================================
Universal locomotion environment.

Wraps Hopper-v5, HalfCheetah-v5, Walker2d-v5, Ant-v5 into a single
fixed-size observation/action interface using Pgraph morphology vectors.

Observation layout (OBS_DIM = 139)
-----------------------------------
Morph tokens  [16 × 5 = 80]   — one token per Pgraph slot:
  [pgraph_norm, jdof_norm, jtype_norm, body_mass_norm, body_mask]

Joint tokens  [ 8 × 6 = 48]   — one token per actuated DOF slot:
  [joint_pos, joint_vel, dof_mask, lim_lo_norm, lim_hi_norm, gear_norm]

Root state    [        11]
  root_lin_vel(3), root_ang_vel(3), root_height(1), root_quat(4)

The interleaved layout (features grouped by token) lets the policy network
directly reshape obs → (16,5) morph tokens and (8,6) joint tokens.

Action space: Box(-1, 1, (8,))
  First n_actuators elements are applied; the rest are masked to zero.
"""

import os, sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_BODIES  = 16    # Ant-v5 Pgraph has exactly 16 entries
MAX_DOF     = 8     # Ant-v5 has 8 actuated DOF
N_MORPH     = MAX_BODIES   # alias used by ppo.py / train.py
N_JOINT     = MAX_DOF      # alias used by ppo.py / train.py
NODE_FEAT   = 5     # per Pgraph slot: pgraph_norm, jdof_norm, jtype_norm, body_mass_norm, body_mask
JOINT_FEAT  = 6     # per DOF slot:   joint_pos, joint_vel, dof_mask, lim_lo_norm, lim_hi_norm, gear_norm
ROOT_FEAT   = 11    # root_lin_vel(3) + root_ang_vel(3) + root_height(1) + root_quat(4)
OBS_DIM     = MAX_BODIES * NODE_FEAT + MAX_DOF * JOINT_FEAT + ROOT_FEAT   # = 139

# Normalization anchors (consistent across all supported robots)
_MASS_MAX  = 10.0    # > max single-body mass observed (~6.25 for HalfCheetah torso)
_GEAR_MAX  = 300.0   # > max gear observed (Ant = 300)
_LIM_SCALE = np.pi   # joint limits in radians → divide by π → approx [-1, 1]

_DOF_PER_JTYPE = {0: 6, 1: 3, 2: 1, 3: 1}   # free, ball, slide, hinge

ROBOT_CONFIGS = {
    'hopper':       'Hopper-v5',
    'halfcheetah':  'HalfCheetah-v5',
    'walker2d':     'Walker2d-v5',
    'ant':          'Ant-v5',
}


# ── Pgraph builder ────────────────────────────────────────────────────────────
def _build_pgraph(model) -> tuple:
    """
    Traverse the MuJoCo kinematic tree with the Pgraph algorithm
    (Yazar & Yesiloglu 2018, Fig. 4) and return:
      pgraph – body-index traversal order
      jdof   – DOF count at each entry (0 for separation-body revisits)
      jtype  – first joint type at each entry (0 if no joints)
    """
    n = model.nbody
    children = {i: [] for i in range(n)}
    for i in range(1, n):
        children[int(model.body_parentid[i])].append(i)

    untraversed = {i: list(cs) for i, cs in children.items()}
    pgraph, jdof, jtype = [], [], []
    sbs: list = []
    curr = 1

    while True:
        pgraph.append(curr)
        adr = int(model.body_jntadr[curr])
        cnt = int(model.body_jntnum[curr])
        if cnt > 0 and adr >= 0:
            dof = sum(_DOF_PER_JTYPE.get(int(model.jnt_type[adr + j]), 0)
                      for j in range(cnt))
            jt  = int(model.jnt_type[adr])
        else:
            dof, jt = 0, 0
        jdof.append(dof)
        jtype.append(jt)

        if len(children[curr]) > 1:
            if not sbs or sbs[-1] != curr:
                sbs.append(curr)

        if untraversed[curr]:
            curr = untraversed[curr].pop(0)
            continue

        while True:
            if not sbs:
                return np.array(pgraph), np.array(jdof), np.array(jtype)
            sep = sbs[-1]
            pgraph.append(sep)
            jdof.append(0)
            sep_adr = int(model.body_jntadr[sep])
            sep_cnt = int(model.body_jntnum[sep])
            jtype.append(int(model.jnt_type[sep_adr])
                         if sep_cnt > 0 and sep_adr >= 0 else 0)
            if untraversed[sep]:
                curr = untraversed[sep].pop(0)
                if not untraversed[sep]:
                    sbs.pop()
                break
            else:
                sbs.pop()


# ── Universal environment ─────────────────────────────────────────────────────
class UniversalLocomotionEnv(gym.Env):
    """
    Fixed obs/action wrapper around standard MuJoCo locomotion envs.

    All morphology vectors (Pgraph, jdof, jtype, body mass, joint limits, gear)
    are precomputed once in __init__ and embedded in every observation, giving
    the policy structural and dynamic context about the robot it is controlling.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, robot_name: str, render_mode=None):
        super().__init__()
        assert robot_name in ROBOT_CONFIGS, \
            f"Unknown robot '{robot_name}'. Choose from {list(ROBOT_CONFIGS)}"

        self.robot_name  = robot_name
        self.render_mode = render_mode

        self._env   = gym.make(ROBOT_CONFIGS[robot_name], render_mode=render_mode)
        self._model = self._env.unwrapped.model
        self._data  = self._env.unwrapped.data

        self.n_actuators = self._model.nu      # 3 / 6 / 6 / 8

        # ── Build static morphology vectors ──────────────────────────
        pgraph, jdof, jtype = _build_pgraph(self._model)
        n_pg = len(pgraph)

        self._pgraph_pad = np.zeros(MAX_BODIES, np.float32)
        self._jdof_pad   = np.zeros(MAX_BODIES, np.float32)
        self._jtype_pad  = np.zeros(MAX_BODIES, np.float32)
        self._body_mask  = np.zeros(MAX_BODIES, np.float32)

        n = min(n_pg, MAX_BODIES)
        max_idx = float(max(pgraph.max(), 1))
        self._pgraph_pad[:n] = pgraph[:n] / max_idx
        self._jdof_pad[:n]   = jdof[:n]   / 6.0
        self._jtype_pad[:n]  = jtype[:n]  / 3.0
        self._body_mask[:n]  = 1.0

        # ── Body mass per Pgraph slot (dynamic parameter) ─────────────
        self._body_mass_norm = np.zeros(MAX_BODIES, np.float32)
        self._body_mass_norm[:n] = (
            self._model.body_mass[pgraph[:n]].astype(np.float32) / _MASS_MAX
        )

        # ── Actuated joint indices in qpos / qvel ────────────────────
        self._qpos_ids: list = []
        self._qvel_ids: list = []
        self._jnt_ids:  list = []
        for a in range(self._model.nu):
            jid = int(self._model.actuator_trnid[a, 0])
            self._qpos_ids.append(int(self._model.jnt_qposadr[jid]))
            self._qvel_ids.append(int(self._model.jnt_dofadr[jid]))
            self._jnt_ids.append(jid)

        # ── Per-actuator dynamic parameters ──────────────────────────
        self._dof_mask   = np.zeros(MAX_DOF, np.float32)
        self._lim_lo     = np.zeros(MAX_DOF, np.float32)
        self._lim_hi     = np.zeros(MAX_DOF, np.float32)
        self._gear_norm  = np.zeros(MAX_DOF, np.float32)

        self._dof_mask[:self.n_actuators] = 1.0
        for a, jid in enumerate(self._jnt_ids):
            lo = float(self._model.jnt_range[jid, 0])
            hi = float(self._model.jnt_range[jid, 1])
            # Joints without limits (jnt_limited=0) keep 0
            if bool(self._model.jnt_limited[jid]):
                self._lim_lo[a] = np.clip(lo / _LIM_SCALE, -2.0, 0.0)
                self._lim_hi[a] = np.clip(hi / _LIM_SCALE,  0.0, 2.0)
            gear = float(self._model.actuator_gear[a, 0])
            self._gear_norm[a] = np.clip(gear / _GEAR_MAX, 0.0, 2.0)

        # ── Spaces ────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(MAX_DOF,), dtype=np.float32
        )

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        d = self._data

        # Actuated joints (dynamic state)
        joint_pos = np.zeros(MAX_DOF, np.float32)
        joint_vel = np.zeros(MAX_DOF, np.float32)
        joint_pos[:self.n_actuators] = np.clip(
            np.array([d.qpos[i] for i in self._qpos_ids], np.float32) / np.pi,
            -1.0, 1.0
        )
        joint_vel[:self.n_actuators] = np.clip(
            np.array([d.qvel[i] for i in self._qvel_ids], np.float32) / 10.0,
            -3.0, 3.0
        )

        # Root body (body 1): cvel = [ang_vel(3), lin_vel(3)]
        root_lin_vel = d.cvel[1][3:6].astype(np.float32)
        root_ang_vel = d.cvel[1][0:3].astype(np.float32)
        root_height  = np.array([d.xpos[1][2]], np.float32)
        root_quat    = d.xquat[1].astype(np.float32)

        # Morph tokens: shape (MAX_BODIES, NODE_FEAT) → flatten to 80
        # Per node: [pgraph_norm, jdof_norm, jtype_norm, body_mass_norm, body_mask]
        morph_tokens = np.stack([
            self._pgraph_pad,
            self._jdof_pad,
            self._jtype_pad,
            self._body_mass_norm,
            self._body_mask,
        ], axis=1).flatten()   # (16, 5) → (80,)

        # Joint tokens: shape (MAX_DOF, JOINT_FEAT) → flatten to 48
        # Per joint: [joint_pos, joint_vel, dof_mask, lim_lo_norm, lim_hi_norm, gear_norm]
        joint_tokens = np.stack([
            joint_pos,
            joint_vel,
            self._dof_mask,
            self._lim_lo,
            self._lim_hi,
            self._gear_norm,
        ], axis=1).flatten()   # (8, 6) → (48,)

        root_state = np.concatenate([root_lin_vel, root_ang_vel, root_height, root_quat])

        return np.concatenate([morph_tokens, joint_tokens, root_state])  # = 139

    # ── Gym interface ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        _, info = self._env.reset(seed=seed, options=options)
        info['robot'] = self.robot_name
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        env_act = np.array(action[:self.n_actuators], dtype=np.float64)
        _, reward, terminated, truncated, info = self._env.step(env_act)
        info['robot'] = self.robot_name
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
