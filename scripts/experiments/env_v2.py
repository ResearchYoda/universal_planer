"""
scripts/experiments/env_v2.py
==============================
V2 universal env (OBS_DIM=99) used for controlled experiments.
Identical to the original v2 obs layout so all experiment variants
are comparable on the same 99-dim observation space.

obs layout:
  pgraph_norm [16]  jdof_norm [16]  jtype_norm [16]  body_mask [16]  → MORPH_DIM=64
  joint_pos   [ 8]  joint_vel  [ 8]  dof_mask   [ 8]               → 24
  root_lin_vel[ 3]  root_ang_vel[ 3]  root_height[ 1]  root_quat [ 4] → 11
  total = 99
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

MAX_BODIES = 16
MAX_DOF    = 8
MORPH_DIM  = MAX_BODIES * 4   # pgraph + jdof + jtype + body_mask
STATE_DIM  = MAX_DOF * 3 + 11 # joint_pos + joint_vel + dof_mask + root_state
OBS_DIM    = MORPH_DIM + STATE_DIM   # 99

_DOF_PER_JTYPE = {0: 6, 1: 3, 2: 1, 3: 1}

ROBOT_CONFIGS = {
    'hopper':      'Hopper-v5',
    'halfcheetah': 'HalfCheetah-v5',
    'walker2d':    'Walker2d-v5',
    'ant':         'Ant-v5',
}


def _build_pgraph(model):
    n = model.nbody
    children = {i: [] for i in range(n)}
    for i in range(1, n):
        children[int(model.body_parentid[i])].append(i)
    untraversed = {i: list(cs) for i, cs in children.items()}
    pgraph, jdof, jtype = [], [], []
    sbs, curr = [], 1
    while True:
        pgraph.append(curr)
        adr = int(model.body_jntadr[curr]); cnt = int(model.body_jntnum[curr])
        if cnt > 0 and adr >= 0:
            dof = sum(_DOF_PER_JTYPE.get(int(model.jnt_type[adr+j]),0) for j in range(cnt))
            jt  = int(model.jnt_type[adr])
        else:
            dof, jt = 0, 0
        jdof.append(dof); jtype.append(jt)
        if len(children[curr]) > 1:
            if not sbs or sbs[-1] != curr: sbs.append(curr)
        if untraversed[curr]:
            curr = untraversed[curr].pop(0); continue
        while True:
            if not sbs:
                return np.array(pgraph), np.array(jdof), np.array(jtype)
            sep = sbs[-1]
            pgraph.append(sep); jdof.append(0)
            sa = int(model.body_jntadr[sep]); sc = int(model.body_jntnum[sep])
            jtype.append(int(model.jnt_type[sa]) if sc > 0 and sa >= 0 else 0)
            if untraversed[sep]:
                curr = untraversed[sep].pop(0)
                if not untraversed[sep]: sbs.pop()
                break
            else:
                sbs.pop()


class UniversalEnvV2(gym.Env):
    """V2 universal env — OBS_DIM=99, pgraph morphology encoding."""

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, robot_name: str, render_mode=None, zero_pgraph: bool = False):
        super().__init__()
        self.robot_name  = robot_name
        self._zero_pgraph = zero_pgraph   # ablation: nullify structural info
        self._env   = gym.make(ROBOT_CONFIGS[robot_name], render_mode=render_mode)
        self._model = self._env.unwrapped.model
        self._data  = self._env.unwrapped.data
        self.n_actuators = self._model.nu

        pgraph, jdof, jtype = _build_pgraph(self._model)
        n = min(len(pgraph), MAX_BODIES)
        max_idx = float(max(pgraph.max(), 1))

        self._pgraph_pad = np.zeros(MAX_BODIES, np.float32)
        self._jdof_pad   = np.zeros(MAX_BODIES, np.float32)
        self._jtype_pad  = np.zeros(MAX_BODIES, np.float32)
        self._body_mask  = np.zeros(MAX_BODIES, np.float32)
        if not zero_pgraph:
            self._pgraph_pad[:n] = pgraph[:n] / max_idx
            self._jdof_pad[:n]   = jdof[:n]   / 6.0
            self._jtype_pad[:n]  = jtype[:n]  / 3.0
        self._body_mask[:n] = 1.0   # mask always present (tells policy valid slots)

        self._qpos_ids, self._qvel_ids = [], []
        for a in range(self._model.nu):
            jid = int(self._model.actuator_trnid[a, 0])
            self._qpos_ids.append(int(self._model.jnt_qposadr[jid]))
            self._qvel_ids.append(int(self._model.jnt_dofadr[jid]))
        self._dof_mask = np.zeros(MAX_DOF, np.float32)
        self._dof_mask[:self.n_actuators] = 1.0

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space      = spaces.Box(-1.0,    1.0,    (MAX_DOF,), np.float32)

    def _get_obs(self):
        d = self._data
        jp = np.zeros(MAX_DOF, np.float32)
        jv = np.zeros(MAX_DOF, np.float32)
        jp[:self.n_actuators] = np.clip(
            np.array([d.qpos[i] for i in self._qpos_ids], np.float32) / np.pi, -1, 1)
        jv[:self.n_actuators] = np.clip(
            np.array([d.qvel[i] for i in self._qvel_ids], np.float32) / 10., -3, 3)
        return np.concatenate([
            self._pgraph_pad, self._jdof_pad, self._jtype_pad, self._body_mask,
            jp, jv, self._dof_mask,
            d.cvel[1][3:6].astype(np.float32),
            d.cvel[1][0:3].astype(np.float32),
            np.array([d.xpos[1][2]], np.float32),
            d.xquat[1].astype(np.float32),
        ])

    def reset(self, seed=None, options=None):
        _, info = self._env.reset(seed=seed, options=options)
        info['robot'] = self.robot_name
        return self._get_obs(), info

    def step(self, action):
        _, rew, term, trunc, info = self._env.step(
            np.array(action[:self.n_actuators], dtype=np.float64))
        info['robot'] = self.robot_name
        return self._get_obs(), float(rew), term, trunc, info

    def close(self): self._env.close()
