"""
src/pgraph.py
=============
Corrected implementation of the Pgraph method.

Reference
---------
Yazar, M. N., & Yesiloglu, S. M. (2018).
Path defined directed graph vector (Pgraph) method for multibody dynamics.
Multibody System Dynamics, 43, 209-227.

Fixes applied vs. previous version
------------------------------------
1. jdof now stores the true DOF count per joint type (free=6, ball=3,
   slide/hinge=1) instead of the number of joints on the body.
2. E_phi (ε_φ) is now assembled by iterating over the Pgraph vector as
   described in the paper's Fig. 6 flowchart:
       for consecutive pairs (j, κ) in Pgraph:
           if κ > j  →  E_phi[κ, j] = φ(κ, j)
   Previously E_phi was built by iterating over MuJoCo body_parentid,
   which gives identical results for simple trees but does NOT implement
   the Pgraph methodology required for closed chains and changing topologies.
3. calculate_jacobian docstring corrected:  J = Φ_T · Φ · H  (Paper Eq.20).
"""

import numpy as np
import mujoco
from .common import form_spatial_transform, quat2mat, get_spatial_inertia

# DOF contributed by each MuJoCo joint type
_DOF_PER_JTYPE: dict[int, int] = {
    0: 6,   # free  joint
    1: 3,   # ball  joint
    2: 1,   # slide joint
    3: 1,   # hinge joint
}


def _body_dof(model, body_id: int) -> int:
    """
    Returns the total generalized-velocity DOF for all joints on body_id.

    body_jntnum counts the number of joints, NOT the number of DOFs.
    A free joint has 1 joint entry but contributes 6 DOFs.
    """
    adr = int(model.body_jntadr[body_id])
    cnt = int(model.body_jntnum[body_id])
    if cnt == 0 or adr < 0:
        return 0
    return sum(
        _DOF_PER_JTYPE.get(int(model.jnt_type[adr + j]), 0)
        for j in range(cnt)
    )


class PgraphModel:
    """
    Computes the Pgraph/jdof traversal vectors and the derived system matrices
    (H, Φ, M_sys) for a MuJoCo multibody model.
    """

    def __init__(self, model, data):
        self.model    = model
        self.data     = data
        self.n_bodies = model.nbody

        self.Pgraph: list[int] = []   # body-index traversal order
        self.jdof:   list[int] = []   # DOF count at each Pgraph entry (0 for revisits)

        self._build_topology()

    # ─────────────────────────────────────────────────────────────────────────
    def _build_topology(self):
        """
        Builds Pgraph and jdof via the flowchart in Paper Fig. 4.

        Algorithm summary (outboard, left-child-priority):
          1. Start at root body (body 1; body 0 = world/ground).
          2. Append current body i to Pgraph; record its DOF as jdof entry.
          3. If i has >1 child (separation body) → push to SBS.
          4. If i has untraversed children → descend to next child.
          5. Else (tip body) → backtrack via SBS:
               - Peek top of SBS (sep_i).
               - Append sep_i to Pgraph with jdof = 0.
               - Descend to next untraversed child of sep_i.
               - If sep_i is exhausted → pop from SBS.
          6. Repeat until SBS is empty.
        """
        model = self.model

        # Build children map: parent body → sorted list of child bodies
        children_map: dict[int, list[int]] = {i: [] for i in range(self.n_bodies)}
        for i in range(1, self.n_bodies):
            children_map[int(model.body_parentid[i])].append(i)

        # Working copy (consumed during traversal)
        untraversed: dict[int, list[int]] = {
            i: list(cs) for i, cs in children_map.items()
        }

        sbs: list[int] = []   # Separation Body Stack (LIFO)
        curr_i: int = 1       # root body  (body 0 is world)

        while True:
            # ── Step 2 ────────────────────────────────────────────────────
            self.Pgraph.append(curr_i)
            self.jdof.append(_body_dof(model, curr_i))   # FIX 1: true DOF count

            # ── Step 3: separation body ───────────────────────────────────
            if len(children_map[curr_i]) > 1:
                if not sbs or sbs[-1] != curr_i:
                    sbs.append(curr_i)

            # ── Step 4: descend or backtrack ──────────────────────────────
            if untraversed[curr_i]:
                curr_i = untraversed[curr_i].pop(0)
                continue

            # curr_i is a tip body – backtrack via SBS
            while True:
                if not sbs:
                    return  # Step 6: algorithm done

                sep_i = sbs[-1]   # peek (do NOT pop yet)

                # Step 5: re-insert separation body as a revisit
                self.Pgraph.append(sep_i)
                self.jdof.append(0)

                if untraversed[sep_i]:
                    curr_i = untraversed[sep_i].pop(0)
                    # Pop from SBS if the separation body is now exhausted
                    if not untraversed[sep_i]:
                        sbs.pop()
                    break   # back to outer loop (Step 2)
                else:
                    sbs.pop()   # sep_i fully processed → remove

    # ─────────────────────────────────────────────────────────────────────────
    def _get_system_matrices(self):
        """
        Returns (H_sys, Phi_sys, M_sys).

        H_sys   – 6n × nv  joint-map matrix
        Phi_sys – 6n × 6n  Spatial Propagation Operator (SPO): Φ = (I−ε_φ)⁻¹
        M_sys   – 6n × 6n  block-diagonal spatial inertia matrix

        FIX 2: E_phi (ε_φ) is now assembled by iterating over the Pgraph
        vector as described in Paper Fig. 6:

            for idx in range(1, len(Pgraph)):
                κ = Pgraph[idx],  j = Pgraph[idx-1]
                if κ > j:   # forward (outward) edge in canonical numbering
                    E_phi[κ, j] = φ(κ, j)

        In MuJoCo's outboard-numbered tree (parent_id < child_id is always
        satisfied), κ > j precisely identifies a parent→child edge, and
        κ < j identifies a tip→separation backtrack (which must be skipped).
        """
        n     = self.n_bodies
        nv    = self.model.nv
        model = self.model
        data  = self.data

        H_sys = np.zeros((6 * n, nv))
        E_phi = np.zeros((6 * n, 6 * n))
        M_sys = np.zeros((6 * n, 6 * n))

        # ── 1. Build H (joint map) and M (spatial inertia) ───────────────
        for i in range(1, n):
            body_row = i * 6
            jnt_adr  = int(model.body_jntadr[i])
            jnt_num  = int(model.body_jntnum[i])

            for j in range(jnt_num):
                jid    = jnt_adr + j
                jtype  = int(model.jnt_type[jid])
                axis   = model.jnt_axis[jid]
                dof_id = int(model.jnt_dofadr[jid])

                if jtype == 0:   # free joint – 6 DOF
                    # MuJoCo qvel ordering: [linX,linY,linZ, angX,angY,angZ]
                    # Spatial vector:       [angX,angY,angZ, linX,linY,linZ]
                    H_sys[body_row + 3, dof_id + 0] = 1.0  # linX
                    H_sys[body_row + 4, dof_id + 1] = 1.0  # linY
                    H_sys[body_row + 5, dof_id + 2] = 1.0  # linZ
                    H_sys[body_row + 0, dof_id + 3] = 1.0  # angX
                    H_sys[body_row + 1, dof_id + 4] = 1.0  # angY
                    H_sys[body_row + 2, dof_id + 5] = 1.0  # angZ

                elif jtype == 1:  # ball joint – 3 angular DOF
                    for k in range(3):
                        h_vec    = np.zeros(6)
                        h_vec[k] = 1.0               # angular axis k
                        H_sys[body_row:body_row + 6, dof_id + k] = h_vec

                elif jtype == 2:  # slide joint – 1 linear DOF
                    h_vec       = np.zeros(6)
                    h_vec[3:6]  = axis
                    H_sys[body_row:body_row + 6, dof_id] = h_vec

                elif jtype == 3:  # hinge joint – 1 angular DOF
                    h_vec       = np.zeros(6)
                    h_vec[0:3]  = axis
                    H_sys[body_row:body_row + 6, dof_id] = h_vec

            # Spatial inertia block for body i
            M_sys[body_row:body_row + 6, body_row:body_row + 6] = \
                get_spatial_inertia(
                    model.body_mass[i],
                    model.body_inertia[i],
                    model.body_ipos[i],
                )

        # ── 2. Build E_phi via Pgraph vector (Paper Fig. 6) ──────────────
        # Cache φ(k, parent) to avoid recomputation for separation revisits.
        phi_cache: dict[int, np.ndarray] = {}

        def _phi(k: int, j: int) -> np.ndarray:
            """6×6 spatial propagation matrix from body j (parent) to k (child)."""
            if k not in phi_cache:
                pos_k = data.xpos[k]
                mat_k = quat2mat(data.xquat[k])
                pos_j = data.xpos[j]
                mat_j = quat2mat(data.xquat[j])
                p_rel = mat_j.T @ (pos_k - pos_j)   # link vector in parent frame
                R_rel = mat_j.T @ mat_k              # rotation child→parent frame
                phi_cache[k] = form_spatial_transform(p_rel, R_rel)
            return phi_cache[k]

        for idx in range(1, len(self.Pgraph)):
            kappa = self.Pgraph[idx]
            j     = self.Pgraph[idx - 1]

            if kappa > j:   # forward edge: κ is child of j
                row = kappa * 6
                col = j * 6
                E_phi[row:row + 6, col:col + 6] = _phi(kappa, j)

        # ── 3. Φ = (I − E_phi)⁻¹  (Paper Eq. 16) ────────────────────────
        I_sys = np.eye(6 * n)
        try:
            Phi_sys = np.linalg.inv(I_sys - E_phi)
        except np.linalg.LinAlgError:
            Phi_sys = np.linalg.pinv(I_sys - E_phi)

        return H_sys, Phi_sys, M_sys

    # ─────────────────────────────────────────────────────────────────────────
    def calculate_jacobian(self, target_body_name: str) -> np.ndarray:
        """
        6×Nv Jacobian for `target_body_name` expressed in world frame.

        Paper Eq. (20):  J = Φ_T · Φ · H
        where Φ_T is the row-selection matrix for the target tip body.
        Extracting rows [tid*6 : tid*6+6] from (Φ·H) is equivalent to
        applying Φ_T and gives the same result.

        Returns
        -------
        J_world : (6, nv) array
            Rows 0-2: angular Jacobian  (world frame)
            Rows 3-5: linear  Jacobian  (world frame)
        """
        H, Phi, _ = self._get_system_matrices()

        tid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, target_body_name
        )
        if tid == -1:
            raise ValueError(f"Body '{target_body_name}' not found in model.")

        # Extract tip rows from Φ·H
        J_body = (Phi @ H)[tid * 6 : (tid + 1) * 6, :]

        # Rotate body-frame Jacobian to world frame
        R = quat2mat(self.data.xquat[tid])
        J_world = np.empty_like(J_body)
        J_world[0:3, :] = R @ J_body[0:3, :]
        J_world[3:6, :] = R @ J_body[3:6, :]

        return J_world

    # ─────────────────────────────────────────────────────────────────────────
    def calculate_mass_matrix(self) -> np.ndarray:
        """
        Generalized (joint-space) mass matrix.
        Paper Eq. (25):  M = H^T · Φ^T · M_sys · Φ · H
        """
        H, Phi, M_sys = self._get_system_matrices()
        return H.T @ Phi.T @ M_sys @ Phi @ H

    # ─────────────────────────────────────────────────────────────────────────
    def calculate_operational_space_mass(self, J_linear: np.ndarray) -> np.ndarray:
        """
        Operational-space (Cartesian) mass matrix.
        Λ = (J · M⁻¹ · J^T)⁻¹

        Parameters
        ----------
        J_linear : (3, nv) array
            Linear (position) part of the Jacobian.
        """
        M    = self.calculate_mass_matrix()
        damp = 1e-4 * np.eye(M.shape[0])
        try:
            M_inv = np.linalg.inv(M + damp)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        JMJ     = J_linear @ M_inv @ J_linear.T
        damp_op = 1e-4 * np.eye(JMJ.shape[0])
        try:
            Lambda = np.linalg.inv(JMJ + damp_op)
        except np.linalg.LinAlgError:
            Lambda = np.linalg.pinv(JMJ)

        return Lambda
