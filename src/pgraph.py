import numpy as np
import mujoco
from .common import form_spatial_transform, quat2mat, get_spatial_inertia

class PgraphModel:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.n_bodies = model.nbody
        
        # Storage for the vectors
        self.Pgraph = []
        self.jdof = []
        
        # Initialize logic
        self._build_topology()

    def _build_topology(self):
        """
        Constructs Pgraph and jdof vectors based on the paper's flowchart.
        """
        children_map = {i: [] for i in range(self.n_bodies)}
        for i in range(1, self.n_bodies):
            parent = self.model.body_parentid[i]
            children_map[parent].append(i)

        untraversed = {i: list(sorted(children)) for i, children in children_map.items()}
        sbs = [] # Separation Body Stack
        
        curr_i = 1 
        
        while True:
            self.Pgraph.append(curr_i)
            dof_count = self.model.body_jntnum[curr_i]
            self.jdof.append(dof_count)

            if len(children_map[curr_i]) > 1:
                if not sbs or sbs[-1] != curr_i:
                     sbs.append(curr_i)

            if len(untraversed[curr_i]) > 0:
                child = untraversed[curr_i].pop(0)
                curr_i = child
                continue
            
            while True:
                if not sbs:
                    return 
                
                parent_i = sbs[-1]
                curr_i = parent_i
                
                self.Pgraph.append(curr_i)
                self.jdof.append(0) 
                
                if len(untraversed[curr_i]) > 0:
                    child = untraversed[curr_i].pop(0)
                    curr_i = child
                    break 
                else:
                    sbs.pop() 
                    continue

    def _get_system_matrices(self):
        """
        Constructs H (Joint Map), Phi (Propagation), and M (Spatial Inertia) matrices.
        Handles Free Joints (Quadruped), Hinge, and Slide joints.
        """
        n = self.n_bodies
        nv = self.model.nv
        
        H_sys = np.zeros((6 * n, nv))
        E_phi = np.zeros((6 * n, 6 * n))
        M_sys = np.zeros((6 * n, 6 * n))
        
        for i in range(1, self.n_bodies):
            body_row = i * 6
            
            # --- 1. Build H (Joint Map) ---
            jnt_adr = self.model.body_jntadr[i]
            jnt_num = self.model.body_jntnum[i]
            
            for j in range(jnt_num):
                jid = jnt_adr + j
                jtype = self.model.jnt_type[jid]
                axis = self.model.jnt_axis[jid]
                dof_id = self.model.jnt_dofadr[jid]
                
                # Check Joint Type
                if jtype == 0: # Free Joint (6 DoF: 3 Lin + 3 Ang)
                    # MuJoCo qvel order: [Linear X,Y,Z, Angular X,Y,Z] (Usually)
                    # Wait, MuJoCo qvel for free joint is: [LinX, LinY, LinZ, AngX, AngY, AngZ]
                    # Spatial Vector V order: [AngX, AngY, AngZ, LinX, LinY, LinZ]
                    
                    # Fill Linear Part (Rows 3,4,5 in Spatial Vector)
                    # Corresponds to Dof 0,1,2 (Linear Velocity)
                    H_sys[body_row+3, dof_id+0] = 1.0 # LinX -> V_linX
                    H_sys[body_row+4, dof_id+1] = 1.0 # LinY -> V_linY
                    H_sys[body_row+5, dof_id+2] = 1.0 # LinZ -> V_linZ
                    
                    # Fill Angular Part (Rows 0,1,2 in Spatial Vector)
                    # Corresponds to Dof 3,4,5 (Angular Velocity)
                    H_sys[body_row+0, dof_id+3] = 1.0 # AngX -> V_angX
                    H_sys[body_row+1, dof_id+4] = 1.0 # AngY -> V_angY
                    H_sys[body_row+2, dof_id+5] = 1.0 # AngZ -> V_angZ

                elif jtype == 2: # Slide
                    h_vec = np.zeros(6)
                    h_vec[3:6] = axis
                    H_sys[body_row:body_row+6, dof_id] = h_vec

                elif jtype == 3: # Hinge
                    h_vec = np.zeros(6)
                    h_vec[0:3] = axis
                    H_sys[body_row:body_row+6, dof_id] = h_vec
            
            # --- 2. Build E_phi (Connectivity) ---
            parent = self.model.body_parentid[i]
            if parent > 0:
                pos_i = self.data.xpos[i]
                mat_i = quat2mat(self.data.xquat[i])
                pos_p = self.data.xpos[parent]
                mat_p = quat2mat(self.data.xquat[parent])
                
                p_rel = mat_p.T @ (pos_i - pos_p)
                R_rel = mat_p.T @ mat_i
                
                phi = form_spatial_transform(p_rel, R_rel)
                
                row_idx = i * 6
                col_idx = parent * 6
                E_phi[row_idx:row_idx+6, col_idx:col_idx+6] = phi

            # --- 3. Build M_sys (Spatial Inertia) ---
            M_i = get_spatial_inertia(
                self.model.body_mass[i], 
                self.model.body_inertia[i], 
                self.model.body_ipos[i]
            )
            M_sys[body_row:body_row+6, body_row:body_row+6] = M_i

        # Compute Phi = (I - E)^-1
        I_sys = np.eye(6 * n)
        try:
            Phi_sys = np.linalg.inv(I_sys - E_phi)
        except np.linalg.LinAlgError:
            Phi_sys = I_sys
            
        return H_sys, Phi_sys, M_sys

    def calculate_jacobian(self, target_body_name):
        """
        Calculates Jacobian J = Phi * H.
        Returns 6xNv Jacobian (Rows 0-2 Angular, 3-5 Linear).
        """
        H, Phi, _ = self._get_system_matrices()
        J_sys = Phi @ H
        
        tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
        if tid == -1:
            raise ValueError(f"Body {target_body_name} not found.")

        row_start = tid * 6
        J_spatial_body = J_sys[row_start:row_start+6, :]
        
        # Rotate from Body Frame to World Frame
        mat_target = quat2mat(self.data.xquat[tid])
        J_world = np.zeros_like(J_spatial_body)
        J_world[0:3, :] = mat_target @ J_spatial_body[0:3, :]
        J_world[3:6, :] = mat_target @ J_spatial_body[3:6, :]
        
        return J_world

    def calculate_mass_matrix(self):
        """
        Computes the Generalized Mass Matrix (M) using Pgraph Eq (25).
        M_gen = H.T * Phi.T * M_sys * Phi * H
        """
        H, Phi, M_sys = self._get_system_matrices()
        
        # Calculate Generalized Mass Matrix
        M_gen = H.T @ Phi.T @ M_sys @ Phi @ H
        return M_gen
    
    def calculate_operational_space_mass(self, J_linear):
        """
        Calculates the Operational Space Mass Matrix (Lambda).
        Lambda = (J * M_inv * J_transpose)^-1
        
        Args:
            J_linear: The 3xNv Linear Jacobian matrix.
        """
        # 1. Get Joint Space Mass Matrix
        M = self.calculate_mass_matrix()
        
        # 2. Invert M (Add small damping for stability near singularities)
        # Using pseudo-inverse or damped inverse is safer
        n = M.shape[0]
        damp = 1e-4 * np.eye(n)
        try:
            M_inv = np.linalg.inv(M + damp)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        # 3. Compute J * M_inv * J.T
        temp = J_linear @ M_inv @ J_linear.T
        
        # 4. Invert result to get Lambda
        # Lambda is usually 3x3 for position control
        damp_op = 1e-4 * np.eye(temp.shape[0])
        try:
            Lambda = np.linalg.inv(temp + damp_op)
        except np.linalg.LinAlgError:
            Lambda = np.linalg.pinv(temp)
            
        return Lambda
