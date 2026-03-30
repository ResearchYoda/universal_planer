import numpy as np
import mujoco

def skew(v):
    """Returns 3x3 skew symmetric matrix for vector v."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def quat2mat(quat):
    """Converts a quaternion to a 3x3 rotation matrix using MuJoCo."""
    res = np.zeros(9)
    mujoco.mju_quat2Mat(res, quat)
    return res.reshape(3, 3)

def form_spatial_transform(pos, rot):
    """
    Constructs the 6x6 Spatial Transform Matrix (phi).
    """
    # Rotation part (R_child_parent^T)
    R_T = rot.T 
    
    # Translation part (cross product matrix)
    S_r = skew(pos)
    
    phi = np.zeros((6, 6))
    phi[0:3, 0:3] = R_T
    phi[3:6, 3:6] = R_T
    phi[3:6, 0:3] = -R_T @ S_r
    return phi

def get_spatial_inertia(mass, inertia_diag, com):
    """
    Constructs the 6x6 Spatial Inertia Matrix (M_b) at the body frame origin.
    
    Args:
        mass (float): Body mass
        inertia_diag (np.array): 3D vector of principal inertias
        com (np.array): 3D vector of Center of Mass relative to body origin
    
    Returns:
        M (6x6 numpy array): Spatial Inertia Matrix
    """
    # 1. Inertia at CoM (3x3)
    I_com = np.diag(inertia_diag)
    
    # 2. Cross product matrix of CoM vector
    c_cross = skew(com)
    
    # 3. Shift Inertia to Body Origin (Parallel Axis Theorem)
    # I_o = I_com - m * c_cross * c_cross
    I_o = I_com - mass * (c_cross @ c_cross)
    
    # 4. Coupling term (m * c_cross)
    mc_cross = mass * c_cross
    
    # 5. Construct 6x6 Matrix
    # [ I_o       mc_cross ]
    # [ mc_cross.T  m*I    ]
    M = np.zeros((6, 6))
    M[0:3, 0:3] = I_o
    M[0:3, 3:6] = mc_cross
    M[3:6, 0:3] = mc_cross.T
    M[3:6, 3:6] = np.eye(3) * mass
    
    return M