"""
Multi-Channel Neural Rodrigues Operator

Implements Equations 6-8 from:
  "Rodrigues Network for Learning Robot Actions" (Zhang et al., 2025)

Classical Rodrigues rotation formula:
  R(ω̂, θ) = I₃ + sin θ [ω̂] + (1 - cos θ)[ω̂]²

Neural generalization — replace fixed coefficients with learnable weights
and generalize joint angle to a multi-dimensional joint feature Θ:

  U[i,j] = W^bias[i,j] + Σ_c (W^cos[i,j,c]·cos(Θ[c]) + W^sin[i,j,c]·sin(Θ[c]))

  F^out[j] = Σ_i (F^in[i] @ U[i,j] + Ū[i,j] @ F^in[i])

where:
  F^in  ∈ R^{B × C_L_in × 4 × 4}  — input  link features  (homogeneous-matrix channels)
  F^out ∈ R^{B × C_L_out × 4 × 4} — output link features
  Θ     ∈ R^{B × C_J}             — joint feature
  U, Ū  ∈ R^{B × C_L_out × C_L_in × 4 × 4} — state-conditioned transform matrices
"""

import torch
import torch.nn as nn


class NeuralRodriguesOperator(nn.Module):
    """Multi-Channel Neural Rodrigues Operator (Sec. 3.3)."""

    def __init__(self, C_L_in: int, C_L_out: int, C_J: int):
        super().__init__()
        self.C_L_in  = C_L_in
        self.C_L_out = C_L_out
        self.C_J     = C_J

        # Primary operator: W* = {W^bias, W^cos, W^sin}
        self.W_bias = nn.Parameter(torch.zeros(C_L_out, C_L_in, 4, 4))
        self.W_cos  = nn.Parameter(torch.zeros(C_L_out, C_L_in, C_J, 4, 4))
        self.W_sin  = nn.Parameter(torch.zeros(C_L_out, C_L_in, C_J, 4, 4))

        # Conjugate operator: W̄* = {W̄^bias, W̄^cos, W̄^sin}
        self.Wb_bias = nn.Parameter(torch.zeros(C_L_out, C_L_in, 4, 4))
        self.Wb_cos  = nn.Parameter(torch.zeros(C_L_out, C_L_in, C_J, 4, 4))
        self.Wb_sin  = nn.Parameter(torch.zeros(C_L_out, C_L_in, C_J, 4, 4))

        self._init_weights()

    def _init_weights(self):
        std = 0.02 / (self.C_L_in ** 0.5)
        for p in self.parameters():
            nn.init.normal_(p, std=std)

    def forward(self, F_in: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_in:  (B, C_L_in, 4, 4)  — parent link feature
            theta: (B, C_J)            — joint feature
        Returns:
            F_out: (B, C_L_out, 4, 4) — transformed feature
        """
        cos_t = torch.cos(theta)  # (B, C_J)
        sin_t = torch.sin(theta)  # (B, C_J)

        # Build U and Ū: (B, C_L_out, C_L_in, 4, 4)
        # einsum 'jicpq,bc->bjipq': contract over C_J dimension c
        U = (self.W_bias
             + torch.einsum('jicpq,bc->bjipq', self.W_cos, cos_t)
             + torch.einsum('jicpq,bc->bjipq', self.W_sin, sin_t))

        Ub = (self.Wb_bias
              + torch.einsum('jicpq,bc->bjipq', self.Wb_cos, cos_t)
              + torch.einsum('jicpq,bc->bjipq', self.Wb_sin, sin_t))

        # F^out[j] = Σ_i ( F^in[i] @ U[i,j]  +  Ū[i,j] @ F^in[i] )
        # term1[b,j,p,r] = Σ_{i,q} F_in[b,i,p,q] · U[b,j,i,q,r]
        term1 = torch.einsum('bipq,bjiqr->bjpr', F_in, U)

        # term2[b,j,p,r] = Σ_{i,q} Ū[b,j,i,p,q] · F_in[b,i,q,r]
        term2 = torch.einsum('bjipq,biqr->bjpr', Ub, F_in)

        return term1 + term2  # (B, C_L_out, 4, 4)
