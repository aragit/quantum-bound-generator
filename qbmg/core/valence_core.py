"""
Type 6 Quantum-Bound Molecular Generator - Differentiable Valence Core
"""
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class Type6ValenceCore(nn.Module):
    """
    A differentiable convex optimization layer that projects raw neural bond logits
    onto the set of physically valid molecular graphs.

    Enforces:
      1. Symmetry (undirected bonds): A = A^T
      2. Non-negativity: A >= 0
      3. Valence bounds: sum_j A_ij <= v_i
    """

    def __init__(self, num_atoms: int):
        super().__init__()
        self.num_atoms = num_atoms

        # CVXPY parameters (inputs from neural network)
        A_hat = cp.Parameter((num_atoms, num_atoms), name="A_hat")
        v_max = cp.Parameter(num_atoms, name="v_max")

        # CVXPY variable (output valid adjacency matrix)
        A_valid = cp.Variable((num_atoms, num_atoms), name="A_valid")

        # Objective: closest valid matrix in Frobenius norm
        objective = cp.Minimize(0.5 * cp.sum_squares(A_valid - A_hat))

        # Physics constraints
        constraints = [
            A_valid == A_valid.T,                 # Symmetry
            A_valid >= 0,                         # Non-negative bond orders
            cp.sum(A_valid, axis=1) <= v_max      # Valence ceiling
        ]

        problem = cp.Problem(objective, constraints)
        self.physics_layer = CvxpyLayer(
            problem,
            parameters=[A_hat, v_max],
            variables=[A_valid]
        )

    def forward(self, raw_logits: torch.Tensor, valence_bounds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_logits: (batch, N, N) or (N, N) unconstrained neural outputs
            valence_bounds: (batch, N) or (N,) hard valence limits per atom

        Returns:
            valid_bonds: Same shape as raw_logits, guaranteed physically valid
        """
        # Symmetrize input to respect undirected graph prior
        sym_logits = (raw_logits + raw_logits.transpose(-1, -2)) / 2.0

        # CvxpyLayer handles batch dimension automatically if present
        if sym_logits.dim() == 2:
            valid_bonds, = self.physics_layer(sym_logits, valence_bounds)
            return valid_bonds
        else:
            # Explicit batch loop for robustness across cvxpylayers versions
            batch_size = sym_logits.size(0)
            outputs = []
            for i in range(batch_size):
                vb, = self.physics_layer(sym_logits[i], valence_bounds[i])
                outputs.append(vb)
            return torch.stack(outputs, dim=0)
