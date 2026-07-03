"""
Valency and thermodynamic validators for generated molecular graphs.
"""
import torch


def check_valency_violations(A: torch.Tensor, v_max: torch.Tensor, tol: float = 1e-5) -> bool:
    """Returns True if any atom exceeds its valence bound."""
    bond_sums = A.sum(dim=-1)
    return (bond_sums > v_max + tol).any().item()


def check_symmetry(A: torch.Tensor, tol: float = 1e-5) -> bool:
    """Returns True if A is symmetric (undirected graph)."""
    return torch.allclose(A, A.transpose(-1, -2), atol=tol)


def check_nonnegativity(A: torch.Tensor, tol: float = -1e-5) -> bool:
    """Returns True if all bond orders are non-negative."""
    return (A >= tol).all().item()


def validate_molecule(A: torch.Tensor, v_max: torch.Tensor) -> dict:
    """
    Comprehensive validation report for a generated adjacency matrix.
    """
    return {
        "symmetric": check_symmetry(A),
        "nonnegative": check_nonnegativity(A),
        "valency_valid": not check_valency_violations(A, v_max),
        "total_bond_energy_proxy": A.sum().item()
    }
