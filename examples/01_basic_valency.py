"""
Example 01: Basic Valency Projection
Demonstrates the Type6ValenceCore guaranteeing 100% valid output at epoch 0.
"""
import torch
from qbmg.core import Type6ValenceCore


def main():
    num_atoms = 10
    physics_layer = Type6ValenceCore(num_atoms=num_atoms)

    # Simulate raw neural backbone output
    raw_neural_guess = torch.randn(1, num_atoms, num_atoms, requires_grad=True)

    # Define valence constraints: C=4, O=2, H=1, etc.
    valence_bounds = torch.tensor([[4., 4., 2., 1., 1., 4., 2., 1., 1., 1.]])

    # Forward pass through the mathematical wormhole
    valid_bonds = physics_layer(raw_neural_guess, valence_bounds)

    print("=" * 60)
    print("Type 6 Quantum-Bound Molecular Generator")
    print("=" * 60)
    print(f"Output shape: {valid_bonds.shape}")
    print(f"Violations detected: {(valid_bonds.sum(dim=-1) > valence_bounds).any().item()}")
    print(f"Symmetry preserved: {torch.allclose(valid_bonds, valid_bonds.transpose(-1, -2))}")
    print(f"Non-negative bonds: {(valid_bonds >= 0).all().item()}")

    # Backward pass: gradient flows through IFT
    loss = valid_bonds.sum()
    loss.backward()
    print(f"Gradient passed to raw guess: {raw_neural_guess.grad is not None}")
    print(f"Gradient norm: {raw_neural_guess.grad.norm().item():.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
