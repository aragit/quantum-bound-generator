<h1 align="center">Quantum-Bound Molecular Generator (QBMG)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/CVXPY-1.3+-9C27B0" alt="CVXPY 1.3+">
  <img src="https://img.shields.io/badge/diffcp-1.0.22+-FF9800" alt="diffcp 1.0.22+">
  <img src="https://img.shields.io/badge/cvxpylayers-0.1+-4CAF50" alt="cvxpylayers">
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900" alt="CUDA 11.8+">
  <img src="https://img.shields.io/badge/pytest-7.0+-0A9EDC" alt="pytest 7.0+">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License MIT">
</p>

<p align="center">
  <b>A fully differentiable, embedded neuro-symbolic architecture for zero-waste molecular generation.</b>
</p>

---

## Overview

Modern deep learning in chemistry relies on a fragmented **"Generate-then-Filter"** (Type 2) paradigm. Neural networks hallucinate physically impossible topologies, and rigid external scripts discard them. This wastes massive compute and deprives the generative model of gradient feedback from the laws of physics.

This repository implements a **Type 6 Neuro[Symbolic] Generator**. By formulating quantum valency and bond geometry as continuous, convex constraints embedded directly within the PyTorch forward pass, this architecture mathematically guarantees **100% physically valid output** at every forward pass.

Originally conceptualized as the structural generative engine powering the AXIOMIS closed-loop clinical intelligence system, this framework is now open-sourced for broader application in biopharma, multi-drug optimization, and materials science.

---

## Architecture: The Single Substrate

QBMG eliminates the discrete API barrier between the neural network and the physics engine. Both exist on the same mathematical substrate.

| Component | Description |
|-----------|-------------|
| **Neural Backbone** | Probabilistic generator producing unconstrained bond-logit adjacency matrices. Modular: swap in SE(3)-Equivariant GNNs or text-conditioned backbones (e.g., MedGemma-4B-IT). |
| **Differentiable Physics Core** | Intercepts raw logits and projects them onto the boundary of a physically valid geometric space via convex optimization. |
| **Implicit Gradient Solver** | Uses the Implicit Function Theorem (IFT) to analytically compute the exact backward gradient of the constraint, shaping the network's latent space during training. |

---

## Mathematical Foundation

To make physical laws differentiable, we frame molecular generation as a constrained projection.

Let $\hat{A}$ be the raw, unconstrained neural output. We project onto the convex set of valid chemistries $\mathcal{C}$:

$$
A^* = \arg\min_{A} \frac{1}{2} \| A - \hat{A} \|_F^2
$$

$$
\text{subject to:} \quad \sum_{j} A_{ij} \le v_i, \quad A_{ij} \ge 0, \quad A = A^T
$$

Where $v_i$ represents the hard valency limits of atom $i$.

### The KKT Conditions and Implicit Function Theorem

Standard Autograd cannot unroll the iterative loops of a convex optimizer without catastrophic memory explosion. Instead, we bypass the loop by differentiating directly at the optimum.

At convergence, the solution $A^*$ satisfies the Karush-Kuhn-Tucker (KKT) equilibrium:

$$
\nabla_A \mathcal{L}(A^*, \lambda^*, \nu^*) = 0
$$

$$
\lambda^* \odot g(A^*) = 0
$$

$$
g(A^*) \le 0, \quad \lambda^* \ge 0, \quad h(A^*) = 0
$$

By applying the **Implicit Function Theorem** to this system, we calculate the exact total derivative at the boundary, yielding the Jacobian $\frac{\partial A^*}{\partial \hat{A}}$.

When `loss.backward()` is called, the gradient passes seamlessly through the KKT equilibrium. The network learns exactly how its raw logits violated physics and shifts its latent space accordingly.

---

## System Requirements

- **OS:** Linux (Ubuntu 20.04/22.04) or macOS
- **Compute:** CUDA 11.8+ compatible GPU (Minimum 8GB VRAM for batched operations)
- **Python:** 3.10+
- **Core Dependencies:** PyTorch >= 2.0.0, CVXPY >= 1.3.0, diffcp >= 1.0.22, cvxpylayers

---

## Installation

We recommend an isolated virtual environment.

```bash
# 1. Clone the repository
git clone https://github.com/aragit/quantum-bound-generator.git
cd quantum-bound-generator

# 2. Create a virtual environment
python3 -m venv qbmg-env
source qbmg-env/bin/activate

# 3. Install PyTorch (adjust CUDA version to your hardware)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 4. Install the core library and dependencies
pip install -e .
```

## Quick Start
```
import torch
from qbmg.core import Type6ValenceCore


# 1. Initialize the embedded constraint layer for a 10-atom molecule
num_atoms = 10
physics_layer = Type6ValenceCore(num_atoms=num_atoms)

# 2. Simulate raw output from a neural backbone
raw_neural_guess = torch.randn(1, num_atoms, num_atoms, requires_grad=True)

# 3. Define physical valence constraints (e.g., C=4, O=2, H=1)
valence_bounds = torch.tensor([[4., 4., 2., 1., 1., 4., 2., 1., 1., 1.]])

# 4. Forward Pass: The mathematical wormhole
valid_bonds = physics_layer(raw_neural_guess, valence_bounds)

print("Violations detected:", (valid_bonds.sum(dim=-1) > valence_bounds).any().item())
# Output: False

# 5. Backward Pass: Gradients flow through the IFT
loss = valid_bonds.sum()
loss.backward()

print("Gradient passed to raw guess:", raw_neural_guess.grad is not None)
# Output: True
```

## Repository Structure
```
quantum-bound-generator/
├── qbmg/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── valence_core.py        # Type 6 Differentiable Convex Layer
│   │   └── implicit_solver.py     # IFT Jacobian computations
│   ├── models/
│   │   └── backbone.py            # Reference PyTorch generators
│   └── utils/
│       └── metrics.py             # Valency and thermodynamic validators
├── examples/
│   ├── 01_basic_valency.py
│   └── 02_llm_conditioned_generation.py
├── tests/
│   └── test_gradients.py          # Autograd gradient checks
├── requirements.txt
├── setup.py
└── README.md
```

## Acknowledgments

Special thanks to the Barnabus Task Force Team for fostering the environment where this architecture was conceived. A specific note of appreciation to Nasrin, whose diligent clinical ML engineering work helped bridge the gap between abstract graph generation and practical, medically viable utility.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
