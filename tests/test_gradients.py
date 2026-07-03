"""
Gradient verification tests using PyTorch Autograd and finite differences.
"""
import torch
import pytest
from qbmg.core import Type6ValenceCore
from qbmg.core.implicit_solver import IFTGradientVerifier


def test_basic_valency_projection():
    n = 8
    core = Type6ValenceCore(num_atoms=n)
    raw = torch.randn(1, n, n, requires_grad=True)
    bounds = torch.tensor([[4., 2., 2., 1., 1., 4., 2., 1.]])

    out = core(raw, bounds)
    assert out.shape == (1, n, n)
    assert not (out.sum(dim=-1) > bounds).any()
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-4)


def test_gradient_flow():
    n = 5
    core = Type6ValenceCore(num_atoms=n)
    raw = torch.randn(1, n, n, requires_grad=True)
    bounds = torch.tensor([[4., 2., 1., 1., 4.]])

    out = core(raw, bounds)
    loss = out.sum()
    loss.backward()

    assert raw.grad is not None
    assert not torch.isnan(raw.grad).any()


def test_ift_gradient_accuracy():
    n = 6
    core = Type6ValenceCore(num_atoms=n)
    verifier = IFTGradientVerifier(num_atoms=n, eps=1e-3)

    A_hat = torch.randn(n, n)
    v_max = torch.tensor([4., 2., 2., 1., 1., 4.])

    analytical, fd, rel_err = verifier.verify(core, A_hat, v_max)
    print(f"Relative gradient error: {rel_err:.6f}")
    assert rel_err < 0.05  # 5% tolerance for finite differences


if __name__ == "__main__":
    test_basic_valency_projection()
    test_gradient_flow()
    test_ift_gradient_accuracy()
    print("All tests passed!")
