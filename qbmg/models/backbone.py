"""
Reference PyTorch generators for the QBMG pipeline.
"""
import torch
import torch.nn as nn


class BondGeneratorBackbone(nn.Module):
    """
    A simple MLP backbone that maps latent noise to unconstrained bond logits.
    Can be replaced with SE(3)-Equivariant GNNs or transformer architectures.
    """

    def __init__(
        self,
        num_atoms: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.latent_dim = latent_dim
        out_dim = num_atoms * num_atoms

        layers = []
        in_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim) noise or conditioning vector
        Returns:
            bond_logits: (batch, num_atoms, num_atoms)
        """
        batch_size = z.size(0)
        flat = self.net(z)
        return flat.view(batch_size, self.num_atoms, self.num_atoms)


class TextConditionedBackbone(nn.Module):
    """
    Placeholder for text-conditioned generation (e.g., MedGemma-4B-IT).
    Currently implements a simple projection from text embedding dim to bond logits.
    """

    def __init__(self, num_atoms: int, text_embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.num_atoms = num_atoms
        self.projection = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms * num_atoms)
        )

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        batch_size = text_embedding.size(0)
        logits = self.projection(text_embedding)
        return logits.view(batch_size, self.num_atoms, self.num_atoms)
