"""
Example 02: LLM-Conditioned Generation (Mocked)
Shows how a text-conditioned backbone feeds into the Type6ValenceCore.
"""
import torch
from qbmg.models.backbone import TextConditionedBackbone
from qbmg.core import Type6ValenceCore
from qbmg.utils.metrics import validate_molecule


def main():
    num_atoms = 15
    batch_size = 2

    # Mock text embeddings (in production, these come from MedGemma-4B-IT)
    mock_text_embed = torch.randn(batch_size, 512)

    backbone = TextConditionedBackbone(num_atoms=num_atoms, text_embed_dim=512)
    physics_layer = Type6ValenceCore(num_atoms=num_atoms)

    # Forward pass
    raw_logits = backbone(mock_text_embed)

    # Mock valence bounds for batch
    valence_bounds = torch.tensor([
        [4., 4., 2., 1., 1., 4., 2., 1., 1., 1., 4., 2., 1., 1., 1.],
        [4., 2., 2., 4., 1., 1., 1., 4., 2., 1., 1., 4., 2., 1., 1.]
    ])

    valid_bonds = physics_layer(raw_logits, valence_bounds)

    for i in range(batch_size):
        report = validate_molecule(valid_bonds[i], valence_bounds[i])
        print(f"Molecule {i+1} validation: {report}")


if __name__ == "__main__":
    main()
