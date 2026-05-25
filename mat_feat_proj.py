"""
MaterialFeatureProjector: 7 → proj_dim projection for material features.

Gives material features meaningful influence in NeRF conditioning.
Without projection: 7-dim vs 256-dim hidden_state → drowned out.
With projection:    64-dim vs 256-dim hidden_state → ~20% bandwidth.
"""

import torch
import torch.nn as nn


class MaterialFeatureProjector(nn.Module):
    def __init__(self, mat_dim: int = 7, proj_dim: int = 64):
        super().__init__()
        self.mat_dim = mat_dim
        self.proj_dim = proj_dim

        self.projector = nn.Sequential(
            nn.Linear(mat_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 7] → [B, proj_dim]"""
        return self.projector(x)