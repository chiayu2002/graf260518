"""
ConditionBottleneck: Drop-in replacement for nn.Linear(1024, 256).

Replaces the single Linear projection with:
    1024 → bottleneck_dim (e.g. 16) → 256

- forward(x) works exactly like the old Linear (backward-compatible)
- encode(x) returns the low-dim bottleneck embedding (for interpolation)
- decode(b) maps bottleneck → 256

Usage in training loop:
    # Normal path (same as before):
    cond_256 = condition_feature(hidden_state)      # [B, 256]

    # Interpolation path:
    b_A = condition_feature.encode(hs_A)            # [B, bottleneck_dim]
    b_B = condition_feature.encode(hs_B)            # [B, bottleneck_dim]
    b_interp = lam * b_A + (1 - lam) * b_B         # [B, bottleneck_dim]
    cond_interp = condition_feature.decode(b_interp)# [B, 256]
"""

import torch
import torch.nn as nn


class ConditionBottleneck(nn.Module):
    """
    Bottleneck projection: hidden_dim → bottleneck_dim → out_dim.

    Parameters
    ----------
    hidden_dim : int
        Input dimension (GRU hidden state size, e.g. 1024).
    out_dim : int
        Output dimension (NeRF netwidth, e.g. 256).
    bottleneck_dim : int
        Bottleneck dimension (e.g. 16). Controls how much the model
        must compress the specimen identity before expanding.
    """

    def __init__(self, hidden_dim: int = 1024, out_dim: int = 256,
                 bottleneck_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder: 1024 → 16
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder: 16 → 256
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, out_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden state to bottleneck space. [B, 1024] → [B, 16]"""
        return self.encoder(x)

    def decode(self, b: torch.Tensor) -> torch.Tensor:
        """Expand bottleneck to conditioning dim. [B, 16] → [B, 256]"""
        return self.decoder(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full projection (backward-compatible). [B, 1024] → [B, 256]"""
        return self.decode(self.encode(x))