"""
ConditionBottleneck: Drop-in replacement for nn.Linear(1024, 256).

Architecture: 1024 → bottleneck_dim (16) → 256

Override mode (for interpolation training):
    The render pipeline expands hidden_state to per-ray before calling
    condition_feature, and calls it multiple times (once per ray chunk).
    Override must handle this:

    1. set_override(pre_decoded_256)  — shape [B, 256], B = batch size
    2. generator() internally: render() expands hs to per-ray, then
       batchify_rays chunks it, and run_network calls condition_feature
       with [chunk_size, 1024] for each chunk.
    3. forward() detects override is set, computes the repeat factor
       from input size, and returns the correctly expanded override.
    4. Override persists across multiple forward() calls (ray chunks).
    5. Call clear_override() after generator() returns.
"""

import torch
import torch.nn as nn


class ConditionBottleneck(nn.Module):
    def __init__(self, hidden_dim: int = 1024, out_dim: int = 256,
                 bottleneck_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, out_dim),
        )

        # Override: persistent until clear_override() is called
        self._override: torch.Tensor | None = None
        self._override_batch_size: int = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1024] → [B, bottleneck_dim]"""
        return self.encoder(x)

    def decode(self, b: torch.Tensor) -> torch.Tensor:
        """[B, bottleneck_dim] → [B, 256]"""
        return self.decoder(b)

    def set_override(self, pre_decoded: torch.Tensor):
        """
        Set a pre-decoded [B, 256] tensor. All subsequent forward() calls
        will return this tensor (expanded to match input batch dim) until
        clear_override() is called.

        Args:
            pre_decoded: [B, out_dim] where B is the number of images
                         (not rays — forward() handles the expansion).
        """
        self._override = pre_decoded
        self._override_batch_size = pre_decoded.shape[0]

    def clear_override(self):
        """Clear override. Must be called after generator() returns."""
        self._override = None
        self._override_batch_size = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [N, 1024] → [N, 256]

        Normal mode: encode → decode.
        Override mode: return override expanded to match N.

        In the render pipeline, N = chunk_size (per-ray), and the override
        is [B, 256] where B = n_interp images. Each image has N/B rays
        in this chunk, so we repeat_interleave by N/B.
        """
        if self._override is not None:
            N = x.shape[0]
            B = self._override_batch_size
            if N == B:
                # Called at batch level (e.g. by discriminator)
                return self._override
            elif N % B == 0:
                # Called at per-ray level: N = B * rays_per_chunk
                repeat_factor = N // B
                return self._override.repeat_interleave(repeat_factor, dim=0)
            else:
                # Fallback: N is not a multiple of B (shouldn't happen
                # in normal usage, but handle gracefully).
                # This can occur if batchify_rays sends a partial last chunk.
                # Expand override to cover all rays, then slice.
                rays_per_image = (N + B - 1) // B  # ceiling division
                expanded = self._override.repeat_interleave(rays_per_image, dim=0)
                return expanded[:N]
        return self.decode(self.encode(x))