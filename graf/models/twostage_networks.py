"""
Two-Stage Networks: SRNetwork + DiscriminatorHigh

Extracted from train_twostage.py for reusability across
train_twostage.py and eval_twostage.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Residual Block (shared by SR)
# ================================================================
class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.net(x)


# ================================================================
# SR Network (64 → 256, 4× via PixelShuffle)
# ================================================================
class SRNetwork(nn.Module):
    """
    Lightweight super-resolution network with global residual.
    Input:  [B, 3, 64, 64]   (NeRF output, range [-1, 1])
    Output: [B, 3, 256, 256] (upscaled, clamped to [-1, 1])

    Global residual: SR only learns the high-freq detail on top of
    a bilinear 4× upsample of the input.
    """

    def __init__(self, ch=64, n_rb=6):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.body = nn.Sequential(*[_ResidualBlock(ch) for _ in range(n_rb)])
        self.up = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True),
        )
        self.tail = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, x):
        # Global residual: SR only needs to learn the difference
        upsampled = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        h = self.head(x)
        h = self.body(h) + h
        h = self.up(h)
        return torch.clamp(self.tail(h) + upsampled, -1, 1)


# ================================================================
# D_high (256×256 discriminator) — same conditioning as D_low
# ================================================================
class DiscriminatorHigh(nn.Module):
    """
    Discriminator for 256×256 SR output (Phase 2).
    Uses the same shared condition projection as D_low.
    """

    def __init__(self, nc=3, ndf=64, hidden_dim=1024, cond_dim=256,
                 num_classes=7, shared_cond_proj=None):
        super().__init__()
        self.nc = nc
        self.cond_dim = cond_dim
        self.num_classes = num_classes

        assert shared_cond_proj is not None, "Must pass shared_cond_proj from NeRF"
        # Wrap in list to avoid nn.Module auto-registration
        self._shared_cond_proj_holder = [shared_cond_proj]

        inp = nc + cond_dim + num_classes  # 3 + 256 + 7 = 266
        self.main = nn.Sequential(
            # 256 → 128
            nn.Conv2d(inp, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 128 → 64
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, True),
            # 64 → 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # 32 → 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # 16 → 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 8 → 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_out = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

    def forward(self, img, label, hidden_state):
        bs = img.size(0)
        h, w = img.size(2), img.size(3)

        cond = self._shared_cond_proj_holder[0](hidden_state)  # [B, 256]
        cond_map = cond.view(bs, self.cond_dim, 1, 1).expand(-1, -1, h, w)
        lab = label[:, 7:14].view(bs, self.num_classes, 1, 1).expand(-1, -1, h, w)

        x = torch.cat([img, lab, cond_map], 1)
        return self.conv_out(self.main(x))


# ================================================================
# Utility
# ================================================================
def nerf_flat_to_img(nerf_flat, batch_size):
    """Convert [B*4096, 3] flat NeRF output to [B, 3, 64, 64] image."""
    return nerf_flat.view(batch_size, 64, 64, 3).permute(0, 3, 1, 2).contiguous()