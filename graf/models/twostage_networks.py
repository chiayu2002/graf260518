"""Two-Stage Networks: SRNetwork + DiscriminatorHigh (FiLM DualProjection 版本)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SRNetwork(nn.Module):
    """PixelShuffle 4× 超解析度網路：64×64 → 256×256"""
    def __init__(self, ch=64, n_rb=6):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1),
            nn.LeakyReLU(0.2, True))
        self.body = nn.Sequential(*[_ResidualBlock(ch) for _ in range(n_rb)])
        self.up   = nn.Sequential(
            nn.Conv2d(ch, ch*4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch*4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True),
        )
        self.tail = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, x):
        h = self.head(x)
        h = self.body(h) + h
        h = self.up(h)
        return torch.clamp(self.tail(h), -1, 1)


class DiscriminatorHigh(nn.Module):
    """256×256 高解析度 Discriminator，共享 FiLM DualProjection conditioning。"""
    def __init__(self, nc=3, ndf=64, hidden_dim=1024, cond_dim=256,
                 shared_cond_proj=None):
        super().__init__()
        self.nc       = nc
        self.cond_dim = cond_dim
        assert shared_cond_proj is not None
        self._shared_cond_proj_holder = [shared_cond_proj]

        inp = nc + cond_dim  # 3 + 256 = 259
        self.main = nn.Sequential(
            nn.Conv2d(inp,     ndf,   4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf,     ndf,   4, 2, 1, bias=False), nn.BatchNorm2d(ndf),   nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf,     ndf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2,   ndf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4,   ndf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8,   ndf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
        )
        self.conv_out = nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)

    def forward(self, img, hidden_state, mat_feat):
        bs   = img.size(0)
        h, w = img.size(2), img.size(3)
        cond     = self._shared_cond_proj_holder[0](hidden_state, mat_feat)  # [B, 256]
        cond_map = cond.view(bs, self.cond_dim, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([img, cond_map], 1)
        return self.conv_out(self.main(x))


def nerf_flat_to_img(nerf_flat, batch_size):
    """NeRF 輸出 flat → [B, 3, 64, 64]"""
    return nerf_flat.view(batch_size, 64, 64, 3).permute(0, 3, 1, 2).contiguous()