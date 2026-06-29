"""Two-Stage Networks: SRNetwork + DiscriminatorHigh (DualProjection version)"""
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
    def __init__(self, ch=64, n_rb=6):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, ch, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.body = nn.Sequential(*[_ResidualBlock(ch) for _ in range(n_rb)])
        self.up = nn.Sequential(
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
    def __init__(self, nc=3, ndf=64, hidden_dim=1024, cond_dim=256,
                 shared_cond_proj=None):
        super().__init__()
        self.nc = nc
        self.cond_dim = cond_dim

        assert shared_cond_proj is not None
        self._shared_cond_proj_holder = [shared_cond_proj]

        inp = nc + cond_dim  # 259
        self.main = nn.Sequential(
            nn.Conv2d(inp, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, ndf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
        )
        self.conv_out = nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)

        # img_encoder + mat_head: predict mat_feat [B, 7]
        # 輸入 256×256，多一層 conv 處理更大的解析度
        # 只看影像，切斷 cond_map 的捷徑
        self.img_encoder = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mat_head = nn.Sequential(
            nn.Linear(ndf * 4, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 7),
            nn.Sigmoid(),   # mat_feat 值域 [0, 1]
        )

    def forward(self, img, hidden_state, mat_feat, return_mat=False):
        """
        img:          [B, 3, H, W]
        hidden_state: [B, 1024]
        mat_feat:     [B, 7]
        return_mat:   若 True，額外回傳 mat_head 預測的 mat_feat（只從影像）
        """
        bs = img.size(0)
        h, w = img.size(2), img.size(3)

        # D 主體：image + cond_map → adversarial score
        cond     = self._shared_cond_proj_holder[0](hidden_state, mat_feat)  # [B, 256]
        cond_map = cond.view(bs, self.cond_dim, 1, 1).expand(-1, -1, h, w)
        x        = torch.cat([img, cond_map], 1)
        out      = self.conv_out(self.main(x))

        if return_mat:
            mat_pred = self.mat_head(self.img_encoder(img))  # [B, 7]，只從影像學
            return out, mat_pred

        return out


def nerf_flat_to_img(nerf_flat, batch_size):
    return nerf_flat.view(batch_size, 64, 64, 3).permute(0, 3, 1, 2).contiguous()