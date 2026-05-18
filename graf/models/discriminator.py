import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, imsize=64, hflip=False,
                 hidden_dim=1024, cond_dim=256, num_classes=14, shared_cond_proj=None):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.imsize = imsize
        self.hflip = hflip
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.num_classes = num_classes

        # ========================================================
        # [共享] 用 list 包起來避免被 nn.Module 自動註冊為 submodule
        # 這樣 D.parameters() 不會包含 shared_cond_proj 的權重
        # → D 的 optimizer 不會更新它，也不會重複儲存在 checkpoint
        # 它只透過 G 的 optimizer 更新（因為它在 NeRF 上是正常 submodule）
        # ========================================================
        assert shared_cond_proj is not None, "Must pass shared_cond_proj from NeRF"
        self._shared_cond_proj_holder = [shared_cond_proj]

        input_nc = nc + cond_dim + num_classes  # 3 + 256 + 7

        blocks = []
        if self.imsize == 64:
            blocks += [
                nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize == 32:
            blocks += [
                nn.Conv2d(input_nc, ndf * 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        blocks += [
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        self.main = nn.Sequential(*blocks)
        self.conv_out = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        self.label_out = nn.Sequential(
            nn.Conv2d(ndf * 8, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, num_classes, 1, bias=False),
        )

        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 8, 32),           # 512 → 32 bottleneck
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, hidden_dim),         # 32 → 1024
        )

    def forward(self, input, label, hidden_state, return_aux=False):
        """
        input: [B, 3, H, W] 格式的影像 (不再接受 flattened pixels)
        label: [B, >=14] — 使用 label[:,7:14] 作為材質條件
        hidden_state: [B, hidden_dim]
        """
        bs = input.size(0)
        h, w = input.size(2), input.size(3)

        if self.hflip:
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (bs, 1, 1, 1), device=input.device).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)

        # 用共享的 condition projection (1024 → 256)
        cond = self._shared_cond_proj_holder[0](hidden_state)  # [B, 256]
        cond_map = cond.view(bs, self.cond_dim, 1, 1).expand(-1, -1, h, w)

        # [修正] 不再硬編碼 batch_size=8 和 spatial=64
        label_cond = label[:, 7:14].view(bs, self.num_classes, 1, 1).expand(-1, -1, h, w)

        x = torch.cat([input, label_cond, cond_map], dim=1)
        features = self.main(x)
        out = self.conv_out(features)

        if return_aux:
            aux_pred = self.aux_head(features)          # [B, 1024]
            label_pred = self.label_out(features)       # [B, num_classes, 1, 1]
            label_pred = label_pred.view(bs, -1)        # [修正] squeeze → [B, num_classes]
            return out, aux_pred, label_pred

        return out