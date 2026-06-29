import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, imsize=64, hflip=False,
                 hidden_dim=1024, cond_dim=256, shared_cond_proj=None):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.imsize = imsize
        self.hflip = hflip
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        assert shared_cond_proj is not None, "Must pass shared_cond_proj (DualProjection) from NeRF"
        self._shared_cond_proj_holder = [shared_cond_proj]

        input_nc = nc + cond_dim  # 3 + 256 = 259

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

        # aux_head: predict hidden_state [B, 1024]
        # 從 features（含 cond_map）學習，合理的捷徑
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 8, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, hidden_dim),
        )

        # img_encoder + mat_head: predict mat_feat [B, 7]
        # 只看影像，切斷 cond_map 的捷徑
        # 強迫 D 從視覺特徵學習結構參數差異，梯度才能有意義地回傳給 G
        self.img_encoder = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
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

    def forward(self, input, hidden_state, mat_feat, return_aux=False, return_mat=False):
        """
        input:        [B, nc*H*W] or [B, C, H, W]
        hidden_state: [B, 1024]
        mat_feat:     [B, 7]
        return_aux:   若 True，額外回傳 aux_head 預測的 hidden_state
        return_mat:   若 True，額外回傳 mat_head 預測的 mat_feat（只從影像）
        """
        input = input[:, :self.nc]
        input = input.view(-1, self.imsize, self.imsize, self.nc).permute(0, 3, 1, 2)

        if self.hflip:
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (len(input), 1, 1, 1)).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)

        # D 主體：image + cond_map → features → adversarial score
        cond = self._shared_cond_proj_holder[0](hidden_state, mat_feat)  # [B, 256]
        cond_map = cond.view(cond.size(0), self.cond_dim, 1, 1).expand(
            -1, -1, input.size(2), input.size(3))

        x        = torch.cat([input, cond_map], dim=1)
        features = self.main(x)
        out      = self.conv_out(features)

        if return_aux and return_mat:
            aux_pred = self.aux_head(features)          # [B, 1024]，從 features 學
            mat_pred = self.mat_head(self.img_encoder(input))  # [B, 7]，只從影像學
            return out, aux_pred, mat_pred

        if return_aux:
            return out, self.aux_head(features)

        if return_mat:
            return out, self.mat_head(self.img_encoder(input))

        return out