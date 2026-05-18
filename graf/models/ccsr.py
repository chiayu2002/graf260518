import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CALayer(nn.Module):
    """通道注意力機制 (Channel Attention)"""
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ConsistencyControllingLatentCode(nn.Module):
    """一致性控制潛在代碼 (CCLC) - 支援動態尺寸"""
    
    def __init__(self, num_views: int):
        super().__init__()
        self.num_views = num_views
        self.latent_codes = nn.Parameter(
            torch.randn(num_views, 3, 64, 64) * 0.1
        )
        
    def forward(self, view_idx: int, target_size: tuple) -> torch.Tensor:
        if isinstance(view_idx, torch.Tensor):
            view_idx = view_idx.item()
        code = self.latent_codes[view_idx % self.num_views]
        return F.interpolate(
            code.unsqueeze(0), size=target_size,
            mode='bilinear', align_corners=False
        ).squeeze(0)


class ConsistencyEnforcingModule(nn.Module):
    def __init__(self, blur_kernel_size: int = 3):
        super().__init__()
        self.blur_kernel_size = blur_kernel_size
        sigma = blur_kernel_size / 3.0
        x = torch.arange(blur_kernel_size) - blur_kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer('blur_kernel', kernel_2d.unsqueeze(0).unsqueeze(0))
    
    def forward(self, sr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        """
        sr_image: CCSR 輸出, [-1, 1]
        lr_image: NeRF 輸入, [-1, 1]
        """
        blurred = F.conv2d(
            sr_image,
            self.blur_kernel.expand(sr_image.size(1), -1, -1, -1),
            padding=self.blur_kernel_size // 2,
            groups=sr_image.size(1)
        )
        downsampled = F.interpolate(
            blurred, size=lr_image.shape[-2:],
            mode='bilinear', align_corners=False
        )
        residual = lr_image - downsampled
        upsampled_residual = F.interpolate(
            residual, size=sr_image.shape[-2:],
            mode='bilinear', align_corners=False
        )
        refined_sr = sr_image + 1.0 * upsampled_residual
        return refined_sr


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.conv(x)


class CCSR(nn.Module):
    """
    一致性控制超分辨率模組
    
    輸入/輸出值域統一為 [-1, 1]
    
    scale_factor=1: 同解析度增強
    scale_factor=2: 32×32 → 64×64 超解析度 (PixelShuffle)
    """
    
    def __init__(self, num_views: int, scale_factor: int = 1):
        super().__init__()
        self.num_views = num_views
        self.scale_factor = scale_factor
        self.cclc = ConsistencyControllingLatentCode(num_views)
        self.cem = ConsistencyEnforcingModule()
        self.sr_network = self._build_sr_network(scale_factor)

    def _build_sr_network(self, scale_factor: int) -> nn.Module:
        if scale_factor == 1:
            return nn.Sequential(
                nn.Conv2d(6, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(64),
                ResBlock(64),
                ResBlock(64),
                nn.Conv2d(64, 3, 3, padding=1),
            )
        elif scale_factor == 2:
            return nn.Sequential(
                nn.Conv2d(6, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(64),
                ResBlock(64),
                ResBlock(64),
                ResBlock(64),
                nn.Conv2d(64, 64 * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(64),
                ResBlock(64),
                nn.Conv2d(64, 3, 3, padding=1),
            )
        else:
            raise ValueError(f"Unsupported scale_factor: {scale_factor}")

    def forward(self, lr_image: torch.Tensor, view_idx: int) -> torch.Tensor:
        """
        Args:
            lr_image: [B, 3, H, W], range [-1, 1]
            view_idx: 視角索引
        Returns:
            [B, 3, H*sf, W*sf], range [-1, 1]
        """
        batch_size, _, h, w = lr_image.shape
        
        # 1. 潛在代碼（跟 lr_image 同尺寸）
        latent_code = self.cclc(view_idx, (h, w))
        latent_code = latent_code.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # 2. 拼接 [B, 6, H, W]
        combined_input = torch.cat([lr_image, latent_code], dim=1)
        
        # 3. SR 網路
        sr_output = self.sr_network(combined_input)
        
        # 4. 殘差連接（輸入輸出都是 [-1,1]，網路只學小殘差）
        if self.scale_factor == 1:
            sr_output = sr_output + lr_image
        else:
            lr_upsampled = F.interpolate(
                lr_image, scale_factor=self.scale_factor,
                mode='bilinear', align_corners=False
            )
            sr_output = sr_output + lr_upsampled
        
        sr_output = torch.clamp(sr_output, -1, 1)
        
        # 5. CEM
        refined_sr = self.cem(sr_output, lr_image)
        return refined_sr