import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from torchsr.models import ninasr_b0, edsr

class CALayer(nn.Module):
    """通道注意力機制 (Channel Attention) - 幫助模型專注於重要細節"""
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
        
        # 初始化為基礎大小 (例如 8x8)，forward 時再動態縮放
        self.latent_codes = nn.Parameter(
            torch.randn(num_views, 3, 64, 64) * 0.1
        )
        
    def forward(self, view_idx: int, target_size: tuple) -> torch.Tensor:
        """獲取特定視角的潛在代碼並縮放到目標尺寸 (H, W)"""
        if isinstance(view_idx, torch.Tensor):
            view_idx = view_idx.item()
        
        code = self.latent_codes[view_idx % self.num_views]
        
        # 動態縮放到與當前輸入圖片一致的大小
        # target_size 預期為 (height, width)
        return F.interpolate(code.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)


class ConsistencyEnforcingModule(nn.Module):
    def __init__(self, blur_kernel_size: int = 3):
        super().__init__()
        self.blur_kernel_size = blur_kernel_size
        # 使用高斯核代替均勻核，這能提供更好的邊緣平滑效果
        sigma = blur_kernel_size / 3.0
        x = torch.arange(blur_kernel_size) - blur_kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer('blur_kernel', kernel_2d.unsqueeze(0).unsqueeze(0))
    
    def forward(self, sr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        # 1. H 矩陣操作：模糊 + 下採樣 (模擬退化過程)
        blurred = F.conv2d(sr_image, self.blur_kernel.expand(sr_image.size(1), -1, -1, -1), 
                          padding=self.blur_kernel_size//2, groups=sr_image.size(1))
        downsampled = F.interpolate(blurred, size=lr_image.shape[-2:], mode='bilinear', align_corners=False)
        
        # 2. 計算數據一致性殘差 (Data Consistency)
        residual = lr_image - downsampled
        upsampled_residual = F.interpolate(residual, size=sr_image.shape[-2:], mode='bilinear', align_corners=False)
        
        # 3. 修正 SR 圖像：這裡權重建議設為 1.0 以嚴格遵守一致性
        refined_sr = sr_image + 1.0 * upsampled_residual
        return refined_sr
    
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            # CALayer(channels)  # 加入注意力層提取高頻紋理
        )
    def forward(self, x):
        return x + self.conv(x)  # 殘差連接

class CCSR(nn.Module):
    """一致性控制超分辨率模組 - 全動態尺寸支援"""
    
    def __init__(self, num_views: int, scale_factor: int = 1):
        super().__init__()
        
        self.num_views = num_views
        self.scale_factor = scale_factor
        
        # CCLC 現在不需要預設的 lr_height/lr_width
        self.cclc = ConsistencyControllingLatentCode(num_views)
        self.cem = ConsistencyEnforcingModule()
        
        self.sr_network = self._build_sr_network(scale_factor)
        # self.activation = nn.Tanh()

    # def _build_sr_network(self, scale_factor: int) -> nn.Module:
    #     blocks = [nn.Conv2d(6, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True)]
    #     for _ in range(12):  # 增加深度至 12 層
    #         blocks.append(ResBlock(64))
    #     blocks.append(nn.Conv2d(64, 3, 3, padding=1))
    #     return nn.Sequential(*blocks)

    def _build_sr_network(self, scale_factor: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64), # 加入殘差塊
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 3, 3, padding=1)
    )   

    # def _build_sr_network(self) -> nn.Module:
    #     return nn.Sequential(
    #         # 1. 特徵提取層 (6通道 -> 64通道)
    #         nn.Conv2d(6, 64, 3, padding=1),
    #         nn.LeakyReLU(0.2, inplace=True),
            
    #         # 2. 上採樣層 (Upsampling)：將特徵圖放大 2 倍
    #         # 使用 PixelShuffle 是目前 SR 最推薦的做法，不會有棋盤效應
    #         nn.Conv2d(64, 64 * 4, 3, padding=1), 
    #         nn.PixelShuffle(2), # 放大 2 倍，通道變回 64
    #         nn.LeakyReLU(0.2, inplace=True),
            
    #         # 3. 殘差塊 (ResBlocks)：在高解析度空間 (2x) 進行細節重建
    #         ResBlock(64),
    #         ResBlock(64),
    #         ResBlock(64),
            
    #         # 4. 降採樣層 (Downsampling)：縮回 1x
    #         # 使用步長為 2 的卷積 (Strided Convolution) 來學習如何壓縮細節
    #         nn.Conv2d(64, 64, 3, stride=2, padding=1),
    #         nn.LeakyReLU(0.2, inplace=True),
            
    #         # 5. 輸出層
    #         nn.Conv2d(64, 3, 3, padding=1)
    #     )

    # def _build_sr_network(self) -> nn.Module:
    #     # 1. Adapter: 處理 6 通道輸入並轉為 3 通道
    #     adapter = nn.Sequential(
    #         nn.Conv2d(6, 64, 3, padding=1),
    #         nn.LeakyReLU(0.2, inplace=True),
    #         nn.Conv2d(64, 3, 3, padding=1)
    #     )

    #     # 2. Pretrained Body: 使用 scale=2 成功下載權重
    #     # 這會將 H, W 變為 2H, 2W
    #     pretrained_body = ninasr_b0(scale=2, pretrained=True)

    #     # 3. Downsampler: 將解析度縮回原始大小 (1:1)
    #     # 使用步長為 2 的卷積或 F.interpolate 都可以
    #     # 這裡建議用卷積，可以學習如何保留預訓練模型產生的細節
    #     downsampler = nn.Sequential(
    #         nn.Conv2d(3, 64, 3, padding=1),
    #         nn.LeakyReLU(0.2, inplace=True),
    #         nn.Conv2d(64, 3, 4, stride=2, padding=1) # 縮小 2 倍
    #     )

    #     return nn.Sequential(adapter, pretrained_body, downsampler)
    
    def forward(self, lr_image: torch.Tensor, view_idx: int) -> torch.Tensor:
        """CCSR 前向傳播：輸出尺寸動態跟隨 lr_image"""
        batch_size, _, h, w = lr_image.shape
        
        # 1. 獲取潛在代碼並動態縮放到輸入影像的大小 (h, w)
        latent_code = self.cclc(view_idx, (h, w))
        latent_code = latent_code.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # 2. 連接輸入 (此處尺寸已完全對齊)
        combined_input = torch.cat([lr_image, latent_code], dim=1)
        
        # 3. 通過網路生成增強影像
        sr_output = self.sr_network(combined_input) + lr_image
        # sr_output = self.activation(sr_output)
        sr_output = torch.clamp(sr_output, -1, 1)
        
        # 4. 應用一致性執行模組 (CEM)
        refined_sr = self.cem(sr_output, lr_image)
        
        return refined_sr