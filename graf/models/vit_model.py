import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PatchEmbed(nn.Module):
    """將圖像分成固定大小的補丁並線性嵌入"""
    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = x.reshape(8, 3, 32, 32)
        B, C, H, W = x.shape
        
        # 投影並重塑
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        
        return x

class ViewConsistencyTransformer(nn.Module):
    """基於 ViT 的視角一致性模型 - 同時預測水平和垂直角度"""
    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=256, 
                 num_heads=8, num_layers=6, mlp_ratio=4):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches
        print(f"初始化 ViT: img_size={img_size}, patch_size={patch_size}, num_patches={num_patches}")

        # CLS 標記和位置編碼
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Transformer 編碼器
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation="gelu"
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 預測頭 - 輸出兩個角度值
        self.angle_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2)  # 預測兩個角度值
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = 8
        
        # 補丁嵌入
        x = self.patch_embed(x)
        
        # 添加 CLS 標記
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置編碼
        x = x + self.pos_embed
        
        # Transformer 處理
        x = x.permute(1, 0, 2)  # (seq_len, batch, dim)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, dim)
        
        # 從 CLS 標記獲取特徵
        cls_feature = x[:, 0]
        
        # 預測角度值
        angles_raw = self.angle_head(cls_feature)
        
        # 分別處理水平和垂直角度，使它們符合各自的範圍
        horizontal_angle = torch.sigmoid(angles_raw[:, 0]).view(-1, 1)  # 0-1 範圍
        vertical_angle = torch.sigmoid(angles_raw[:, 1]).view(-1, 1) * 0.5  # 0-0.5 範圍
        
        # 合併兩個角度值
        angles = torch.cat([horizontal_angle, vertical_angle], dim=1)
        
        return angles