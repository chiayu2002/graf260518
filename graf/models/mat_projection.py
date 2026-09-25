"""
MatProjection — 純 mat_feat conditioning（移除 GRU）
=====================================================
架構：
  mat_feat(7) → _expand_mat → [x, x², (1-x)²](21維)
              → mat_scale(21) → MLP(21→128→256→256) → embedding(256)

移除 GRU 的原因：
  RS315 和 RS330 的 GRU hidden state cos_sim ≈ 0.97，
  GRU 認為兩者幾乎一樣，導致 conditioning 無法區分兩個 specimen。
  mat_feat 的 force 維度差距（RS315=0.411, RS330=1.000）足夠讓模型
  學到視覺差異，因此改成純 mat_feat conditioning。

RS315 推論時直接輸入 RS315 的 mat_feat，不需要 GRU 或插值。

set_override / clear_override 供 InterpCache 插值使用，接口不變。
out_dim = 256，下游接口不變。
"""

import torch
import torch.nn as nn


class MatProjection(nn.Module):
    def __init__(self, mat_dim=7, out_dim=256):
        super().__init__()
        self.mat_dim = mat_dim
        self.out_dim = out_dim

        mat_in = mat_dim * 3   # 21（_expand_mat 展開後）

        # mat_scale：可學習的 per-dimension 重要性權重
        self.mat_scale = nn.Parameter(torch.ones(mat_in))

        # MLP：21 → 128 → 256 → 256
        # 比原本的 mat_film 更深，補足移除 hs_proj 後的表達能力
        self.mat_mlp = nn.Sequential(
            nn.Linear(mat_in, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, out_dim),
        )

        # 初始化最後一層，讓初始 embedding 接近零（穩定訓練初期）
        nn.init.zeros_(self.mat_mlp[-1].weight)
        nn.init.zeros_(self.mat_mlp[-1].bias)

        # override 機制（InterpCache 使用）
        self._override = None

    @staticmethod
    def _expand_mat(mat_feat):
        """
        mat_feat: [B, 7]，值域 [0, 1]
        欄位順序: displacement, force, corner_σ, corner_ε, core_σ, core_ε, cover_ε

        RS315 vs RS330 的差距分析：
          force(idx=1):    0.411 vs 1.000  差距 0.589 ← 最大，放大 3x
          corner_ε(idx=3): 0.725 vs 1.000  差距 0.275 ← 次大，放大 2x
          core_ε(idx=5):   0.727 vs 1.000  差距 0.273 ← 次大，放大 2x
          cover_ε(idx=6):  0.751 vs 1.000  差距 0.249 ← 次大，放大 2x
          core_σ(idx=4):   0.017 vs 0.000  差距 0.017 ← 最小，不放大

        放大後再做非線性展開，強制讓模型看到 RS315 和 RS330 的差異。
        clamp(0,1) 確保值域不超出範圍。
        """
        # 手動放大差異最大的維度
        weights = torch.ones(mat_feat.shape[-1],
                             dtype=mat_feat.dtype, device=mat_feat.device)
        weights[1] = 3.0   # force：差距最大
        weights[3] = 2.0   # corner_ε
        weights[5] = 2.0   # core_ε
        weights[6] = 2.0   # cover_ε

        mat_w = (mat_feat * weights).clamp(0, 1)   # 放大後 clamp 回 [0,1]

        x   = mat_w
        x2  = mat_w ** 2
        ix2 = (1.0 - mat_w) ** 2
        return torch.cat([x, x2, ix2], dim=-1)   # [B, 21]

    def encode(self, mat_feat, hs=None):
        """
        mat_feat: [B, 7]
        hs: 忽略（保留參數相容性，供舊程式碼呼叫）
        return:   [B, 256]
        """
        mat_exp = self._expand_mat(mat_feat.float())           # [B, 21]
        mat_exp = mat_exp * self.mat_scale.abs().clamp(min=0.5)
        return self.mat_mlp(mat_exp)                           # [B, 256]

    def forward(self, hs, mat_feat):
        """
        hs:       忽略（保留接口相容性）
        mat_feat: [B, 7]
        return:   [B, 256]
        """
        if self._override is not None:
            return self._override
        return self.encode(mat_feat)

    def set_override(self, embedding):
        self._override = embedding

    def clear_override(self):
        self._override = None