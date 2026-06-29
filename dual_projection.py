"""
FiLM-based Conditioning Projection
====================================
架構：
  hs_proj:  Linear(1024 → 256)
  mat_film: Linear(21 → 128) → LeakyReLU → Linear(128 → 512) → split → gamma[256] + beta[256]
  encode:   gamma * hs_proj(hs) + beta  →  [B, 256]

mat_feat 輸入加入非線性擴展（原始 + 平方 + (1-x)^2），放大 RS315/RS330 中間值與極值的差距。
mat_feat 必須透過 modulate hs_proj 的輸出才能影響結果，lookup table 捷徑被結構性消除。

gamma 初始化為 1，beta 初始化為 0，訓練初期行為接近純 hs_proj（恆等調製）。

set_override / clear_override 供 InterpCache 插值使用，接口不變。
"""

import torch
import torch.nn as nn


class DualProjection(nn.Module):
    def __init__(self, hs_dim=1024, hs_out=256, mat_dim=7, mat_out=256):
        """
        hs_out  : hs_proj 的輸出維度，同時也是 out_dim（256）
        mat_out : FiLM 版本中以 hs_out 為準，此參數保留以維持接口相容
        mat_dim : mat_feat 的原始維度（7），內部會擴展到 mat_dim*3（21）
        """
        super().__init__()
        self.hs_dim  = hs_dim
        self.mat_dim = mat_dim
        self.hs_out  = hs_out
        self.mat_out = hs_out   # FiLM: gamma/beta 維度 == hs_out
        self.out_dim = hs_out   # 256，下游接口不變

        # ── hs 投影：1024 → 256 ──────────────────────────────────
        self.hs_proj = nn.Linear(hs_dim, hs_out)

        # ── mat → FiLM 參數 ──────────────────────────────────────
        # 輸入：[x, x^2, (1-x)^2]，維度 7*3 = 21
        # 輸出：gamma[256] | beta[256]，共 512
        mat_in = mat_dim * 3  # 21
        self.mat_film = nn.Sequential(
            nn.Linear(mat_in, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, hs_out * 2),   # → 512
        )

        # gamma 初始化為 1，beta 初始化為 0（恆等調製）
        nn.init.zeros_(self.mat_film[-1].weight)
        bias_init = torch.zeros(hs_out * 2)
        bias_init[:hs_out] = 1.0          # gamma 部分初始為 1
        self.mat_film[-1].bias = nn.Parameter(bias_init)

        # ── override 機制（InterpCache 使用）─────────────────────
        self._override = None

    # ── 非線性輸入擴展 ────────────────────────────────────────────
    @staticmethod
    def _expand_mat(mat_feat):
        """
        mat_feat: [B, 7]，值域 [0, 1]
        return:   [B, 21]  = [x | x^2 | (1-x)^2]
        """
        x   = mat_feat
        x2  = mat_feat ** 2
        ix2 = (1.0 - mat_feat) ** 2
        return torch.cat([x, x2, ix2], dim=-1)

    # ── 核心編碼 ──────────────────────────────────────────────────
    def encode(self, hs, mat_feat):
        """
        hs:       [B, 1024]
        mat_feat: [B, 7]
        return:   [B, 256]
        """
        hs_emb     = self.hs_proj(hs)                        # [B, 256]
        mat_exp    = self._expand_mat(mat_feat.float())      # [B, 21]
        film_params = self.mat_film(mat_exp)                 # [B, 512]
        gamma, beta = film_params.chunk(2, dim=-1)           # 各 [B, 256]
        return gamma * hs_emb + beta                         # [B, 256]

    # ── 前向傳播 ──────────────────────────────────────────────────
    def forward(self, hs, mat_feat):
        """
        正常推論路徑。
        若 set_override 已設定，直接返回預計算的插值 embedding。
        注意：override 返回 [n_interp, 256]，此時 hs 也應是 [n_interp, 1024]。
        render() 層負責後續的 repeat_interleave 展開。
        """
        if self._override is not None:
            return self._override
        return self.encode(hs, mat_feat)

    def set_override(self, embedding):
        """InterpCache 注入預插值 embedding，跳過 encode 計算。"""
        self._override = embedding

    def clear_override(self):
        self._override = None