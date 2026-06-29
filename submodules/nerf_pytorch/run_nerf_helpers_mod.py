"""
NeRF helpers — FiLM DualProjection 版本
========================================
conditioning 維度：256（FiLM out_dim）
pts_linears 輸入：pos_enc(63) + z_feat(256) + conditioning(256) = 575
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

# 確保同目錄的 dual_projection 可以被找到
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dual_projection import DualProjection

relu = partial(F.relu, inplace=True)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
        max_freq = self.kwargs['max_freq_log2']
        N_freqs  = self.kwargs['num_freqs']
        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim   = out_dim

    def embed(self, inputs):
        with torch.cuda.amp.autocast(enabled=False):
            inputs = inputs.float()
            return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    embed_kwargs = {
        'include_input'  : True,
        'input_dims'     : 3,
        'max_freq_log2'  : multires - 1,
        'num_freqs'      : multires,
        'log_sampling'   : True,
        'periodic_fns'   : [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3,
                 output_ch=4, skips=[4], use_viewdirs=False, **kwargs):
        super(NeRF, self).__init__()
        self.D              = D
        self.W              = W
        self.input_ch       = input_ch       # pos_enc(63) + z_feat(256) = 319
        self.input_ch_views = input_ch_views
        self.skips          = skips
        self.use_viewdirs   = use_viewdirs

        # FiLM DualProjection：hs(1024→256) modulated by mat(7→γ/β[256]) → 256
        # out_dim = 256，與 W 相同
        self.condition_feature = DualProjection(
            hs_dim=1024, hs_out=256,
            mat_dim=7,   mat_out=256,
        )

        # pts_linears 輸入維度：
        #   input_ch     = pos_enc(63) + z_feat(256) = 319
        #   conditioning = 256  (FiLM out_dim)
        #   → first layer input = 319 + 256 = 575
        cond_dim = self.condition_feature.out_dim  # 256

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch + cond_dim, W)] +
            [nn.Linear(W, W) if i not in self.skips
             else nn.Linear(W + input_ch + cond_dim, W)
             for i in range(D - 1)]
        )
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W // 2)]
        )

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear   = nn.Linear(W, 1)
            self.rgb_linear     = nn.Linear(W // 2, 3)
        else:
            self.output_linear  = nn.Linear(W, output_ch)

    def forward(self, x, conditioning):
        """
        x            : [N_pts, input_ch + input_ch_views]
        conditioning : [N_pts, 256]  已由 render() 層展開
        """
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1)
        conditioning = conditioning.to(input_pts.device)

        # input_pts = [pos_enc(63) | z_feat(256)]
        input_o, input_shape = torch.split(input_pts, [63, 256], dim=-1)
        conditioned_input = torch.cat([input_o, input_shape, conditioning], dim=-1)
        # conditioned_input: [N_pts, 63+256+256] = [N_pts, 575]

        h = conditioned_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = relu(h)
            if i in self.skips:
                h = torch.cat([h, conditioned_input], -1)

        if self.use_viewdirs:
            alpha   = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = relu(h)
            rgb     = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


# ── Ray helpers ──────────────────────────────────────────────────────────────

def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t(); j = j.t()
    x = (i - W*.5) / focal
    y = -(j - H*.5) / focal
    z = -torch.ones_like(i)
    dirs   = torch.stack([x, y, z], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    t      = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 =  1. + 2.*near / rays_o[..., 2]
    d0 = -1./(W/(2.*focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2.*near / rays_o[..., 2]
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    return rays_o, rays_d


def get_rays_ortho(H, W, c2w, size_h, size_w):
    rays_d = -c2w[:3, 2].view(1, 1, 3).expand(W, H, -1)
    i, j   = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t(); j = j.t()
    rays_o = torch.stack([(i-W*.5), -(j-H*.5), torch.zeros_like(i)], -1)
    rays_o = rays_o * torch.tensor([size_w/W, size_h/H, 1]).view(1, 1, 3)
    rays_o = torch.sum(rays_o[..., None, :] * c2w[:3, :3], -1)
    rays_o = rays_o + c2w[:3, -1].view(1, 1, 3)
    return rays_o, rays_d


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    weights = weights + 1e-5
    pdf     = weights / torch.sum(weights, -1, keepdim=True)
    cdf     = torch.cumsum(pdf, -1)
    cdf     = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0., 1., steps=N_samples).expand(
            list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        u = np.linspace(0., 1., N_samples) if det else np.random.rand(*new_shape)
        u = torch.Tensor(u)
    u    = u.contiguous()
    inds = torch.searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1)*torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g  = torch.gather(cdf.unsqueeze(1).expand(matched_shape),  2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom  = (cdf_g[..., 1] - cdf_g[..., 0])
    denom  = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t      = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples