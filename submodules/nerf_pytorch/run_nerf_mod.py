"""
NeRF rendering — FiLM DualProjection 版本
==========================================
關鍵修正：condition_feature() 在 render() 層呼叫（此時 hs 是乾淨的 [B, 1024]），
展開到 per-ray 後再 chunk 進 batchify_rays，避免 per-ray hs 造成的 shape 錯誤。

資料流：
  render():
    condition_feature(hs[B], mat[B]) → cond[B, 256]
    → repeat → cond_full[total_rays, 256]
  batchify_rays():
    cond_full[chunk_rays, 256]  → render_rays()
  render_rays():
    cond_chunk[N_rays, 256] → expand → cond_pts[N_rays*N_samples, 256]
  run_network():
    直接使用 conditioning_expanded[total_pts, 256]，不再呼叫 condition_feature
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# ── 確保同目錄的 helpers 可以被找到 ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_nerf_helpers_mod import get_embedder, NeRF, get_rays, ndc_rays

# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
relu = partial(F.relu, inplace=True)


# ── batchify ─────────────────────────────────────────────────────────────────

def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs, conditioning):
        return torch.cat([
            fn(inputs[i:i+chunk], conditioning[i:i+chunk])
            for i in range(0, inputs.shape[0], chunk)
        ], 0)
    return ret


# ── run_network ───────────────────────────────────────────────────────────────

def run_network(inputs, viewdirs, fn, conditioning_expanded,
                embed_fn, embeddirs_fn, features=None, netchunk=1024*64):
    """
    conditioning_expanded: [total_pts, 256]，已由 render() 層計算並展開。
    不在此處呼叫 condition_feature，避免 per-ray hs 的 shape 問題。
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded    = embed_fn(inputs_flat)

    if features is not None:
        features = features.unsqueeze(1).expand(-1, inputs.shape[1], -1).flatten(0, 1)
        embedded = torch.cat([embedded, features], -1)

    if viewdirs is not None:
        input_dirs      = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs   = embeddirs_fn(input_dirs_flat)
        embedded        = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded, conditioning_expanded)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


# ── batchify_rays ─────────────────────────────────────────────────────────────

def batchify_rays(rays_flat, hidden_state, mat_feat, chunk=1024*32,
                  conditioning_full=None, **kwargs):
    """
    conditioning_full: [total_rays, 256]，在 render() 層計算好傳入。
    每個 chunk 取對應片段傳給 render_rays。
    """
    all_ret  = {}
    features = kwargs.get('features')

    for i in range(0, rays_flat.shape[0], chunk):
        if features is not None:
            kwargs['features'] = features[i:i+chunk]

        ret = render_rays(
            rays_flat[i:i+chunk],
            hidden_state[i:i+chunk],
            mat_feat[i:i+chunk],
            conditioning_chunk=conditioning_full[i:i+chunk],
            **kwargs
        )
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


# ── render ────────────────────────────────────────────────────────────────────

def render(H, W, focal, hidden_state, mat_feat, chunk=1024*32, rays=None,
           c2w=None, ndc=True, near=0., far=1., use_viewdirs=False,
           c2w_staticcam=None, **kwargs):
    """
    hidden_state : [B, 1024]  —— 乾淨的 batch-level tensor
    mat_feat     : [B, 7]
    """
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    rays_o   = torch.reshape(rays_o, [-1, 3]).float()
    rays_d   = torch.reshape(rays_d, [-1, 3]).float()
    near_t   = near * torch.ones_like(rays_d[..., :1])
    far_t    = far  * torch.ones_like(rays_d[..., :1])
    rays_cat = torch.cat([rays_o, rays_d, near_t, far_t], -1)

    bs           = hidden_state.shape[0]
    n_total_rays = rays_cat.shape[0]
    rays_per_img = n_total_rays // bs

    hs_per_ray  = hidden_state.unsqueeze(1).repeat(1, rays_per_img, 1).view(
        -1, hidden_state.shape[-1])
    mat_per_ray = mat_feat.unsqueeze(1).repeat(1, rays_per_img, 1).view(
        -1, mat_feat.shape[-1])

    # ★ 在 render 層用乾淨的 [B] 計算 conditioning，然後展開到 per-ray
    network_fn        = kwargs['network_fn']
    conditioning      = network_fn.condition_feature(hidden_state, mat_feat)  # [B, 256]
    conditioning_full = conditioning.unsqueeze(1).repeat(1, rays_per_img, 1).view(
        -1, conditioning.shape[-1])                              # [total_rays, 256]

    if use_viewdirs:
        rays_cat = torch.cat([rays_cat, viewdirs], -1)

    if kwargs.get('features') is not None:
        bs_f   = kwargs['features'].shape[0]
        N_rays = sh[0] // bs_f
        kwargs['features'] = kwargs['features'].unsqueeze(1).expand(
            -1, N_rays, -1).flatten(0, 1)

    all_ret = batchify_rays(
        rays_cat, hs_per_ray, mat_per_ray, chunk,
        conditioning_full=conditioning_full,
        **kwargs
    )

    for k in all_ret:
        k_sh       = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list  = [all_ret[k] for k in k_extract]
    ret_dict  = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


# ── create_nerf ───────────────────────────────────────────────────────────────

def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    input_ch      += args.feat_dim   # pos_enc(63) + z_feat(256) = 319
    input_ch_views = 0
    embeddirs_fn   = None

    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips     = [3]

    model = NeRF(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
    )
    grad_vars    = list(model.parameters())
    named_params = list(model.named_parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
        )
        grad_vars    += list(model_fine.parameters())
        named_params += list(model_fine.named_parameters())

    # ★ network_query_fn 接收 conditioning_expanded（已展開的 [pts, 256]）
    network_query_fn = lambda inputs, viewdirs, network_fn, conditioning_expanded, features: \
        run_network(
            inputs, viewdirs, network_fn, conditioning_expanded,
            features=features, embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn, netchunk=args.netchunk,
        )

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb'          : args.perturb,
        'N_importance'     : args.N_importance,
        'network_fine'     : model_fine,
        'N_samples'        : args.N_samples,
        'network_fn'       : model,
        'use_viewdirs'     : args.use_viewdirs,
        'raw_noise_std'    : args.raw_noise_std,
        'ndc'              : False,
        'lindisp'          : False,
    }
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb']       = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, grad_vars, named_params


# ── raw2outputs ───────────────────────────────────────────────────────────────

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, pytest=False):
    raw2alpha = lambda raw, dists, act_fn=relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists,
                       torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb   = torch.sigmoid(raw[..., :3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha   = raw2alpha(raw[..., 3] + noise, dists)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]

    rgb_map   = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map  = 1. / torch.max(
        1e-10 * torch.ones_like(depth_map),
        depth_map / (torch.sum(weights, -1) + 1e-10))
    acc_map   = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map


# ── render_rays ───────────────────────────────────────────────────────────────

def render_rays(ray_batch, hidden_state, mat_feat,
                network_fn, network_query_fn, N_samples,
                conditioning_chunk=None,
                features=None, retraw=False, lindisp=False,
                perturb=0., N_importance=0, network_fine=None,
                raw_noise_std=0., verbose=False, pytest=False):
    """
    conditioning_chunk : [N_rays, 256]，從 batchify_rays 傳入的 per-ray conditioning。
    在此展開到 per-point 後交給 network_query_fn。
    """
    N_rays   = ray_batch.shape[0]
    rays_o   = ray_batch[:, 0:3]
    rays_d   = ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds   = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1./near * (1. - t_vals) + 1./far * t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        mids   = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper  = torch.cat([mids, z_vals[..., -1:]], -1)
        lower  = torch.cat([z_vals[..., :1], mids],  -1)
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # ★ conditioning_chunk: [N_rays, 256] → [N_rays*N_samples, 256]
    cond_pts = conditioning_chunk.unsqueeze(1).expand(
        -1, N_samples, -1).flatten(0, 1)

    raw = network_query_fn(pts, viewdirs, network_fn, cond_pts, features)

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    return ret