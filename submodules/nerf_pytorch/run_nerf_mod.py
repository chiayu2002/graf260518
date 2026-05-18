import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial

import matplotlib.pyplot as plt

from .run_nerf_helpers_mod import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

relu = partial(F.relu, inplace=True)            # saves a lot of memory


def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs, hidden_state):
        return torch.cat([
            fn(inputs[i:i+chunk], hidden_state[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, hidden_state, embed_fn, embeddirs_fn,
                features=None, netchunk=1024*64,
                mat_feat=None):
    """輸出 rgb and sigma — 不再需要 label"""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    num_total_points = inputs_flat.shape[0]
    batch_size = hidden_state.shape[0]
    points_per_batch = num_total_points // batch_size

    # 預先投影 1024 → 256 再 expand,省記憶體
    hidden_state_projected = fn.condition_feature(hidden_state)

    # ── 新增：concat specimen_onehot 和 mat_feat ──────────────
    extra_list = [hidden_state_projected]
    # if specimen_onehot is not None:
    #     extra_list.append(specimen_onehot.float())          # [B, 3]
    if mat_feat is not None:
        extra_list.append(mat_feat.float())                 # [B, 7]
    hidden_state_enhanced = torch.cat(extra_list, dim=-1)  # [B, 256+3+7=266]

    hidden_state_expanded = hidden_state_enhanced.repeat_interleave(
        points_per_batch, dim=0
    ).float()

    if features is not None:
        features = features.unsqueeze(1).expand(-1, inputs.shape[1], -1).flatten(0, 1)
        embedded = torch.cat([embedded, features], -1)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded, hidden_state_expanded)
    outputs = torch.reshape(outputs_flat,
                            list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, hidden_state,  mat_feat,chunk=1024*32, **kwargs):
    all_ret = {}
    features = kwargs.get('features')
    for i in range(0, rays_flat.shape[0], chunk):
        if features is not None:
            kwargs['features'] = features[i:i+chunk]
        ret = render_rays(rays_flat[i:i+chunk], hidden_state[i:i+chunk], mat_feat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, hidden_state, mat_feat, chunk=1024*32, rays=None, c2w=None,
           ndc=True, near=0., far=1., use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
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

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    bs = hidden_state.shape[0]
    n_total_rays = rays.shape[0]
    rays_per_img = n_total_rays // bs
    hidden_state_per_ray = hidden_state.unsqueeze(1).repeat(1, rays_per_img, 1).view(
        -1, hidden_state.shape[-1]
    )

    # specimen_onehot = torch.tensor(specimen_onehot, dtype=torch.float32).to(device)
    # specimen_onehot_per_ray = specimen_onehot.unsqueeze(1).repeat(1, rays_per_img, 1).view(
    #     -1, specimen_onehot.shape[-1])

    mat_feat = torch.tensor(mat_feat, dtype=torch.float32).to(device)
    mat_feat_per_ray = mat_feat.unsqueeze(1).repeat(1, rays_per_img, 1).view(
        -1, mat_feat.shape[-1])

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    if kwargs.get('features') is not None:
        bs_f = kwargs['features'].shape[0]
        N_rays = sh[0] // bs_f
        kwargs['features'] = kwargs['features'].unsqueeze(1).expand(-1, N_rays, -1).flatten(0, 1)

    all_ret = batchify_rays(rays, hidden_state_per_ray, mat_feat_per_ray, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch += args.feat_dim
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [3]

    # ── 新增：cond_extra_dim = specimen(3) + mat_feat(7) ──
    cond_extra_dim = getattr(args, 'num_classes', 10)

    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, numclasses=cond_extra_dim)
    grad_vars = list(model.parameters())
    named_params = list(model.named_parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        grad_vars += list(model_fine.parameters())
        named_params = list(model_fine.named_parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn, hidden_state, features, \
                               mat_feat=None: run_network(
    inputs, viewdirs, network_fn, hidden_state,
    features=features,
    embed_fn=embed_fn,
    embeddirs_fn=embeddirs_fn,
    netchunk=args.netchunk,
    # specimen_onehot=specimen_onehot,
    mat_feat=mat_feat,
    )
#specimen_onehot=None,
    render_kwargs_train = {             
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
        'ndc': False,
        'lindisp': False,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, grad_vars, named_params


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, pytest=False):
    raw2alpha = lambda raw, dists, act_fn=relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1)+1e-10))
    acc_map = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch, hidden_state, mat_feat,
                network_fn, network_query_fn, N_samples,
                features=None, retraw=False, lindisp=False,
                perturb=0., N_importance=0, network_fine=None,
                raw_noise_std=0., verbose=False, pytest=False):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape)
        if pytest:
            np.random.seed(0)
            t_rand = torch.Tensor(np.random.rand(*list(z_vals.shape)))
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    raw = network_query_fn(pts, viewdirs, network_fn, hidden_state, features, mat_feat)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, pytest=pytest
    )
    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    return ret