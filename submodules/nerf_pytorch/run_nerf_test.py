import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
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
    def ret(inputs, label):
        label_oftype = label[:,0]
        return torch.cat([fn(inputs[i:i+chunk], label_oftype) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, label, embed_fn, embeddirs_fn, features=None, netchunk=1024*64):   #輸出rgb and sigma
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) #524288 3
    embedded = embed_fn(inputs_flat)
    #print(f"0embedded.shape: {embedded.shape}") 524288 63
    if features is not None:
        # expand features to shape of flattened inputs  524288 256
        features = features.unsqueeze(1).expand(-1, inputs.shape[1], -1).flatten(0, 1)
        features_shape = features

        embedded = torch.cat([embedded, features_shape], -1)
        #print(f"features: {features_shape}")  319

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape) #8192 64 3
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])  #524288 3
        embedded_dirs = embeddirs_fn(input_dirs_flat) #524288 27
        embedded = torch.cat([embedded, embedded_dirs], -1)  #524288 346

    outputs_flat = batchify(fn, netchunk)(embedded, label)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])  #8192 64 4
    return outputs


# def batchify_rays(rays_flat, label, chunk=1024*32, **kwargs):  #批次render rays

#     all_ret = {}
#     features = kwargs.get('features')
#     for i in range(0, rays_flat.shape[0], chunk):
#         if features is not None:
#             kwargs['features'] = features[i:i+chunk]
#         ret = render_rays(rays_flat[i:i+chunk],label,  **kwargs)
#         for k in ret:
#             if k not in all_ret:
#                 all_ret[k] = []
#             all_ret[k].append(ret[k])

#     all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
#     return all_ret

def batchify_rays(rays_flat, label, chunk=1024*32, **kwargs):
    """按label分类收集渲染结果的批处理函数
    Args:
        rays_flat: 展平的光线数据
        label: 标签数据
        chunk: 批次大小
        **kwargs: 其他参数
    Returns:
        dict: 按label分类的渲染结果
    """
    label_results = {}
    features = kwargs.get('features')

    # 计算每个batch的rays數量
    batch_size = label.shape[0]
    rays_per_batch = rays_flat.shape[0] // batch_size

    # print(f"Debug - Total rays: {rays_flat.shape[0]}, Batch size: {batch_size}, Rays per batch: {rays_per_batch}")

    # 確保label形狀正確
    if len(label.shape) > 1:
        label = label[:, 0]  # 使用第一列作为标签

    # 獲取唯一的label值
    unique_labels = torch.unique(label)
    # print(f"Debug - Unique labels: {unique_labels}")
    
    # 為每個label初始化结果字典
    for l in unique_labels:
        label_results[l.item()] = {'rgb_map': [], 'disp_map': [], 'acc_map': []}

    # 按chunk處理rays
    for i in range(0, rays_flat.shape[0], chunk):
        chunk_end = min(i + chunk, rays_flat.shape[0])
        chunk_rays = rays_flat[i:chunk_end]
        
        # 计算当前chunk对应的batch indices
        batch_start_idx = i // rays_per_batch
        batch_end_idx = (chunk_end - 1) // rays_per_batch
        chunk_labels = []

        # 收集當前 chunk 的所有相關 labels
        for batch_idx in range(batch_start_idx, min(batch_end_idx + 1, batch_size)):
            rays_in_this_batch = min(rays_per_batch, chunk_end - i - (batch_idx - batch_start_idx) * rays_per_batch)
            chunk_labels.extend([label[batch_idx].item()] * rays_in_this_batch)
        
        chunk_label = torch.tensor(chunk_labels, device=label.device)
        
        if features is not None:
            kwargs['features'] = features[i:chunk_end]

        # print(f"Debug - Processing chunk {i}:{chunk_end}, chunk_label shape: {chunk_label.shape}")
            
        # 渲染当前chunk的rays
        ret = render_rays(chunk_rays, chunk_label.unsqueeze(1), **kwargs)

        # print(f"Debug - Render return keys: {ret.keys()}")
        
        # 按label分类收集结果
        for l in unique_labels:
            l_idx = (chunk_label == l)
            
            if l_idx.any():
                for k in ['rgb_map', 'disp_map', 'acc_map']:
                    if k in ret:
                        filtered_data = ret[k][l_idx]
                        if len(ret[k].shape) > 1:
                            filtered_data = filtered_data.reshape(-1, ret[k].shape[-1])
                        label_results[l.item()][k].append(filtered_data)

    # 合併每個label的结果
    final_results = {}
    for l in unique_labels:
        l_item = l.item()
        final_results[l_item] = {
            k: torch.cat(label_results[l_item][k], 0) if label_results[l_item][k] else None
            for k in ['rgb_map', 'disp_map', 'acc_map']
        }

    return final_results

def render(H, W, focal, label, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays #8192 3
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float() #8192 3

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    #print("rays values:", rays) #torch.Size([8192, 8])
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) #torch.Size([8192, 11])
        #print("rays values:", rays)

    # Expand features to shape of rays
    if kwargs.get('features') is not None:
        bs = kwargs['features'].shape[0]
        N_rays = sh[0] // bs
        kwargs['features'] = kwargs['features'].unsqueeze(1).expand(-1, N_rays, -1).flatten(0, 1)

    # Render and reshape
    all_ret = batchify_rays(rays, label, chunk, **kwargs)

    # 創建按標籤分類的結果字典
    label_results = {}
    
    # 為每個標籤收集結果
    for label_idx in all_ret:
        curr_results = all_ret[label_idx]
        
        # 建立當前標籤的結果字典
        label_results[label_idx] = {
            'rgb': curr_results['rgb_map'],
            'disp': curr_results['disp_map'],
            'acc': curr_results['acc_map']
        }
        
        # 收集extras
        extras = {}
        for k in curr_results:
            if k not in ['rgb_map', 'disp_map', 'acc_map']:
                extras[k] = curr_results[k]
        label_results[label_idx]['extras'] = extras

    return label_results

def render_path(render_poses, hwf, chunk, render_kwargs, label, features=None, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        feature = None if features is None else features[i]
        rgb, disp, acc, _ = render(H, W, focal, label, features=feature, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch += args.feat_dim 
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=(args.use_viewdirs or args.feat_dim_appearance > 0))
    grad_vars = list(model.parameters())
    named_params = list(model.named_parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        grad_vars += list(model_fine.parameters())
        named_params = list(model_fine.named_parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn, label, features: run_network(inputs, viewdirs, network_fn, label,
                                                                                  features=features,
                                                                                  embed_fn=embed_fn,
                                                                                  embeddirs_fn=embeddirs_fn,
                                                                                  netchunk=args.netchunk,
                                                                                  )

    render_kwargs_train = {             
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'ndc': False,
        'lindisp': False,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, grad_vars, named_params


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """ A helper function for `render_rays`.
    """
    raw2alpha = lambda raw, dists, act_fn=relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1)+1e-10))     # add eps to avoid division by zero
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                label,
                network_fn,
                network_query_fn,
                N_samples,
                features=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn, label, features)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn, label, features)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret