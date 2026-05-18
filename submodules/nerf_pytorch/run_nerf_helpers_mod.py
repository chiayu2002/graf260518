import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial



# Misc
relu = partial(F.relu, inplace=True)            # saves a lot of memory


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        # ========================================================
        # [AMP safe] 高頻 sin/cos 在 fp16 下會失精,強制 fp32
        # 對 bfloat16 也適用 — 保守起見統一走 fp32
        # ========================================================
        with torch.cuda.amp.autocast(enabled=False):
            inputs = inputs.float()
            return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3,
                 output_ch=4, skips=[4], numclasses=4, use_viewdirs=False, **kwargs):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.numclasses = numclasses

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch + W + numclasses, W)] +
            [nn.Linear(W, W) if i not in self.skips
             else nn.Linear(W + input_ch + W + numclasses, W) for i in range(D-1)]
        )
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        # ========================================================
        # [移除] condition_embedding 從未被呼叫,刪掉
        # ========================================================

        # 共享給 D 用 — 1024 → W (=256)
        self.condition_feature = nn.Linear(1024, W)

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, hidden_state):
        """
        x: [N_pts, input_ch + W]  (positional encoding + projected features)
        hidden_state: [N_pts, W]  (已經由 run_network 預先投影)
        """
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        hidden_state = hidden_state.to(input_pts.device)

        # 直接使用已投影的 hidden_state
        feature_embedding = hidden_state

        input_o, input_shape = torch.split(input_pts, [63, 256], dim=-1)
        conditioned_input = torch.cat([input_o, input_shape, feature_embedding], dim=-1)
        h = conditioned_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = relu(h)
            if i in self.skips:
                h = torch.cat([h, conditioned_input], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = relu(h)
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        return outputs   


# ========================================================
# [修正2] 移除 get_rays 中的 debug 程式碼
# 原本每次呼叫都會:
#   1. rays_d.detach().cpu() → GPU→CPU 複製 256×256×3 tensor，
#      強制 GPU pipeline 同步等待，嚴重影響效能
#   2. 計算角度值但結果從未使用（print 被註解掉了）
# 每個 iteration 呼叫 get_rays 至少 8 次（batch_size=8）
# ========================================================
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    x = (i-W*.5)/focal
    y = -(j-H*.5)/focal
    z = -torch.ones_like(i)

    dirs = torch.stack([x, y, z], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)

    # [修正2] 已移除 debug 程式碼:
    # rays_d_cpu = rays_d.detach().cpu()   ← 每次呼叫都做 GPU→CPU 複製!
    # calculate_ray_angle(...)              ← 計算但從不使用
    # center_ray, left_ray, right_ray      ← 計算但從不使用
    
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):   #把射線原點移到near平面
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


def get_rays_ortho(H, W, c2w, size_h, size_w):
    """Similar structure to 'get_rays' in submodules/nerf_pytorch/run_nerf_helpers.py"""
    # # Rotate ray directions from camera frame to the world frame
    rays_d = -c2w[:3, 2].view(1, 1, 3).expand(W, H, -1)  # direction to center in world coordinates 到世界座標中心的方向

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()

    # Translation from center for origins 從原點中心平移
    rays_o = torch.stack([(i - W * .5), -(j - H * .5), torch.zeros_like(i)], -1)

    # Normalize to [-size_h/2, -size_w/2]
    rays_o = rays_o * torch.tensor([size_w / W, size_h / H, 1]).view(1, 1, 3)

    # Rotate origins to the world frame 將原點旋轉到世界座標系
    rays_o = torch.sum(rays_o[..., None, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate origins to the world frame 將原點平移到世界座標系
    rays_o = rays_o + c2w[:3, -1].view(1, 1, 3)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples