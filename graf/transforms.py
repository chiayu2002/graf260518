import torch
from math import sqrt, exp

from submodules.nerf_pytorch.run_nerf_helpers_mod import get_rays, get_rays_ortho


class ImgToPatch(object):
    def __init__(self, ray_sampler, hwf):
        self.ray_sampler = ray_sampler
        self.hwf = hwf      # camera intrinsics

    def __call__(self, img):
        rgbs = []
        for img_i in img:
            pose = torch.eye(4)         # use dummy pose to infer pixel values
            _, selected_idcs, pixels_i = self.ray_sampler(H=self.hwf[0], W=self.hwf[1], focal=self.hwf[2], pose=pose)
            if selected_idcs is not None:
                rgbs_i = img_i.flatten(1, 2).t()[selected_idcs]
            else:
                rgbs_i = torch.nn.functional.grid_sample(img_i.unsqueeze(0), 
                                     pixels_i.unsqueeze(0), mode='bilinear', align_corners=True)[0]
                rgbs_i = rgbs_i.flatten(1, 2).t()
            rgbs.append(rgbs_i)

        rgbs = torch.cat(rgbs, dim=0)       # (B*N)x3

        return rgbs


class RaySampler(object): #生成相機射線
    def __init__(self, N_samples, orthographic=False):
        super(RaySampler, self).__init__()
        self.N_samples = N_samples
        self.scale = torch.ones(1,).float()
        self.return_indices = True
        self.orthographic = orthographic

    def __call__(self, H, W, focal, pose):
        if self.orthographic: #正射投影
            size_h, size_w = focal      # Hacky
            rays_o, rays_d = get_rays_ortho(H, W, pose, size_h, size_w)
        else: #透射投影
            rays_o, rays_d = get_rays(H, W, focal, pose)

        select_inds = self.sample_rays(H, W) #取樣射線

        if self.return_indices:
            rays_o = rays_o.view(-1, 3)[select_inds] #光線原點
            rays_d = rays_d.view(-1, 3)[select_inds] #光線方向

            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds %  W) / float(W) - 0.5

            hw = torch.stack([h,w]).t()

        else:
            rays_o = torch.nn.functional.grid_sample(rays_o.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_d = torch.nn.functional.grid_sample(rays_d.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_o = rays_o.permute(1,2,0).view(-1, 3)
            rays_d = rays_d.permute(1,2,0).view(-1, 3)

            hw = select_inds
            select_inds = None

        return torch.stack([rays_o, rays_d]), select_inds, hw

    def sample_rays(self, H, W):
        raise NotImplementedError


class FullRaySampler(RaySampler):
    def __init__(self, **kwargs):
        super(FullRaySampler, self).__init__(N_samples=None, **kwargs)

    def sample_rays(self, H, W):
        return torch.arange(0, H*W) #生成了一個從 0 到 H*W-1 的長度為 H*W 的一維張量



class FlexGridRaySampler(RaySampler):
    def __init__(self, N_samples, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1,
                 **kwargs):
        self.N_samples_sqrt = int(sqrt(N_samples))
        super(FlexGridRaySampler, self).__init__(self.N_samples_sqrt**2, **kwargs)

        self.random_shift = random_shift
        self.random_scale = random_scale

        self.min_scale = min_scale
        self.max_scale = max_scale

        # nn.functional.grid_sample grid value range in [-1,1]
        self.w, self.h = torch.meshgrid([torch.linspace(-1,1,self.N_samples_sqrt),
                                         torch.linspace(-1,1,self.N_samples_sqrt)])
        self.h = self.h.unsqueeze(2)
        self.w = self.w.unsqueeze(2)

        # directly return grid for grid_sample
        self.return_indices = False

        self.iterations = 0
        self.scale_anneal = scale_anneal

    def sample_rays(self, H, W):

        if self.scale_anneal > 0:
            k_iter = self.iterations // 1000 * 3
            min_scale = max(self.min_scale, self.max_scale * exp(-k_iter*self.scale_anneal))
            min_scale = min(0.9, min_scale)
        else:
            min_scale = self.min_scale

        scale = 1
        if self.random_scale:
            scale = torch.Tensor(1).uniform_(min_scale, self.max_scale)
            h = self.h * scale 
            w = self.w * scale 

        if self.random_shift:
            max_offset = 1 - scale.item()

            # ========================================================
            # [Bottom-biased] 垂直 offset 偏向下半部(損傷集中區),
            # 水平維持均勻
            #
            # 設計:用 Beta(2, 5) 分布抽樣,mean ~ 0.286 偏向小值
            # 映射方式:
            #   beta_sample 小 (常見) → h_offset 大 → patch 往下
            #   beta_sample 大 (罕見) → h_offset 小/負 → patch 往上(偶爾看上半)
            # 
            # y=-1 是頂,y=+1 是底 (grid_sample 座標慣例)
            # 所以 h_offset > 0 代表 patch 中心往下移
            # ========================================================
            # 水平:均勻左右
            w_offset = torch.Tensor(1).uniform_(0, max_offset) * \
                    (torch.randint(2, (1,)).float() - 0.5) * 2

            # 垂直:偏下的 beta 分布
            beta_dist = torch.distributions.Beta(2.0, 5.0)
            beta_sample = beta_dist.sample((1,)).to(h.device)   # [0, 1],偏向小值
            # 映射到 [-0.3*max_offset, +max_offset]
            # beta=0 → h_offset = max_offset (最下)
            # beta=1 → h_offset = -0.3*max_offset (略上)
            h_offset = max_offset * (1.0 - beta_sample * 1.3)

            h += h_offset
            w += w_offset

        self.scale = scale

        if self.iterations < 100 and self.iterations % 10 == 0:
            h_center = h.mean().item()
            # print(f"[sampler] it={self.iterations}, h_center={h_center:.3f} "
            #     f"(>0 = 偏下, <0 = 偏上)")

        return torch.cat([h, w], dim=2) #torch.Size([32, 32, 2])


# def sample_rays(self, H, W):

#         if self.scale_anneal > 0:
#             k_iter = self.iterations // 1000 * 3
#             min_scale = max(self.min_scale, self.max_scale * exp(-k_iter*self.scale_anneal))
#             min_scale = min(0.9, min_scale)
#         else:
#             min_scale = self.min_scale

#         scale = 1
#         if self.random_scale:
#             scale = torch.Tensor(1).uniform_(min_scale, self.max_scale)
#             h = self.h * scale 
#             w = self.w * scale 

#         if self.random_shift:
#             max_offset = 1 - scale.item()

#             # ========================================================
#             # [Bottom-biased] 垂直 offset 偏向下半部(損傷集中區),
#             # 水平維持均勻
#             #
#             # 設計:用 Beta(2, 5) 分布抽樣,mean ~ 0.286 偏向小值
#             # 映射方式:
#             #   beta_sample 小 (常見) → h_offset 大 → patch 往下
#             #   beta_sample 大 (罕見) → h_offset 小/負 → patch 往上(偶爾看上半)
#             # 
#             # y=-1 是頂,y=+1 是底 (grid_sample 座標慣例)
#             # 所以 h_offset > 0 代表 patch 中心往下移
#             # ========================================================
#             # 水平:均勻左右
#             w_offset = torch.Tensor(1).uniform_(0, max_offset) * \
#                     (torch.randint(2, (1,)).float() - 0.5) * 2

#             # 垂直:偏下的 beta 分布
#             beta_dist = torch.distributions.Beta(2.0, 5.0)
#             beta_sample = beta_dist.sample((1,)).to(h.device)   # [0, 1],偏向小值
#             # 映射到 [-0.3*max_offset, +max_offset]
#             # beta=0 → h_offset = max_offset (最下)
#             # beta=1 → h_offset = -0.3*max_offset (略上)
#             h_offset = max_offset * (1.0 - beta_sample * 1.3)

#             h += h_offset
#             w += w_offset

#         self.scale = scale

#         if self.iterations < 100 and self.iterations % 10 == 0:
#             h_center = h.mean().item()
#             print(f"[sampler] it={self.iterations}, h_center={h_center:.3f} "
#                 f"(>0 = 偏下, <0 = 偏上)")

#         return torch.cat([h, w], dim=2) #torch.Size([32, 32, 2])

# def sample_rays(self, H, W):

#         if self.scale_anneal>0:
#             k_iter = self.iterations // 1000 * 3
#             min_scale = max(self.min_scale, self.max_scale * exp(-k_iter*self.scale_anneal))
#             min_scale = min(0.9, min_scale)
#         else:
#             min_scale = self.min_scale

#         scale = 1
#         if self.random_scale:
#             scale = torch.Tensor(1).uniform_(min_scale, self.max_scale)
#             h = self.h * scale 
#             w = self.w * scale 

#         if self.random_shift:
#             max_offset = 1-scale.item()
#             h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2
#             w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2

#             h += h_offset
#             w += w_offset

#         self.scale = scale

#         return torch.cat([h, w], dim=2) #torch.Size([32, 32, 2])