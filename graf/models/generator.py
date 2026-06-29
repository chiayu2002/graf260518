import numpy as np
import torch
from ..utils import sample_on_sphere, look_at, to_sphere
from ..transforms import FullRaySampler
from submodules.nerf_pytorch.run_nerf_mod import render
from functools import partial
import torch.nn.functional as F
from graf.models.ccsr import CCSR
import os


class Generator(object):
    def __init__(self, H, W, focal, radius, ray_sampler, render_kwargs_train, render_kwargs_test,
                 parameters, named_parameters,
                 range_u=(0,1), range_v=(0.01,0.49), v=0, chunk=None, device='cuda',
                 orthographic=False, use_default_rays=False, use_ccsr=True, num_views=8):
        self.device = device
        self.H = int(H); self.W = int(W)
        self.focal = focal; self.radius = radius
        self.range_u = range_u; self.range_v = range_v
        self.chunk = chunk; self.v = v
        self.use_default_rays = use_default_rays
        self.use_ccsr = use_ccsr

        coords = torch.from_numpy(np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1))
        self.coords = coords.view(-1, 2)

        self.ray_sampler = ray_sampler
        self.val_ray_sampler = FullRaySampler(orthographic=orthographic)
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.initial_raw_noise_std = self.render_kwargs_train['raw_noise_std']
        self._parameters = parameters
        self._named_parameters = named_parameters
        self.module_dict = {'generator': self.render_kwargs_train['network_fn']}
        for name, module in [('generator_fine', self.render_kwargs_train['network_fine'])]:
            if module is not None:
                self.module_dict[name] = module

        self._v_list = [float(x.strip()) for x in self.v.split(",")]

        if self.use_ccsr:
            self.ccsr = CCSR(num_views=num_views, scale_factor=1).to(device)
            self.module_dict['ccsr'] = self.ccsr

        for name, module in self.module_dict.items():
            if name in ['generator', 'generator_fine']:
                continue
            self._parameters += list(module.parameters())
            self._named_parameters += list(module.named_parameters())

        self.parameters = lambda: self._parameters
        self.named_parameters = lambda: self._named_parameters
        self.use_test_kwargs = False
        self.render = partial(render, H=self.H, W=self.W, focal=self.focal, chunk=self.chunk)

    def __call__(self, z, label, hidden_state, rays=None, return_ccsr_output=False):
        """
        label: [B, 16] — label[:,0:7]=vec, label[:,7:14]=mat_feat, label[:,14]=h_idx, label[:,15]=a_idx
        hidden_state: [B, 1024]
        mat_feat is extracted from label[:,7:14].
        """
        bs = z.shape[0]

        # Extract mat_feat from label
        mat_feat = label[:, 7:14].float()  # [B, 7]

        if rays is None:
            if self.use_default_rays:
                rays = torch.cat([self.sample_rays() for _ in range(bs)], dim=1)
            else:
                all_rays = []
                for i in range(label.size(0)):
                    height_idx = int(label[i, 14].item())
                    angle_idx = int(label[i, 15].item())
                    selected_u = angle_idx / 360
                    selected_v = self._v_list[height_idx]
                    rays_i = self.sample_select_rays(selected_u, selected_v)
                    all_rays.append(rays_i)
                rays = torch.cat(all_rays, dim=1)

        render_kwargs = self.render_kwargs_test if self.use_test_kwargs else self.render_kwargs_train
        render_kwargs = dict(render_kwargs)
        render_kwargs['features'] = z

        # Render with hidden_state + mat_feat
        rgb, disp, acc, extras = render(
            self.H, self.W, self.focal,
            hidden_state, mat_feat,
            chunk=self.chunk, rays=rays,
            **render_kwargs
        )

        rays_to_output = lambda x: x.view(len(x), -1) * 2 - 1
        rgb_nerf = rays_to_output(rgb)
        disp_out = rays_to_output(disp)
        acc_out = rays_to_output(acc)

        ccsr_output = None
        if self.use_ccsr and return_ccsr_output:
            patch_size = int(np.sqrt(rgb.shape[0] // bs))
            nerf_patch = rgb.view(bs, patch_size, patch_size, 3).permute(0, 3, 1, 2).contiguous()
            angle_indices = label[:, 15].int()
            view_indices = (angle_indices * self.ccsr.num_views) // 360
            unique_views = torch.unique(view_indices)
            ccsr_results = torch.zeros_like(nerf_patch)
            for vidx in unique_views:
                mask = (view_indices == vidx)
                batch_input = nerf_patch[mask]
                batch_output = self.ccsr(batch_input, vidx.item())
                ccsr_results[mask] = batch_output
            ccsr_output = ccsr_results

        if self.use_test_kwargs:
            if return_ccsr_output:
                return rgb_nerf, disp_out, acc_out, extras, ccsr_output
            return rgb_nerf, disp_out, acc_out, extras
        else:
            if return_ccsr_output:
                return rgb_nerf, rays, ccsr_output
            return rgb_nerf, rays

    def decrease_nerf_noise(self, it):
        end_it = 5000
        if it < end_it:
            noise_std = self.initial_raw_noise_std - self.initial_raw_noise_std / end_it * it
            self.render_kwargs_train['raw_noise_std'] = noise_std

    def sample_pose(self):
        loc = sample_on_sphere(self.range_u, self.range_v)
        radius = self.radius
        if isinstance(radius, tuple):
            radius = np.random.uniform(*radius)
        loc = loc * radius
        R = look_at(loc)[0]
        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        return torch.Tensor(RT.astype(np.float32))

    def sample_select_pose(self, u, v):
        radius = self.radius
        loc = to_sphere(u, v) * radius
        R = look_at(loc)[0]
        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        return torch.Tensor(RT.astype(np.float32))

    def sample_rays(self):
        pose = self.sample_pose()
        sampler = self.val_ray_sampler if self.use_test_kwargs else self.ray_sampler
        batch_rays, _, _ = sampler(self.H, self.W, self.focal, pose)
        return batch_rays

    def sample_select_rays(self, u, v):
        pose = self.sample_select_pose(u, v)
        sampler = self.val_ray_sampler if self.use_test_kwargs else self.ray_sampler
        batch_rays, _, _ = sampler(self.H, self.W, self.focal, pose)
        return batch_rays

    def to(self, device):
        self.render_kwargs_train['network_fn'].to(device)
        self.device = device
        return self

    def train(self):
        self.use_test_kwargs = False
        self.render_kwargs_train['network_fn'].train()

    def eval(self):
        self.use_test_kwargs = True
        self.render_kwargs_train['network_fn'].eval()