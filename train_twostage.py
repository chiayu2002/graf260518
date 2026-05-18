"""
Two-Stage Training: NeRF 64×64 Full Image + SR 256×256

Phase 1 (iter 0 ~ phase2_start):
  - NeRF renders 64×64 full image (FullRaySampler, H=W=64)
  - D_low judges 64×64 (your existing Discriminator)
  - Real = resize(256→64)
  - Reverse-TTUR: lr_g=0.0003 > lr_d=0.0001

Phase 2 (iter phase2_start ~):
  - Phase 1 continues (D_low still active)
  - SR network: 64→256 via PixelShuffle
  - D_high judges 256×256
  - End-to-end gradient: D_high → SR → NeRF
  - Multi-scale discriminator (D_low + D_high)

[INTERP] Conditioning Interpolation Regularization:
  - Bottleneck: condition_feature now uses 1024→16→256
  - During G step, randomly interpolate bottleneck embeddings
    between specimen pairs with Beta(0.2, 0.2) mixing
  - Only G adversarial loss on interpolated samples (Strategy 2)
  - D never sees interpolated samples

Performance Optimizations:
  - Pre-computed full-image rays cached per (angle, height) pair
  - x_real_64 computed once per iteration
  - Discriminator forward consolidated (single forward for real/fake aux)
  - torch.no_grad() scopes minimized
  - Reduced redundant .to(device) calls
"""
import argparse, os, importlib, json, glob, time, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from contextlib import contextmanager

torch.set_default_tensor_type('torch.cuda.FloatTensor')
import wandb
sys.path.append('submodules')
import torchvision.utils as vutils

from graf.gan_training import Evaluator
from graf.config import get_data, build_models, load_config, save_config, build_lr_scheduler
from graf.utils import get_zdist
from graf.train_step import compute_grad2, compute_loss, toggle_grad, CCSRLoss
from GAN_stability.gan_training.checkpoints_mod import CheckpointIO

# Import extracted networks
from graf.models.twostage_networks import SRNetwork, DiscriminatorHigh, nerf_flat_to_img


# ================================================================
# Helpers
# ================================================================
def setup_directories(config):
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return out_dir, checkpoint_dir


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def initialize_training(config, device):
    extractor_path = config['data']['extractor_path']
    extractor_args = json.load(open(glob.glob(
        "/Data/home/vicky/graf260108_im64/HystereticGRU/2026-03-24_17-13-01/args.json",
        recursive=True)[0], "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module(
        "graf.models.HystereticPrediction"
    ).__dict__[extractor_args.architecture](**vars(extractor_args))
    extractor = extractor.to(device)
    state_dict = torch.load(glob.glob(extractor_path, recursive=True)[0])["state_dict"]
    status = extractor.load_state_dict(state_dict)
    print("Extractor Loading Status:", status)

    train_dataset, hwfr = get_data(config, extractor, extractor_args)
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far'] - config['data']['near'],) * 2
        hwfr[2] = hw_ortho
    config['data']['hwfr'] = hwfr

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['training']['batch_size'],
        num_workers=config['training']['nworkers'], shuffle=True,
        pin_memory=True, sampler=None, drop_last=True,
        generator=torch.Generator(device='cuda:0'))

    generator, discriminator = build_models(config)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    return train_loader, train_dataset, generator, discriminator


# ================================================================
# Ray Cache — avoid recomputing full-image rays every iteration
# ================================================================
class RayCache:
    def __init__(self, generator, v_list, focal_64):
        self._cache = {}
        self._generator = generator
        self._v_list = v_list
        self._focal_64 = focal_64
        self._n_heights = len(v_list)

    def get_rays(self, angle_idx, height_idx):
        key = (angle_idx, height_idx)
        if key not in self._cache:
            u = angle_idx / 360.0
            v = self._v_list[height_idx % self._n_heights]
            pose = self._generator.sample_select_pose(u, v)
            rays, _, _ = self._generator.val_ray_sampler(64, 64, self._focal_64, pose)
            self._cache[key] = rays
        return self._cache[key]

    def batch_rays(self, label_batch, bs):
        """Build batched rays from label batch. Returns [2, bs*4096, 3]."""
        rays_list = []
        for i in range(bs):
            h_idx = int(label_batch[i, 14].item())
            a_idx = int(label_batch[i, 15].item())
            rays_list.append(self.get_rays(a_idx, h_idx))
        return torch.cat(rays_list, dim=1)


# ================================================================
# [INTERP] Interpolation utilities
# ================================================================
def build_interp_cache(cached_hs, specimen_specs, cond_bottleneck, device):
    """
    Pre-compute bottleneck embeddings and material features for each
    training specimen. Returns a list of dicts for interpolation sampling.
    """
    interp_data = []
    with torch.no_grad():
        for sn, (vec, mf) in specimen_specs.items():
            if sn not in cached_hs:
                continue
            hs = cached_hs[sn]
            b = cond_bottleneck.encode(hs.unsqueeze(0))
            m = torch.tensor(mf, dtype=torch.float32, device=device)
            v = torch.tensor(vec, dtype=torch.float32, device=device)
            interp_data.append({
                'name': sn,
                'bottleneck': b.squeeze(0),   # [16]
                'mat_feat': m,                # [7]
                'label_vec': v,               # [7]
                'hidden_state': hs,           # [1024]
            })
    return interp_data


def sample_interpolated_conditioning(interp_data, n_interp, device):
    """
    Generate n_interp interpolated conditioning samples.
    Uses Beta(0.2, 0.2) for mixing — U-shaped, mostly near 0 or 1,
    occasionally in the middle.

    Returns tuple: (bottleneck, mat, label, hidden_state) tensors,
    or (None, None, None, None) if < 2 specimens.
    """
    n_specimens = len(interp_data)
    if n_specimens < 2:
        return None, None, None, None

    beta_dist = torch.distributions.Beta(0.2, 0.2)
    bottlenecks, mats, labels, hss = [], [], [], []

    for _ in range(n_interp):
        idx_a, idx_b = random.sample(range(n_specimens), 2)
        a, b = interp_data[idx_a], interp_data[idx_b]
        lam = beta_dist.sample().item()

        bottlenecks.append(lam * a['bottleneck'] + (1 - lam) * b['bottleneck'])
        mats.append(lam * a['mat_feat'] + (1 - lam) * b['mat_feat'])
        hss.append(lam * a['hidden_state'] + (1 - lam) * b['hidden_state'])

        v_interp = lam * a['label_vec'] + (1 - lam) * b['label_vec']
        m_interp = mats[-1]
        h_idx = random.randint(0, 3)
        a_idx = random.randint(0, 359)
        label_interp = torch.cat([v_interp, m_interp,
                                  torch.tensor([h_idx, a_idx], device=device,
                                               dtype=torch.float32)])
        labels.append(label_interp)

    return (torch.stack(bottlenecks), torch.stack(mats),
            torch.stack(labels), torch.stack(hss))


# ================================================================
# Main
# ================================================================
def main():
    set_random_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/twostage.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_wandb_id', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])
    restart_every = config['training']['restart_every']
    batch_size    = config['training']['batch_size']
    fid_every     = config['training']['fid_every']
    save_best     = config['training']['save_best']
    reg_param     = config['training']['reg_param']
    aux_loss_weight = config['training']['label_param']
    mat_loss_weight = config['training'].get('mat_param', 0.1)
    print_every   = config['training']['print_every']
    sample_every  = config['training']['sample_every']
    save_every_s  = config['training']['save_every']
    device = torch.device("cuda:0")

    # AMP
    use_amp = config['training'].get('use_amp', False)
    amp_dtype_str = config['training'].get('amp_dtype', 'bfloat16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None

    # Two-stage config
    phase2_start    = config['training'].get('phase2_start_iter', 100000)
    lambda_d_high   = config['training'].get('lambda_d_high', 1.0)
    lambda_recon_sr = config['training'].get('lambda_recon_sr', 1.0)
    recon_config    = config.get('reconstruction', {})
    use_recon       = recon_config.get('enabled', False)
    lambda_recon    = recon_config.get('lambda_recon', 0.1)

    # [INTERP] Interpolation config — can be set in YAML or use defaults
    interp_config  = config.get('interpolation', {})
    use_interp     = interp_config.get('enabled', True)
    n_interp       = interp_config.get('n_samples', 2)
    lambda_interp  = interp_config.get('lambda', 0.5)
    interp_start   = interp_config.get('start_iter', 5000)

    print(f"[TwoStage] Phase1: 0~{phase2_start}  Phase2: {phase2_start}+")
    print(f"[Recon] enabled={use_recon}, λ={lambda_recon}")
    print(f"[Weights] aux={aux_loss_weight}, mat={mat_loss_weight}, reg={reg_param}")
    print(f"[INTERP] enabled={use_interp}, n={n_interp}, λ={lambda_interp}, start={interp_start}")

    out_dir, checkpoint_dir = setup_directories(config)
    save_config(os.path.join(out_dir, 'config.yaml'), config)

    # WandB
    wandb_kw = dict(project=config['wandb']['project'],
                    name=config['wandb']['name'], config=config)
    if args.resume_wandb_id:
        wandb_kw.update(id=args.resume_wandb_id, resume='must')
    wandb.init(**wandb_kw)

    # Models
    train_loader, train_dataset, generator, discriminator = initialize_training(config, device)

    nerf_model = generator.render_kwargs_train['network_fn']
    shared_cond_proj = nerf_model.condition_feature

    sr_network = SRNetwork(ch=64, n_rb=6).to(device)
    d_high = DiscriminatorHigh(
        nc=3, ndf=config['discriminator']['ndf'],
        hidden_dim=config['discriminator'].get('hidden_dim', 1024),
        cond_dim=config['discriminator'].get('cond_dim', 256),
        num_classes=config['nerf']['num_classes'],
        shared_cond_proj=shared_cond_proj,
    ).to(device)

    print(f"[SR] {sum(p.numel() for p in sr_network.parameters()):,} params")
    print(f"[D_high] {sum(p.numel() for p in d_high.parameters()):,} params")

    # [INTERP] Verify bottleneck is present
    if hasattr(shared_cond_proj, 'bottleneck_dim'):
        print(f"[Bottleneck] {shared_cond_proj.hidden_dim} "
              f"→ {shared_cond_proj.bottleneck_dim} → {shared_cond_proj.out_dim}")
    else:
        print(f"[Bottleneck] Not using ConditionBottleneck (plain Linear)")
        if use_interp:
            print(f"[WARNING] Interpolation requires ConditionBottleneck. Disabling.")
            use_interp = False

    # Optimizers
    lr_g = config['training']['lr_g']
    lr_d = config['training']['lr_d']
    g_optimizer      = optim.RMSprop(generator.parameters(), lr=lr_g, alpha=0.99, eps=1e-8)
    d_optimizer      = optim.RMSprop(discriminator.parameters(), lr=lr_d, alpha=0.99, eps=1e-8)
    sr_optimizer     = optim.Adam(sr_network.parameters(), lr=lr_g, betas=(0.0, 0.99))
    d_high_optimizer = optim.RMSprop(d_high.parameters(), lr=lr_d, alpha=0.99, eps=1e-8)

    recon_loss_fn = CCSRLoss(device=device)

    # Checkpoint
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
    checkpoint_io.register_modules(
        discriminator=discriminator,
        g_optimizer=g_optimizer, d_optimizer=d_optimizer,
        sr_network=sr_network, d_high=d_high,
        sr_optimizer=sr_optimizer, d_high_optimizer=d_high_optimizer,
        **generator.module_dict)

    # Distributions
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    # Evaluator
    evaluator = Evaluator(fid_every > 0, generator, zdist, None,
                          batch_size=batch_size, device=device, inception_nsamples=33)
    if fid_every > 0:
        evaluator.inception_eval.initialize_target(
            train_loader,
            cache_file=os.path.join(out_dir, 'fid_cache_train.npz'),
            act_cache_file=os.path.join(out_dir, 'kid_cache_train.npz'))

    cached_hs = {n: h.to(device) for n, h in train_dataset.hidden_state.items()}
    print(f"Cached hidden states: {list(cached_hs.keys())}")

    # Ray cache
    v_list = [float(x.strip()) for x in config['data']['v'].split(",")]
    n_heights = len(v_list)
    focal_64 = generator.focal * (64.0 / generator.H)
    ray_cache = RayCache(generator, v_list, focal_64)

    # State
    fid_best = kid_best = float('inf')
    it = epoch_idx = -1
    tstart = t0 = time.time()
    phase2_flag = False

    # Resume
    if args.resume:
        sc = checkpoint_io.load(args.resume)
        it = sc.get('it', -1)
        epoch_idx = sc.get('epoch_idx', -1)
        fid_best = sc.get('fid_best', float('inf'))
        kid_best = sc.get('kid_best', float('inf'))
        print(f"[Resume] it={it}")

    g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
    d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

    @contextmanager
    def amp_ctx():
        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                yield
        else:
            yield

    # Pre-compute specimen info for sampling
    specimen_specs = {
        'RS307': ([1, 0, 1, 0, 0, 0, 1], [0, 0.1906, 0.8342, 0.1, 1, 0.0589, 0.1081]),
        'RS330': ([1, 0, 0, 0, 1, 1, 0], [0.0088, 1, 1, 1, 0, 1, 1]),
        'RS615': ([0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0.5826, 0, 0]),
        'RS315': ([1, 0, 0, 1, 0, 1, 0], [0.0062, 0.4365, 0.602, 0.6261, 0.0064, 0.6284, 0.653]),
    }
    sample_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    sample_poses = torch.stack([generator.sample_select_pose(i / 8, 0.5) for i in range(8)])

    # [INTERP] Build initial interpolation cache
    if use_interp:
        interp_data = build_interp_cache(cached_hs, specimen_specs, shared_cond_proj, device)
        print(f"[INTERP] Cached {len(interp_data)} specimens: "
              f"{[d['name'] for d in interp_data]}")

    last_reg_lo = 0.0
    last_reg_hi = 0.0

    # ================================================================
    # Training Loop
    # ================================================================
    while True:
        epoch_idx += 1
        for x_real, label, hidden_state in tqdm(train_loader, desc=f"Epoch {epoch_idx}"):
            it += 1
            in_p2 = (it >= phase2_start)
            if in_p2 and not phase2_flag:
                print(f"\n{'=' * 60}\n[Phase 2] iter {it}: SR + D_high active\n{'=' * 60}")
                phase2_flag = True

            # [INTERP] Refresh interp cache periodically
            if use_interp and it >= interp_start and it % 1000 == 0:
                interp_data = build_interp_cache(
                    cached_hs, specimen_specs, shared_cond_proj, device)

            # --- Data prep ---
            x_real = x_real.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            hidden_state = hidden_state.to(device, non_blocking=True)
            mat = label[:, 7:14]
            x_real_64 = F.interpolate(x_real, size=(64, 64), mode='bilinear', align_corners=True)

            generator.ray_sampler.iterations = it
            do_r1 = (it % 16 == 0)

            # ============================================================
            # D_low step — UNCHANGED (D never sees interp samples)
            # ============================================================
            toggle_grad(generator, False)
            toggle_grad(discriminator, True)
            generator.train()
            discriminator.train()
            d_optimizer.zero_grad(set_to_none=True)

            x_real_d = x_real_64.detach().requires_grad_(True)
            z = zdist.sample((batch_size,))

            with amp_ctx():
                d_real_lo, aux_real, lab_real = discriminator(
                    x_real_d, label, hidden_state, return_aux=True)
                dloss_real_lo = compute_loss(d_real_lo, 1)
                aux_loss_real = F.mse_loss(aux_real, hidden_state)
                mat_loss_real = F.mse_loss(lab_real, mat)

            if do_r1:
                dloss_real_lo.backward(retain_graph=True)
                reg_lo = reg_param * 16 * compute_grad2(d_real_lo.float(), x_real_d).mean()
                reg_lo.backward(retain_graph=True)
                last_reg_lo = reg_lo.item()
            else:
                dloss_real_lo.backward(retain_graph=True)
                reg_lo = torch.tensor(0., device=device)

            aux_mat_loss = aux_loss_weight * aux_loss_real + mat_loss_weight * mat_loss_real
            aux_mat_loss.backward()

            with torch.no_grad(), amp_ctx():
                rays_f = ray_cache.batch_rays(label, batch_size)
                nerf_flat, _ = generator(z, label, hidden_state, rays=rays_f)

            nerf_img_64 = nerf_flat_to_img(nerf_flat, batch_size)

            with amp_ctx():
                d_fake_lo = discriminator(nerf_img_64, label, hidden_state)
                dloss_fake_lo = compute_loss(d_fake_lo, 0)

            dloss_fake_lo.backward()

            if use_scaler:
                scaler.step(d_optimizer)
                scaler.update()
            else:
                d_optimizer.step()
            d_scheduler.step()

            # ============================================================
            # D_high step — UNCHANGED
            # ============================================================
            dloss_real_hi = dloss_fake_hi = reg_hi = torch.tensor(0., device=device)
            if in_p2:
                toggle_grad(d_high, True)
                toggle_grad(sr_network, False)
                d_high.train()
                d_high_optimizer.zero_grad(set_to_none=True)

                xr_dh = x_real.detach().requires_grad_(True)
                with amp_ctx():
                    dr_hi = d_high(xr_dh, label, hidden_state)
                    dloss_real_hi = compute_loss(dr_hi, 1)

                if do_r1:
                    reg_hi = reg_param * 16 * compute_grad2(dr_hi.float(), xr_dh).mean()
                    last_reg_hi = reg_hi.item()
                else:
                    reg_hi = torch.tensor(0., device=device)

                with torch.no_grad():
                    sr256 = sr_network(nerf_img_64)
                with amp_ctx():
                    df_hi = d_high(sr256, label, hidden_state)
                    dloss_fake_hi = compute_loss(df_hi, 0)

                dhi_loss = dloss_real_hi + dloss_fake_hi + reg_hi
                if use_scaler:
                    scaler.scale(dhi_loss).backward()
                    scaler.step(d_high_optimizer)
                    scaler.update()
                else:
                    dhi_loss.backward()
                    d_high_optimizer.step()

            # ============================================================
            # Generator + SR step (with interpolation regularization)
            # ============================================================
            if config['nerf']['decrease_noise']:
                generator.decrease_nerf_noise(it)

            toggle_grad(generator, True)
            toggle_grad(discriminator, False)
            generator.train()
            discriminator.train()
            g_optimizer.zero_grad(set_to_none=True)

            if in_p2:
                toggle_grad(sr_network, True)
                toggle_grad(d_high, False)
                sr_network.train()
                sr_optimizer.zero_grad(set_to_none=True)

            z = zdist.sample((batch_size,))
            with amp_ctx():
                rays_g = ray_cache.batch_rays(label, batch_size)
                nerf_g, _ = generator(z, label, hidden_state, rays=rays_g)
                nerf_g_img = nerf_flat_to_img(nerf_g, batch_size)

                dfl_g, aux_f, lab_f = discriminator(
                    nerf_g_img, label, hidden_state, return_aux=True)
                g_adv_lo = compute_loss(dfl_g, 1)
                g_aux = F.mse_loss(aux_f, hidden_state)
                mat_loss_f = F.mse_loss(lab_f, mat)

                recon64 = torch.tensor(0., device=device)
                if use_recon:
                    recon64 = recon_loss_fn(nerf_g_img, x_real_64)

                g_adv_hi = sr_rec = torch.tensor(0., device=device)
                if in_p2:
                    sr_g = sr_network(nerf_g_img)
                    dfh_g = d_high(sr_g, label, hidden_state)
                    g_adv_hi = compute_loss(dfh_g, 1)
                    sr_rec = recon_loss_fn(sr_g, x_real)

                gloss = (g_adv_lo
                         + lambda_recon * recon64
                         + aux_loss_weight * g_aux
                         + mat_loss_weight * mat_loss_f)
                if in_p2:
                    gloss = gloss + lambda_d_high * g_adv_hi + lambda_recon_sr * sr_rec

                # ====================================================
                # [INTERP] Interpolation regularization (Strategy 2)
                # D is frozen → only G adversarial loss on interp samples
                # ====================================================
                g_interp_loss = torch.tensor(0., device=device)
                if (use_interp and it >= interp_start
                        and hasattr(shared_cond_proj, 'encode')):
                    interp_result = sample_interpolated_conditioning(
                        interp_data, n_interp, device)
                    if interp_result[0] is not None:
                        i_bneck, i_mat, i_label, i_hs = interp_result

                        # Generate with interpolated conditioning
                        z_interp = zdist.sample((n_interp,))
                        rays_interp = ray_cache.batch_rays(i_label, n_interp)
                        nerf_interp, _ = generator(
                            z_interp, i_label, i_hs, rays=rays_interp)
                        nerf_interp_img = nerf_flat_to_img(nerf_interp, n_interp)

                        # G adversarial loss only — D doesn't train on this
                        d_interp_lo = discriminator(
                            nerf_interp_img, i_label, i_hs)
                        g_interp_loss = compute_loss(d_interp_lo, 1)

                        if in_p2:
                            sr_interp = sr_network(nerf_interp_img)
                            d_interp_hi = d_high(sr_interp, i_label, i_hs)
                            g_interp_loss = (g_interp_loss
                                             + compute_loss(d_interp_hi, 1))

                        gloss = gloss + lambda_interp * g_interp_loss

            if use_scaler:
                scaler.scale(gloss).backward()
                scaler.step(g_optimizer)
                if in_p2:
                    scaler.step(sr_optimizer)
                scaler.update()
            else:
                gloss.backward()
                g_optimizer.step()
                if in_p2:
                    sr_optimizer.step()
            g_scheduler.step()

            # ============================================================
            # Logging
            # ============================================================
            log_dict = None
            if (it + 1) % print_every == 0:
                lr_g_now = g_optimizer.param_groups[0]['lr']
                lr_d_now = d_optimizer.param_groups[0]['lr']
                log_dict = {
                    "loss/g_total": gloss.item(),
                    "loss/g_adv_low": g_adv_lo.item(),
                    "loss/g_label": g_aux.item(),
                    "loss/g_mat": mat_loss_f.item(),
                    "loss/d_low": (dloss_real_lo.item() + dloss_fake_lo.item()
                                   + last_reg_lo
                                   + aux_loss_weight * aux_loss_real.item()
                                   + mat_loss_weight * mat_loss_real.item()),
                    "loss/d_reallabel": aux_loss_real.item(),
                    "loss/d_mat": mat_loss_real.item(),
                    "loss/dloss_real_lo": dloss_real_lo.item(),
                    "loss/dloss_fake_lo": dloss_fake_lo.item(),
                    "loss/reg_lo": last_reg_lo,
                    "loss/recon64": recon64.item(),
                    "lr/g": lr_g_now,
                    "lr/d": lr_d_now,
                    "training/phase": 2 if in_p2 else 1,
                    "loss/g_interp": g_interp_loss.item(),  # [INTERP]
                }
                if in_p2:
                    log_dict.update({
                        "loss/g_adv_high": g_adv_hi.item(),
                        "loss/dloss_real_hi": dloss_real_hi.item(),
                        "loss/dloss_fake_hi": dloss_fake_hi.item(),
                        "loss/reg_hi": last_reg_hi,
                        "loss/sr_recon256": sr_rec.item(),
                    })

            # ============================================================
            # Samples
            # ============================================================
            if (it % sample_every == 0) or (it < 5000 and it % 200 == 0):
                ztest = zdist.sample((8,))
                rgb_pan, dep_pan, acc_pan, sr_pan, shown = [], [], [], [], []
                for sn, (vec, mf) in specimen_specs.items():
                    if sn not in cached_hs:
                        continue
                    ll = [vec + mf + [float(a)] for a in sample_angles]
                    lt = torch.tensor(ll, dtype=torch.float32, device=device)
                    ht = cached_hs[sn].unsqueeze(0).expand(8, -1)
                    rgb, dep, acc = evaluator.create_samples(
                        ztest.to(device), lt, ht, sample_poses)

                    if dep.dim() == 3: dep = dep.unsqueeze(1)
                    if dep.shape[1] == 1: dep = dep.expand(-1, 3, -1, -1)
                    if acc.dim() == 3: acc = acc.unsqueeze(1)
                    if acc.shape[1] == 1: acc = acc.expand(-1, 3, -1, -1)

                    rgb_pan.append(rgb.cpu())
                    dep_pan.append(dep.cpu())
                    acc_pan.append(acc.cpu())
                    shown.append(sn)

                    if in_p2:
                        with torch.no_grad():
                            r4sr = rgb.to(device)
                            if r4sr.shape[2] != 64:
                                r4sr = F.interpolate(r4sr, size=(64, 64), mode='bilinear')
                            sr_pan.append(sr_network(r4sr).cpu())

                if rgb_pan:
                    cap = f"it {it} | {'/'.join(shown)}"
                    if log_dict is None:
                        log_dict = {}
                    log_dict["sample/rgb64"] = wandb.Image(
                        vutils.make_grid(torch.cat(rgb_pan), nrow=8, normalize=True), caption=cap)
                    log_dict["sample/depth"] = wandb.Image(
                        vutils.make_grid(torch.cat(dep_pan), nrow=8, normalize=True), caption=cap)
                    log_dict["sample/acc"] = wandb.Image(
                        vutils.make_grid(torch.cat(acc_pan), nrow=8, normalize=True), caption=cap)
                    if sr_pan:
                        log_dict["sample/sr256"] = wandb.Image(
                            vutils.make_grid(torch.cat(sr_pan), nrow=8, normalize=True),
                            caption=f"SR|{cap}")

            # ============================================================
            # FID / KID
            # ============================================================
            if fid_every > 0 and (it + 1) % fid_every == 0:
                nd = len(train_dataset)

                def mgen():
                    while True:
                        idx = np.random.choice(nd, batch_size, replace=False)
                        ll, hl = [], []
                        for i in idx:
                            _, li, hi = train_dataset[i]
                            ll.append(li)
                            hl.append(hi)
                        lb = torch.stack(ll).to(device)
                        hb = torch.stack(hl).to(device)
                        ps = torch.stack([
                            generator.sample_select_pose(
                                int(lb[i, 15].item()) / 360.,
                                v_list[int(lb[i, 14].item()) % n_heights])
                            for i in range(batch_size)])
                        zf = zdist.sample((batch_size,))
                        with torch.no_grad():
                            r, _, _ = evaluator.create_samples(zf, lb, hb, ps)
                        r = (r / 2 + .5).mul_(255).clamp_(0, 255).to(
                            torch.uint8).float() / 255. * 2 - 1
                        yield r.cpu()

                fid, kid = evaluator.compute_fid_kid(None, None, sample_generator=mgen())
                if log_dict is None:
                    log_dict = {}
                log_dict["val/fid"] = fid
                log_dict["val/kid"] = kid
                torch.cuda.empty_cache()

                if save_best == 'fid' and fid < fid_best:
                    fid_best = fid
                    wandb.run.summary["best_fid"] = fid_best
                    checkpoint_io.save('model_best.pt', it=it, epoch_idx=epoch_idx,
                                       fid_best=fid_best, kid_best=kid_best, save_to_wandb=True)
                elif save_best == 'kid' and kid < kid_best:
                    kid_best = kid
                    wandb.run.summary["best_kid"] = kid_best
                    checkpoint_io.save('model_best.pt', it=it, epoch_idx=epoch_idx,
                                       fid_best=fid_best, kid_best=kid_best, save_to_wandb=True)

            if log_dict is not None:
                wandb.log(log_dict, step=it)

            # ============================================================
            # Checkpoint saving
            # ============================================================
            if (it + 1) % 10000 == 0:
                checkpoint_io.save('model_%08d.pt' % it, it=it, epoch_idx=epoch_idx,
                                   fid_best=fid_best, kid_best=kid_best, save_to_wandb=True)
            if time.time() - t0 > save_every_s:
                checkpoint_io.save(config['training']['model_file'], it=it,
                                   epoch_idx=epoch_idx, fid_best=fid_best,
                                   kid_best=kid_best, save_to_wandb=True)
                t0 = time.time()
                if restart_every > 0 and t0 - tstart > restart_every:
                    return


if __name__ == '__main__':
    main()