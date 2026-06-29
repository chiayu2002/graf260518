"""
Evaluate a trained Two-Stage GRAF model (FiLM DualProjection version).

Supports both:
  - NeRF 64×64 output (Phase 1)
  - SR 256×256 output (Phase 2, if SR checkpoint exists)

Usage:
  python eval_twostage.py --config configs/twostage.yaml --checkpoint path/to/model.pt --all
  python eval_twostage.py --config configs/twostage.yaml --checkpoint path/to/model.pt --all --sr
  python eval_twostage.py --config configs/twostage.yaml --checkpoint path/to/model.pt --quick --sr
"""
import argparse
import csv
import json
import glob
import importlib
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

torch.set_default_tensor_type('torch.cuda.FloatTensor')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('submodules')

from GAN_stability.gan_training.checkpoints_mod import CheckpointIO
from graf.gan_training import Evaluator
from graf.config import get_data, build_models, load_config
from graf.utils import get_zdist
from graf.models.twostage_networks import SRNetwork


# =============================================================================
# Metric implementations
# =============================================================================

def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x / 2 + 0.5).clamp(0, 1)


def compute_mse(pred, target):
    return torch.mean((pred - target) ** 2).item()


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)).item()


def compute_r2(pred, target):
    target_mean = target.mean()
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    if ss_tot.item() == 0:
        return float('nan')
    return (1 - ss_res / ss_tot).item()


def compute_ssim(pred, target, window_size=11, sigma=1.5,
                 C1=0.01**2, C2=0.03**2):
    channels = pred.shape[1]
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords = coords - window_size // 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    window = (g1d[:, None] * g1d[None, :])
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(pred,   window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = F.conv2d(pred*pred,     window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target*target, window, padding=pad, groups=channels) - mu2_sq
    sigma12   = F.conv2d(pred*target,   window, padding=pad, groups=channels) - mu12
    ssim_map  = ((2*mu12 + C1) * (2*sigma12 + C2)) / \
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# =============================================================================
# Helpers
# =============================================================================

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pose_from_label(generator, label_row, v_list):
    """
    label 是 16-dim：
      [0:7]  = structural vec
      [7:14] = mat_feat
      [14]   = height index
      [15]   = angle index (0~359)
    """
    v_idx   = int(label_row[14].item())
    ang_idx = int(label_row[15].item())
    u = ang_idx / 360.0
    v = v_list[v_idx % len(v_list)]
    return generator.sample_select_pose(u, v)


def identify_specimen(hs, cached_hs):
    best_name = 'unknown'
    best_cos  = -1.0
    for name, cached in cached_hs.items():
        cos = F.cosine_similarity(hs.unsqueeze(0), cached.unsqueeze(0)).item()
        if cos > best_cos:
            best_cos  = cos
            best_name = name
    return best_name


def aggregate(values, name):
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(arr) == 0:
        return {f'{name}_mean': float('nan'), f'{name}_std': float('nan'), f'{name}_n': 0}
    return {
        f'{name}_mean': float(arr.mean()),
        f'{name}_std':  float(arr.std()),
        f'{name}_min':  float(arr.min()),
        f'{name}_max':  float(arr.max()),
        f'{name}_n':    int(len(arr)),
    }


# =============================================================================
# Core evaluation
# =============================================================================

def eval_paired_metrics(generator, evaluator, sr_network, loader, zdist, v_list,
                        cached_hs, n_eval, device, use_sr=False):
    generator.eval()
    if sr_network is not None:
        sr_network.eval()

    results_nerf  = {'psnr': [], 'ssim': [], 'mse': [], 'r2': []}
    results_sr    = {'psnr': [], 'ssim': [], 'mse': [], 'r2': []}
    per_spec_nerf = {}
    per_spec_sr   = {}
    collected = 0

    pbar = tqdm(total=n_eval, desc='Paired metrics', ncols=80)
    for batch in loader:
        if collected >= n_eval:
            break

        real_imgs, labels, hidden_states = batch
        real_imgs     = real_imgs.to(device)
        labels        = labels.to(device)
        hidden_states = hidden_states.to(device)
        bs = real_imgs.size(0)

        # ★ 正確的 pose 提取：label 是 16-dim，pose index 在 [14] 和 [15]
        poses = torch.stack([
            pose_from_label(generator, labels[i], v_list) for i in range(bs)])

        z = zdist.sample((bs,))
        with torch.no_grad():
            rgb_nerf, _, _ = evaluator.create_samples(z, labels, hidden_states, poses)

        nerf_h, nerf_w = rgb_nerf.shape[2], rgb_nerf.shape[3]
        real_for_nerf  = F.interpolate(real_imgs, size=(nerf_h, nerf_w),
                                       mode='bilinear', align_corners=True)
        fake_nerf_01 = to_01(rgb_nerf).cpu()
        real_nerf_01 = to_01(real_for_nerf).cpu()

        if use_sr and sr_network is not None:
            with torch.no_grad():
                nerf_for_sr = rgb_nerf.to(device)
                if nerf_h != 64 or nerf_w != 64:
                    nerf_for_sr = F.interpolate(nerf_for_sr, size=(64, 64),
                                                mode='bilinear', align_corners=True)
                sr_256 = sr_network(nerf_for_sr)
            fake_sr_01 = to_01(sr_256).cpu()
            real_sr_01 = to_01(real_imgs).cpu()

        for i in range(min(bs, n_eval - collected)):
            spec_name = identify_specimen(
                hidden_states[i].cpu(),
                {k: v.cpu() for k, v in cached_hs.items()})

            fn, rn = fake_nerf_01[i:i+1], real_nerf_01[i:i+1]
            psnr_n = compute_psnr(fn, rn)
            ssim_n = compute_ssim(fn, rn)
            mse_n  = compute_mse(fn, rn)
            r2_n   = compute_r2(fn, rn)
            results_nerf['psnr'].append(psnr_n)
            results_nerf['ssim'].append(ssim_n)
            results_nerf['mse'].append(mse_n)
            results_nerf['r2'].append(r2_n)
            if spec_name not in per_spec_nerf:
                per_spec_nerf[spec_name] = {'psnr': [], 'ssim': [], 'mse': [], 'r2': []}
            per_spec_nerf[spec_name]['psnr'].append(psnr_n)
            per_spec_nerf[spec_name]['ssim'].append(ssim_n)
            per_spec_nerf[spec_name]['mse'].append(mse_n)
            per_spec_nerf[spec_name]['r2'].append(r2_n)

            if use_sr and sr_network is not None:
                fs, rs = fake_sr_01[i:i+1], real_sr_01[i:i+1]
                psnr_s = compute_psnr(fs, rs)
                ssim_s = compute_ssim(fs, rs)
                mse_s  = compute_mse(fs, rs)
                r2_s   = compute_r2(fs, rs)
                results_sr['psnr'].append(psnr_s)
                results_sr['ssim'].append(ssim_s)
                results_sr['mse'].append(mse_s)
                results_sr['r2'].append(r2_s)
                if spec_name not in per_spec_sr:
                    per_spec_sr[spec_name] = {'psnr': [], 'ssim': [], 'mse': [], 'r2': []}
                per_spec_sr[spec_name]['psnr'].append(psnr_s)
                per_spec_sr[spec_name]['ssim'].append(ssim_s)
                per_spec_sr[spec_name]['mse'].append(mse_s)
                per_spec_sr[spec_name]['r2'].append(r2_s)

        collected += bs
        pbar.update(min(bs, n_eval - collected + bs))
    pbar.close()

    metrics_nerf = {}
    for name in ['psnr', 'ssim', 'mse', 'r2']:
        metrics_nerf.update(aggregate(results_nerf[name], name))

    metrics_sr = {}
    if use_sr and sr_network is not None:
        for name in ['psnr', 'ssim', 'mse', 'r2']:
            metrics_sr.update(aggregate(results_sr[name], f'sr_{name}'))

    return (metrics_nerf, results_nerf, per_spec_nerf,
            metrics_sr, results_sr, per_spec_sr)


def eval_fid_kid(generator, evaluator, sr_network, train_dataset, zdist,
                 v_list, out_dir, device, batch_size, use_sr=False):
    loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0,
        shuffle=True, pin_memory=False, drop_last=False,
        generator=torch.Generator(device='cuda:0'))

    fid_cache = os.path.join(out_dir, 'fid_cache_train.npz')
    kid_cache = os.path.join(out_dir, 'kid_cache_train.npz')
    print("  Initializing real image features...")
    evaluator.inception_eval.initialize_target(
        loader, cache_file=fid_cache, act_cache_file=kid_cache)

    n_dataset = len(train_dataset)

    def matched_sample_gen():
        while True:
            indices = np.random.choice(n_dataset, size=batch_size, replace=False)
            labels_list, hs_list = [], []
            for idx in indices:
                _, label_i, hs_i = train_dataset[idx]
                labels_list.append(label_i)
                hs_list.append(hs_i)
            label_batch = torch.stack(labels_list).to(device)  # [B, 16]
            hs_batch    = torch.stack(hs_list).to(device)      # [B, 1024]

            # ★ 正確的 pose 提取
            poses = torch.stack([
                pose_from_label(generator, label_batch[i], v_list)
                for i in range(batch_size)])

            z = zdist.sample((batch_size,))
            with torch.no_grad():
                rgb, _, _ = evaluator.create_samples(z, label_batch, hs_batch, poses)
                rgb = rgb.to(device)
                if use_sr and sr_network is not None:
                    nerf_h = rgb.shape[2]
                    if nerf_h != 64:
                        rgb = F.interpolate(rgb, size=(64, 64),
                                            mode='bilinear', align_corners=True)
                    rgb = sr_network(rgb)
            rgb = (rgb / 2 + 0.5).mul_(255).clamp_(0, 255) \
                  .to(torch.uint8).to(torch.float) / 255. * 2 - 1
            yield rgb.cpu()

    fid, kid = evaluator.compute_fid_kid(None, None, sample_generator=matched_sample_gen())
    return float(fid), float(kid)


def save_viewpoint_grid(generator, evaluator, sr_network, zdist, cached_hs,
                        eval_dir, it, device, use_sr=False):
    """8 個視角 × 4 個 specimen 的生成結果。"""
    N_views = 8
    # 等間隔視角，v=0.5（中間高度）
    angle_positions = [(i / N_views+0.25, 0.35) for i in range(N_views)]
    angles = [int(u * 360) for u, _ in angle_positions]

    # ★ 正確的 specimen_specs（16-dim label = vec[7] + mat_feat[7] + [h_idx, a_idx]）
    # 這裡只定義 vec 和 mat_feat，a_idx 在迴圈中填入
    specimen_specs = {
        'RS307': ([1,0,1,0,0,0,1], [0.000000,0.156717,0.918975,0.339361,0.498170,0.310937,0.360617]),
        'RS330': ([1,0,0,0,1,1,0], [0.008831,1.000000,1.000000,1.000000,0.000000,1.000000,1.000000]),
        'RS615': ([0,1,0,1,0,1,0], [1.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000]),
        'RS315': ([1,0,0,1,0,1,0], [0.006158,0.411209,0.538894,0.725360,0.017321,0.727661,0.751007]),
    }

    ztest = zdist.sample((N_views,))
    rgb_panels_nerf = []
    rgb_panels_sr   = []
    spec_names      = []

    for spec_name, (vec, mat_feat) in specimen_specs.items():
        if spec_name not in cached_hs:
            continue

        # 組成 16-dim label：vec(7) + mat_feat(7) + h_idx(0) + a_idx
        # h_idx 固定為 0（v=0.5 對應第0個 v_list entry）
        labels_list = [vec + mat_feat + [0.0, float(a)] for a in angles]
        label_batch = torch.tensor(labels_list, dtype=torch.float32, device=device)
        hs_batch    = cached_hs[spec_name].unsqueeze(0).expand(N_views, -1)
        poses       = torch.stack([
            generator.sample_select_pose(u, v) for u, v in angle_positions])

        with torch.no_grad():
            rgb, _, _ = evaluator.create_samples(
                ztest.to(device), label_batch, hs_batch, poses)

        rgb_panels_nerf.append(rgb.detach().cpu())
        spec_names.append(spec_name)

        if use_sr and sr_network is not None:
            with torch.no_grad():
                r4sr = rgb.to(device)
                if r4sr.shape[2] != 64:
                    r4sr = F.interpolate(r4sr, size=(64, 64), mode='bilinear')
                sr_out = sr_network(r4sr)
            rgb_panels_sr.append(sr_out.detach().cpu())

    if rgb_panels_nerf:
        rgb_all  = torch.cat(rgb_panels_nerf, dim=0)
        grid     = to_01(rgb_all)
        out_path = os.path.join(eval_dir, f'grid_nerf_v035_it{it}.png')
        save_image(grid, out_path, nrow=N_views)
        print(f"  Saved NeRF grid ({' / '.join(spec_names)}) -> {out_path}")
        for i, sn in enumerate(spec_names):
            save_image(to_01(rgb_panels_nerf[i]),
                       os.path.join(eval_dir, f'nerf_{sn}_v035_it{it}.png'), nrow=N_views)

    if rgb_panels_sr:
        sr_all      = torch.cat(rgb_panels_sr, dim=0)
        out_path_sr = os.path.join(eval_dir, f'grid_sr256_v035_it{it}.png')
        save_image(to_01(sr_all), out_path_sr, nrow=N_views)
        print(f"  Saved SR 256 grid -> {out_path_sr}")
        for i, sn in enumerate(spec_names):
            save_image(to_01(rgb_panels_sr[i]),
                       os.path.join(eval_dir, f'sr256_{sn}_v035_it{it}.png'), nrow=N_views)


# =============================================================================
# CSV output
# =============================================================================

def write_csv(csv_path, metrics_nerf, per_sample_nerf, per_spec_nerf,
              metrics_sr, per_sample_sr, per_spec_sr,
              fid_nerf=None, kid_nerf=None,
              fid_sr=None,  kid_sr=None):
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)

        w.writerow(['=== NeRF Paired Metrics ==='])
        w.writerow(['metric', 'value'])
        for k, v in metrics_nerf.items():
            w.writerow([k, v])

        if per_spec_nerf:
            w.writerow([])
            w.writerow(['=== NeRF Per-Specimen ==='])
            w.writerow(['specimen', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std',
                        'mse_mean', 'mse_std', 'r2_mean', 'r2_std', 'count'])
            for sn in sorted(per_spec_nerf.keys()):
                m = per_spec_nerf[sn]
                w.writerow([sn,
                    f"{np.mean(m['psnr']):.4f}", f"{np.std(m['psnr']):.4f}",
                    f"{np.mean(m['ssim']):.6f}", f"{np.std(m['ssim']):.6f}",
                    f"{np.mean(m['mse']):.6f}",  f"{np.std(m['mse']):.6f}",
                    f"{np.mean(m['r2']):.6f}",   f"{np.std(m['r2']):.6f}",
                    len(m['psnr'])])

        if metrics_sr:
            w.writerow([])
            w.writerow(['=== SR 256 Paired Metrics ==='])
            w.writerow(['metric', 'value'])
            for k, v in metrics_sr.items():
                w.writerow([k, v])

            if per_spec_sr:
                w.writerow([])
                w.writerow(['=== SR 256 Per-Specimen ==='])
                w.writerow(['specimen', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std',
                            'mse_mean', 'mse_std', 'r2_mean', 'r2_std', 'count'])
                for sn in sorted(per_spec_sr.keys()):
                    m = per_spec_sr[sn]
                    w.writerow([sn,
                        f"{np.mean(m['psnr']):.4f}", f"{np.std(m['psnr']):.4f}",
                        f"{np.mean(m['ssim']):.6f}", f"{np.std(m['ssim']):.6f}",
                        f"{np.mean(m['mse']):.6f}",  f"{np.std(m['mse']):.6f}",
                        f"{np.mean(m['r2']):.6f}",   f"{np.std(m['r2']):.6f}",
                        len(m['psnr'])])

        w.writerow([])
        w.writerow(['=== FID / KID ==='])
        if fid_nerf is not None:
            w.writerow(['nerf_fid',       f"{fid_nerf:.4f}"])
            w.writerow(['nerf_kid_x100',  f"{kid_nerf * 100:.4f}"])
        if fid_sr is not None:
            w.writerow(['sr256_fid',      f"{fid_sr:.4f}"])
            w.writerow(['sr256_kid_x100', f"{kid_sr * 100:.4f}"])

        w.writerow([])
        w.writerow(['=== NeRF Per-Sample Detail ==='])
        w.writerow(['idx', 'psnr', 'ssim', 'mse', 'r2'])
        for i in range(len(per_sample_nerf['psnr'])):
            w.writerow([i,
                f"{per_sample_nerf['psnr'][i]:.4f}",
                f"{per_sample_nerf['ssim'][i]:.6f}",
                f"{per_sample_nerf['mse'][i]:.6f}",
                f"{per_sample_nerf['r2'][i]:.6f}"])

        if per_sample_sr and per_sample_sr['psnr']:
            w.writerow([])
            w.writerow(['=== SR 256 Per-Sample Detail ==='])
            w.writerow(['idx', 'psnr', 'ssim', 'mse', 'r2'])
            for i in range(len(per_sample_sr['psnr'])):
                w.writerow([i,
                    f"{per_sample_sr['psnr'][i]:.4f}",
                    f"{per_sample_sr['ssim'][i]:.6f}",
                    f"{per_sample_sr['mse'][i]:.6f}",
                    f"{per_sample_sr['r2'][i]:.6f}"])

    print(f"  Saved metrics -> {csv_path}")


def save_comparison_plots(per_spec_nerf, per_spec_sr, eval_dir, it):
    if not per_spec_nerf:
        return
    os.makedirs(os.path.join(eval_dir, 'comparisons'), exist_ok=True)

    for tag, per_spec in [('nerf', per_spec_nerf), ('sr256', per_spec_sr)]:
        if not per_spec:
            continue
        metric_names = ['psnr', 'ssim', 'mse', 'r2']
        titles       = ['PSNR (dB)', 'SSIM', 'MSE', 'R²']
        fig, axes    = plt.subplots(1, 4, figsize=(20, 5))
        for ax, mn, title in zip(axes, metric_names, titles):
            for sn in sorted(per_spec.keys()):
                vals = per_spec[sn].get(mn, [])
                if vals:
                    ax.hist(vals, alpha=0.5, label=sn, bins=15)
            ax.set_title(f'{title} ({tag})')
            ax.set_xlabel(title)
            ax.set_ylabel('Count')
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f'dist_{tag}_it{it}.png'), dpi=150)
        plt.close()
    print(f"  Saved distribution plots")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Two-Stage GRAF model (FiLM version).')
    parser.add_argument('--config',     type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_pairs',  type=int, default=100)
    parser.add_argument('--sr',         action='store_true',
                        help='Also evaluate SR 256×256 (requires Phase 2 checkpoint)')
    parser.add_argument('--fid_kid',       action='store_true')
    parser.add_argument('--psnr_r2',       action='store_true')
    parser.add_argument('--create_sample', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='Fast check: all metrics, n=50')
    parser.add_argument('--all',   action='store_true',
                        help='All metrics, n=num_pairs')
    parser.add_argument('--gpu',   type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.quick:
        args.fid_kid = args.psnr_r2 = args.create_sample = True
        args.num_pairs = 50
    if args.all:
        args.fid_kid = args.psnr_r2 = args.create_sample = True
    if not any([args.fid_kid, args.psnr_r2, args.create_sample]):
        args.fid_kid = args.psnr_r2 = args.create_sample = True

    print("\n" + "=" * 60)
    print("Two-Stage Evaluation (FiLM DualProjection)")
    print("=" * 60)
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  SR eval:    {args.sr}")
    print(f"  Paired:     {args.psnr_r2} (n={args.num_pairs})")
    print(f"  FID/KID:    {args.fid_kid}")
    print(f"  Samples:    {args.create_sample}")
    print("=" * 60 + "\n")

    set_random_seed(0)
    config = load_config(args.config)
    config['data']['fov']             = float(config['data']['fov'])
    config['training']['nworkers']    = 0

    batch_size = config['training']['batch_size']
    out_dir    = os.path.join(config['training']['outdir'], config['expname'])
    eval_dir   = os.path.join(out_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    device = torch.device('cuda:0')

    # ── Extractor ────────────────────────────────────────────────────────────
    print("Loading GRU extractor...")
    extractor_path = config['data']['extractor_path']
    extractor_args = json.load(open(
        "/Data/home/vicky/graf260108_im64/HystereticGRU/2026-03-24_17-13-01/args.json", "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module(
        "graf.models.HystereticPrediction"
    ).__dict__[extractor_args.architecture](**vars(extractor_args))
    extractor = extractor.to(device)
    state_dict = torch.load(glob.glob(extractor_path, recursive=True)[0])["state_dict"]
    extractor.load_state_dict(state_dict)
    print("Extractor loaded.")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_dataset, hwfr = get_data(config, extractor, extractor_args)
    config['data']['hwfr'] = hwfr
    print(f"Dataset: {len(train_dataset)} images")

    cached_hs = {n: h.to(device) for n, h in train_dataset.hidden_state.items()}
    print(f"Cached hidden states: {list(cached_hs.keys())}")

    loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0,
        shuffle=True, pin_memory=False, drop_last=False,
        generator=torch.Generator(device='cuda:0'))

    # ── Models ───────────────────────────────────────────────────────────────
    generator, _ = build_models(config, disc=False)
    generator    = generator.to(device)

    sr_network = SRNetwork(ch=64, n_rb=6).to(device) if args.sr else None

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.dirname(args.checkpoint))
    checkpoint_io.register_modules(**generator.module_dict)
    if sr_network is not None:
        checkpoint_io.register_modules(sr_network=sr_network)

    print(f"Registered modules: {list(checkpoint_io.module_dict.keys())}")
    load_dict = checkpoint_io.load(os.path.basename(args.checkpoint))
    it = load_dict.get('it', -1)
    print(f"Loaded checkpoint iter={it}")

    if sr_network is not None:
        if 'sr_network' in load_dict:
            print("[SR] Weights loaded from checkpoint.")
        else:
            print("[SR] WARNING: sr_network not found in checkpoint!")
            print("[SR]          SR results will be meaningless. Use Phase 2 checkpoint.")

    # ★ FiLM sanity check：印出 mat_film 最後一層的 bias 前幾個值
    # 正常情況下前 256 個（gamma 部分）應接近 1，後 256 個（beta）接近 0
    nerf_model   = generator.render_kwargs_train['network_fn']
    film_bias    = nerf_model.condition_feature.mat_film[-1].bias.data
    gamma_mean   = film_bias[:256].mean().item()
    beta_mean    = film_bias[256:].mean().item()
    hs_proj_norm = nerf_model.condition_feature.hs_proj.weight.norm().item()
    print(f"[FiLM Sanity] mat_film bias: gamma_mean={gamma_mean:.4f} "
          f"(init=1.0), beta_mean={beta_mean:.4f} (init=0.0)")
    print(f"[FiLM Sanity] hs_proj weight norm={hs_proj_norm:.4f}")

    # ── Embedding similarity ──────────────────────────────────────────────────
    specimen_specs = {
        'RS307': ([1,0,1,0,0,0,1], [0.000000,0.156717,0.918975,0.339361,0.498170,0.310937,0.360617]),
        'RS330': ([1,0,0,0,1,1,0], [0.008831,1.000000,1.000000,1.000000,0.000000,1.000000,1.000000]),
        'RS615': ([0,1,0,1,0,1,0], [1.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000]),
        'RS315': ([1,0,0,1,0,1,0], [0.006158,0.411209,0.538894,0.725360,0.017321,0.727661,0.751007]),
    }
    print("\n[Embedding Cosine Similarity]")
    cond_proj = nerf_model.condition_feature
    emb_dict  = {}
    with torch.no_grad():
        for sn, (vec, mf) in specimen_specs.items():
            if sn not in cached_hs:
                # RS315 用 cached_hs['RS330'] 的 hs 代理（結構最近）
                hs_proxy = cached_hs.get('RS330', list(cached_hs.values())[0])
            else:
                hs_proxy = cached_hs[sn]
            mf_t      = torch.tensor(mf, dtype=torch.float32, device=device).unsqueeze(0)
            emb_dict[sn] = cond_proj.encode(hs_proxy.unsqueeze(0), mf_t).squeeze(0)

    pairs = [('RS307','RS315'),('RS307','RS330'),('RS307','RS615'),
             ('RS315','RS330'),('RS315','RS615'),('RS330','RS615')]
    for k1, k2 in pairs:
        if k1 in emb_dict and k2 in emb_dict:
            cos = F.cosine_similarity(
                emb_dict[k1].unsqueeze(0), emb_dict[k2].unsqueeze(0)).item()
            print(f"  {k1} vs {k2}: Emb256_cos={cos:.4f}")

    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)
    evaluator = Evaluator(args.fid_kid, generator, zdist, None,
                          batch_size=batch_size, device=device)
    v_list = [float(x.strip()) for x in config['data']['v'].split(',')]

    # =========================================================================
    # 1. Paired metrics (PSNR / SSIM / MSE / R²)
    # =========================================================================
    metrics_nerf    = {}
    per_sample_nerf = {'psnr': [], 'ssim': [], 'mse': [], 'r2': []}
    per_spec_nerf   = {}
    metrics_sr      = {}
    per_sample_sr   = {'psnr': [], 'ssim': [], 'mse': [], 'r2': []}
    per_spec_sr     = {}

    if args.psnr_r2:
        print(f"\n[1] Paired metrics on {args.num_pairs} samples...")
        t0 = time.time()
        (metrics_nerf, per_sample_nerf, per_spec_nerf,
         metrics_sr,   per_sample_sr,   per_spec_sr) = eval_paired_metrics(
            generator, evaluator, sr_network, loader, zdist, v_list,
            cached_hs, args.num_pairs, device, use_sr=args.sr)

        print(f"\n  NeRF 64×64 metrics:")
        print(f"  {'Metric':<12} {'Mean':>10} {'Std':>10}")
        print("  " + "-" * 32)
        for name in ['psnr', 'ssim', 'mse', 'r2']:
            print(f"  {name.upper():<12} "
                  f"{metrics_nerf.get(f'{name}_mean', 0):>10.4f} "
                  f"{metrics_nerf.get(f'{name}_std',  0):>10.4f}")

        if metrics_sr:
            print(f"\n  SR 256×256 metrics:")
            print(f"  {'Metric':<12} {'Mean':>10} {'Std':>10}")
            print("  " + "-" * 32)
            for name in ['psnr', 'ssim', 'mse', 'r2']:
                print(f"  {name.upper():<12} "
                      f"{metrics_sr.get(f'sr_{name}_mean', 0):>10.4f} "
                      f"{metrics_sr.get(f'sr_{name}_std',  0):>10.4f}")

        print(f"\n  Took {time.time() - t0:.1f}s")

    # =========================================================================
    # 2. FID / KID
    # =========================================================================
    fid_nerf = kid_nerf = fid_sr = kid_sr = None

    if args.fid_kid:
        print(f"\n[2a] NeRF 64×64 FID/KID...")
        t0 = time.time()
        fid_nerf, kid_nerf = eval_fid_kid(
            generator, evaluator, None, train_dataset, zdist, v_list,
            out_dir, device, batch_size, use_sr=False)
        print(f"     NeRF  FID={fid_nerf:.2f}  KID×100={kid_nerf*100:.2f}  "
              f"({time.time()-t0:.1f}s)")
        torch.cuda.empty_cache()

        if args.sr and sr_network is not None:
            print(f"\n[2b] SR 256×256 FID/KID...")
            t0 = time.time()
            fid_sr, kid_sr = eval_fid_kid(
                generator, evaluator, sr_network, train_dataset, zdist, v_list,
                out_dir, device, batch_size, use_sr=True)
            print(f"     SR    FID={fid_sr:.2f}  KID×100={kid_sr*100:.2f}  "
                  f"({time.time()-t0:.1f}s)")
            torch.cuda.empty_cache()

    # =========================================================================
    # 3. Sample grids
    # =========================================================================
    if args.create_sample:
        print(f"\n[3] Generating sample grids...")
        t0 = time.time()
        generator.eval()
        save_viewpoint_grid(generator, evaluator, sr_network, zdist, cached_hs,
                            eval_dir, it, device, use_sr=args.sr)
        generator.train()
        print(f"    Took {time.time()-t0:.1f}s")

    # =========================================================================
    # 4. Save CSV + plots
    # =========================================================================
    csv_path = os.path.join(eval_dir, f'metrics_it{it}.csv')
    write_csv(csv_path,
              metrics_nerf, per_sample_nerf, per_spec_nerf,
              metrics_sr,   per_sample_sr,   per_spec_sr,
              fid_nerf=fid_nerf, kid_nerf=kid_nerf,
              fid_sr=fid_sr,     kid_sr=kid_sr)

    save_comparison_plots(per_spec_nerf, per_spec_sr, eval_dir, it)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results -> {eval_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()