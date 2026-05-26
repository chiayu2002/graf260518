"""
Evaluate a trained GRAF model (GRU conditioning version).

Metrics:
  - Paired (fake rendered at real pose vs real image): PSNR, SSIM, MSE, R²
  - Set-level (comparable with CCSR eval):             FID, KID
  - Per-specimen FID (diagnostic, opt-in):             per-spec FID, KID
  - Qualitative:                                       multi-specimen viewpoint grid

FID comparability with CCSR version:
  Both versions reconstruct poses from labels via pose_from_label(),
  ensuring the same viewpoint distribution for generated images.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

torch.set_default_tensor_type('torch.cuda.FloatTensor')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('submodules')

from submodules.GAN_stability.gan_training.checkpoints_mod import CheckpointIO
from graf.gan_training import Evaluator
from graf.config import get_data, build_models, load_config
from graf.utils import count_trainable_parameters, get_zdist
from graf.transforms import ImgToPatch


# =============================================================================
# Metric implementations
# =============================================================================
def to_01(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]"""
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
    """SSIM for [B, C, H, W] images in [0, 1]."""
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

    ssim_map = ((2*mu12 + C1) * (2*sigma12 + C2)) / \
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
    """Reconstruct the pose that corresponds to one label row.
    Identical logic to the CCSR eval version — ensures FID comparability."""
    v_idx   = int(label_row[7].item())
    ang_idx = int(label_row[8].item())
    u = ang_idx / 360.0
    v = v_list[v_idx]
    return generator.sample_select_pose(u, v)


def identify_specimen(hs, cached_hs):
    """Identify which specimen a hidden_state belongs to (by cosine similarity)."""
    best_name = 'unknown'
    best_cos = -1
    for name, cached in cached_hs.items():
        cos = F.cosine_similarity(hs.unsqueeze(0), cached.unsqueeze(0)).item()
        if cos > best_cos:
            best_cos = cos
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
# Core evaluation routines
# =============================================================================
def eval_paired_metrics(generator, evaluator, loader, zdist, v_list,
                        cached_hs, n_eval, device):
    """
    Batch-ized paired metrics (PSNR / SSIM / MSE / R²).
    Renders full images at real poses — same approach as CCSR eval version.
    """
    generator.eval()
    results = {'psnr': [], 'ssim': [], 'mse': [], 'r2': []}
    per_spec = {}   # spec_name → {psnr: [], ssim: [], ...}
    collected = 0

    pbar = tqdm(total=n_eval, desc='Paired metrics (batch)', ncols=80)
    for batch in loader:
        if collected >= n_eval:
            break

        real_imgs, labels, hidden_states = batch
        real_imgs = real_imgs.to(device)
        labels = labels.to(device)
        hidden_states = hidden_states.to(device)
        bs = real_imgs.size(0)

        # Batch poses — same as CCSR version
        poses = torch.stack([
            pose_from_label(generator, labels[i], v_list) for i in range(bs)
        ])

        z = zdist.sample((bs,))
        with torch.no_grad():
            rgb, _, _ = evaluator.create_samples(z, labels, hidden_states, poses)

        fake01 = to_01(rgb).cpu()
        real01 = to_01(real_imgs).cpu()

        for i in range(min(bs, n_eval - collected)):
            f, r = fake01[i:i+1], real01[i:i+1]

            psnr_val = compute_psnr(f, r)
            ssim_val = compute_ssim(f, r)
            mse_val  = compute_mse(f, r)
            r2_val   = compute_r2(f, r)

            results['psnr'].append(psnr_val)
            results['ssim'].append(ssim_val)
            results['mse'].append(mse_val)
            results['r2'].append(r2_val)

            # Per-specimen grouping
            spec_name = identify_specimen(hidden_states[i].cpu(), 
                                          {k: v.cpu() for k, v in cached_hs.items()})
            if spec_name not in per_spec:
                per_spec[spec_name] = {'psnr': [], 'ssim': [], 'mse': [], 'r2': []}
            per_spec[spec_name]['psnr'].append(psnr_val)
            per_spec[spec_name]['ssim'].append(ssim_val)
            per_spec[spec_name]['mse'].append(mse_val)
            per_spec[spec_name]['r2'].append(r2_val)

        collected += bs
        pbar.update(min(bs, n_eval - collected + bs))
    pbar.close()

    # Aggregate
    metrics = {}
    for name in ['psnr', 'ssim', 'mse', 'r2']:
        metrics.update(aggregate(results[name], name))

    return metrics, results, per_spec


def eval_fid_kid(generator, evaluator, train_dataset, zdist, v_list,
                 cached_hs, out_dir, device, batch_size):
    """
    Marginal FID with matched conditioning + pose from label.
    
    This matches the CCSR eval version's approach:
      - Conditions sampled from the dataset (matched to real distribution)
      - Poses reconstructed from labels via pose_from_label()
      - uint8 quantization applied (same as training FID)
    """
    # Real image features (cached)
    loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0,
        shuffle=True, pin_memory=False, drop_last=False,
        generator=torch.Generator(device='cuda:0'),
    )
    fid_cache = os.path.join(out_dir, 'fid_cache_train.npz')
    kid_cache = os.path.join(out_dir, 'kid_cache_train.npz')
    print("  Initializing real image features (cached if available)...")
    evaluator.inception_eval.initialize_target(
        loader, cache_file=fid_cache, act_cache_file=kid_cache,
    )

    n_dataset = len(train_dataset)

    def matched_sample_generator():
        """
        每次 yield 一個 batch 的 fake。
        條件從 dataset 隨機抽,pose 從 label 重建。
        """
        while True:
            indices = np.random.choice(n_dataset, size=batch_size, replace=False)
            labels_list, hs_list = [], []
            for idx in indices:
                _, label_i, hs_i = train_dataset[idx]
                labels_list.append(label_i)
                hs_list.append(hs_i)
            label_batch = torch.stack(labels_list).to(device)  # [B, 9]
            hs_batch = torch.stack(hs_list).to(device)         # [B, 1024]

            # Reconstruct poses from labels (CCSR-compatible)
            poses = torch.stack([
                pose_from_label(generator, label_batch[i], v_list)
                for i in range(batch_size)
            ])

            z = zdist.sample((batch_size,))
            with torch.no_grad():
                rgb, _, _ = evaluator.create_samples(z, label_batch, hs_batch, poses)

            # uint8 quantization (same as train.py)
            rgb = (rgb / 2 + 0.5).mul_(255).clamp_(0, 255) \
                  .to(torch.uint8).to(torch.float) / 255. * 2 - 1
            yield rgb.cpu()

    fid, kid = evaluator.compute_fid_kid(
        real_label=None, hidden_state=None,
        sample_generator=matched_sample_generator(),
    )
    return float(fid), float(kid)


def eval_per_spec_fid(generator, evaluator, train_dataset, zdist, v_list,
                      cached_hs, device, batch_size):
    """Per-specimen FID (diagnostic, biased high for small n)."""
    v_list_eval = v_list
    n_heights = len(v_list_eval)

    fid_results = {}
    kid_results = {}

    spec_items = [(n, h) for n, h in cached_hs.items()
                  if n + '_n' in train_dataset.specimen_props]

    for spec_name, hs in tqdm(spec_items, desc="Per-spec FID", ncols=80):
        props = train_dataset.specimen_props[spec_name + '_n']
        label_vec_7d = props['AR'] + props['LR'] + props['TR']

        def per_spec_gen(label_7d, hs_single):
            while True:
                full_labels = []
                for _ in range(batch_size):
                    h_idx = float(np.random.randint(0, n_heights))
                    a_idx = float(np.random.randint(0, 360))
                    full_labels.append(torch.tensor(
                        label_7d + [h_idx, a_idx], dtype=torch.float32))
                label_batch = torch.stack(full_labels).to(device)
                hs_batch = hs_single.unsqueeze(0).expand(batch_size, -1).to(device)

                # Reconstruct poses
                poses = torch.stack([
                    pose_from_label(generator, label_batch[i], v_list_eval)
                    for i in range(batch_size)
                ])

                z = zdist.sample((batch_size,))
                with torch.no_grad():
                    rgb, _, _ = evaluator.create_samples(z, label_batch, hs_batch, poses)
                rgb = (rgb / 2 + 0.5).mul_(255).clamp_(0, 255) \
                      .to(torch.uint8).to(torch.float) / 255. * 2 - 1
                yield rgb.cpu()

        fid, kid = evaluator.compute_fid_kid(
            None, None, sample_generator=per_spec_gen(label_vec_7d, hs))
        fid_results[spec_name] = fid
        kid_results[spec_name] = kid
        tqdm.write(f"    {spec_name}: FID={fid:.2f}, KID×100={kid*100:.2f}")
        torch.cuda.empty_cache()

    return fid_results, kid_results


def save_viewpoint_grid(generator, evaluator, zdist, cached_hs,
                        eval_dir, it, device):
    """Save multi-specimen viewpoint grid (8 views × N specimens)."""
    N_views = 8
    angle_positions = [(i / N_views+0.25, 0.5) for i in range(N_views)]
    angles = [int(u * 360) for u, _ in angle_positions]

    specimens = {
        'RS307': [1.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0],
        'RS330': [1.0, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0],
        'RS615': [0.0, 1.0,  0.0, 1.0, 0.0,  1.0, 0.0],
        'RS315': [1.0, 0.0,  0.0, 1.0, 0.0,  1.0, 0.0],
    }

    ztest = zdist.sample((N_views,))
    rgb_panels = []
    spec_names = []

    for spec_name, vec in specimens.items():
        if spec_name not in cached_hs:
            continue

        labels_list = [vec + [0.5, float(a)] for a in angles]
        label_batch = torch.tensor(labels_list, dtype=torch.float32, device=device)
        hs_batch = cached_hs[spec_name].unsqueeze(0).expand(N_views, -1)

        poses = torch.stack([
            generator.sample_select_pose(u, v) for u, v in angle_positions
        ])

        with torch.no_grad():
            rgb, _, _ = evaluator.create_samples(
                ztest.to(device), label_batch, hs_batch, poses)
        rgb_panels.append(rgb.detach().cpu())
        spec_names.append(spec_name)

    if rgb_panels:
        rgb_all = torch.cat(rgb_panels, dim=0)
        grid = to_01(rgb_all)
        grid = (grid * 255).to(torch.uint8).float() / 255
        out_path = os.path.join(eval_dir, f'grid_specimens_it{it}.png')
        save_image(grid, out_path, nrow=N_views)
        print(f"  Saved grid ({' / '.join(spec_names)}) -> {out_path}")

        # Also save per-specimen grids
        for i, spec_name in enumerate(spec_names):
            spec_rgb = rgb_panels[i]
            spec_grid = to_01(spec_rgb)
            spec_path = os.path.join(eval_dir, f'samples_{spec_name}_iter{it}.png')
            save_image(spec_grid, spec_path, nrow=N_views)


# =============================================================================
# CSV output
# =============================================================================
def write_csv(csv_path, metrics, per_sample, per_spec_metrics,
              fid=None, kid=None, fid_results=None, kid_results=None):
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)

        # Aggregate paired metrics
        w.writerow(['=== Aggregate Paired Metrics ==='])
        w.writerow(['metric', 'value'])
        for k, v in metrics.items():
            w.writerow([k, v])

        # Per-specimen paired metrics
        if per_spec_metrics:
            w.writerow([])
            w.writerow(['=== Per-Specimen Paired Metrics ==='])
            w.writerow(['specimen', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std',
                        'mse_mean', 'mse_std', 'r2_mean', 'r2_std', 'count'])
            for spec_name in sorted(per_spec_metrics.keys()):
                m = per_spec_metrics[spec_name]
                w.writerow([
                    spec_name,
                    f"{np.mean(m['psnr']):.4f}", f"{np.std(m['psnr']):.4f}",
                    f"{np.mean(m['ssim']):.6f}", f"{np.std(m['ssim']):.6f}",
                    f"{np.mean(m['mse']):.6f}", f"{np.std(m['mse']):.6f}",
                    f"{np.mean(m['r2']):.6f}", f"{np.std(m['r2']):.6f}",
                    len(m['psnr']),
                ])

        # FID / KID
        if fid is not None:
            w.writerow([])
            w.writerow(['=== FID / KID ==='])
            w.writerow(['marginal_fid', f"{fid:.4f}"])
            w.writerow(['marginal_kid_x100', f"{kid*100:.4f}"])

        if fid_results:
            w.writerow([])
            w.writerow(['=== Per-Specimen FID (diagnostic) ==='])
            w.writerow(['specimen', 'fid', 'kid_x100'])
            for name in fid_results:
                w.writerow([name, f"{fid_results[name]:.4f}", f"{kid_results[name]*100:.4f}"])

        # Per-sample detail
        w.writerow([])
        w.writerow(['=== Per-Sample Detail ==='])
        w.writerow(['idx', 'psnr', 'ssim', 'mse', 'r2'])
        for i in range(len(per_sample['psnr'])):
            w.writerow([i,
                        f"{per_sample['psnr'][i]:.4f}",
                        f"{per_sample['ssim'][i]:.6f}",
                        f"{per_sample['mse'][i]:.6f}",
                        f"{per_sample['r2'][i]:.6f}"])

    print(f"  Saved metrics -> {csv_path}")


# =============================================================================
# Comparison plots
# =============================================================================
def save_comparison_plots(per_spec_metrics, eval_dir, it):
    """Save per-specimen comparison plots and metric distributions."""
    if not per_spec_metrics:
        return

    compare_dir = os.path.join(eval_dir, 'comparisons')
    os.makedirs(compare_dir, exist_ok=True)

    # Distribution plots
    metric_names = ['psnr', 'ssim', 'mse', 'r2']
    titles = ['PSNR (dB)', 'SSIM', 'MSE', 'R²']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, mname, title in zip(axes, metric_names, titles):
        for spec_name in sorted(per_spec_metrics.keys()):
            vals = per_spec_metrics[spec_name][mname]
            if vals:
                ax.hist(vals, alpha=0.5, label=spec_name, bins=15)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel('Count')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f'metrics_distribution_iter{it}.png'), dpi=150)
    plt.close()
    print(f"  Saved distribution plot")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate trained GRAF model (GRU version).')
    parser.add_argument('--config', type=str,
                        default='./results/column20260417_gru_disc_Dbn_lrg3_aux_sampling_speed/config.yaml')
    parser.add_argument('--checkpoint', type=str,
                        default='./results/column20260417_gru_disc_Dbn_lrg3_aux_sampling_speed/chkpts/model_00209999.pt')
    parser.add_argument('--num_pairs', type=int, default=100,
                        help='Number of paired samples for PSNR/SSIM/R².')

    # Individual flags
    parser.add_argument('--fid_kid', action='store_true', help='Marginal FID/KID.')
    parser.add_argument('--per_spec_fid', action='store_true', help='Per-specimen FID (SLOW).')
    parser.add_argument('--psnr_r2', action='store_true', help='Paired metrics.')
    parser.add_argument('--create_sample', action='store_true', help='Sample grids.')

    # Combo flags
    parser.add_argument('--quick', action='store_true', help='FID + 50 pairs + samples.')
    parser.add_argument('--all', action='store_true', help='All except per-spec FID.')
    parser.add_argument('--full', action='store_true', help='Everything.')
    parser.add_argument('--gpu', type=str, default='4')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.quick:
        args.fid_kid = args.psnr_r2 = args.create_sample = True
        args.num_pairs = 50
    if args.all:
        args.fid_kid = args.psnr_r2 = args.create_sample = True
    if args.full:
        args.fid_kid = args.per_spec_fid = args.psnr_r2 = args.create_sample = True
    if not any([args.fid_kid, args.per_spec_fid, args.psnr_r2, args.create_sample]):
        print("[INFO] No flags provided, defaulting to --all.")
        args.fid_kid = args.psnr_r2 = args.create_sample = True

    print("\n" + "="*60)
    print("Evaluation Configuration")
    print("="*60)
    print(f"  Config:       {args.config}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Samples:      {args.create_sample}")
    print(f"  Marginal FID: {args.fid_kid}")
    print(f"  Per-spec FID: {args.per_spec_fid}  {'(SLOW)' if args.per_spec_fid else ''}")
    print(f"  Paired:       {args.psnr_r2}  (n={args.num_pairs})")
    print("="*60 + "\n")

    set_random_seed(0)

    # --- Config ---
    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])
    config['training']['nworkers'] = 0

    batch_size = config['training']['batch_size']
    out_dir    = os.path.join(config['training']['outdir'], config['expname'])
    eval_dir   = os.path.join(out_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device('cuda:0')

    # --- GRU Extractor ---
    print("Loading GRU extractor...")
    extractor_path = config['data']['extractor_path']
    extractor_args = json.load(open("HystereticGRU/2026-03-24_17-13-01/args.json", "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module("graf.models.HystereticPrediction").__dict__[
        extractor_args.architecture
    ](**vars(extractor_args))
    extractor = extractor.to(device)
    state_dict = torch.load(glob.glob(extractor_path, recursive=True)[0])["state_dict"]
    status = extractor.load_state_dict(state_dict)
    print(f"Extractor loaded: {status}")

    # --- Data ---
    train_dataset, hwfr = get_data(config, extractor, extractor_args)
    config['data']['hwfr'] = hwfr
    print(f"Dataset: {len(train_dataset)} images, HW={hwfr[0]}x{hwfr[1]}")

    cached_hs = {}
    for exp_name, hs in train_dataset.hidden_state.items():
        cached_hs[exp_name] = hs.to(device)
    print(f"Cached hidden states: {list(cached_hs.keys())}")

    loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0,
        shuffle=True, pin_memory=False, drop_last=False,
        generator=torch.Generator(device='cuda:0'),
    )

    # --- Model ---
    generator, _ = build_models(config, disc=False)
    generator = generator.to(device)
    print(f"Generator params: {count_trainable_parameters(generator)}")

    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.dirname(args.checkpoint))
    checkpoint_io.register_modules(**generator.module_dict)
    print(f"Registered modules: {list(checkpoint_io.module_dict.keys())}")
    assert len(checkpoint_io.module_dict) > 0, "No modules registered!"

    load_dict = checkpoint_io.load(os.path.basename(args.checkpoint))
    it = load_dict.get('it', -1)
    print(f"Loaded checkpoint iter={it}")

    # Sanity check
    nerf_model = generator.render_kwargs_train['network_fn']
    norm_val = nerf_model.condition_feature.weight.norm().item()
    print(f"[Sanity] condition_feature weight norm = {norm_val:.4f}")

    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)
    evaluator = Evaluator(
        args.fid_kid or args.per_spec_fid, generator, zdist, None,
        batch_size=batch_size, device=device,
    )

    v_list = [float(x.strip()) for x in config['data']['v'].split(',')]

    # =================================================================
    # 1. Paired metrics (batch-ized, full images)
    # =================================================================
    metrics = {}
    per_sample = {'psnr': [], 'ssim': [], 'mse': [], 'r2': []}
    per_spec_metrics = {}

    if args.psnr_r2:
        print(f"\n[1] Paired metrics on {args.num_pairs} samples (batch-ized)...")
        t0 = time.time()
        metrics, per_sample, per_spec_metrics = eval_paired_metrics(
            generator, evaluator, loader, zdist, v_list,
            cached_hs, args.num_pairs, device,
        )
        # Print results
        print(f"\n  {'Metric':<12} {'Mean':>10} {'Std':>10}")
        print("  " + "-"*32)
        for name in ['psnr', 'ssim', 'mse', 'r2']:
            print(f"  {name.upper():<12} {metrics[f'{name}_mean']:>10.4f} {metrics[f'{name}_std']:>10.4f}")

        if per_spec_metrics:
            print(f"\n  Per-specimen breakdown:")
            print(f"  {'Specimen':<12} {'PSNR':>8} {'SSIM':>8} {'MSE':>8} {'R²':>8} {'N':>5}")
            print("  " + "-"*49)
            for spec in sorted(per_spec_metrics.keys()):
                m = per_spec_metrics[spec]
                print(f"  {spec:<12} "
                      f"{np.mean(m['psnr']):>8.2f} "
                      f"{np.mean(m['ssim']):>8.4f} "
                      f"{np.mean(m['mse']):>8.4f} "
                      f"{np.mean(m['r2']):>8.4f} "
                      f"{len(m['psnr']):>5}")

        print(f"\n  Paired metrics took {time.time() - t0:.1f}s")

    # =================================================================
    # 2. FID / KID (with pose from label — CCSR-compatible)
    # =================================================================
    fid = kid = None
    fid_results = kid_results = None

    if args.fid_kid:
        print(f"\n[2] Marginal FID / KID (pose from label — CCSR-compatible)...")
        t0 = time.time()
        fid, kid = eval_fid_kid(
            generator, evaluator, train_dataset, zdist, v_list,
            cached_hs, out_dir, device, batch_size,
        )
        print(f"    Marginal FID = {fid:.2f}")
        print(f"    Marginal KID×100 = {kid*100:.2f}")
        print(f"    FID/KID took {time.time() - t0:.1f}s")
        torch.cuda.empty_cache()

    if args.per_spec_fid:
        print(f"\n[2b] Per-specimen FID (diagnostic)...")
        t0 = time.time()
        fid_results, kid_results = eval_per_spec_fid(
            generator, evaluator, train_dataset, zdist, v_list,
            cached_hs, device, batch_size,
        )
        print(f"    Per-spec FID took {time.time() - t0:.1f}s")

    # =================================================================
    # 3. Sample grids
    # =================================================================
    if args.create_sample:
        print(f"\n[3] Generating sample grids...")
        t0 = time.time()
        generator.eval()
        save_viewpoint_grid(generator, evaluator, zdist, cached_hs,
                            eval_dir, it, device)
        generator.train()
        print(f"    Sample generation took {time.time() - t0:.1f}s")

    # =================================================================
    # 4. Save results
    # =================================================================
    csv_path = os.path.join(eval_dir, f'metrics_it{it}.csv')
    write_csv(csv_path, metrics, per_sample, per_spec_metrics,
              fid=fid, kid=kid, fid_results=fid_results, kid_results=kid_results)

    if per_spec_metrics:
        save_comparison_plots(per_spec_metrics, eval_dir, it)

    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"All results in: {eval_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()