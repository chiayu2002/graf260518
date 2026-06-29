"""
Generate rotation / elevation videos from a trained Two-Stage GRAF model.
Uses the same data pipeline and create_samples(z, label, hidden_states, poses)
as eval_twostage.py.

Supports:
  - NeRF 64×64 output
  - SR 256×256 output (with --sr flag, if SR checkpoint exists)
  - Hidden-state interpolation grids
"""
import argparse
import glob
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')
sys.path.append('submodules')

from submodules.GAN_stability.gan_training.checkpoints_mod import CheckpointIO
from graf.gan_training import Evaluator
from graf.config import get_data, build_models, load_config
from graf.utils import get_zdist
from graf.models.twostage_networks import SRNetwork
import imageio


# ─────────────────────────────────────────────────────────────
# Specimen definitions: (AR/LR/TR 7-dim, mat_feat 7-dim)
# Label = vec + mat_feat + [height_idx, angle_idx] = 16-dim
# ─────────────────────────────────────────────────────────────
SPECIMENS = {
    'RS307': {
        'vec':      [1, 0, 1, 0, 0, 0, 1],
        'mat_feat': [0.000000, 0.156717, 0.918975, 0.339361, 0.498170, 0.310937, 0.360617],
    },
    'RS330': {
        'vec':      [1, 0, 0, 0, 1, 1, 0],
        'mat_feat': [0.008831, 1.000000, 1.000000, 1.000000, 0.000000, 1.000000, 1.000000],
    },
    'RS615': {
        'vec':      [0, 1, 0, 1, 0, 1, 0],
        'mat_feat': [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000],
    },
    'RS315': {
        'vec':      [1, 0, 0, 1, 0, 1, 0],
        'mat_feat': [0.006158, 0.411209, 0.538894, 0.725360, 0.017321, 0.727661, 0.751007],
    },
}

def make_label(spec_name, height_idx=0, angle_idx=0, N=1, device='cuda'):
    """Build a 16-dim label tensor for a specimen.
    Returns: [N, 16] float tensor.
    """
    s = SPECIMENS[spec_name]
    row = s['vec'] + s['mat_feat'] + [float(height_idx), float(angle_idx)]
    return torch.tensor([row] * N, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────
# Video helpers
# ─────────────────────────────────────────────────────────────
def save_video(imgs, fname, as_gif=False, fps=24, quality=8):
    """Save a tensor of images [N, C, H, W] as mp4 video."""
    imgs_np = (255 * np.clip(
        imgs.permute(0, 2, 3, 1).detach().cpu().numpy() / 2 + 0.5, 0, 1)
    ).astype(np.uint8)

    writer = imageio.get_writer(fname, fps=fps, quality=quality,
                                format='FFMPEG')
    for frame in imgs_np:
        writer.append_data(frame)
    writer.close()

    if as_gif:
        gif_name = os.path.splitext(fname)[0] + ".gif"
        os.system(
            f'ffmpeg -y -i {fname} -r 15 '
            f'-vf "scale=512:-1,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" '
            f'{gif_name}')
    print(f'  Saved: {fname}')


def make_video_with_hs(evaluator, basename, z, label, hidden_states, poses,
                       sr_network=None, as_gif=False):
    """
    Render video frame by frame.

    z             : (N_samples, z_dim)
    label         : (N_samples, 16)
    hidden_states : (N_samples, hs_dim)
    poses         : (N_frames, 3, 4)
    sr_network    : optional SRNetwork for 256×256 output
    """
    from tqdm import tqdm as _tqdm

    N_samples, N_frames = len(z), len(poses)
    all_rgbs, all_depths = [], []
    all_sr = [] if sr_network is not None else None

    for f_idx in _tqdm(range(N_frames), desc='Rendering frames'):
        pose_f = poses[f_idx:f_idx + 1].expand(N_samples, -1, -1)

        rgb_f, depth_f, _ = evaluator.create_samples(
            z, label, hidden_states, poses=pose_f)
        all_rgbs.append(rgb_f.cpu())
        all_depths.append(depth_f.cpu())

        if sr_network is not None:
            with torch.no_grad():
                r4sr = rgb_f.to(z.device)
                if r4sr.shape[2] != 64:
                    r4sr = F.interpolate(r4sr, size=(64, 64), mode='bilinear')
                sr_out = sr_network(r4sr)
            all_sr.append(sr_out.cpu())

    rgbs = torch.stack(all_rgbs, dim=1)     # (N_samples, N_frames, C, H, W)
    depths = torch.stack(all_depths, dim=1)
    print(f'Done, saving {rgbs.shape}')

    fps = min(int(N_frames / 2.), 25)
    for i in range(N_samples):
        save_video(rgbs[i],   basename + f'{i:04d}_rgb.mp4',   as_gif=as_gif, fps=fps)
        save_video(depths[i], basename + f'{i:04d}_depth.mp4', as_gif=as_gif, fps=fps)

    if all_sr is not None:
        sr_stack = torch.stack(all_sr, dim=1)
        for i in range(N_samples):
            save_video(sr_stack[i], basename + f'{i:04d}_sr256.mp4',
                       as_gif=as_gif, fps=fps)


# ─────────────────────────────────────────────────────────────
# Hidden-state interpolation
# ─────────────────────────────────────────────────────────────
def lerp(hs_a, hs_b, t):
    return (1 - t) * hs_a + t * hs_b


def eval_hs_interpolation(generator, evaluator, zdist, cached_hs,
                          sr_network, eval_dir, device, N_steps=11):
    """
    Hidden-state interpolation: grid image per specimen pair.
    Rows = interpolation steps (t=0→1), Columns = viewpoints.
    """
    from itertools import combinations
    from torchvision.utils import save_image as _save_image
    from tqdm import tqdm as _tqdm

    interp_dir = os.path.join(eval_dir, 'hs_interpolation')
    os.makedirs(interp_dir, exist_ok=True)

    available = [name for name in SPECIMENS if name in cached_hs]
    pairs = list(combinations(available, 2))
    print(f"  Specimens: {available}")
    print(f"  Pairs: {pairs}")

    z = zdist.sample((1,))

    view_angles = [
        (0.0, 0.5),    # front
        (0.25, 0.5),   # right 90°
        (0.5, 0.5),    # back 180°
    ]
    N_views = len(view_angles)
    ts = np.linspace(0.0, 1.0, N_steps)

    for name_a, name_b in pairs:
        print(f"\n  {name_a} → {name_b}  ({N_steps} steps × {N_views} views)")
        hs_a = cached_hs[name_a]
        hs_b = cached_hs[name_b]

        # Use name_a's label as base (angle/height are dummy here)
        label = make_label(name_a, height_idx=0, angle_idx=0, N=1, device=device)

        all_nerf = []
        all_sr = []
        for t in _tqdm(ts, desc=f'    interp'):
            hs_t = lerp(hs_a.unsqueeze(0), hs_b.unsqueeze(0), t)

            for u, v in view_angles:
                pose = generator.sample_select_pose(u, v).unsqueeze(0)
                with torch.no_grad():
                    rgb, _, _ = evaluator.create_samples(z, label, hs_t, poses=pose)
                all_nerf.append(rgb.cpu())

                if sr_network is not None:
                    with torch.no_grad():
                        r4sr = rgb.to(device)
                        if r4sr.shape[2] != 64:
                            r4sr = F.interpolate(r4sr, size=(64, 64), mode='bilinear')
                        sr_out = sr_network(r4sr)
                    all_sr.append(sr_out.cpu())

        # Save NeRF grid
        grid = torch.cat(all_nerf, dim=0)
        grid_01 = (grid / 2 + 0.5).clamp(0, 1)
        out_path = os.path.join(interp_dir, f'{name_a}_to_{name_b}_nerf.png')
        _save_image(grid_01, out_path, nrow=N_views)
        print(f'    Saved: {out_path}')

        # Save SR grid
        if all_sr:
            grid_sr = torch.cat(all_sr, dim=0)
            grid_sr_01 = (grid_sr / 2 + 0.5).clamp(0, 1)
            out_sr = os.path.join(interp_dir, f'{name_a}_to_{name_b}_sr256.png')
            _save_image(grid_sr_01, out_sr, nrow=N_views)
            print(f'    Saved: {out_sr}')

        torch.cuda.empty_cache()

    print(f"\n  Results in: {interp_dir}/")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Two-Stage GRAF - Video Generation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory (default: <checkpoint_dir>/eval_videos)')
    parser.add_argument('--sr', action='store_true',
                        help='Also render SR 256×256 videos (requires SR in checkpoint)')
    parser.add_argument('--rotation', action='store_true',
                        help='Generate horizontal rotation video (fixed elevation)')
    parser.add_argument('--elevation', action='store_true',
                        help='Generate elevation sweep video (fixed azimuth)')
    parser.add_argument('--rotation_elevation', action='store_true',
                        help='Generate both rotation and elevation videos')
    parser.add_argument('--N_frames', type=int, default=40,
                        help='Number of frames in video')
    parser.add_argument('--as_gif', action='store_true',
                        help='Also output gif format')
    parser.add_argument('--interpolate_hs', action='store_true',
                        help='Generate hidden-state interpolation grids')
    parser.add_argument('--interp_steps', type=int, default=11,
                        help='Interpolation steps (including endpoints)')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not (args.rotation or args.elevation or args.rotation_elevation
            or args.interpolate_hs):
        args.rotation_elevation = True

    # ================================================================
    # 1. Load config
    # ================================================================
    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])
    config['training']['nworkers'] = 0
    batch_size = config['training']['batch_size']
    device = torch.device('cuda:0')

    # ================================================================
    # 2. Load GRU extractor
    # ================================================================
    print("Loading GRU extractor...")
    extractor_path = config['data']['extractor_path']
    extractor_args = json.load(
        open("HystereticGRU/2026-03-24_17-13-01/args.json", "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module(
        "graf.models.HystereticPrediction"
    ).__dict__[extractor_args.architecture](**vars(extractor_args))
    extractor = extractor.to(device)
    state_dict = torch.load(
        glob.glob(extractor_path, recursive=True)[0])["state_dict"]
    status = extractor.load_state_dict(state_dict)
    print(f"Extractor loaded: {status}")

    # ================================================================
    # 3. Load dataset (for hwfr and cached hidden states)
    # ================================================================
    train_dataset, hwfr = get_data(config, extractor, extractor_args)
    config['data']['hwfr'] = hwfr
    print(f"Dataset: {len(train_dataset)} images, HW={hwfr[0]}x{hwfr[1]}")

    cached_hs = {n: h.to(device) for n, h in train_dataset.hidden_state.items()}
    print(f"Cached hidden states: {list(cached_hs.keys())}")

    # ================================================================
    # 4. Build generator + SR, load checkpoint
    # ================================================================
    generator, _ = build_models(config, disc=False)
    generator = generator.to(device)

    sr_network = SRNetwork(ch=64, n_rb=6).to(device) if args.sr else None

    checkpoint_io = CheckpointIO(
        checkpoint_dir=os.path.dirname(args.checkpoint))
    checkpoint_io.register_modules(**generator.module_dict)
    if sr_network is not None:
        checkpoint_io.register_modules(sr_network=sr_network)

    print(f"Registered: {list(checkpoint_io.module_dict.keys())}")
    load_dict = checkpoint_io.load(os.path.basename(args.checkpoint))
    it = load_dict.get('it', -1)
    print(f"Loaded checkpoint iter={it}")

    if sr_network is not None:
        if 'sr_network' in load_dict:
            print("[SR] Weights loaded from checkpoint")
        else:
            print("[SR] WARNING: sr_network not found in checkpoint, using random weights!")
            print("[SR] Videos will be meaningless. Use --sr only with Phase 2 checkpoints.")

    generator.eval()
    if sr_network is not None:
        sr_network.eval()

    # ================================================================
    # 5. Setup evaluator / zdist
    # ================================================================
    zdist = get_zdist(config['z_dist']['type'],
                      config['z_dist']['dim'], device=device)
    evaluator = Evaluator(False, generator, zdist, None,
                          batch_size=batch_size, device=device)

    # Output directory
    if args.outdir is None:
        eval_dir = os.path.join(os.path.dirname(args.checkpoint), 'eval_videos')
    else:
        eval_dir = args.outdir
    os.makedirs(eval_dir, exist_ok=True)

    N_frames = args.N_frames

    # ================================================================
    # 6. Generate videos per specimen
    # ================================================================
    for spec_name in SPECIMENS:
        if spec_name not in cached_hs:
            print(f"  [SKIP] {spec_name} not in cached_hs")
            continue

        print(f"\n{'=' * 50}")
        print(f"  Specimen: {spec_name}")
        print(f"{'=' * 50}")

        N_samples = 1
        z = zdist.sample((N_samples,))
        hs = cached_hs[spec_name].unsqueeze(0).expand(N_samples, -1)
        label = make_label(spec_name, height_idx=0, angle_idx=0,
                           N=N_samples, device=device)

        # ── Horizontal rotation video ──
        if args.rotation or args.rotation_elevation:
            print(f'\n  --- {spec_name}: Rotation video ---')
            poses = torch.stack([
                generator.sample_select_pose(i / N_frames, 0.5)
                for i in range(N_frames)])

            outpath = os.path.join(eval_dir, f'{spec_name}_rotation_v05/')
            os.makedirs(outpath, exist_ok=True)
            make_video_with_hs(evaluator, outpath, z, label, hs, poses,
                               sr_network=sr_network, as_gif=args.as_gif)
            torch.cuda.empty_cache()

        # ── Elevation sweep video ──
        if args.elevation or args.rotation_elevation:
            print(f'\n  --- {spec_name}: Elevation video ---')
            vmin = config['data'].get('vmin', 0)
            vmax = config['data'].get('vmax', 1)

            poses = torch.stack([
                generator.sample_select_pose(0.0, vmin + (vmax - vmin) * i / N_frames)
                for i in range(N_frames)])

            outpath = os.path.join(eval_dir, f'{spec_name}_elevation/')
            os.makedirs(outpath, exist_ok=True)
            make_video_with_hs(evaluator, outpath, z, label, hs, poses,
                               sr_network=sr_network, as_gif=args.as_gif)
            torch.cuda.empty_cache()

    # ================================================================
    # 7. Hidden-state interpolation
    # ================================================================
    if args.interpolate_hs:
        print(f"\n{'=' * 50}")
        print(f"  Hidden-state interpolation ({args.interp_steps} steps)")
        print(f"{'=' * 50}")
        eval_hs_interpolation(
            generator, evaluator, zdist, cached_hs,
            sr_network, eval_dir, device,
            N_steps=args.interp_steps,
        )

    print(f'\n{"=" * 50}')
    print(f'All done!')
    print(f'Output: {eval_dir}')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    main()