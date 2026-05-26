"""
Generate rotation / elevation videos from a trained GRAF model (GRU version).
Uses the same data pipeline and create_samples(z, label, hidden_states, poses)
as the main eval.py.
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
from graf.config import get_data, build_models, load_config, get_render_poses
from graf.utils import get_zdist, to_phi, to_theta
import imageio


def save_video(imgs, fname, as_gif=False, fps=24, quality=8):
    """save_video 本地版：明確使用 imageio-ffmpeg 避免 tifffile plugin 錯誤。"""
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


# ─────────────────────────────────────────────────────────────
# Custom make_video：支援 hidden_states 參數
# ─────────────────────────────────────────────────────────────
def make_video_with_hs(evaluator, basename, z, label, hidden_states, poses, as_gif=False):
    """
    逐幀渲染影片，避免 create_samples 內部 batch split 導致
    z / label / hidden_state 尺寸不對齊。

    z             : (N_samples, z_dim)
    label         : (N_samples, 9)          ← 7 specimen props + h_idx + a_idx
    hidden_states : (N_samples, hs_dim)     ← GRU hidden state
    poses         : (N_frames, 3, 4)
    """
    from tqdm import tqdm as _tqdm

    N_samples, N_frames = len(z), len(poses)
    all_rgbs, all_depths = [], []

    for f_idx in _tqdm(range(N_frames), desc='Rendering frames'):
        # 每幀：z, label, hs 維持 (N_samples, ...) ，pose 只取第 f_idx 幀
        pose_f = poses[f_idx:f_idx+1]                       # (1, 3, 4)
        pose_f = pose_f.expand(N_samples, -1, -1)           # (N_samples, 3, 4)

        rgb_f, depth_f, _ = evaluator.create_samples(
            z, label, hidden_states, poses=pose_f)           # each (N_samples, C, H, W)
        all_rgbs.append(rgb_f.cpu())
        all_depths.append(depth_f.cpu())

    # (N_samples, N_frames, C, H, W)
    rgbs   = torch.stack(all_rgbs,  dim=1)
    depths = torch.stack(all_depths, dim=1)
    print(f'Done, saving {rgbs.shape}')

    fps = min(int(N_frames / 2.), 25)
    for i in range(N_samples):
        save_video(rgbs[i],   basename + f'{i:04d}_rgb.mp4',   as_gif=as_gif, fps=fps)
        save_video(depths[i], basename + f'{i:04d}_depth.mp4', as_gif=as_gif, fps=fps)


# ─────────────────────────────────────────────────────────────
# Hidden-state interpolation
# ─────────────────────────────────────────────────────────────
def lerp(hs_a, hs_b, t):
    """Linear interpolation: (1-t)*a + t*b"""
    return (1 - t) * hs_a + t * hs_b


def eval_hs_interpolation(generator, evaluator, zdist, cached_hs,
                          specimens, eval_dir, device, N_steps=11):
    """
    Hidden-state 內插：每對 specimen 生成一張 grid 圖。
    每一列是同一個視角，每一行是不同的 t (0→1)。

    輸出範例 (RS307→RS615, N_steps=11, 3 views):
        row 0: t=0.0 (= RS307)  view0  view1  view2
        row 1: t=0.1            view0  view1  view2
        ...
        row 10: t=1.0 (= RS615) view0  view1  view2
    """
    from itertools import combinations
    from torchvision.utils import save_image as _save_image
    from tqdm import tqdm as _tqdm

    interp_dir = os.path.join(eval_dir, 'hs_interpolation')
    os.makedirs(interp_dir, exist_ok=True)

    available = [name for name in specimens if name in cached_hs]
    pairs = list(combinations(available, 2))
    print(f"  Specimens: {available}")
    print(f"  Pairs: {pairs}")

    # 固定 z（所有 pair 共用）
    z = zdist.sample((1,))

    # 3 個代表性視角
    view_angles = [
        (0.0,  0.5),   # 正面
        (0.25, 0.5),   # 右側 90°
        (0.5,  0.5),   # 背面 180°
    ]
    N_views = len(view_angles)
    ts = np.linspace(0.0, 1.0, N_steps)

    for name_a, name_b in pairs:
        print(f"\n  {name_a} → {name_b}  ({N_steps} steps × {N_views} views)")
        hs_a = cached_hs[name_a]  # (hs_dim,)
        hs_b = cached_hs[name_b]

        label_7d = specimens[name_a]
        label = torch.tensor(
            [label_7d + [0.5, 0.0]], dtype=torch.float32, device=device)

        all_imgs = []  # 收集所有圖片，最後一次存
        for t in _tqdm(ts, desc=f'    interp'):
            hs_t = lerp(hs_a.unsqueeze(0), hs_b.unsqueeze(0), t)

            for u, v in view_angles:
                pose = generator.sample_select_pose(u, v).unsqueeze(0)
                with torch.no_grad():
                    rgb, _, _ = evaluator.create_samples(
                        z, label, hs_t, poses=pose)
                all_imgs.append(rgb.cpu())

        # (N_steps * N_views, C, H, W) → grid: nrow=N_views 讓每列=同一個 t
        grid = torch.cat(all_imgs, dim=0)
        grid_01 = (grid / 2 + 0.5).clamp(0, 1)

        out_path = os.path.join(interp_dir, f'{name_a}_to_{name_b}.png')
        _save_image(grid_01, out_path, nrow=N_views)
        print(f'    Saved: {out_path}')
        torch.cuda.empty_cache()

    print(f"\n  Results in: {interp_dir}/")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='GRAF (GRU) - Video Generation')
    parser.add_argument('--config', type=str,
                        default='./results/column20260417_gru_disc_Dbn_lrg3_aux_sampling_speed/config.yaml')
    parser.add_argument('--checkpoint', type=str,
                        default='./results/column20260417_gru_disc_Dbn_lrg3_aux_sampling_speed/chkpts/model_00209999.pt')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory (default: <checkpoint_dir>/eval_videos)')
    parser.add_argument('--rotation', action='store_true',
                        help='生成水平旋轉影片 (固定仰角，環繞一圈)')
    parser.add_argument('--elevation', action='store_true',
                        help='生成仰角變化影片 (固定水平角，改變仰角)')
    parser.add_argument('--rotation_elevation', action='store_true',
                        help='同時生成旋轉和仰角影片')
    parser.add_argument('--N_frames', type=int, default=40,
                        help='Number of frames in video')
    parser.add_argument('--as_gif', action='store_true',
                        help='同時輸出 gif 格式')
    parser.add_argument('--interpolate_hs', action='store_true',
                        help='生成 hidden state 內插 grid 圖')
    parser.add_argument('--interp_steps', type=int, default=11,
                        help='內插步數 (含兩端點，e.g. 11 → t=0.0,0.1,...,1.0)')
    parser.add_argument('--gpu', type=str, default='4')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 預設至少要有一種模式
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
    # 2. Load GRU extractor  (跟 eval.py 一模一樣)
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
    # 3. Load dataset (為了取得 hwfr 和 cached hidden states)
    # ================================================================
    train_dataset, hwfr = get_data(config, extractor, extractor_args)
    config['data']['hwfr'] = hwfr
    print(f"Dataset: {len(train_dataset)} images, HW={hwfr[0]}x{hwfr[1]}")

    cached_hs = {}
    for exp_name, hs in train_dataset.hidden_state.items():
        cached_hs[exp_name] = hs.to(device)
    print(f"Cached hidden states: {list(cached_hs.keys())}")

    # ================================================================
    # 4. Build generator & load checkpoint
    # ================================================================
    generator, _ = build_models(config, disc=False)
    generator = generator.to(device)

    checkpoint_io = CheckpointIO(
        checkpoint_dir=os.path.dirname(args.checkpoint))
    checkpoint_io.register_modules(**generator.module_dict)
    load_dict = checkpoint_io.load(os.path.basename(args.checkpoint))
    it = load_dict.get('it', -1)
    print(f"Loaded checkpoint iter={it}")

    generator.eval()

    # ================================================================
    # 5. Setup evaluator / zdist
    # ================================================================
    zdist = get_zdist(config['z_dist']['type'],
                      config['z_dist']['dim'], device=device)
    evaluator = Evaluator(False, generator, zdist, None,
                          batch_size=batch_size, device=device)

    v_list = [float(x.strip()) for x in config['data']['v'].split(',')]

    # 輸出目錄
    if args.outdir is None:
        eval_dir = os.path.join(os.path.dirname(args.checkpoint), 'eval_videos')
    else:
        eval_dir = args.outdir
    os.makedirs(eval_dir, exist_ok=True)

    # render radius
    render_radius = config['data']['radius']
    if isinstance(render_radius, str):
        render_radius = float(render_radius.split(',')[1])
    elif isinstance(render_radius, tuple):
        render_radius = render_radius[1]

    umin = config['data'].get('umin', 0)
    umax = config['data'].get('umax', 1)
    N_frames = args.N_frames

    # ================================================================
    # 6. 為每個 specimen 生成影片
    # ================================================================
    # specimen label 定義 (跟 eval.py save_viewpoint_grid 一致)
    specimens = {
        'RS307': [1.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0],
        'RS330': [1.0, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0],
        'RS615': [0.0, 1.0,  0.0, 1.0, 0.0,  1.0, 0.0],
        'RS315': [1.0, 0.0,  0.0, 1.0, 0.0,  1.0, 0.0],
    }

    for spec_name, label_7d in specimens.items():
        if spec_name not in cached_hs:
            print(f"  [SKIP] {spec_name} not in cached_hs")
            continue

        print(f"\n{'='*50}")
        print(f"  Specimen: {spec_name}")
        print(f"{'='*50}")

        N_samples = 1  # 每個 specimen 生成 1 個 sample
        z = zdist.sample((N_samples,))
        hs = cached_hs[spec_name].unsqueeze(0).expand(N_samples, -1)  # (1, hs_dim)

        # label: 7d + [v_idx=0.5, angle=0.0] (angle 在影片裡會由 poses 決定，這裡只是佔位)
        label = torch.tensor(
            [label_7d + [0.5, 0.0]] * N_samples,
            dtype=torch.float32, device=device)

        # ── 水平旋轉影片 ──
        if args.rotation or args.rotation_elevation:
            print(f'\n  --- {spec_name}: 水平旋轉影片 ---')
            plist = []
            for i in range(N_frames):
                u = i / N_frames
                v = 0.5
                plist.append(generator.sample_select_pose(u, v))
            poses = torch.stack(plist)

            outpath = os.path.join(eval_dir, f'{spec_name}_rotation/')
            os.makedirs(outpath, exist_ok=True)
            make_video_with_hs(evaluator, outpath, z, label, hs, poses,
                               as_gif=args.as_gif)
            torch.cuda.empty_cache()
            print(f'    Saved to: {outpath}')

        # ── 仰角變化影片 ──
        if args.elevation or args.rotation_elevation:
            print(f'\n  --- {spec_name}: 仰角變化影片 ---')
            vmin = config['data'].get('vmin', 0)
            vmax = config['data'].get('vmax', 1)

            plist = []
            for i in range(N_frames):
                u = 0.0  # 固定正前方
                v = vmin + (vmax - vmin) * i / N_frames
                plist.append(generator.sample_select_pose(u, v))
            poses = torch.stack(plist)

            outpath = os.path.join(eval_dir, f'{spec_name}_elevation/')
            os.makedirs(outpath, exist_ok=True)
            make_video_with_hs(evaluator, outpath, z, label, hs, poses,
                               as_gif=args.as_gif)
            torch.cuda.empty_cache()
            print(f'    Saved to: {outpath}')

    # ================================================================
    # 7. Hidden-state 內插
    # ================================================================
    if args.interpolate_hs:
        print(f"\n{'='*50}")
        print(f"  Hidden-state interpolation ({args.interp_steps} steps)")
        print(f"{'='*50}")
        eval_hs_interpolation(
            generator, evaluator, zdist, cached_hs,
            specimens, eval_dir, device,
            N_steps=args.interp_steps,
        )

    print(f'\n{"="*50}')
    print(f'所有生成完成！')
    print(f'輸出目錄: {eval_dir}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()