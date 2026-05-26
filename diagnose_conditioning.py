"""
診斷腳本：驗證 RS315 vs RS330 生成結果相同的原因
=================================================

測試項目：
  1. Representation 分析 — hidden state / bottleneck / decoded conditioning 的距離
  2. Swap test — 交叉 hidden state 與 mat_feat，看哪個通道真正影響輸出
  3. Random hidden state — 隨機 1024-dim 向量是否也產生一樣的輸出（G 是否忽略 conditioning）
  4. Mat_feat isolation — 將 mat_feat 歸零或替換，觀察輸出變化
  5. Bottleneck interpolation sweep — 在 bottleneck 空間內插，檢查輸出是否連續變化
  6. Override test — 用 set_override 注入不同 conditioning，確認 override 路徑是否生效

用法：
  python diagnose_conditioning.py --config configs/twostage.yaml --checkpoint path/to/model.pt
"""
import argparse
import os
import sys
import json
import glob
import importlib

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

sys.path.append('submodules')

from graf.config import get_data, build_models, load_config
from graf.utils import get_zdist
from graf.gan_training import Evaluator
from graf.models.twostage_networks import nerf_flat_to_img


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_all(config_path, checkpoint_path, device):
    """載入 config、extractor、dataset、generator、checkpoint。"""
    config = load_config(config_path)
    config['data']['fov'] = float(config['data']['fov'])
    config['training']['nworkers'] = 0

    # Extractor
    extractor_path = config['data']['extractor_path']
    extractor_args = json.load(open(glob.glob(
        "HystereticGRU/2026-03-24_17-13-01/args.json", recursive=True)[0], "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module(
        "graf.models.HystereticPrediction"
    ).__dict__[extractor_args.architecture](**vars(extractor_args))
    extractor = extractor.to(device)
    state_dict = torch.load(glob.glob(extractor_path, recursive=True)[0])["state_dict"]
    extractor.load_state_dict(state_dict)

    # Dataset & Generator
    train_dataset, hwfr = get_data(config, extractor, extractor_args)
    config['data']['hwfr'] = hwfr
    generator, _ = build_models(config, disc=False)
    generator = generator.to(device)

    # Checkpoint
    from GAN_stability.gan_training.checkpoints_mod import CheckpointIO
    ckpt_dir = os.path.dirname(checkpoint_path)
    checkpoint_io = CheckpointIO(checkpoint_dir=ckpt_dir)
    checkpoint_io.register_modules(**generator.module_dict)
    load_dict = checkpoint_io.load(os.path.basename(checkpoint_path))
    it = load_dict.get('it', -1)
    print(f"[載入] checkpoint iter={it}")

    # Cached hidden states
    cached_hs = {n: h.to(device) for n, h in train_dataset.hidden_state.items()}

    v_list = [float(x.strip()) for x in config['data']['v'].split(",")]
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    return config, generator, cached_hs, v_list, zdist, train_dataset


# ================================================================
# Specimen 定義
# ================================================================
SPECIMEN_SPECS = {
    'RS307': ([1, 0, 1, 0, 0, 0, 1], [0.000000, 0.156717, 0.918975, 0.339361, 0.498170, 0.310937, 0.360617]),
    'RS330': ([1, 0, 0, 0, 1, 1, 0], [0.008831, 1.000000, 1.000000, 1.000000, 0.000000, 1.000000, 1.000000]),
    'RS615': ([0, 1, 0, 1, 0, 1, 0], [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000]),
    'RS315': ([1, 0, 0, 1, 0, 1, 0], [0.006158, 0.411209, 0.538894, 0.725360, 0.017321, 0.727661, 0.751007]),
}


def make_label(vec, mat, h_idx=1, a_idx=0):
    """建立 [vec(7), mat(7), h_idx(1), a_idx(1)] = 16-dim label。"""
    return vec + mat + [float(h_idx), float(a_idx)]


def render_specimen(generator, evaluator, zdist, cached_hs, specimen_name, device,
                    override_hs=None, override_mat=None, override_vec=None,
                    n_views=8, z_fixed=None):
    """
    渲染指定 specimen 的 n_views 張圖。
    可選 override hidden state / mat_feat / label vec。
    回傳 [n_views, 3, H, W] 的 tensor。
    """
    vec_orig, mat_orig = SPECIMEN_SPECS[specimen_name]
    vec = override_vec if override_vec is not None else vec_orig
    mat = override_mat if override_mat is not None else mat_orig

    hs = override_hs if override_hs is not None else cached_hs[specimen_name]

    angles = [i * 360 // n_views for i in range(n_views)]
    labels_list = []
    for a in angles:
        labels_list.append(make_label(vec, mat, h_idx=1, a_idx=a))
    label_t = torch.tensor(labels_list, dtype=torch.float32, device=device)
    hs_t = hs.unsqueeze(0).expand(n_views, -1)

    poses = torch.stack([
        generator.sample_select_pose(a / 360.0, 0.5) for a in angles
    ])

    z = z_fixed if z_fixed is not None else zdist.sample((n_views,))

    with torch.no_grad():
        rgb, _, _ = evaluator.create_samples(z.to(device), label_t, hs_t, poses)
    return rgb  # [n_views, 3, H, W]


def pixel_diff(img_a, img_b):
    """計算兩組圖片的 L2、MAE、max-abs 差異。"""
    l2 = (img_a - img_b).pow(2).mean().item()
    mae = (img_a - img_b).abs().mean().item()
    max_abs = (img_a - img_b).abs().max().item()
    return {'MSE': l2, 'MAE': mae, 'MaxAbs': max_abs}


def save_grid(images_dict, path, nrow=8):
    """
    images_dict: {name: [N, 3, H, W]} — 按行排列。
    """
    rows = []
    captions = []
    for name, imgs in images_dict.items():
        rows.append(imgs.cpu())
        captions.append(name)
    all_imgs = torch.cat(rows, dim=0)
    grid = vutils.make_grid(all_imgs, nrow=nrow, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, path)
    print(f"  儲存: {path}  (rows: {', '.join(captions)})")


# ================================================================
# 測試函數
# ================================================================

def test_1_representation(cached_hs, nerf_model, device):
    """分析所有 specimen pair 在不同空間的距離。"""
    print("\n" + "=" * 70)
    print("TEST 1: Representation 分析")
    print("=" * 70)

    cond_proj = nerf_model.condition_feature
    has_bottleneck = hasattr(cond_proj, 'encode')

    names = sorted(cached_hs.keys())
    print(f"\n{'Pair':<18} | {'HS L2':<10} | {'HS Cos':<10}", end="")
    if has_bottleneck:
        print(f" | {'Bneck L2':<10} | {'Bneck Cos':<10} | {'Cond L2':<10} | {'Cond Cos':<10}", end="")
    print()
    print("-" * (80 if has_bottleneck else 42))

    with torch.no_grad():
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                h1, h2 = cached_hs[n1], cached_hs[n2]
                l2 = (h1 - h2).norm().item()
                cos = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
                print(f"  {n1} vs {n2:<6} | {l2:<10.4f} | {cos:<10.4f}", end="")

                if has_bottleneck:
                    b1 = cond_proj.encode(h1.unsqueeze(0))
                    b2 = cond_proj.encode(h2.unsqueeze(0))
                    bl2 = (b1 - b2).norm().item()
                    bcos = F.cosine_similarity(b1, b2).item()

                    c1 = cond_proj(h1.unsqueeze(0))
                    c2 = cond_proj(h2.unsqueeze(0))
                    cl2 = (c1 - c2).norm().item()
                    ccos = F.cosine_similarity(c1, c2).item()
                    print(f" | {bl2:<10.4f} | {bcos:<10.4f} | {cl2:<10.4f} | {ccos:<10.4f}", end="")
                print()

    # Mat feat projection 分析
    if hasattr(nerf_model, 'mat_feat_proj'):
        print(f"\n  Mat feat projection 分析:")
        with torch.no_grad():
            for i, n1 in enumerate(names):
                for n2 in names[i + 1:]:
                    if n1 not in SPECIMEN_SPECS or n2 not in SPECIMEN_SPECS:
                        continue
                    m1 = torch.tensor(SPECIMEN_SPECS[n1][1], dtype=torch.float32, device=device)
                    m2 = torch.tensor(SPECIMEN_SPECS[n2][1], dtype=torch.float32, device=device)
                    p1 = nerf_model.mat_feat_proj(m1.unsqueeze(0))
                    p2 = nerf_model.mat_feat_proj(m2.unsqueeze(0))
                    ml2 = (p1 - p2).norm().item()
                    mcos = F.cosine_similarity(p1, p2).item()
                    raw_l2 = (m1 - m2).norm().item()
                    print(f"    {n1} vs {n2}: raw_mat L2={raw_l2:.4f}  "
                          f"proj L2={ml2:.4f}  proj cos={mcos:.4f}")


def test_2_swap(generator, evaluator, zdist, cached_hs, device, out_dir):
    """交叉 hidden state 與 mat_feat，看哪個通道影響輸出。"""
    print("\n" + "=" * 70)
    print("TEST 2: Swap Test (RS315 vs RS330)")
    print("=" * 70)

    z_fixed = zdist.sample((8,))
    targets = ['RS315', 'RS330']
    results = {}

    # 原始
    for sn in targets:
        results[f'{sn}_orig'] = render_specimen(
            generator, evaluator, zdist, cached_hs, sn, device, z_fixed=z_fixed)

    # Swap hidden state: RS315 的 hs + RS330 的 label/mat
    results['RS315hs_RS330mat'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS330', device,
        override_hs=cached_hs['RS315'], z_fixed=z_fixed)

    # Swap hidden state: RS330 的 hs + RS315 的 label/mat
    results['RS330hs_RS315mat'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS315', device,
        override_hs=cached_hs['RS330'], z_fixed=z_fixed)

    # Swap mat_feat only
    results['RS315_mat330'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS315', device,
        override_mat=SPECIMEN_SPECS['RS330'][1], z_fixed=z_fixed)
    results['RS330_mat315'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS330', device,
        override_mat=SPECIMEN_SPECS['RS315'][1], z_fixed=z_fixed)

    # Swap vec only
    results['RS315_vec330'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS315', device,
        override_vec=SPECIMEN_SPECS['RS330'][0], z_fixed=z_fixed)
    results['RS330_vec315'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS330', device,
        override_vec=SPECIMEN_SPECS['RS315'][0], z_fixed=z_fixed)

    # 比較
    print(f"\n  {'Comparison':<40} | {'MSE':<12} | {'MAE':<12} | {'MaxAbs':<12}")
    print("  " + "-" * 80)

    comparisons = [
        ("RS315_orig vs RS330_orig",           'RS315_orig', 'RS330_orig'),
        ("RS315_orig vs RS315hs+RS330mat",     'RS315_orig', 'RS315hs_RS330mat'),
        ("RS330_orig vs RS330hs+RS315mat",     'RS330_orig', 'RS330hs_RS315mat'),
        ("RS315_orig vs RS315+mat330",         'RS315_orig', 'RS315_mat330'),
        ("RS330_orig vs RS330+mat315",         'RS330_orig', 'RS330_mat315'),
        ("RS315_orig vs RS315+vec330",         'RS315_orig', 'RS315_vec330'),
        ("RS330_orig vs RS330+vec315",         'RS330_orig', 'RS330_vec315'),
        ("RS315hs+RS330mat vs RS330_orig",     'RS315hs_RS330mat', 'RS330_orig'),
        ("RS330hs+RS315mat vs RS315_orig",     'RS330hs_RS315mat', 'RS315_orig'),
    ]
    for desc, a, b in comparisons:
        d = pixel_diff(results[a], results[b])
        print(f"  {desc:<40} | {d['MSE']:<12.6f} | {d['MAE']:<12.6f} | {d['MaxAbs']:<12.6f}")

    # 儲存圖片
    save_grid(results, os.path.join(out_dir, 'test2_swap.png'))


def test_3_random_hs(generator, evaluator, zdist, cached_hs, device, out_dir):
    """用隨機 hidden state 測試 G 是否完全忽略 conditioning。"""
    print("\n" + "=" * 70)
    print("TEST 3: Random Hidden State（G 是否忽略 conditioning?）")
    print("=" * 70)

    z_fixed = zdist.sample((8,))
    results = {}

    # 原始 RS315, RS330
    for sn in ['RS315', 'RS330']:
        results[f'{sn}_orig'] = render_specimen(
            generator, evaluator, zdist, cached_hs, sn, device, z_fixed=z_fixed)

    # 隨機 hidden state (3 組不同的隨機向量)
    for trial in range(3):
        rand_hs = torch.randn(1024, device=device)
        results[f'RS315_randHS_{trial}'] = render_specimen(
            generator, evaluator, zdist, cached_hs, 'RS315', device,
            override_hs=rand_hs, z_fixed=z_fixed)

    # 零向量 hidden state
    zero_hs = torch.zeros(1024, device=device)
    results['RS315_zeroHS'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS315', device,
        override_hs=zero_hs, z_fixed=z_fixed)

    print(f"\n  {'Comparison':<40} | {'MSE':<12} | {'MAE':<12} | {'MaxAbs':<12}")
    print("  " + "-" * 80)
    ref = results['RS315_orig']
    for key in results:
        if key == 'RS315_orig':
            continue
        d = pixel_diff(ref, results[key])
        print(f"  RS315_orig vs {key:<24} | {d['MSE']:<12.6f} | {d['MAE']:<12.6f} | {d['MaxAbs']:<12.6f}")

    save_grid(results, os.path.join(out_dir, 'test3_random_hs.png'))


def test_4_mat_isolation(generator, evaluator, zdist, cached_hs, device, out_dir):
    """將 mat_feat 歸零或替換極端值，觀察輸出變化。"""
    print("\n" + "=" * 70)
    print("TEST 4: Mat feat isolation")
    print("=" * 70)

    z_fixed = zdist.sample((8,))
    results = {}

    results['RS315_orig'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS315', device, z_fixed=z_fixed)

    # 歸零 mat
    results['RS315_zeroMat'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS315', device,
        override_mat=[0.0] * 7, z_fixed=z_fixed)

    # 全 1 mat
    results['RS315_onesMat'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS315', device,
        override_mat=[1.0] * 7, z_fixed=z_fixed)

    # 用 RS307 的 mat (距離最遠的)
    results['RS315_mat307'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS315', device,
        override_mat=SPECIMEN_SPECS['RS307'][1], z_fixed=z_fixed)

    # 用 RS615 的 mat
    results['RS315_mat615'] = render_specimen(
        generator, evaluator, zdist, cached_hs, 'RS315', device,
        override_mat=SPECIMEN_SPECS['RS615'][1], z_fixed=z_fixed)

    print(f"\n  {'Comparison':<40} | {'MSE':<12} | {'MAE':<12} | {'MaxAbs':<12}")
    print("  " + "-" * 80)
    ref = results['RS315_orig']
    for key in results:
        if key == 'RS315_orig':
            continue
        d = pixel_diff(ref, results[key])
        print(f"  RS315_orig vs {key:<24} | {d['MSE']:<12.6f} | {d['MAE']:<12.6f} | {d['MaxAbs']:<12.6f}")

    save_grid(results, os.path.join(out_dir, 'test4_mat_isolation.png'))


def test_5_bottleneck_sweep(generator, evaluator, zdist, cached_hs, device, out_dir):
    """在 bottleneck 空間中從 RS315 插值到 RS330，觀察輸出是否連續變化。"""
    print("\n" + "=" * 70)
    print("TEST 5: Bottleneck 插值 sweep (RS315 → RS330)")
    print("=" * 70)

    nerf_model = generator.render_kwargs_train['network_fn']
    cond_proj = nerf_model.condition_feature

    if not hasattr(cond_proj, 'encode') or not hasattr(cond_proj, 'set_override'):
        print("  [SKIP] ConditionBottleneck 不支援 encode/set_override。")
        return

    z_fixed = zdist.sample((8,))
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    results = {}

    with torch.no_grad():
        b_315 = cond_proj.encode(cached_hs['RS315'].unsqueeze(0))  # [1, 16]
        b_330 = cond_proj.encode(cached_hs['RS330'].unsqueeze(0))  # [1, 16]

    # 每個 alpha 都用 override 注入
    for alpha in alphas:
        b_interp = alpha * b_330 + (1 - alpha) * b_315  # [1, 16]
        cond_interp = cond_proj.decode(b_interp)  # [1, 256]

        # 用 set_override 注入 — 需要擴展到 n_views
        n_views = 8
        cond_expanded = cond_interp.expand(n_views, -1)  # [8, 256]

        # 準備 label 和 rays（用 RS315 的 label 結構）
        vec, mat = SPECIMEN_SPECS['RS315']
        angles = [i * 360 // n_views for i in range(n_views)]
        labels_list = [make_label(vec, mat, h_idx=1, a_idx=a) for a in angles]
        label_t = torch.tensor(labels_list, dtype=torch.float32, device=device)
        hs_t = cached_hs['RS315'].unsqueeze(0).expand(n_views, -1)
        poses = torch.stack([generator.sample_select_pose(a / 360.0, 0.5) for a in angles])

        # 注入 override（每個 view 獨立）
        # 注意：evaluator.create_samples 內部呼叫 generator，
        # generator 會呼叫 render → run_network → condition_feature(hs)
        # 如果 set_override 設了值，condition_feature 會回傳 override 而非重新 encode
        cond_proj.set_override(cond_expanded)
        with torch.no_grad():
            rgb, _, _ = evaluator.create_samples(z_fixed.to(device), label_t, hs_t, poses)
        cond_proj.clear_override()

        key = f'alpha={alpha:.1f}'
        results[key] = rgb

    # 計算相鄰 alpha 之間的差異
    print(f"\n  {'Pair':<30} | {'MSE':<12} | {'MAE':<12}")
    print("  " + "-" * 58)
    keys = list(results.keys())
    for i in range(len(keys) - 1):
        d = pixel_diff(results[keys[i]], results[keys[i + 1]])
        print(f"  {keys[i]} → {keys[i+1]:<14} | {d['MSE']:<12.6f} | {d['MAE']:<12.6f}")

    # 首尾差異
    d = pixel_diff(results[keys[0]], results[keys[-1]])
    print(f"  {keys[0]} → {keys[-1]:<14} | {d['MSE']:<12.6f} | {d['MAE']:<12.6f}  ← 總差")

    save_grid(results, os.path.join(out_dir, 'test5_bottleneck_sweep.png'))


def test_6_all_specimens(generator, evaluator, zdist, cached_hs, device, out_dir):
    """生成所有 specimen 的圖片並互相比較，建立完整的差異矩陣。"""
    print("\n" + "=" * 70)
    print("TEST 6: 全 specimen 差異矩陣")
    print("=" * 70)

    z_fixed = zdist.sample((8,))
    results = {}

    for sn in SPECIMEN_SPECS:
        if sn in cached_hs:
            results[sn] = render_specimen(
                generator, evaluator, zdist, cached_hs, sn, device, z_fixed=z_fixed)

    names = sorted(results.keys())
    print(f"\n  MSE matrix:")
    print(f"  {'':>8}", end="")
    for n in names:
        print(f" {n:>10}", end="")
    print()
    for n1 in names:
        print(f"  {n1:>8}", end="")
        for n2 in names:
            if n1 == n2:
                print(f" {'---':>10}", end="")
            else:
                d = pixel_diff(results[n1], results[n2])
                print(f" {d['MSE']:>10.6f}", end="")
        print()

    save_grid(results, os.path.join(out_dir, 'test6_all_specimens.png'))


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Conditioning 診斷工具")
    parser.add_argument('--config', type=str,
                        default='configs/twostage.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint 路徑 (e.g. results/.../chkpts/model_best.pt)')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tests', type=str, default='1,2,3,4,5,6',
                        help='要執行的測試編號 (逗號分隔)')
    parser.add_argument('--outdir', type=str, default=None,
                        help='輸出目錄 (預設: checkpoint 同層的 diagnose/)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0')
    set_seed(42)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # 輸出目錄
    if args.outdir is None:
        ckpt_parent = os.path.dirname(os.path.dirname(args.checkpoint))
        out_dir = os.path.join(ckpt_parent, 'diagnose')
    else:
        out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[輸出目錄] {out_dir}")

    # 載入
    config, generator, cached_hs, v_list, zdist, train_dataset = load_all(
        args.config, args.checkpoint, device)

    batch_size = config['training']['batch_size']
    evaluator = Evaluator(False, generator, zdist, None,
                          batch_size=batch_size, device=device)

    nerf_model = generator.render_kwargs_train['network_fn']

    tests_to_run = [int(t.strip()) for t in args.tests.split(',')]

    if 1 in tests_to_run:
        test_1_representation(cached_hs, nerf_model, device)

    if 2 in tests_to_run:
        test_2_swap(generator, evaluator, zdist, cached_hs, device, out_dir)

    if 3 in tests_to_run:
        test_3_random_hs(generator, evaluator, zdist, cached_hs, device, out_dir)

    if 4 in tests_to_run:
        test_4_mat_isolation(generator, evaluator, zdist, cached_hs, device, out_dir)

    if 5 in tests_to_run:
        test_5_bottleneck_sweep(generator, evaluator, zdist, cached_hs, device, out_dir)

    if 6 in tests_to_run:
        test_6_all_specimens(generator, evaluator, zdist, cached_hs, device, out_dir)

    print("\n" + "=" * 70)
    print("所有測試完成！")
    print(f"圖片已儲存至: {out_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()