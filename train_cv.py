"""
Cross Validation (Leave-One-Out) Two-Stage Training
====================================================
基於版本 A (Damage Consistency Loss 單一約束) 的 Leave-One-Out CV 框架。

四個 fold（四筆資料均有真實影像）：
  fold1: test=RS307, train=[RS315, RS330, RS615]
  fold2: test=RS315, train=[RS307, RS330, RS615]
  fold3: test=RS330, train=[RS307, RS315, RS615]
  fold4: test=RS615, train=[RS307, RS315, RS330]

使用方式：
  python train_twostage_cv.py --config configs/cv_fold1.yaml
  python train_twostage_cv.py --config configs/cv_fold2.yaml
  python train_twostage_cv.py --config configs/cv_fold2.yaml --resume path/to/model.pt

YAML 設定範例（cv_fold2.yaml）：
  cross_validation:
    test_specimen: RS315
    train_specimens: [RS307, RS330, RS615]
  data:
    datadir: [data/RS307_n, data/RS330_n, data/RS615_n]
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
from graf.models.twostage_networks import SRNetwork, DiscriminatorHigh, nerf_flat_to_img


# ── 目錄與種子 ────────────────────────────────────────────────────────────────

def setup_directories(config):
    out_dir        = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    os.makedirs(out_dir,        exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return out_dir, checkpoint_dir


def set_random_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed);    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


# ── 初始化（CV 版本：dataset 只載入 train specimens）─────────────────────────

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
    status     = extractor.load_state_dict(state_dict)
    print("Extractor Loading Status:", status)

    # ★ CV 關鍵：dataset 的 datadir 只包含 train specimens（由 YAML 控制）
    # ImageDataset 內部仍會為所有 4 個 specimens 建立 hidden state，
    # 但 real images 只來自 train_specimens 的資料夾
    train_dataset, hwfr = get_data(config, extractor, extractor_args)
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far'] - config['data']['near'],) * 2
        hwfr[2]  = hw_ortho
    config['data']['hwfr'] = hwfr

    # ★ pin_memory=False：因為 torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = config['training']['batch_size'],
        num_workers = config['training']['nworkers'],
        shuffle     = True,
        pin_memory  = False,
        sampler     = None,
        drop_last   = True,
        generator   = torch.Generator(device='cuda:0'),
    )

    generator, discriminator = build_models(config)
    generator     = generator.to(device)
    discriminator = discriminator.to(device)
    return train_loader, train_dataset, generator, discriminator


# ── RayCache ──────────────────────────────────────────────────────────────────

class RayCache:
    def __init__(self, generator, v_list, focal_64):
        self._cache     = {}
        self._generator = generator
        self._v_list    = v_list
        self._focal_64  = focal_64
        self._n_heights = len(v_list)

    def get_rays(self, angle_idx, height_idx):
        key = (angle_idx, height_idx)
        if key not in self._cache:
            u    = angle_idx / 360.0
            v    = self._v_list[height_idx % self._n_heights]
            pose = self._generator.sample_select_pose(u, v)
            rays, _, _ = self._generator.val_ray_sampler(64, 64, self._focal_64, pose)
            self._cache[key] = rays
        return self._cache[key]

    def batch_rays(self, label_batch, bs):
        rays_list = []
        for i in range(bs):
            h_idx = int(label_batch[i, 14].item())
            a_idx = int(label_batch[i, 15].item())
            rays_list.append(self.get_rays(a_idx, h_idx))
        return torch.cat(rays_list, dim=1)


# ── InterpCache（CV 版本：只在 train specimens 之間插值）─────────────────────

class InterpCache:
    """
    CV 版本的插值快取：
    - 只在 train_specimens 之間做插值（test specimen 不參與）
    - hidden state 為所有 4 個 specimens 都有（ImageDataset 建立）
    """

    def __init__(self, cached_hs, specimen_specs, dual_proj, device, train_specimens):
        self.names           = []
        self.train_specimens = train_specimens
        embed_list, vec_list, mat_list, hs_list = [], [], [], []

        with torch.no_grad():
            # ★ 只對 train_specimens 建立 embedding
            for sn in train_specimens:
                if sn not in specimen_specs or sn not in cached_hs:
                    continue
                vec, mf = specimen_specs[sn]
                hs   = cached_hs[sn]
                mf_t = torch.tensor(mf, dtype=torch.float32, device=device)
                emb  = dual_proj.encode(hs.unsqueeze(0), mf_t.unsqueeze(0)).squeeze(0)

                self.names.append(sn)
                embed_list.append(emb)
                vec_list.append(torch.tensor(vec, dtype=torch.float32, device=device))
                mat_list.append(mf_t)
                hs_list.append(hs)

        self.n = len(self.names)
        if self.n > 0:
            self.embeddings    = torch.stack(embed_list)
            self.label_vecs    = torch.stack(vec_list)
            self.mat_feats     = torch.stack(mat_list)
            self.hidden_states = torch.stack(hs_list)

        print(f"[InterpCache] train specimens: {self.names} ({self.n} total)")

    def sample(self, n_interp, beta_dist, device):
        if self.n < 2:
            return None, None, None, None
        idx_a = torch.randint(0, self.n,     (n_interp,), device=device)
        idx_b = torch.randint(0, self.n - 1, (n_interp,), device=device)
        idx_b = idx_b + (idx_b >= idx_a).long()
        lam   = beta_dist.sample((n_interp,)).to(device).unsqueeze(-1)

        i_emb = lam * self.embeddings[idx_a]    + (1 - lam) * self.embeddings[idx_b]
        i_hs  = lam * self.hidden_states[idx_a] + (1 - lam) * self.hidden_states[idx_b]
        i_mat = lam * self.mat_feats[idx_a]     + (1 - lam) * self.mat_feats[idx_b]
        i_vec = lam * self.label_vecs[idx_a]    + (1 - lam) * self.label_vecs[idx_b]

        h_idx   = torch.randint(0,   4, (n_interp, 1), device=device, dtype=torch.float32)
        a_idx   = torch.randint(0, 360, (n_interp, 1), device=device, dtype=torch.float32)
        i_label = torch.cat([i_vec, i_mat, h_idx, a_idx], dim=-1)

        return i_emb, i_label, i_hs, i_mat


# ── Damage Consistency Loss（版本 A：單一約束）───────────────────────────────

def compute_damage_consistency_loss(nerf_img, mat_feat, margin=0.05, damage_thresh=0.15):
    """
    版本 A：單一約束，整體暗色比例排序。
    Proxy: RS307=0.268, RS615=0.350, RS315=0.494, RS330=0.752
    """
    bs = nerf_img.size(0)
    if bs < 2:
        return torch.tensor(0., device=nerf_img.device)

    img_01     = (nerf_img + 1.0) / 2.0
    brightness = img_01.mean(dim=1)
    dark_ratio = (1.0 - brightness).mean(dim=[1, 2])

    expected_damage = (
        mat_feat[:, 0] * 0.20 +   # displacement
        mat_feat[:, 3] * 0.15 +   # corner_rebar_ε
        mat_feat[:, 4] * 0.15 +   # core_concrete_σ
        mat_feat[:, 5] * 0.25 +   # core_concrete_ε
        mat_feat[:, 6] * 0.25     # cover_concrete_ε
    )

    loss  = torch.tensor(0., device=nerf_img.device)
    count = 0

    for i in range(bs):
        for j in range(i + 1, bs):
            damage_diff = expected_damage[i] - expected_damage[j]
            if damage_diff.abs().item() < damage_thresh:
                continue
            dark_diff = dark_ratio[i] - dark_ratio[j]
            loss  = loss + F.relu(-damage_diff.sign() * dark_diff + margin)
            count += 1

    if count > 0:
        loss = loss / count
    return loss


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    set_random_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',          default='configs/cv_fold2.yaml')
    parser.add_argument('--resume',          type=str, default=None)
    parser.add_argument('--resume_wandb_id', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])

    # ★ CV 設定
    cv_config       = config.get('cross_validation', {})
    test_specimen   = cv_config.get('test_specimen', 'RS315')
    train_specimens = cv_config.get('train_specimens', ['RS307', 'RS330', 'RS615'])
    print(f"\n[CV] test_specimen   = {test_specimen}")
    print(f"[CV] train_specimens = {train_specimens}")

    restart_every   = config['training']['restart_every']
    batch_size      = config['training']['batch_size']
    fid_every       = config['training']['fid_every']
    save_best       = config['training']['save_best']
    reg_param       = config['training']['reg_param']
    aux_loss_weight = config['training']['label_param']
    print_every     = config['training']['print_every']
    sample_every    = config['training']['sample_every']
    save_every_s    = config['training']['save_every']
    device          = torch.device("cuda:0")

    use_amp       = config['training'].get('use_amp', False)
    amp_dtype_str = config['training'].get('amp_dtype', 'bfloat16')
    amp_dtype     = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
    use_scaler    = use_amp and amp_dtype == torch.float16
    scaler        = torch.cuda.amp.GradScaler() if use_scaler else None

    phase2_start    = config['training'].get('phase2_start_iter', 150000)
    lambda_d_high   = config['training'].get('lambda_d_high', 0.5)
    lambda_recon_sr = config['training'].get('lambda_recon_sr', 0.5)
    phase2_warmup   = config['training'].get('phase2_warmup_steps', 20000)

    interp_config = config.get('interpolation', {})
    use_interp    = interp_config.get('enabled', True)
    n_interp      = interp_config.get('n_samples', 4)
    lambda_interp = interp_config.get('lambda', 1.0)
    interp_start  = interp_config.get('start_iter', 5000)

    damage_config = config.get('damage_consistency', {})
    use_damage    = damage_config.get('enabled', True)
    lambda_damage = damage_config.get('lambda', 0.3)
    damage_start  = damage_config.get('start_iter', 5000)
    damage_thresh = damage_config.get('threshold', 0.15)

    print(f"[TwoStage] Phase1: 0~{phase2_start}  Phase2: {phase2_start}+")
    print(f"[INTERP]   λ={lambda_interp}, start={interp_start}")
    print(f"[DAMAGE]   λ={lambda_damage}, start={damage_start}, thresh={damage_thresh}")

    out_dir, checkpoint_dir = setup_directories(config)
    save_config(os.path.join(out_dir, 'config.yaml'), config)

    wandb_kw = dict(project=config['wandb']['project'],
                    name=config['wandb']['name'], config=config)
    if args.resume_wandb_id:
        wandb_kw.update(id=args.resume_wandb_id, resume='must')
    wandb.init(**wandb_kw)

    train_loader, train_dataset, generator, discriminator = initialize_training(config, device)

    nerf_model       = generator.render_kwargs_train['network_fn']
    shared_cond_proj = nerf_model.condition_feature

    sr_network = SRNetwork(ch=64, n_rb=6).to(device)
    d_high     = DiscriminatorHigh(
        nc=3, ndf=config['discriminator']['ndf'],
        hidden_dim=config['discriminator'].get('hidden_dim', 1024),
        cond_dim=shared_cond_proj.out_dim,
        shared_cond_proj=shared_cond_proj,
    ).to(device)

    lr_g = config['training']['lr_g']
    lr_d = config['training']['lr_d']
    g_optimizer      = optim.RMSprop(generator.parameters(),     lr=lr_g, alpha=0.99, eps=1e-8)
    d_optimizer      = optim.RMSprop(discriminator.parameters(), lr=lr_d, alpha=0.99, eps=1e-8)
    sr_optimizer     = optim.Adam(sr_network.parameters(),       lr=lr_g, betas=(0.0, 0.99))
    d_high_optimizer = optim.RMSprop(d_high.parameters(),        lr=lr_d, alpha=0.99, eps=1e-8)

    recon_loss_fn = CCSRLoss(device=device)

    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
    checkpoint_io.register_modules(
        discriminator=discriminator,
        g_optimizer=g_optimizer, d_optimizer=d_optimizer,
        sr_network=sr_network, d_high=d_high,
        sr_optimizer=sr_optimizer, d_high_optimizer=d_high_optimizer,
        **generator.module_dict,
    )

    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    evaluator = Evaluator(fid_every > 0, generator, zdist, None,
                          batch_size=batch_size, device=device, inception_nsamples=33)
    if fid_every > 0:
        evaluator.inception_eval.initialize_target(
            train_loader,
            cache_file     = os.path.join(out_dir, 'fid_cache_train.npz'),
            act_cache_file = os.path.join(out_dir, 'kid_cache_train.npz'),
        )

    # ★ cached_hs 包含所有 4 個 specimens（ImageDataset 永遠為全部建立 hidden state）
    cached_hs = {n: h.to(device) for n, h in train_dataset.hidden_state.items()}
    print(f"\nCached hidden states for: {list(cached_hs.keys())}")

    # 全部 specimen specs（含 test specimen，用於 sample grid 視覺化）
    specimen_specs = {
        'RS307': ([1,0,1,0,0,0,1], [0.000000,0.156717,0.918975,0.339361,0.498170,0.310937,0.360617]),
        'RS330': ([1,0,0,0,1,1,0], [0.008831,1.000000,1.000000,1.000000,0.000000,1.000000,1.000000]),
        'RS615': ([0,1,0,1,0,1,0], [1.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000]),
        'RS315': ([1,0,0,1,0,1,0], [0.006158,0.411209,0.538894,0.725360,0.017321,0.727661,0.751007]),
    }

    # ── Similarity 分析 ───────────────────────────────────────────────────────
    print("-" * 60)
    for i, k1 in enumerate(list(cached_hs.keys())):
        for k2 in list(cached_hs.keys())[i+1:]:
            hs1, hs2 = cached_hs[k1], cached_hs[k2]
            diff    = (hs1 - hs2).norm().item()
            cos_sim = F.cosine_similarity(hs1.unsqueeze(0), hs2.unsqueeze(0)).item()
            with torch.no_grad():
                mf1 = torch.tensor(specimen_specs[k1][1], device=device).unsqueeze(0)
                mf2 = torch.tensor(specimen_specs[k2][1], device=device).unsqueeze(0)
                e1  = shared_cond_proj.encode(hs1.unsqueeze(0), mf1)
                e2  = shared_cond_proj.encode(hs2.unsqueeze(0), mf2)
                emb_cos = F.cosine_similarity(e1, e2).item()
            tag = "TEST" if k1 == test_specimen or k2 == test_specimen else "train"
            print(f"  [{tag}] {k1} vs {k2}: HS_cos={cos_sim:.4f}  Emb256_cos={emb_cos:.4f}")
    print("-" * 60 + "\n")

    v_list    = [float(x.strip()) for x in config['data']['v'].split(",")]
    n_heights = len(v_list)
    focal_64  = generator.focal * (64.0 / generator.H)
    ray_cache = RayCache(generator, v_list, focal_64)

    fid_best = kid_best = float('inf')
    it = epoch_idx = -1
    tstart = t0 = time.time()
    phase2_flag = False

    if args.resume:
        sc        = checkpoint_io.load(args.resume)
        it        = sc.get('it',        -1)
        epoch_idx = sc.get('epoch_idx', -1)
        fid_best  = sc.get('fid_best',  float('inf'))
        kid_best  = sc.get('kid_best',  float('inf'))
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

    sample_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    sample_poses  = torch.stack([generator.sample_select_pose(i/8, 0.5) for i in range(8)])

    # ★ InterpCache：只在 train specimens 之間插值
    interp_cache = None
    beta_dist    = None
    if use_interp:
        interp_cache = InterpCache(
            cached_hs, specimen_specs, shared_cond_proj, device, train_specimens)
        beta_dist = torch.distributions.Beta(
            torch.tensor(0.2, device=device),
            torch.tensor(0.2, device=device),
        )

    zero        = torch.tensor(0., device=device)
    last_reg_lo = 0.0
    last_reg_hi = 0.0

    # ================================================================
    # 訓練迴圈
    # ================================================================
    while True:
        epoch_idx += 1
        for x_real, label, hidden_state in tqdm(train_loader, desc=f"Epoch {epoch_idx}"):
            it    += 1
            in_p2  = (it >= phase2_start)
            if in_p2 and not phase2_flag:
                print(f"\n{'='*60}\n[Phase 2] iter {it}: SR + D_high active\n{'='*60}")
                phase2_flag = True

            # ★ InterpCache 更新（只用 train specimens）
            if use_interp and it >= interp_start and it % 1000 == 0:
                interp_cache = InterpCache(
                    cached_hs, specimen_specs, shared_cond_proj, device, train_specimens)

            x_real       = x_real.to(device, non_blocking=True)
            label        = label.to(device,  non_blocking=True)
            hidden_state = hidden_state.to(device, non_blocking=True)
            mat_feat     = label[:, 7:14].float()
            x_real_64    = F.interpolate(x_real, size=(64, 64),
                                         mode='bilinear', align_corners=True)

            generator.ray_sampler.iterations = it
            do_r1 = (it % 16 == 0)

            # ── D_low step ───────────────────────────────────────────
            toggle_grad(generator, False)
            toggle_grad(discriminator, True)
            generator.train(); discriminator.train()
            d_optimizer.zero_grad(set_to_none=True)

            x_real_d = x_real_64.detach().requires_grad_(True)
            z        = zdist.sample((batch_size,))

            with amp_ctx():
                d_real_lo, aux_real = discriminator(
                    x_real_d, hidden_state, mat_feat, return_aux=True)
                dloss_real_lo = compute_loss(d_real_lo, 1)
                aux_loss_real = F.mse_loss(aux_real, hidden_state)

            if do_r1:
                dloss_real_lo.backward(retain_graph=True)
                reg_lo = reg_param * 16 * compute_grad2(d_real_lo.float(), x_real_d).mean()
                reg_lo.backward(retain_graph=True)
                last_reg_lo = reg_lo.item()
            else:
                dloss_real_lo.backward(retain_graph=True)

            (aux_loss_weight * aux_loss_real).backward()

            with torch.no_grad(), amp_ctx():
                rays_f       = ray_cache.batch_rays(label, batch_size)
                nerf_flat, _ = generator(z, label, hidden_state, rays=rays_f)

            nerf_img_64 = nerf_flat_to_img(nerf_flat, batch_size)

            with amp_ctx():
                d_fake_lo     = discriminator(nerf_img_64, hidden_state, mat_feat)
                dloss_fake_lo = compute_loss(d_fake_lo, 0)

            dloss_fake_lo.backward()
            d_optimizer.step()
            d_scheduler.step()

            # ── D_high step ──────────────────────────────────────────
            dloss_real_hi = dloss_fake_hi = zero
            if in_p2:
                toggle_grad(d_high, True); toggle_grad(sr_network, False)
                d_high.train()
                d_high_optimizer.zero_grad(set_to_none=True)

                xr_dh = x_real.detach().requires_grad_(True)
                with amp_ctx():
                    dr_hi         = d_high(xr_dh, hidden_state, mat_feat)
                    dloss_real_hi = compute_loss(dr_hi, 1)

                reg_hi = zero
                if do_r1:
                    reg_hi = reg_param * 16 * compute_grad2(dr_hi.float(), xr_dh).mean()
                    last_reg_hi = reg_hi.item()

                with torch.no_grad():
                    sr256 = sr_network(nerf_img_64)
                with amp_ctx():
                    df_hi         = d_high(sr256, hidden_state, mat_feat)
                    dloss_fake_hi = compute_loss(df_hi, 0)

                dhi_loss = dloss_real_hi + dloss_fake_hi + reg_hi
                dhi_loss.backward()
                d_high_optimizer.step()

            # ── G + SR step ──────────────────────────────────────────
            if config['nerf']['decrease_noise']:
                generator.decrease_nerf_noise(it)

            toggle_grad(generator, True); toggle_grad(discriminator, False)
            generator.train(); discriminator.train()
            g_optimizer.zero_grad(set_to_none=True)

            if in_p2:
                toggle_grad(sr_network, True); toggle_grad(d_high, False)
                sr_network.train()
                sr_optimizer.zero_grad(set_to_none=True)

            z = zdist.sample((batch_size,))
            with amp_ctx():
                rays_g     = ray_cache.batch_rays(label, batch_size)
                nerf_g, _  = generator(z, label, hidden_state, rays=rays_g)
                nerf_g_img = nerf_flat_to_img(nerf_g, batch_size)

                dfl_g, aux_f = discriminator(
                    nerf_g_img, hidden_state, mat_feat, return_aux=True)
                g_adv_lo = compute_loss(dfl_g, 1)
                g_aux    = F.mse_loss(aux_f, hidden_state)

                g_adv_hi = sr_rec = zero
                if in_p2:
                    p2_elapsed = it - phase2_start
                    p2_ramp    = min(1.0, p2_elapsed / max(1, phase2_warmup))
                    sr_g       = sr_network(nerf_g_img)
                    dfh_g      = d_high(sr_g, hidden_state, mat_feat)
                    g_adv_hi   = compute_loss(dfh_g, 1)
                    sr_rec     = recon_loss_fn(sr_g, x_real)

                gloss = g_adv_lo + aux_loss_weight * g_aux
                if in_p2:
                    gloss = (gloss
                             + lambda_d_high   * p2_ramp * g_adv_hi
                             + lambda_recon_sr * p2_ramp * sr_rec)

                # ── Damage Consistency Loss（版本 A）─────────────────
                g_damage_loss = zero
                if use_damage and it >= damage_start:
                    g_damage_loss = compute_damage_consistency_loss(
                        nerf_g_img, mat_feat,
                        margin=0.05, damage_thresh=damage_thresh,
                    )
                    gloss = gloss + lambda_damage * g_damage_loss

                    if in_p2:
                        sr_damage = compute_damage_consistency_loss(
                            sr_g, mat_feat,
                            margin=0.05, damage_thresh=damage_thresh,
                        )
                        gloss = gloss + lambda_damage * p2_ramp * sr_damage

                # ── Interpolation Loss（只在 train specimens 之間）────
                g_interp_loss = zero
                if use_interp and it >= interp_start and interp_cache is not None:
                    interp_result = interp_cache.sample(n_interp, beta_dist, device)
                    i_emb, i_label, i_hs, i_mat = interp_result

                    if i_emb is not None:
                        shared_cond_proj.set_override(i_emb)
                        z_interp       = zdist.sample((n_interp,))
                        rays_interp    = ray_cache.batch_rays(i_label, n_interp)
                        nerf_interp, _ = generator(z_interp, i_label, i_hs,
                                                   rays=rays_interp)
                        nerf_interp_img = nerf_flat_to_img(nerf_interp, n_interp)
                        shared_cond_proj.clear_override()

                        d_interp_lo   = discriminator(nerf_interp_img, i_hs, i_mat)
                        g_interp_loss = compute_loss(d_interp_lo, 1)

                        if in_p2:
                            sr_interp     = sr_network(nerf_interp_img)
                            d_interp_hi   = d_high(sr_interp, i_hs, i_mat)
                            g_interp_loss = (g_interp_loss
                                             + p2_ramp * compute_loss(d_interp_hi, 1))

                        gloss = gloss + lambda_interp * g_interp_loss

            gloss.backward()
            g_optimizer.step()
            if in_p2:
                sr_optimizer.step()
            g_scheduler.step()

            # ── Logging ───────────────────────────────────────────────
            log_dict = None
            if (it + 1) % print_every == 0:
                log_dict = {
                    "loss/g_total"      : gloss.item(),
                    "loss/g_adv_low"    : g_adv_lo.item(),
                    "loss/g_label"      : g_aux.item(),
                    "loss/g_damage"     : g_damage_loss.item(),
                    "loss/g_interp"     : g_interp_loss.item(),
                    "loss/d_low"        : (dloss_real_lo.item() + dloss_fake_lo.item()
                                           + last_reg_lo
                                           + aux_loss_weight * aux_loss_real.item()),
                    "loss/d_reallabel"  : aux_loss_real.item(),
                    "loss/dloss_real_lo": dloss_real_lo.item(),
                    "loss/dloss_fake_lo": dloss_fake_lo.item(),
                    "loss/reg_lo"       : last_reg_lo,
                    "lr/g"              : g_optimizer.param_groups[0]['lr'],
                    "lr/d"              : d_optimizer.param_groups[0]['lr'],
                    "training/phase"    : 2 if in_p2 else 1,
                    "cv/test_specimen"  : test_specimen,
                }
                if in_p2:
                    log_dict.update({
                        "loss/g_adv_high"   : g_adv_hi.item(),
                        "loss/dloss_real_hi": dloss_real_hi.item(),
                        "loss/dloss_fake_hi": dloss_fake_hi.item(),
                        "loss/reg_hi"       : last_reg_hi,
                        "loss/sr_recon256"  : sr_rec.item(),
                        "training/p2_ramp"  : p2_ramp,
                    })

            # ── Samples（含 test specimen）────────────────────────────
            if (it % sample_every == 0) or (it < 5000 and it % 200 == 0):
                ztest = zdist.sample((8,))
                rgb_pan, dep_pan, acc_pan, sr_pan, shown = [], [], [], [], []

                for sn, (vec, mf) in specimen_specs.items():
                    # ★ 所有 4 個 specimens 都生成，test specimen 標記 [TEST]
                    if sn not in cached_hs:
                        continue
                    ll  = [vec + mf + [float(a)] for a in sample_angles]
                    lt  = torch.tensor(ll, dtype=torch.float32, device=device)
                    ht  = cached_hs[sn].unsqueeze(0).expand(8, -1)
                    rgb, dep, acc = evaluator.create_samples(
                        ztest.to(device), lt, ht, sample_poses)

                    if dep.dim() == 3: dep = dep.unsqueeze(1)
                    if dep.shape[1] == 1: dep = dep.expand(-1, 3, -1, -1)
                    if acc.dim() == 3: acc = acc.unsqueeze(1)
                    if acc.shape[1] == 1: acc = acc.expand(-1, 3, -1, -1)

                    rgb_pan.append(rgb.cpu())
                    dep_pan.append(dep.cpu())
                    acc_pan.append(acc.cpu())
                    tag = f"{sn}[TEST]" if sn == test_specimen else sn
                    shown.append(tag)

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
                        vutils.make_grid(torch.cat(rgb_pan), nrow=8, normalize=True),
                        caption=cap)
                    log_dict["sample/depth"] = wandb.Image(
                        vutils.make_grid(torch.cat(dep_pan), nrow=8, normalize=True),
                        caption=cap)
                    log_dict["sample/acc"]   = wandb.Image(
                        vutils.make_grid(torch.cat(acc_pan), nrow=8, normalize=True),
                        caption=cap)
                    if sr_pan:
                        log_dict["sample/sr256"] = wandb.Image(
                            vutils.make_grid(torch.cat(sr_pan), nrow=8, normalize=True),
                            caption=f"SR|{cap}")

            # ── Embedding cosine similarity（每 10k steps）────────────
            if (it + 1) % 10000 == 0:
                with torch.no_grad():
                    emb_dict = {}
                    for sn, (vec, mf) in specimen_specs.items():
                        mf_t = torch.tensor(mf, device=device).unsqueeze(0)
                        if sn in cached_hs:
                            emb_dict[sn] = shared_cond_proj.encode(
                                cached_hs[sn].unsqueeze(0), mf_t)

                    pairs = [
                        ('RS307', 'RS315'), ('RS307', 'RS330'), ('RS307', 'RS615'),
                        ('RS315', 'RS330'), ('RS315', 'RS615'), ('RS330', 'RS615'),
                    ]
                    if log_dict is None:
                        log_dict = {}
                    for k1, k2 in pairs:
                        if k1 in emb_dict and k2 in emb_dict:
                            cos = F.cosine_similarity(emb_dict[k1], emb_dict[k2]).item()
                            log_dict[f"emb_cos/{k1}_vs_{k2}"] = cos

            # ── FID / KID（只對 train specimens 計算）────────────────
            if fid_every > 0 and (it + 1) % fid_every == 0:
                nd = len(train_dataset)

                def mgen():
                    while True:
                        idx = np.random.choice(nd, batch_size, replace=False)
                        ll, hl = [], []
                        for i in idx:
                            _, li, hi = train_dataset[i]
                            ll.append(li); hl.append(hi)
                        lb = torch.stack(ll).to(device)
                        hb = torch.stack(hl).to(device)
                        ps = torch.stack([
                            generator.sample_select_pose(
                                int(lb[i, 15].item()) / 360.,
                                v_list[int(lb[i, 14].item()) % n_heights])
                            for i in range(batch_size)
                        ])
                        zf = zdist.sample((batch_size,))
                        with torch.no_grad():
                            r, _, _ = evaluator.create_samples(zf, lb, hb, ps)
                        r = ((r / 2 + .5).mul_(255).clamp_(0, 255)
                             .to(torch.uint8).float() / 255. * 2 - 1)
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
                                       fid_best=fid_best, kid_best=kid_best,
                                       save_to_wandb=True)
                elif save_best == 'kid' and kid < kid_best:
                    kid_best = kid
                    wandb.run.summary["best_kid"] = kid_best
                    checkpoint_io.save('model_best.pt', it=it, epoch_idx=epoch_idx,
                                       fid_best=fid_best, kid_best=kid_best,
                                       save_to_wandb=True)

            if log_dict is not None:
                wandb.log(log_dict, step=it)

            save_interval = 10000 if it < 100000 else 5000

            if (it + 1) % save_interval == 0:
                checkpoint_io.save(
                    'model_%08d.pt' % it,
                    it=it,
                    epoch_idx=epoch_idx,
                    fid_best=fid_best,
                    kid_best=kid_best,
                    save_to_wandb=True
    )
            if time.time() - t0 > save_every_s:
                checkpoint_io.save(config['training']['model_file'], it=it,
                                   epoch_idx=epoch_idx, fid_best=fid_best,
                                   kid_best=kid_best, save_to_wandb=True)
                t0 = time.time()
                if restart_every > 0 and t0 - tstart > restart_every:
                    return


if __name__ == '__main__':
    main()