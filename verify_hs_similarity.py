"""
驗證假設:RS315 的 hidden_state 跟訓練集哪個 specimen 最接近
如果 RS315 跟 RS330 cos_sim 最高 → 證實「nearest-neighbor lookup」假設
"""
import os, sys, json, glob, argparse, importlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('submodules')

from graf.datasets import HystereticDataset

# ============================================================
# 1. 載入 GRU extractor
# ============================================================
print("Loading GRU extractor...")
extractor_path = "/Data/home/vicky/graf260108_im64/HystereticGRU/2026-03-24_17-13-01/Opensees Based damage Patterns Generation/c7d0zrxf/checkpoints/best.ckpt"
extractor_args_path = "/Data/home/vicky/graf260108_im64/HystereticGRU/2026-03-24_17-13-01/args.json"

extractor_args = json.load(open(extractor_args_path, "r"))
extractor_args = argparse.Namespace(**extractor_args)
extractor = importlib.import_module("graf.models.HystereticPrediction").__dict__[
    extractor_args.architecture
](**vars(extractor_args))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = extractor.to(device)
state_dict = torch.load(glob.glob(extractor_path, recursive=True)[0])["state_dict"]
status = extractor.load_state_dict(state_dict)
print(f"Extractor loaded: {status}")
extractor.eval()

# ============================================================
# 2. 計算所有 4 個 specimen 的 hidden_state
# ============================================================
all_specimens = ['RS307', 'RS315', 'RS330', 'RS615']

dataset = HystereticDataset(
    root="/Data/home/vicky/graf260108_im64/Data/Hysteresis",
    specified=all_specimens,
    **vars(extractor_args)
)
loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                    num_workers=0, collate_fn=dataset.collate_fn)

hidden_states = {}
with torch.no_grad():
    for batch in tqdm(loader, desc="Extracting hidden states"):
        simulated_loop, _, _, _, exp = batch
        simulated_loop = simulated_loop.to(device)
        output, _ = extractor.L(simulated_loop)
        hidden_states[exp[0]] = output[0, -1, :].cpu().squeeze()  # [1024]

print(f"\nExtracted hidden states for: {list(hidden_states.keys())}")

# ============================================================
# 3. 計算 cos_sim 跟 L2 distance 矩陣
# ============================================================
print("\n" + "="*70)
print("Pairwise Cosine Similarity (越接近 1 越相似)")
print("="*70)

print(f"{'':<10}", end="")
for s in all_specimens:
    print(f"{s:<10}", end="")
print()

for s1 in all_specimens:
    print(f"{s1:<10}", end="")
    for s2 in all_specimens:
        if s1 == s2:
            print(f"{'1.000':<10}", end="")
        else:
            cos = F.cosine_similarity(
                hidden_states[s1].unsqueeze(0),
                hidden_states[s2].unsqueeze(0)
            ).item()
            print(f"{cos:.4f}    ", end="")
    print()

print("\n" + "="*70)
print("Pairwise L2 Distance (越小越相似)")
print("="*70)

print(f"{'':<10}", end="")
for s in all_specimens:
    print(f"{s:<10}", end="")
print()

for s1 in all_specimens:
    print(f"{s1:<10}", end="")
    for s2 in all_specimens:
        if s1 == s2:
            print(f"{'0.000':<10}", end="")
        else:
            l2 = (hidden_states[s1] - hidden_states[s2]).norm().item()
            print(f"{l2:.4f}    ", end="")
    print()

# ============================================================
# 4. 對 RS315 做 nearest-neighbor 分析
# ============================================================
print("\n" + "="*70)
print("RS315 vs 訓練集 specimen 的 nearest-neighbor 分析")
print("="*70)

train_specs = ['RS307', 'RS330', 'RS615']
hs_315 = hidden_states['RS315']

scores = {}
for s in train_specs:
    cos = F.cosine_similarity(
        hs_315.unsqueeze(0),
        hidden_states[s].unsqueeze(0)
    ).item()
    l2 = (hs_315 - hidden_states[s]).norm().item()
    scores[s] = (cos, l2)
    print(f"RS315 vs {s}:  cos_sim = {cos:.4f},  L2 = {l2:.4f}")

# 判斷誰最近
nearest_by_cos = max(scores.items(), key=lambda x: x[1][0])[0]
nearest_by_l2 = min(scores.items(), key=lambda x: x[1][1])[0]

print(f"\nNearest by cosine similarity: {nearest_by_cos}")
print(f"Nearest by L2 distance:       {nearest_by_l2}")

# 結論
print("\n" + "="*70)
print("結論")
print("="*70)
if nearest_by_cos == 'RS330' or nearest_by_l2 == 'RS330':
    print("⚠️ 假設成立:RS315 的 hidden_state 跟 RS330 最接近")
    print("   → 這解釋了為什麼生成的 RS315 看起來像 RS330")
    print("   → 證實 G 學到的是 'nearest-neighbor lookup' 而非物理泛化")
else:
    print(f"假設不成立:RS315 跟 {nearest_by_cos} 最接近,不是 RS330")
    print("   → 需要重新分析為什麼生成結果像 RS330")
    print("   → 可能是 G 學到的「視覺鄰近性」跟「hidden_state 鄰近性」不一致")