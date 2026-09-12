"""孪生（逐位相同）embedding 组在 raw / Sinkhorn 两种口径下的碰撞存活率。

用法：
    .venv/Scripts/python.exe scripts/multimodal/probe_twin_sinkhorn.py <IandS|VG>

回答两个问题：
  1. 纯 argmin（sid_raw）下，孪生组是否必然全组同码？（预期：100%）
  2. 导出期 Sinkhorn（sid_sk）下，孪生组能被拆开多少？
     —— 若能拆开，说明"Sinkhorn 只是运气"或"Sinkhorn 会牺牲保真度强行拆"，
        需要看被拆物品的量化误差增量。
"""
import sys
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SHORT = sys.argv[1] if len(sys.argv) > 1 else "IandS"

EMB = os.path.join(ROOT, "data", "Amazon23", SHORT, "emb", "long", "emb_fused_gate_e60.npy")
SID_DIR = os.path.join(ROOT, "results", "sid_e5000", SHORT, "gate__init8192")

emb = np.load(EMB).astype(np.float32)
raw = np.load(os.path.join(SID_DIR, "sid_raw.npy"))
sk = np.load(os.path.join(SID_DIR, "sid_sk.npy"))
N = len(emb)
print(f"[{SHORT}] N={N}  emb={emb.shape[1]}d  sid={raw.shape}")

# --- 逐位相同的 embedding 分组 ---
_, inv, cnt = np.unique(emb, axis=0, return_inverse=True, return_counts=True)
groups = {}
for i, g in enumerate(inv):
    groups.setdefault(int(g), []).append(i)
twins = {g: idx for g, idx in groups.items() if len(idx) > 1}
n_twin_items = sum(len(v) for v in twins.values())
print(f"孪生组={len(twins)}  涉及物品={n_twin_items}  ICR(argmin)上限={(N - n_twin_items + len(twins)) / N:.4f}")

# --- 三种口径下，孪生组是否整组同码 ---
def collide_report(codes, name):
    still = 0
    residual_items = 0   # 孪生组内仍剩余的"多余份数" Σ(m - k)
    for idx in twins.values():
        ks = {tuple(codes[i]) for i in idx}
        if len(ks) == 1:
            still += 1
        residual_items += len(idx) - len(ks)
    print(f"  {name:10s} 仍整组同码={still:4d}/{len(twins)}  "
          f"({still / len(twins) * 100:5.1f}%)   组内残余碰撞物品={residual_items}")
    return still

print("\n孪生组存活情况：")
collide_report(raw, "sid_raw")
collide_report(sk, "sid_sk")

# --- Sinkhorn 为拆孪生付出的量化误差代价 ---
def code_mse(codes):
    _, inv2, cnt2 = np.unique(codes, axis=0, return_inverse=True, return_counts=True)
    return inv2

# 用码本重建 MSE 需要模型，这里退而求其次：统计"被改码"的孪生物品
twin_ids = np.array([i for v in twins.values() for i in v])
changed = np.array([not np.array_equal(raw[i], sk[i]) for i in twin_ids])
print(f"\n孪生物品被 Sinkhorn 改码：{changed.sum()}/{len(twin_ids)}  ({changed.mean() * 100:.1f}%)")
all_changed = np.array([not np.array_equal(raw[i], sk[i]) for i in range(N)])
print(f"全体物品被改码：{all_changed.sum()}/{N}  ({all_changed.mean() * 100:.1f}%)")

# --- 对照组：非孪生但 raw 下碰撞的物品 ---
_, inv_r, cnt_r = np.unique(raw, axis=0, return_inverse=True, return_counts=True)
twin_set = set(twin_ids.tolist())
non_twin_collided = []
for g in np.unique(inv_r):
    idx = np.where(inv_r == g)[0]
    if len(idx) > 1 and not (set(idx.tolist()) & twin_set):
        non_twin_collided.extend(idx.tolist())
print(f"raw 下碰撞但非孪生的物品：{len(non_twin_collided)}")
if non_twin_collided:
    a = np.array(non_twin_collided)
    c = np.array([not np.array_equal(raw[i], sk[i]) for i in a])
    print(f"  其中被 Sinkhorn 改码：{c.sum()}/{len(a)}  ({c.mean() * 100:.1f}%)")
