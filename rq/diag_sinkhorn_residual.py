#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断：为什么 Sinkhorn 碰撞消解消不完最后 21 个碰撞？
=====================================================
假设：sk_epsilon=0.003 远小于「最近码 vs 次近码」的距离差（归一化后），
exp(-d/eps) 饱和成 one-hot，Sinkhorn 的行列归一化只能同比例缩放、无法把质量
搬到次近码 -> argmax 恒定 -> 同一批近重复物品每轮都回到同一码（不动点）。

验证：
  A) 复现 FAISS RQ + 找出 sid_sk 里残留的碰撞组，报告每个物品的
     d1(最近) / gap(d2-d1) 原始值与归一化值（center_distance_for_constraint 口径）
  B) 对这批卡住物品做 eps 扫描 {0.003, 0.01, 0.03, 0.1, 0.3}：
     各 eps 下 argmax 消掉多少碰撞 + 多少物品被迫离开最近码（保真度代价代理）
"""
import collections
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))          # rq/
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from rqkmeans_faiss import train_faiss_rq, get_rq_codebooks, encode_with_rq  # noqa: E402
from models.layers import sinkhorn_algorithm                                 # noqa: E402
from models.vq import VectorQuantizer                                        # noqa: E402

RUN = os.path.join(ROOT, "results", "sid_e5000", "IandS", "rqkmeans")
EMB = os.path.join(ROOT, "data", "Amazon23", "IandS", "emb", "long", "emb_fused_gate_e60.npy")

emb = np.load(EMB).astype(np.float32)
N = emb.shape[0]
raw = np.load(os.path.join(RUN, "sid_raw.npy"))
sk = np.load(os.path.join(RUN, "sid_sk.npy"))

print("== 复现 FAISS RQ（验证确定性）==")
rq = train_faiss_rq(emb, 3, 256, verbose=False)
cbs = get_rq_codebooks(rq)
codes = encode_with_rq(rq, emb, 256, verbose=False)
print("raw 复现一致:", bool((codes.astype(int) == raw.astype(int)).all()))

keys = [tuple(r) for r in sk.tolist()]
cnt = collections.Counter(keys)
coll_codes = sorted(k for k, v in cnt.items() if v > 1)
coll_items = [i for i, k in enumerate(keys) if cnt[k] > 1]
print(f"\n== A) 残留碰撞：{len(coll_codes)} 组 / {len(coll_items)} 物品 ==")

# 残差（前两层码不变）-> 末层码本距离
resid = emb.copy()
for l in range(2):
    resid -= cbs[l][sk[:, l]]
C = cbs[2]
K = C.shape[0]


def dists(items):
    r = torch.from_numpy(resid[items])
    c = torch.from_numpy(C)
    return torch.sum(r ** 2, 1, keepdim=True) + torch.sum(c ** 2, 1) - 2 * r @ c.T


print(f"{'组码':<14}{'组大小':>5}  各成员 d1        归一化 gap(d2-d1)")
norm_gaps = []
for k in coll_codes:
    items = [i for i, kk in enumerate(keys) if kk == k]
    d = dists(items).numpy()
    d1 = d.min(axis=1)
    part = np.partition(d, 1, axis=1)[:, :2]
    gap = part[:, 1] - part[:, 0]
    amp = d.max(axis=1) - d.min(axis=1) + 1e-5        # center_distance_for_constraint 的 amplitude
    gnorm = gap / amp
    norm_gaps.extend(gnorm.tolist())
    print(f"{str(k):<14}{len(items):>5}  {np.round(d1, 4).tolist()}  {np.round(gnorm, 4).tolist()}")

norm_gaps = np.array(norm_gaps)
print(f"\n归一化 gap: min={norm_gaps.min():.4f}  median={np.median(norm_gaps):.4f}  "
      f"max={norm_gaps.max():.4f}   |  eps=0.003")
print(f"gap > 10*eps 的物品占比: {(norm_gaps > 0.03).mean():.1%}   "
      f"（gap >> eps -> exp(-d/eps) 饱和，Sinkhorn 无法搬质量）")

print("\n== B) 对这批卡住物品做 eps 扫描（一个 batch，sk_iters=50）==")
d = dists(coll_items)
d_argmin = torch.argmin(d, dim=-1)
mse_before = float(((resid[coll_items] - C[sk[coll_items, 2]]) ** 2).sum(1).mean())

print(f"{'eps':>7}{'argmax 后仍碰撞组':>12}{'离开最近码物品':>12}{'组内 MSE 变化':>14}")
for eps in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
    dd = VectorQuantizer.center_distance_for_constraint(d).double()
    Q = sinkhorn_algorithm(dd, eps, 50)
    new = torch.argmax(Q, dim=-1).numpy()
    newkeys = collections.Counter()
    base = {i: sk[i] for i in coll_items}
    newkeys = collections.Counter(
        tuple([base[i][0], base[i][1], int(c)]) for i, c in zip(coll_items, new))
    n_still = sum(1 for v in newkeys.values() if v > 1)
    moved = int((new != d_argmin.numpy()).sum())
    mse_new = float(((resid[coll_items] - C[new]) ** 2).sum(1).mean())
    print(f"{eps:>7}{n_still:>12}{moved:>12}{mse_new - mse_before:>+14.6f}")
