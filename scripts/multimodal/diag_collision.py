#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断 Sinkhorn 之后仍未消除的碰撞，到底属于哪一类。

三类假设：
  A. 硬下界：物品 embedding 逐位相同（重复商品），任何确定性量化都必然同码
  B. batch 内均匀分配的必然结果：Sinkhorn 让 batch 内 B 个物品均分到 K 个码，
     B=2048 >> K=256 时期望每码 8 个 -> 结构性碰撞
  C. 迭代未收敛：还在爬升，轮次不够

用法：
  python scripts/multimodal/diag_collision.py \
      --emb data/Amazon23/IandS/emb/long/emb_fused_text.npy \
      --sid results/sid/IandS/text/sid_sk.npy \
      --sid_raw results/sid/IandS/text/sid_raw.npy
"""
import argparse
import collections
import os
import sys

import numpy as np


class _Tee(object):
    """同时写 stdout 和 utf-8 文件（Windows 控制台 GBK 会炸，落盘最稳）。"""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.s = sys.stdout

    def write(self, s):
        self.s.write(s)
        self.f.write(s)

    def flush(self):
        self.s.flush()
        self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--sid", required=True, help="sinkhorn 后的 codes")
    ap.add_argument("--sid_raw", default=None, help="argmin 的 codes（可选，对照）")
    ap.add_argument("--codebook_K", type=int, default=256, help="最后一层码本大小")
    ap.add_argument("--batch", type=int, default=2048, help="导出时的 batch（Sinkhorn 作用范围）")
    ap.add_argument("--out", default=None, help="同时把结果写入该 utf-8 文件")
    args = ap.parse_args()

    if args.out:
        sys.stdout = _Tee(args.out)

    emb = np.load(args.emb)
    sid = np.load(args.sid)
    N = len(emb)
    print(f"[data] emb={emb.shape}  sid={sid.shape}  N={N:,}")

    # ---------- 假设 A：逐位相同的 embedding ----------
    # 用 bytes 视图做精确去重（避免浮点哈希不稳定问题）
    b = np.ascontiguousarray(emb).view(np.void(emb.dtype.itemsize * emb.shape[1]))
    uniq_emb, inv, cnt = np.unique(b.ravel(), return_inverse=True, return_counts=True)
    n_dup_items = int(cnt[cnt > 1].sum())
    n_dup_groups = int((cnt > 1).sum())
    print(f"\n[A] 逐位相同的 embedding：{n_dup_items:,} 个物品落在 {n_dup_groups:,} 个重复组里 "
          f"（硬下界，占全库 {n_dup_items/N:.2%}）")
    print(f"    理论上界 ICR <= {uniq_emb.size/N:.4f}")

    # ---------- 当前碰撞状况 ----------
    keys = [",".join(map(str, row)) for row in sid]
    counter = collections.Counter(keys)
    n_uniq = len(counter)
    collided_items = sum(v for v in counter.values() if v > 1)
    sizes = collections.Counter(counter.values())
    print(f"\n[now] sid_sk: unique={n_uniq:,}/{N:,}  ICR={n_uniq/N:.4f}  "
          f"碰撞物品={collided_items:,}  碰撞组={sum(v for k,v in sizes.items() if k>1):,}")
    print(f"      碰撞组大小分布: {dict(sorted((k, v) for k, v in sizes.items() if k > 1))}")

    # ---------- 碰撞组里，有多少是「embedding 完全相同」的硬碰撞 ----------
    groups = collections.defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    hard_items, soft_items = 0, 0
    hard_groups, soft_groups = 0, 0
    soft_examples = []
    for k, idxs in groups.items():
        if len(idxs) < 2:
            continue
        sub = emb[idxs]
        if np.all(sub == sub[0]):          # 全组 embedding 相同 -> 无解
            hard_items += len(idxs)
            hard_groups += 1
        else:
            soft_items += len(idxs)
            soft_groups += 1
            if len(soft_examples) < 5:
                d = sub @ sub.T
                sims = d[np.triu_indices(len(idxs), 1)]
                soft_examples.append((len(idxs), float(sims.min()), float(sims.max())))
    tot_groups = hard_groups + soft_groups
    print(f"\n[B] 剩余碰撞拆解：")
    print(f"    硬碰撞（embedding 逐位相同，任何量化都无解）: {hard_items:,} 物品 / {hard_groups:,} 组 "
          f"({hard_groups/max(tot_groups,1):.1%} 的组)")
    print(f"    软碰撞（embedding 不同但量化同码，本可解）  : {soft_items:,} 物品 / {soft_groups:,} 组 "
          f"({soft_groups/max(tot_groups,1):.1%} 的组)")
    if soft_examples:
        print(f"    软碰撞样例 (组大小, 组内最小余弦, 最大余弦): {soft_examples}")

    # ---------- 假设 B：Sinkhorn batch 内均分的数学下界 ----------
    K, B = args.codebook_K, args.batch
    print(f"\n[C] Sinkhorn 作用范围：batch={B}，最后一层码本 K={K}")
    print(f"    双随机约束下 batch 内期望每码 {B/K:.1f} 个物品 -> 单轮内结构性碰撞不可避免；")
    print(f"    迭代之所以能爬升，是因为 collided 集合每轮变小、分批边界变化 = 重新洗牌。")
    print(f"    => 把导出 batch 降到 <= K ({K}) 可让 Sinkhorn 有机会输出近似置换（但会慢 {B//K}x）")

    # ---------- 假设 C：是否还在收敛 ----------
    log = os.path.join(os.path.dirname(os.path.abspath(args.sid)), "build_sid.log")
    if os.path.exists(log):
        lines = [l for l in open(log, encoding="utf-8") if "round" in l and "ICR=" in l]
        if lines:
            print(f"\n[D] 迭代收敛曲线（共 {len(lines)} 轮）:")
            prev = None
            for l in lines:
                icr = float(l.split("ICR=")[1].split()[0])
                delta = "" if prev is None else f"  (+{icr-prev:.4f})"
                print(f"    {l.strip()}{delta}")
                prev = icr
            if len(lines) >= 2:
                first = float(lines[0].split("ICR=")[1].split()[0])
                # 线性外推：按最后 5 轮的平均增量估 100 轮
                tail = [float(l.split("ICR=")[1].split()[0]) for l in lines[-6:]]
                rate = (tail[-1] - tail[0]) / (len(tail) - 1)
                print(f"\n    末段每轮增量 ≈ {rate:+.5f} -> 外推再跑 80 轮约 +{rate*80:+.4f}，"
                      f"封顶 ICR≈{min(1.0, tail[-1]+rate*80):.4f}")
                print(f"    首轮 {first:.4f} 已拿到 {(first-0.8295)/(prev-0.8295):.0%} 的总增益 "
                      f"-> 边际收益递减，不是单纯轮次问题")

    # ---------- raw 对照 ----------
    if args.sid_raw and os.path.exists(args.sid_raw):
        sid_raw = np.load(args.sid_raw)
        kr = [",".join(map(str, row)) for row in sid_raw]
        cr = collections.Counter(kr)
        gr = collections.defaultdict(list)
        for i, k in enumerate(kr):
            gr[k].append(i)
        hard_r = sum(len(v) for v in gr.values()
                     if len(v) > 1 and np.all(emb[v] == emb[v][0]))
        print(f"\n[ref] sid_raw 中硬碰撞物品 = {hard_r:,}（应与 sk 的 {hard_items:,} 接近，"
              f"因为硬碰撞不可解）")


if __name__ == "__main__":
    main()
