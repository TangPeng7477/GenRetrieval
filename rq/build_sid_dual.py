#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双版本 SID 导出：同一个 RQ-VAE ckpt -> sid_raw + sid_sk（互不覆盖）
==================================================================

背景（为什么要解耦）
--------------------
`rq/build_sid.py` 只导出"碰撞消解后"的最终 SID，原始量化结果被覆盖掉了，
于是 Sinkhorn 的代价（保真度损失）无法在同一 ckpt 上量化。
本脚本把两步拆开，各存一份：

    sid_raw  —— `get_indices(use_sk=False)` 纯最近邻量化（argmin）
                训练期 sk_epsilons=0，所以这就是训练时看到的 codes
    sid_sk   —— 对**冲突物品**施加最后一层 Sinkhorn 做均衡重分配，迭代至无碰撞

这样消融是干净的：同权重、同数据，唯一变量是"要不要 Sinkhorn"。

为什么 Sinkhorn 只开最后一层
----------------------------
esci-ai-search 实测：逐层 Sinkhorn 能把 ICR 拉到 100%，但 **LCP（前缀语义保持）反而变差**
—— 前缀被打散，层次聚类结构被破坏。最后一层 Sinkhorn 是 ICR / LCP 的甜点区。
想做对照可传 `--sk_layers all`。

产物（全部落在 --out_dir）
-------------------------
    sid_raw.npy            (N, L) int32
    sid_raw.json           {item_idx: ["<a_5>","<b_12>",...]}  MiniOneRec 格式
    sid_raw.stats.json
    sid_sk.npy / sid_sk.json / sid_sk.stats.json

用法
----
    python rq/build_sid_dual.py \
        --ckpt results/sid/IandS/gate/ckpt/best_collision_model.pth \
        --emb  data/Amazon23/IandS/emb/long/emb_fused_gate_e60.npy \
        --out_dir results/sid/IandS/gate
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from datasets import EmbDataset                      # noqa: E402
from models.rqvae import RQVAE                       # noqa: E402


def build_model(a, in_dim, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = RQVAE(
        in_dim=in_dim,
        num_emb_list=a.num_emb_list, e_dim=a.e_dim, layers=a.layers,
        dropout_prob=getattr(a, "dropout_prob", 0.0), bn=getattr(a, "bn", False),
        loss_type=getattr(a, "loss_type", "mse"),
        quant_loss_weight=getattr(a, "quant_loss_weight", 1.0),
        beta=getattr(a, "beta", 0.25),
        kmeans_init=getattr(a, "kmeans_init", True),
        kmeans_iters=getattr(a, "kmeans_iters", 100),
        sk_epsilons=getattr(a, "sk_epsilons", [0.0] * len(a.num_emb_list)),
        sk_iters=getattr(a, "sk_iters", 50),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def quantize(model, data, idxs, device, batch, use_sk):
    out = np.zeros((len(idxs), len(model.num_emb_list)), dtype=np.int64)
    sub = torch.utils.data.Subset(data, list(map(int, idxs)))
    loader = DataLoader(sub, batch_size=batch, shuffle=False,
                        num_workers=0, pin_memory=True)
    p = 0
    for d in loader:
        d = d.to(device)
        ind = model.get_indices(d, use_sk=use_sk)
        ind = ind.view(-1, ind.shape[-1]).cpu().numpy()
        out[p: p + len(ind)] = ind
        p += len(ind)
    return out


def key_of(row):
    return ",".join(str(int(x)) for x in row)


def dump(out_dir, stem, codes, prefixes, N, extra_stats, t0):
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, stem + ".npy")
    json_path = os.path.join(out_dir, stem + ".json")
    np.save(npy_path, codes.astype(np.int32))

    idx_map = {i: [f"<{prefixes[l]}_{int(codes[i, l])}>" for l in range(codes.shape[1])]
               for i in range(N)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(idx_map, f, ensure_ascii=False)

    counter = collections.Counter(key_of(r) for r in codes)
    stats = {
        "n_items": int(N),
        "n_layers": int(codes.shape[1]),
        "n_unique_codes": int(len(counter)),
        "icr": round(len(counter) / N, 6),
        "collision_rate": round(1 - len(counter) / N, 6),
        "max_code_frequency": int(max(counter.values())),
        "n_collided_items": int(sum(v for v in counter.values() if v > 1)),
        "elapsed_sec": round(time.time() - t0, 1),
        "npy": npy_path,
        "json": json_path,
    }
    stats.update(extra_stats)
    with open(os.path.join(out_dir, stem + ".stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[dump] {stem}: ICR={stats['icr']:.4f}  unique={stats['n_unique_codes']:,}/{N:,}  "
          f"最大冲突={stats['max_code_frequency']}  冲突物品={stats['n_collided_items']:,}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--emb", default="", help="默认取 ckpt args 里的 emb")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    # ⚠️ 这不是训练 batch，而是 Sinkhorn 的作用范围，对 ICR 影响极大（见 docs/KNOWLEDGE_BASE.md §5.7）。
    #    Sinkhorn 在 [B, K] 距离矩阵上做双随机归一化后 argmax：B=2048 >> K=256 时每码期望 8 个物品，
    #    结构性碰撞不可避免，实测 ICR 卡在 0.935（跑 100 轮也才 0.953）。
    #    B=64（< K）时每码期望 0.25 个物品，可近似置换 -> ICR 0.9997，且第 17 轮就早停。
    #    上游 MiniOneRec rq/generate_indices.py:83 用的就是 64，我们早年误用训练 batch 2048。
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_rounds", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3,
                    help="连续多少轮 ICR 无提升就早停")
    ap.add_argument("--sk_epsilon", type=float, default=0.003,
                    help="碰撞消解时的 Sinkhorn epsilon")
    ap.add_argument("--sk_layers", default="last", choices=["last", "all"],
                    help="last=只在最后一层开（默认，保 LCP）；all=逐层开（保 ICR 伤 LCP）")
    ap.add_argument("--no_sk", action="store_true", help="跳过 Sinkhorn，只导出 sid_raw")
    ap.add_argument("--prefixes", default="a,b,c,d,e")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    emb_path = args.emb or getattr(a, "emb", "")
    if not emb_path or not os.path.exists(emb_path):
        raise SystemExit(f"找不到输入向量: {emb_path!r}（用 --emb 指定）")

    data = EmbDataset(emb_path)
    N = len(data)
    model, _ = build_model(a, data.dim, args.ckpt, device)
    L = len(model.num_emb_list)
    prefixes = [p.strip() for p in args.prefixes.split(",")]
    print(f"[build_sid_dual] emb={emb_path}\n"
          f"                 N={N:,}  in_dim={data.dim}  L={L}  "
          f"num_emb={model.num_emb_list}  e_dim={a.e_dim}\n"
          f"                 ckpt={args.ckpt}")

    # ================= pass 1: 纯最近邻量化 -> sid_raw =================
    t1 = time.time()
    codes_raw = quantize(model, data, np.arange(N), device, args.batch, use_sk=False)
    stats_raw = dump(args.out_dir, "sid_raw", codes_raw, prefixes, N, {
        "version": "raw",
        "quantize": "argmin(nearest codebook entry), use_sk=False",
        "sk_epsilon": 0.0,
        "sk_layers": "none",
        "ckpt": args.ckpt,
        "emb": emb_path,
        "pass_sec": round(time.time() - t1, 1),
    }, t0)

    if args.no_sk:
        print("[skip] --no_sk：只导出 sid_raw")
        return

    # ================= pass 2: Sinkhorn 碰撞消解 -> sid_sk =================
    eps = [0.0] * L
    if args.sk_layers == "last":
        eps[-1] = args.sk_epsilon
    else:
        eps = [args.sk_epsilon] * L
    for vq, e in zip(model.rq.vq_layers, eps):
        vq.sk_epsilon = e
    print(f"[sinkhorn] layers={args.sk_layers}  epsilons={eps}  iters={model.rq.vq_layers[0].sk_iters}")

    codes_sk = codes_raw.copy()
    keys = [key_of(r) for r in codes_sk]
    rounds = []
    best_uniq = len(set(keys))
    patience = 0
    t2 = time.time()
    for rnd in range(1, args.max_rounds + 1):
        counter = collections.Counter(keys)
        collided = [i for i, k in enumerate(keys) if counter[k] > 1]
        if not collided:
            print(f"[sinkhorn] 第 {rnd} 轮前已无碰撞 ✔")
            break
        new = quantize(model, data, np.array(collided), device, args.batch, use_sk=True)
        for pos, item in enumerate(collided):
            codes_sk[item] = new[pos]
            keys[item] = key_of(new[pos])
        uniq = len(set(keys))
        rounds.append({"round": rnd, "n_collided_before": len(collided),
                       "n_unique_after": int(uniq), "icr": round(uniq / N, 6)})
        print(f"[sinkhorn] round {rnd}: 冲突物品={len(collided):,} -> "
              f"unique={uniq:,} ICR={uniq/N:.4f} ({time.time()-t2:.0f}s)")
        if uniq > best_uniq:
            best_uniq, patience = uniq, 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[sinkhorn] 连续 {patience} 轮无提升 -> 早停"
                      f"（码本上限 {int(np.prod(model.num_emb_list)):,} vs N={N:,}）")
                break

    # 量化 sid_sk 相对 sid_raw 改动了多少物品（= 保真度代价的代理）
    changed = int(np.sum(np.any(codes_sk != codes_raw, axis=1)))
    stats_sk = dump(args.out_dir, "sid_sk", codes_sk, prefixes, N, {
        "version": "sinkhorn",
        "quantize": "argmin + last-layer sinkhorn collision resolution",
        "sk_epsilon": args.sk_epsilon,
        "sk_layers": args.sk_layers,
        "sk_iters": int(model.rq.vq_layers[0].sk_iters),
        "rounds": rounds,
        "n_items_changed_vs_raw": changed,
        "changed_ratio": round(changed / N, 6),
        "icr_raw": stats_raw["icr"],
        "icr_gain": round(len(set(key_of(r) for r in codes_sk)) / N - stats_raw["icr"], 6),
        "ckpt": args.ckpt,
        "emb": emb_path,
        "pass_sec": round(time.time() - t2, 1),
    }, t0)

    summary = {"raw": stats_raw, "sinkhorn": stats_sk,
               "note": "同一 ckpt 的两版 SID；唯一变量是是否施加 Sinkhorn 碰撞消解"}
    with open(os.path.join(args.out_dir, "sid_stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] raw ICR={stats_raw['icr']:.4f} -> sk ICR={stats_sk['icr']:.4f}  "
          f"({changed:,} 个物品被改码, {changed/N*100:.1f}%)  共 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
