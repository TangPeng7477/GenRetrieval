#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2: 训练好的量化器 ckpt -> 全体物品的 SID codes（含碰撞消解）
==============================================================
替代上游 `rq/generate_indices.py`（那份是硬编码 `xxx` 路径的草稿，无法直接运行）。

流程
----
1. 第一遍：`get_indices(use_sk=False)` —— 纯最近邻量化，快；
2. 碰撞消解：把**冲突物品**挑出来，只对它们开最后一层 Sinkhorn
   `get_indices(use_sk=True)`，利用 Sinkhorn 的均衡分配把它们推给不同码；
   迭代至无碰撞或达 --max_rounds。
   （上游做法；逐层 Sinkhorn 会伤 LCP，所以只开最后一层 —— 见 UPGRADE_PLAN §4.2）

为什么只开最后一层
------------------
esci 项目实测：逐层 Sinkhorn 能把 ICR 拉到 100%，但 LCP 反而变差
（前缀被打散，语义聚类被破坏）。最后一层 Sinkhorn 是 ICR / LCP 的甜点。

输出
----
    <out_dir>/<short>.index_<tag>.npy     (N, L) int32 —— 便于 eval_sid.py 直接吃
    <out_dir>/<short>.index_<tag>.json    {item_idx: ["<a_5>","<b_12>",...]} —— MiniOneRec 格式
    <out_dir>/<short>.index_<tag>.stats.json

用法
----
    python rq/build_sid.py --short IandS --tag rqvae_gate \
        --ckpt rq/index/output/IandS/<ts>/best_collision_model.pth \
        --emb data/Amazon23/IandS/emb/emb_fused_gate.npy
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

from datasets import EmbDataset                       # noqa: E402
from models.rqvae import RQVAE                        # noqa: E402


def read_ckpt(ckpt_path):
    """只读 ckpt 的 args（用于拿 data_path / 超参），不建模型"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["args"], ckpt


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
    model = model.to(device).eval()
    return model, a, ckpt

@torch.no_grad()
def quantize(model, data, idxs, device, batch, use_sk):
    """返回 idxs 位置的 codes (len(idxs), L)"""
    out = np.zeros((len(idxs), len(model.num_emb_list)), dtype=np.int64)
    sub = torch.utils.data.Subset(data, list(idxs))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--short", default="IandS")
    ap.add_argument("--tag", default="rqvae", help="输出文件名后缀，如 rqvae_gate")
    ap.add_argument("--emb", default="", help="默认取 ckpt 里记录的 args.data_path")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max_rounds", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3,
                    help="连续多少轮 ICR 无提升就早停")
    ap.add_argument("--sk_last", type=float, default=0.003,
                    help="碰撞消解时最后一层的 Sinkhorn epsilon")
    ap.add_argument("--prefixes", default="a,b,c,d,e")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # --- 数据路径 ---
    a, _ckpt = read_ckpt(args.ckpt)                       # 先只读 args
    emb_path = args.emb or getattr(a, "data_path", "")
    if not emb_path or not os.path.exists(emb_path):
        raise SystemExit(f"找不到输入向量: {emb_path!r}（用 --emb 指定）")

    data = EmbDataset(emb_path)
    N = len(data)
    model, a, _ = build_model(a, data.dim, args.ckpt, device)
    L = len(model.num_emb_list)
    prefixes = [p.strip() for p in args.prefixes.split(",")]
    print(f"[build_sid] emb={emb_path}  N={N:,}  in_dim={data.dim}  L={L}  "
          f"num_emb={model.num_emb_list}  e_dim={a.e_dim}")

    # --- pass 1: 最近邻量化 ---
    t1 = time.time()
    codes = quantize(model, data, np.arange(N), device, args.batch, use_sk=False)
    keys = [key_of(r) for r in codes]
    uniq = len(set(keys))
    print(f"[pass1] {time.time()-t1:.1f}s  unique={uniq:,}/{N:,} "
          f"ICR={uniq/N:.4f}  碰撞率={1-uniq/N:.4f}")

    # --- 碰撞消解：只对冲突物品开最后一层 Sinkhorn ---
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    model.rq.vq_layers[-1].sk_epsilon = args.sk_last

    round_stats = []
    best_uniq = uniq
    patience = 0
    for rnd in range(1, args.max_rounds + 1):
        counter = collections.Counter(keys)
        groups = [k for k, v in counter.items() if v > 1]
        if not groups:
            print(f"[collision] 第 {rnd} 轮前已无碰撞 ✔")
            break
        collided = [i for i, k in enumerate(keys) if counter[k] > 1]
        t2 = time.time()
        new = quantize(model, data, np.array(collided), device, args.batch, use_sk=True)
        for pos, item in enumerate(collided):
            codes[item] = new[pos]
            keys[item] = key_of(new[pos])
        uniq = len(set(keys))
        round_stats.append({"round": rnd, "n_collided_before": len(collided),
                            "n_unique_after": uniq, "icr": round(uniq / N, 6)})
        print(f"[collision] round {rnd}: 冲突物品={len(collided):,} -> "
              f"unique={uniq:,} ICR={uniq/N:.4f} ({time.time()-t2:.1f}s)")

        # 早停：码本容量不足时（Π K_l < N）继续迭代只是让 Sinkhorn 反复重排，
        # 唯一性不再提升，但保真度会被持续拉低 -> 无收益就停。
        if uniq > best_uniq:
            best_uniq, patience = uniq, 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[collision] 连续 {patience} 轮无提升 -> 早停"
                      f"（码本上限 {int(np.prod(model.num_emb_list)):,} vs N={N:,}）")
                break

    uniq = len(set(keys))
    counter = collections.Counter(keys)
    max_freq = max(counter.values())

    # --- 输出 ---
    out_dir = args.out_dir or os.path.join("data", "Amazon23", args.short)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{args.short}.index_{args.tag}" if args.tag else f"{args.short}.index"
    npy_path = os.path.join(out_dir, stem + ".npy")
    json_path = os.path.join(out_dir, stem + ".json")
    np.save(npy_path, codes.astype(np.int32))

    idx_map = {}
    for i in range(N):
        idx_map[i] = [f"<{prefixes[l]}_{int(codes[i, l])}>" for l in range(L)]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(idx_map, f, ensure_ascii=False)

    stats = {
        "short": args.short, "tag": args.tag, "ckpt": args.ckpt, "emb": emb_path,
        "n_items": int(N), "n_layers": int(L),
        "num_emb_list": [int(x) for x in model.num_emb_list],
        "e_dim": int(a.e_dim),
        "icr": round(uniq / N, 6),
        "collision_rate": round(1 - uniq / N, 6),
        "n_unique_codes": int(uniq),
        "max_code_frequency": int(max_freq),
        "sk_last": args.sk_last,
        "rounds": round_stats,
        "elapsed_sec": round(time.time() - t0, 1),
        "npy": npy_path, "json": json_path,
    }
    with open(os.path.join(out_dir, stem + ".stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[done] ICR={stats['icr']:.4f}  最大冲突={max_freq}  "
          f"耗时={stats['elapsed_sec']}s")
    print(f"       {npy_path}")
    print(f"       {json_path}")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
