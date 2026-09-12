#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ-KMeans（MiniOneRec 原版代码）+ 与 RQ-VAE 完全相同的 Sinkhorn 碰撞消解
========================================================================

目的
----
RQ-VAE（gate+init8192+5000 轮）已有五组结果；本脚本用 **MiniOneRec 复制过来的
FAISS ResidualQuantizer**（rq/rqkmeans_faiss.py，即上游原版：train_type=Train_default、
max_beam_size=1、3 层 x 256 码）在**同一份 embedding** 上生成 SID，
再施加与 `build_sid_dual.py` **完全相同**的导出期 Sinkhorn：

    * 只对碰撞物品重分配
    * 只开最后一层（sk_epsilon=0.003, sk_iters=50）
    * batch=64（Sinkhorn 作用范围，B<K 是 ICR 的关键，见 KNOWLEDGE_BASE §5.7）
    * 最多 20 轮、patience=3 早停
    * 同一个 sinkhorn_algorithm（rq/models/layers.py）+ center_distance_for_constraint

唯一差别：RQ-KMeans 没有 encoder/decoder，量化直接发生在 1024 维输入空间
（MiniOneRec 口径），RQ-VAE 是 32 维 latent。所以「保真度」一栏不可直接横比
（详见 compare 脚本注释），ICR / 死码 / LCP / cohesion 完全可比。

产物（--out_dir，与 build_sid_dual 同构）
----------------------------------------
    sid_raw.npy / sid_raw.json / sid_raw.stats.json
    sid_sk.npy  / sid_sk.json  / sid_sk.stats.json
    summary.json（键名与 make_sid_summary.py 对齐，fidelity 换成 emb 空间量化 MSE）

用法
----
    .venv/Scripts/python.exe rq/build_sid_rqkmeans.py \
        --emb data/Amazon23/IandS/emb/long/emb_fused_gate_e60.npy \
        --out_dir results/sid_e5000/IandS/rqkmeans
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# MiniOneRec 原版 RQ-KMeans（复制过来的代码，原样调用，不重新实现）
from rqkmeans_faiss import (                       # noqa: E402
    train_faiss_rq, encode_with_rq, get_rq_codebooks,
)
# 与 build_sid_dual 完全相同的产物格式（npy + MiniOneRec json + stats.json）
from build_sid_dual import dump, key_of            # noqa: E402
# 同一个 Sinkhorn 实现（训练/导出共用）
from models.layers import sinkhorn_algorithm       # noqa: E402
from models.vq import VectorQuantizer              # noqa: E402  (center_distance_for_constraint)


def kmeans_quant_mse(emb, codebooks, codes, batch=8192):
    """emb 空间直接量化误差：||x - Σ_l C_l[code_l]||²（RQ-KMeans 无 decoder，
    这就是它的"重建"；与 RQ-VAE 的 decoder 重建不可横比，仅作自身 raw/sk 对照）。"""
    N, d = emb.shape
    var = float(emb.var())
    out = []
    for k in range(1, codes.shape[1] + 1):
        recon = np.zeros((N, d), dtype=np.float32)
        for l in range(k):
            recon += codebooks[l][codes[:, l]]
        se = float(((emb - recon) ** 2).sum())
        mse = se / (N * d)
        out.append({
            "n_layers_used": k,
            "quant_mse": round(mse, 8),
            "r2": round(1.0 - mse / var, 6) if var > 0 else None,
        })
    return {"input_var": round(var, 8), "cumulative": out,
            "final": out[-1] if out else None}


def sinkhorn_resolve_last_layer(emb, codebooks, codes_raw, *, device,
                                sk_epsilon=0.003, sk_iters=50, batch=64,
                                max_rounds=20, patience=3):
    """与 build_sid_dual.py pass-2 逐行对齐的碰撞消解，只是量化器换成了 FAISS 码本。
    前面各层码不动（argmin 确定性），仅对碰撞物品在**最后一层**做 Sinkhorn 重分配。"""
    L = codes_raw.shape[1]
    N = codes_raw.shape[0]
    K = codebooks[-1].shape[0]
    C_last = torch.from_numpy(np.ascontiguousarray(codebooks[-1])).to(device)

    # 残差只依赖前 L-1 层码（本轮不会变），算一次即可
    resid = emb.astype(np.float32).copy()
    for l in range(L - 1):
        resid -= codebooks[l][codes_raw[:, l]]
    resid_t = torch.from_numpy(np.ascontiguousarray(resid)).to(device)

    codes_sk = codes_raw.copy()
    keys = [key_of(r) for r in codes_sk]
    best_uniq = len(set(keys))
    pat, rounds = 0, []
    t0 = time.time()

    for rnd in range(1, max_rounds + 1):
        counter = collections.Counter(keys)
        collided = [i for i, k in enumerate(keys) if counter[k] > 1]
        if not collided:
            print(f"[sinkhorn] 第 {rnd} 轮前已无碰撞 ✔")
            break
        idx = torch.tensor(collided, dtype=torch.long, device=device)
        for s in range(0, len(idx), batch):
            b = idx[s: s + batch]
            d = torch.sum(resid_t[b] ** 2, dim=1, keepdim=True) + \
                torch.sum(C_last ** 2, dim=1).unsqueeze(0) - \
                2.0 * resid_t[b] @ C_last.t()
            d = VectorQuantizer.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, sk_epsilon, sk_iters)
            new_last = torch.argmax(Q, dim=-1)
            codes_sk[b.cpu().numpy(), L - 1] = new_last.cpu().numpy()
        for i in collided:
            keys[i] = key_of(codes_sk[i])
        uniq = len(set(keys))
        rounds.append({"round": rnd, "n_collided_before": len(collided),
                       "n_unique_after": int(uniq), "icr": round(uniq / N, 6)})
        print(f"[sinkhorn] round {rnd}: 冲突物品={len(collided):,} -> "
              f"unique={uniq:,} ICR={uniq/N:.4f} ({time.time()-t0:.0f}s)")
        if uniq > best_uniq:
            best_uniq, pat = uniq, 0
        else:
            pat += 1
            if pat >= patience:
                print(f"[sinkhorn] 连续 {pat} 轮无提升 -> 早停"
                      f"（码本上限 {K**L:,} vs N={N:,}）")
                break
    return codes_sk, rounds, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="RQ-KMeans(MiniOneRec) + same Sinkhorn export")
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_levels", type=int, default=3)
    ap.add_argument("--codebook_size", type=int, default=256)
    # 以下四个与 build_sid_dual.py 的导出期默认值完全一致
    ap.add_argument("--sk_epsilon", type=float, default=0.003)
    ap.add_argument("--sk_iters", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64,
                    help="Sinkhorn 作用范围（非训练 batch），必须 < K=256")
    ap.add_argument("--max_rounds", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--prefixes", default="a,b,c,d,e")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    emb = np.load(args.emb).astype(np.float32)
    N, dim = emb.shape
    prefixes = [p.strip() for p in args.prefixes.split(",")]
    print(f"[rqkmeans] emb={args.emb}\n           N={N:,}  dim={dim}  "
          f"levels={args.num_levels}  K={args.codebook_size}  device={device}")

    # ---------- 1) MiniOneRec 原版 RQ-KMeans 训练 ----------
    tq = time.time()
    rq = train_faiss_rq(emb, args.num_levels, args.codebook_size)
    train_sec = round(time.time() - tq, 1)
    codebooks = get_rq_codebooks(rq)                  # (L, K, dim)

    # ---------- 2) raw 码（argmin/beam=1 最近邻）----------
    codes_raw = encode_with_rq(rq, emb, args.codebook_size, verbose=True).astype(np.int32)
    q_raw = kmeans_quant_mse(emb, codebooks, codes_raw)
    stats_raw = dump(args.out_dir, "sid_raw", codes_raw, prefixes, N, {
        "version": "raw",
        "quantizer": f"FAISS ResidualQuantizer (MiniOneRec rqkmeans_faiss.py, beam=1, {args.num_levels}x{args.codebook_size})",
        "quantize": "nearest codebook entry (beam search, max_beam_size=1)",
        "sk_epsilon": 0.0,
        "sk_layers": "none",
        "kmeans_quant": q_raw,
        "train_sec": train_sec,
        "emb": args.emb,
        "pass_sec": round(time.time() - tq, 1),
    }, t0)

    # ---------- 3) 与 RQ-VAE 相同的 Sinkhorn 碰撞消解 ----------
    codes_sk, rounds, sk_sec = sinkhorn_resolve_last_layer(
        emb, codebooks, codes_raw, device=device,
        sk_epsilon=args.sk_epsilon, sk_iters=args.sk_iters, batch=args.batch,
        max_rounds=args.max_rounds, patience=args.patience)
    changed = int(np.sum(np.any(codes_sk != codes_raw, axis=1)))
    q_sk = kmeans_quant_mse(emb, codebooks, codes_sk)
    stats_sk = dump(args.out_dir, "sid_sk", codes_sk, prefixes, N, {
        "version": "sinkhorn",
        "quantizer": "FAISS ResidualQuantizer (MiniOneRec)",
        "quantize": "argmin + last-layer sinkhorn collision resolution (同 build_sid_dual)",
        "sk_epsilon": args.sk_epsilon,
        "sk_iters": args.sk_iters,
        "sk_layers": "last",
        "batch": args.batch,
        "rounds": rounds,
        "n_items_changed_vs_raw": changed,
        "changed_ratio": round(changed / N, 6),
        "icr_raw": stats_raw["icr"],
        "icr_gain": round(len(set(key_of(r) for r in codes_sk)) / N - stats_raw["icr"], 6),
        "kmeans_quant": q_sk,
        "emb": args.emb,
        "pass_sec": round(sk_sec, 1),
    }, t0)

    # ---------- 4) summary.json（键名与 make_sid_summary.py 对齐）----------
    def ev_row(codes, q):
        used0 = len(np.unique(codes[:, 0]))
        return {
            "icr": round(len(set(key_of(r) for r in codes)) / codes.shape[0], 6),
            "dead_code_L1": round(1.0 - used0 / args.codebook_size, 6),
            "cohesion_L1": None,      # 由 eval_sid.py 补；此处只放导出期能算的
            "fidelity_mse": q["final"]["quant_mse"],
            "fidelity_mse_argmin": q_raw["final"]["quant_mse"],
            "fidelity_cost": round(q["final"]["quant_mse"] - q_raw["final"]["quant_mse"], 8),
            "recon_r2": q["final"]["r2"],
            "_fidelity_note": "emb 空间直接量化 MSE（无 decoder），与 RQ-VAE 的 decoder 重建不可横比",
        }

    summary = {
        "short": os.path.basename(os.path.dirname(args.out_dir.rstrip("/\\"))),
        "mode": os.path.basename(args.out_dir.rstrip("/\\")),
        "emb": args.emb,
        "out_dir": args.out_dir,
        "train": {  # 对齐 compare_sid_modes.py 的「训练侧」表
            "epochs": None,
            "batch_size": args.batch,
            "lr": None,
            "best_collision": None,
            "best_collision_epoch": None,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "quantizer": {
            "type": "faiss_rq_kmeans_minionerec",
            "levels": args.num_levels, "codebook": args.codebook_size,
            "train_sec": train_sec,
        },
        "sid_raw": ev_row(codes_raw, q_raw),
        "sid_sk": ev_row(codes_sk, q_sk),
        "sinkhorn_cost": {
            "n_items_changed": changed,
            "changed_ratio": round(changed / N, 6),
            "icr_gain": stats_sk["icr_gain"],
        },
        "note": "RQ-KMeans(MiniOneRec 原版代码) + 与 RQ-VAE 相同的导出期 Sinkhorn"
                "（碰撞物品 / 末层 / eps=0.003 / batch=64 / sk_iters=50 / 20轮 patience3）",
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] raw ICR={stats_raw['icr']:.4f} -> sk ICR={stats_sk['icr']:.4f}  "
          f"({changed:,} 个物品被改码, {changed/N*100:.1f}%)  共 {time.time()-t0:.0f}s")
    print(f"[note] fidelity 为 emb 空间量化 MSE：raw={q_raw['final']['quant_mse']:.8f} "
          f"sk={q_sk['final']['quant_mse']:.8f}")


if __name__ == "__main__":
    main()
