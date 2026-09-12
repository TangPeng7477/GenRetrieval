#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 RQ-VAE 的 latent 有效维度与 L0 码本覆盖
=================================================
回答一个问题：**为什么某个域的 L0 会出死码，另一个域不会？**

零训练、只做一次前向。需要 torch + numpy，用项目 venv 跑：

    ./.venv/Scripts/python.exe scripts/multimodal/probe_latent_rank.py IandS
    ./.venv/Scripts/python.exe scripts/multimodal/probe_latent_rank.py VG

输出三块（`docs/SID_PIPELINE.md` §2.8.2 表格的数字来源）：
  1) 输入空间（1024 维融合向量）协方差谱：有效秩 / 各 PC 方差占比
  2) latent 空间（32 维，encoder 输出）同样的谱 —— 量化真正面对的点云
  3) L0 码本覆盖：每个质心到"最近样本"的距离
     死码 = 没有任何样本以它为 argmin ⟺ 附近没有样本 ⟹ 它的 min 距离应当显著更大，
     这正是"元胞落进没有数据的空方向"的直接证据。
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RQ = os.path.join(ROOT, "rq")
if RQ not in sys.path:
    sys.path.insert(0, RQ)

from models.rqvae import RQVAE  # noqa: E402


def spectrum_report(name, X):
    """返回 (participation_ratio, stable_rank, {分位方差}, {达阈值所需PC数})。"""
    Xc = X - X.mean(0, keepdims=True)
    n = Xc.shape[0]
    ev = np.linalg.eigvalsh((Xc.T @ Xc) / n)[::-1]
    ev = np.clip(ev, 0, None)
    cum = np.cumsum(ev) / ev.sum()
    pr = float(ev.sum() ** 2 / np.sum(ev ** 2))          # participation ratio
    sr = float(ev.sum() / ev[0])                          # stable rank
    need = {f"{int(t * 100)}%": int(np.searchsorted(cum, t) + 1)
            for t in (0.5, 0.9)}
    at = {f"@{k}": float(cum[k - 1]) for k in (8, 16, 32) if k <= len(cum)}
    print(f"  [{name}] dim={X.shape[1]}  trace(var)={ev.sum():.4f}  "
          f"有效秩(PR)={pr:.1f}  stable_rank={sr:.1f}")
    print(f"          方差占比 {at}  |  达 50%/90% 方差所需 PC 数 {need}")
    return pr, sr, at, need


def main():
    dom = sys.argv[1] if len(sys.argv) > 1 else "IandS"
    run = sys.argv[2] if len(sys.argv) > 2 else f"results/sid_e5000/{dom}/gate__init8192"
    run = os.path.join(ROOT, run) if not os.path.isabs(run) else run

    meta = json.load(open(os.path.join(run, "train_metrics.json"), encoding="utf-8"))
    a = meta["args"]
    emb_path = os.path.join(ROOT, a["emb"])
    ckpt_path = os.path.join(run, "ckpt", "selected_model.pth")

    X = np.load(emb_path).astype(np.float32)
    in_dim = X.shape[1]
    print("=" * 74)
    print(f"{dom}  N={len(X):,}  in_dim={in_dim}  ckpt=selected_model.pth "
          f"(ep{meta['selected_epoch']})")

    print("\n[1] 输入空间（1024 维融合向量）")
    spectrum_report("input", X.astype(np.float64))

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = RQVAE(in_dim=in_dim, num_emb_list=a["num_emb_list"], e_dim=a["e_dim"],
                  layers=a["layers"], dropout_prob=a.get("dropout_prob", 0.0),
                  bn=a.get("bn", False), loss_type=a.get("loss_type", "mse"),
                  beta=a.get("beta", 0.25), kmeans_init=a.get("kmeans_init", True),
                  sk_epsilons=a.get("sk_epsilons", [0.0] * len(a["num_emb_list"])))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(dev).eval()

    with torch.no_grad():
        zs = []
        for i in range(0, len(X), 4096):
            xb = torch.tensor(X[i:i + 4096], dtype=torch.float32, device=dev)
            zs.append(model.encoder(xb).cpu().numpy())
        z = np.concatenate(zs, 0)

    print("\n[2] latent 空间（量化真正面对的点云）")
    spectrum_report("latent", z.astype(np.float64))

    # 逐层残差（与 train.log 的 [init] 行同口径，但这里是训练后的 encoder）
    with torch.no_grad():
        zt = torch.tensor(z, dtype=torch.float32, device=dev)
        residual = zt
        print("\n[3] 各层量化前后残差模长（训练后 encoder）")
        for li, layer in enumerate(model.rq.vq_layers):
            x_res, _, _ = layer(residual, use_sk=False)
            residual = residual - x_res
            print(f"      L{li}: 残差‖·‖={float(residual.norm(dim=1).mean()):.4f}")

    # ---- L0 码本覆盖：质心到最近样本的距离 ----
    W = model.rq.vq_layers[0].embedding.weight.detach().cpu().numpy()
    K = W.shape[0]
    D = torch.cdist(zt, torch.tensor(W, dtype=torch.float32, device=dev))  # (N,K)
    dist = D.min(dim=1).values.cpu().numpy()            # 每个样本到最近质心的距离
    assign = D.min(dim=1).indices.cpu().numpy()
    min_d = D.min(dim=0).values.cpu().numpy()          # 每个质心到最近样本的距离
    order = np.argsort(min_d)
    used = np.unique(assign)
    dead = np.array([k for k in range(K) if k not in set(used.tolist())])

    print(f"\n[4] L0 码本覆盖：K={K}  实际被选中={len(used)}  死码={len(dead)}"
          f"  ({len(dead) / K:.2%})")
    print(f"      逐点最近质心距离: mean={float(dist.mean()):.4f} "
          f"p50={float(np.median(dist)):.4f} p99={float(np.percentile(dist, 99)):.4f}")
    if len(dead):
        print(f"      死码 {sorted(dead.tolist())}")
        print(f"      死码质心到最近样本距离: "
              f"min={min_d[dead].min():.4f} mean={min_d[dead].mean():.4f} max={min_d[dead].max():.4f}")
        alive = np.setdiff1d(np.arange(K), dead)
        print(f"      活码同指标            : "
              f"min={min_d[alive].min():.4f} mean={min_d[alive].mean():.4f} max={min_d[alive].max():.4f}")
        rank = {int(k): int(np.where(order == k)[0][0]) + 1 for k in dead}
        print(f"      死码在'离样本最远质心'榜上的名次: "
              f"{sorted(rank.values())} (共 {K} 名，越大=越远)")
    counts = np.bincount(assign, minlength=K)
    nz = counts[counts > 0]
    print(f"      占用分布: max={nz.max()} p50={int(np.median(nz))} min={nz.min()} "
          f"mean={nz.mean():.1f}")
    p = counts / counts.sum()
    ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(K)
    print(f"      usage 熵（归一化）={ent:.4f}")

    # 参照尺度：码本铺得比数据云更"开"多少（>1 说明边角质心在数据云之外）
    cloud = float(np.linalg.norm(z - z.mean(0), axis=1).mean())
    cb = float(np.linalg.norm(W - W.mean(0), axis=1).mean())
    print(f"      尺度参照: 样本到样本均值的平均距离={cloud:.4f}  "
          f"质心到质心均值的平均距离={cb:.4f}  比值={cb / cloud:.2f}")
    print(f"      码本质心模长均值={float(np.linalg.norm(W, axis=1).mean()):.4f}  "
          f"样本模长均值={float(np.linalg.norm(z, axis=1).mean()):.4f}")


if __name__ == "__main__":
    main()
