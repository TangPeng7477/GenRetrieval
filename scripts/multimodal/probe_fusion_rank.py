#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""融合模式 vs 各向异性：谁真的把点云「摊开」了（零训练）
================================================================
只读已落盘的融合向量，算协方差谱。回答一个问题：

  **`docs/UPGRADE_PLAN.md` §4.6.3 报的「PCA768 + 两侧白化 → 跨模态互检索 +39~44%」，
  对 `gate` 这种可学习融合还有没有空间？**

推理链（script 要检验的就是它）：
  · §4.6.3 的代理指标测的是**固定岭回归映射**的跨模态还原能力；
  · `gate` / `mlp` 是用 72 万共现对**监督训练**出来的非线性融合，
    其第一层本就是可学习线性变换 —— 对输入做白化 ≈ 对该层做固定重参数化，
    **不改变可实现函数类，只改善优化条件数**；
  · 若监督融合确实吸收了「输入各向异性」这个病灶，
    则 **PR(gate) 应显著高于 PR(concat) ≈ PR(text)**
    （concat 是 PCA，无监督线性，无能力自行修正各向异性）。

反之，若 PR(gate) ≈ PR(text)，说明融合并没有改善点云几何 →
V4-H 的期望值应当上调，白化必须前置。

用法：
    ./.venv/Scripts/python.exe scripts/multimodal/probe_fusion_rank.py
    ./.venv/Scripts/python.exe scripts/multimodal/probe_fusion_rank.py --domains IandS
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

TAGS = [("text", "emb_fused_text.npy"),
        ("concat", "emb_fused_concat.npy"),
        ("mlp", "emb_fused_mlp_e60.npy"),
        ("gate", "emb_fused_gate_e60.npy")]


def mean_cos(X, n_sample=3000, seed=42):
    """窄锥效应（narrow cone）判据：随机采样子集的平均成对余弦。
    越大越窄 —— 与 modality gap 文献里 "cone effect" 同口径。"""
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=min(n_sample, X.shape[0]), replace=False)
    Z = X[idx]
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    S = Z @ Z.T
    m = ~np.eye(len(Z), dtype=bool)
    return float(S[m].mean())


def spectrum(X, name):
    """返回 (PR, stable_rank, 达50%/90%方差所需PC, 前32PC累计方差, 平均成对余弦)。"""
    Xc = X - X.mean(0, keepdims=True)
    n = Xc.shape[0]
    ev = np.linalg.eigvalsh((Xc.T @ Xc) / n)[::-1]
    ev = np.clip(ev, 0, None)
    cum = np.cumsum(ev) / ev.sum()
    pr = float(ev.sum() ** 2 / np.sum(ev ** 2))
    sr = float(ev.sum() / ev[0])
    n50 = int(np.searchsorted(cum, 0.5) + 1)
    n90 = int(np.searchsorted(cum, 0.9) + 1)
    at32 = float(cum[31]) if len(cum) > 32 else float(cum[-1])
    mc = mean_cos(X)
    return dict(tag=name, dim=int(X.shape[1]), n=int(n), PR=round(pr, 1),
                stable_rank=round(sr, 1), pc50=n50, pc90=n90, var32=round(at32, 4),
                mean_cos=round(mc, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="IandS,VG")
    ap.add_argument("--out", default="results/fusion_rank.json")
    args = ap.parse_args()

    report = {}
    for dom in [d.strip() for d in args.domains.split(",") if d.strip()]:
        d = os.path.join(ROOT, "data/Amazon23", dom, "emb/long")
        print(f"\n===== {dom} =====")
        rows = []
        for tag, fn in TAGS:
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                print(f"  [skip] {tag}: 缺 {fn}")
                continue
            X = np.load(p).astype(np.float32)
            r = spectrum(X, tag)
            rows.append(r)
            print(f"  {tag:>7s}  dim={r['dim']:>4d}  PR={r['PR']:>7.1f}  "
                  f"stable_rank={r['stable_rank']:>7.1f}  "
                  f"PC@50%={r['pc50']:>4d}  PC@90%={r['pc90']:>4d}  "
                  f"前32PC方差={r['var32']:.4f}  平均成对余弦={r['mean_cos']:.4f}")
        if rows:
            base = next((r for r in rows if r["tag"] == "text"), rows[0])
            gate = next((r for r in rows if r["tag"] == "gate"), None)
            if gate:
                print(f"  → PR 增益 text→gate：×{gate['PR'] / base['PR']:.2f}")
            report[dom] = rows

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n落盘：{out}")


if __name__ == "__main__":
    main()
