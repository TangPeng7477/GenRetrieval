#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 RQ-VAE 训练损失曲线（总损失 / 重建损失 / 量化损失 三相结构）
================================================================
数据来源（读盘，不硬编码）：
  results/sid_e5000/IandS/<run>/train_metrics.json  -> curve[] / warmup / selected_epoch

产出：docs/figures/train_curve_3phase.png

用法:
  ./.venv/Scripts/python.exe scripts/multimodal/plot_train_curve.py [--run gate__init8192]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_TOTAL = "#5A5A5A"
C_RECON = "#2E8B57"
C_RQ = "#D2691E"
C_GRID = "#D8D8D8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/sid_e5000/IandS")
    ap.add_argument("--run", default="gate__init8192")
    ap.add_argument("--out_dir", default="docs/figures")
    args = ap.parse_args()

    with open(os.path.join(args.root, args.run, "train_metrics.json"), encoding="utf-8") as f:
        j = json.load(f)
    curve = j["curve"]
    ep = np.array([c["epoch"] for c in curve], dtype=float)
    loss = np.array([c["loss"] for c in curve], dtype=float) * 1e4
    recon = np.array([c["recon_loss"] for c in curve], dtype=float) * 1e4
    rq = loss - recon
    bpe = j.get("batches_per_epoch") or 1
    warm_ep = (j.get("warmup_steps") or 0) / bpe
    sel = j.get("selected_epoch")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    ax.plot(ep, loss, color=C_TOTAL, lw=2.0, marker="o", ms=2.6, label="总损失（表观指标）")
    ax.plot(ep, recon, color=C_RECON, lw=2.0, marker="o", ms=2.6, label="重建损失 recon")
    ax.plot(ep, rq, color=C_RQ, lw=2.0, ls="--", marker="o", ms=2.6,
            label="量化损失 rq = 总 − recon")

    if warm_ep:
        ax.axvline(warm_ep, color="#9AA0A6", ls=":", lw=1.3)
        ax.text(warm_ep * 1.1, ax.get_ylim()[1] * 0.93, f"warmup 结束\n(ep{warm_ep:.0f})",
                fontsize=9, color="#666666", va="top")

    ax.set_xscale("log")
    ax.set_xticks([1, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000])
    ax.set_xticklabels(["1", "10", "50", "100", "500", "1000", "2000", "3000", "4000", "5000"])
    ax.set_xlabel("epoch（对数轴）")
    ax.set_ylabel("损失（x1e-4）")
    ax.grid(alpha=0.35, color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.annotate("① 早期：三线同降\n（重建主导）",
                xy=(40, recon[39] if len(recon) > 39 else recon[-1]),
                xytext=(4, loss[0] * 0.72), fontsize=9.5, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.1))
    ax.annotate("② 中后期：recon 仍降，rq 反向抬升\n（rate-distortion 记账位移）",
                xy=(2500, rq[2499] if len(rq) > 2499 else rq[-1]),
                xytext=(200, rq[-1] * 1.9), fontsize=9.5, color="#8A4B1E",
                arrowprops=dict(arrowstyle="->", color="#C08A5E", lw=1.1))

    if sel:
        ax.axvline(sel, color="#2E8B57", ls="-.", lw=1.2, alpha=0.8)
        ax.text(sel * 0.55, ax.get_ylim()[1] * 0.55, f"选中 ckpt\nep{sel}",
                fontsize=9, color="#2E8B57", ha="center", va="top")

    ax.set_title(f"RQ-VAE 训练曲线三相结构（{args.run}，{int(ep[-1])} 轮）",
                 fontsize=12.5, pad=12, color="#1A1A1A")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "train_curve_3phase.png")
    fig.savefig(out, facecolor="white", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"recon: {recon[0]:.2f} -> {recon[-1]:.2f}  rq: {rq[0]:.2f} -> {rq[-1]:.2f}  -> {out}")


if __name__ == "__main__":
    main()
