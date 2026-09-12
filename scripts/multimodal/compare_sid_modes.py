#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
横向对比同一域下各融合模式的 SID 质量，输出 Markdown + JSON。

用法：
  python scripts/multimodal/compare_sid_modes.py results/sid/IandS [text mlp gate]

产物：
  results/sid/IandS/compare.md    横向对比表（可直接粘进报告）
  results/sid/IandS/compare.json  机器可读
"""
import json
import os
import sys

ROWS = [
    ("ICR", "icr", "{:.4f}"),
    ("第0层死码率", "dead_code_L1", "{:.4f}"),
    ("LCP ratio", "lcp_ratio", "{:.2f}"),
    ("LCP(nn)", "lcp_nn", "{:.4f}"),
    ("prefix-1 内聚比", "cohesion_L1", "{:.4f}"),
    ("保真度 MSE", "fidelity_mse", "{:.6f}"),
    ("**重建 R²（尺度无关）**", "recon_r2", "{:.4f}"),
    ("argmin 理论 MSE", "fidelity_mse_argmin", "{:.6f}"),
    ("保真度代价 ΔMSE", "fidelity_cost", "{:.2e}"),
]

EXTRA = [
    ("改码物品数", "n_items_changed", "{:,}"),
    ("改码比例", "changed_ratio", "{:.2%}"),
]


def load_r2(root, mode, version):
    """取 3 层累积重建的 R²（尺度无关，跨模式可比）。"""
    p = os.path.join(root, mode, f"eval_{version}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    d = d.get(list(d.keys())[0], d) if "fidelity" not in d else d
    cum = d.get("fidelity", {}).get("cumulative", [])
    return cum[-1].get("r2") if cum else None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/sid/IandS"
    modes = sys.argv[2].split() if len(sys.argv) > 2 else ["text", "mlp", "gate"]

    data = {}
    for m in modes:
        p = os.path.join(root, m, "summary.json")
        if not os.path.exists(p):
            print(f"[skip] 无 {p}")
            continue
        s = json.load(open(p, encoding="utf-8"))
        for v, key in (("raw", "sid_raw"), ("sk", "sid_sk")):
            if key in s:
                s[key]["recon_r2"] = load_r2(root, m, v)
        data[m] = s

    if not data:
        print("没有任何 summary.json，退出")
        return

    lines = []
    short = list(data.values())[0].get("short", "?")
    lines.append(f"# SID 三模式横向对比（{short}）")
    lines.append("")
    ep = list(data.values())[0].get("train", {}).get("epochs", "?")
    lines.append(f"> 同一 RQ-VAE 配方（{ep} 轮 / bs 2048 / lr 1e-3），"
                 "raw = 纯 argmin，sk = 末层 Sinkhorn（导出 batch=64）；init_samples / 训练期 "
                 "Sinkhorn 等消融差异见目录名标签（mode__tag）与各 run 的 train_metrics.json。")
    lines.append("")

    # 主表：每模式两列
    hdr = "| 指标 | " + " | ".join(f"{m} raw | {m} sk" for m in data) + " |"
    lines.append(hdr)
    lines.append("|---" * (1 + 2 * len(data)) + "|")
    for label, key, fmt in ROWS:
        cells = []
        for m in data:
            for v in ("sid_raw", "sid_sk"):
                x = data[m].get(v, {}).get(key)
                cells.append(fmt.format(x) if isinstance(x, (int, float)) else "n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## 碰撞消解代价")
    lines.append("")
    lines.append("| 项 | " + " | ".join(data) + " |")
    lines.append("|---" * (1 + len(data)) + "|")
    for label, key, fmt in EXTRA:
        cells = []
        for m in data:
            x = data[m].get("sinkhorn_cost", {}).get(key)
            cells.append(fmt.format(x) if isinstance(x, (int, float)) else "n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    cells = []
    for m in data:
        g = data[m].get("sinkhorn_cost", {}).get("icr_gain")
        cells.append(f"{g:+.4f}" if isinstance(g, (int, float)) else "n/a")
    lines.append(f"| ICR 增益 | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## 训练侧")
    lines.append("")
    lines.append("| 项 | " + " | ".join(data) + " |")
    lines.append("|---" * (1 + len(data)) + "|")
    for label, key, fmt in [("训练轮数", "epochs", "{:,}"),
                            ("best_collision", "best_collision", "{:.6f}"),
                            ("命中轮次", "best_collision_epoch", "{:,}"),
                            ("耗时(秒)", "elapsed_sec", "{:.1f}")]:
        cells = []
        for m in data:
            x = data[m].get("train", {}).get(key)
            cells.append(fmt.format(x) if isinstance(x, (int, float)) else "n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("### 读法提醒")
    lines.append("")
    lines.append("- **top-5 首 token 命中率不可跨模式横比**：它受第 0 层码本使用率影响"
                 "（用码越少、组越大，命中率天然越高）。跨模式请只看 **LCP ratio**（相对随机基线的倍数）。")
    lines.append("- **prefix-2 / prefix-3 的 cohesion 不可信**：前缀基数大导致每组只剩 1~2 个样本，"
                 "`eval_sid.py` 已打上「退化」标记。")
    lines.append("- **raw ICR 的差距反映 embedding 分布的可量化性**，不是 SID 本身的优劣："
                 "raw 越高说明融合后的向量越容易被码本区分。")
    lines.append("- **跨模式比重建要看 R² 而不是 MSE**：本报告三模式的 `input_var` 实测几乎一致"
                 "（≈0.0009765），故 MSE 也可比；但换数据集后务必确认，否则用 R²。")

    md = "\n".join(lines) + "\n"
    out_md = os.path.join(root, "compare.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(root, "compare.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(md)
    print(f"[saved] {out_md}")


if __name__ == "__main__":
    main()
