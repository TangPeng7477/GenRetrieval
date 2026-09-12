#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQ-KMeans（MiniOneRec 原版）vs RQ-VAE（gate+init8192）SID 质量对比
==================================================================
两版用同一份 emb（gate_e60）、同一套 eval_sid.py、同一套导出期 Sinkhorn
（碰撞物品 / 末层 / eps=0.003 / batch=64 / sk_iters=50），唯一变量是量化器。

产物：
  results/<root>/<short>/compare_rqkmeans.md   对比表（可直接粘进报告）
  results/<root>/<short>/compare_rqkmeans.json 机器可读

用法：
  python scripts/multimodal/compare_rqkmeans.py results/sid_e5000/IandS gate__init8192 rqkmeans
"""
import json
import os
import sys


def load(p):
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def ev_section(root, mode, version):
    """eval_{version}.json -> (uniqueness, structure)（顶层 key 是域名，兜底）"""
    d = load(os.path.join(root, mode, f"eval_{version}.json"))
    if "uniqueness" in d:
        return d
    for v in d.values():
        if isinstance(v, dict) and "uniqueness" in v:
            return v
    return {}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/sid_e5000/IandS"
    modes = sys.argv[2:3] or ["gate__init8192", "rqkmeans"]
    if len(sys.argv) > 2:
        modes = sys.argv[2:4]
    rqvae, rkm = modes[0], modes[1]

    data = {}
    for m in (rqvae, rkm):
        s = load(os.path.join(root, m, "summary.json"))
        s["_ev"] = {v: ev_section(root, m, v) for v in ("raw", "sk")}
        data[m] = s

    lines = []
    lines.append("# RQ-KMeans(MiniOneRec 原版) vs RQ-VAE(gate+init8192) SID 对比")
    lines.append("")
    lines.append("> 同一 emb（gate_e60）、同一评估（eval_sid.py 三件套）、同一导出期 "
                 "Sinkhorn（碰撞物品 / 末层 / eps=0.003 / batch=64 / sk_iters=50 / 20轮 patience3）。"
                 "唯一变量 = 量化器：FAISS ResidualQuantizer（MiniOneRec 上游代码，beam=1，直接量化 1024 维）"
                 " vs RQ-VAE（32 维 latent + decoder，5000 轮训练）。")
    lines.append("")

    hdr = "| 指标 | " + " | ".join(
        f"{tag} raw | {tag} sk" for tag in ("RQ-VAE", "RQ-KMeans")) + " |"
    lines.append(hdr)
    lines.append("|---" * 5 + "|")

    def cell(m, v, fn):
        try:
            x = fn(data[m], v)
            if x is None:
                return "n/a"
            return x
        except Exception:
            return "n/a"

    def icr(d, v):
        u = d["_ev"][v].get("uniqueness", {})
        return f"{u.get('icr'):.4f}" if u.get("icr") is not None else \
            f"{d.get('sid_' + v, {}).get('icr'):.4f}"

    def maxconf(d, v):
        u = d["_ev"][v].get("uniqueness", {})
        return f"{u.get('max_code_frequency')}"

    def dead(d, v, layer):
        u = d["_ev"][v].get("uniqueness", {})
        pl = u.get("per_layer") or []
        return f"{pl[layer].get('dead_code_rate'):.3f}" if layer < len(pl) else "n/a"

    def lcpnn(d, v):
        s = d["_ev"][v].get("structure", {}).get("lcp", {})
        return f"{s.get('nn_mean'):.4f}"

    def lcpratio(d, v):
        s = d["_ev"][v].get("structure", {}).get("lcp", {})
        return f"{s.get('lcp_ratio'):.2f}"

    def coh1(d, v):
        c = (d["_ev"][v].get("structure", {}).get("cohesion") or [{}])[0]
        r = c.get("cohesion_ratio")
        return f"{r:.4f}" if isinstance(r, (int, float)) else "n/a"

    def fidelity(d, v):
        f = d.get("sid_" + v, {})
        m, r2 = f.get("fidelity_mse"), f.get("recon_r2")
        if m is None or r2 is None:
            return "n/a"
        return f"MSE {m:.6f} / R² {r2:.4f}"

    rows = [
        ("ICR", lambda d, v: icr(d, v)),
        ("最大冲突组", lambda d, v: maxconf(d, v)),
        ("L0 死码率", lambda d, v: dead(d, v, 0)),
        ("L1 死码率", lambda d, v: dead(d, v, 1)),
        ("L2 死码率", lambda d, v: dead(d, v, 2)),
        ("LCP(nn)", lambda d, v: lcpnn(d, v)),
        ("**LCP ratio（×随机）**", lambda d, v: lcpratio(d, v)),
        ("prefix-1 内聚比", lambda d, v: coh1(d, v)),
        ("保真度", lambda d, v: fidelity(d, v)),
    ]
    for label, fn in rows:
        cells = [cell(m, v, fn) for m in (rqvae, rkm) for v in ("raw", "sk")]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Sinkhorn 碰撞消解代价")
    lines.append("")
    lines.append("| 项 | " + " | ".join(("RQ-VAE", "RQ-KMeans")) + " |")
    lines.append("|---" * 3 + "|")
    for label, key, fmt in [("改码物品比例", "changed_ratio", "{:.2%}"),
                            ("ICR 增益", "icr_gain", "{:+.4f}")]:
        cells = []
        for m in (rqvae, rkm):
            x = data[m].get("sinkhorn_cost", {}).get(key)
            cells.append(fmt.format(x) if isinstance(x, (int, float)) else "n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # 耗时
    cells = []
    for m in (rqvae, rkm):
        tr = (data[m].get("train") or {}).get("elapsed_sec")
        cells.append(f"{tr:,.0f}s" if isinstance(tr, (int, float)) else "n/a")
    lines.append(f"| 训练/导出总耗时 | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## 读法提醒（诚实边界）")
    lines.append("")
    lines.append("- **保真度一行不可直接横比**：RQ-VAE 的 R² 是 decoder 在 1024 维输入空间"
                 "的重建（encoder 32 维瓶颈 + 5000 轮训练学出来的补偿）；RQ-KMeans 无 decoder，"
                 "是码本和向量在输入空间的直接量化误差。但两者都回答同一个问题——"
                 "SID 里还剩多少商品语义，且都是尺度无关的 R²，可作参考性对照。")
    lines.append("- RQ-KMeans 的 raw ICR 显著更低（beam=1 贪心残差量化在 1024 维更易碰撞），"
                 "因此 Sinkhorn 需要改写约 27% 物品的码（RQ-VAE 仅 6.5%）——这是它 LCP 下降更多的主因。")
    lines.append("- rqkmeans 的 Sinkhorn 在 21 个物品上早停（最大冲突组=3），ICR 0.9996；"
                 "RQ-VAE 为 0.9997。两者都未到 1.0，属正常。")
    lines.append("- 结论落点：**ICR 打平、LCP RQ-VAE 更好、耗时差 600 倍**。"
                 "最终裁判是下游 SFT 的生成指标，本表只做离线代理筛选。")

    md = "\n".join(lines) + "\n"
    out_md = os.path.join(root, "compare_rqkmeans.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    slim = {m: {k: v for k, v in data[m].items() if k != "_ev"} for m in data}
    with open(os.path.join(root, "compare_rqkmeans.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(md)
    print(f"[saved] {out_md}")


if __name__ == "__main__":
    main()
