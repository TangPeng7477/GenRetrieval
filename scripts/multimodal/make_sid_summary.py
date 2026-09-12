#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从一次 SID 实验的产物目录重新生成 summary.json。

为什么单独成脚本：
  run_sid_exp.sh 里原本内联了一段 python 做汇总，但它用「融合模式(mode)」去取
  eval_sid.py 报告的顶层 key，而报告的 key 其实是「域名(short)」→ 除 icr 外的
  三件套指标全落成了 null。抽出来后不必重训即可修复，也方便单独重跑汇总。

用法：
  python scripts/multimodal/make_sid_summary.py --out_dir results/sid/IandS/text
  python scripts/multimodal/make_sid_summary.py --out_dir results/sid/IandS/mlp --short IandS --mode mlp
"""
import argparse
import json
import os


def load(p, default=None):
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--short", default=None, help="域名，默认从路径倒数第二段推断")
    ap.add_argument("--mode", default=None, help="融合模式，默认从路径最后一段推断")
    args = ap.parse_args()

    out = args.out_dir.rstrip("/\\")
    parts = os.path.normpath(out).split(os.sep)
    mode = args.mode or (parts[-1] if parts else "")
    short = args.short or (parts[-2] if len(parts) >= 2 else "")

    tm = load(os.path.join(out, "train_metrics.json"), {})

    # eval_sid.py 的报告顶层 key 是「域名(short)」，不是融合模式(mode)，两种都兜一遍
    ev_raw_all = load(os.path.join(out, "eval_raw.json"), {})
    ev_sk_all = load(os.path.join(out, "eval_sk.json"), {})
    raw = ev_raw_all.get(short) or ev_raw_all.get(mode) or ev_raw_all
    sk = ev_sk_all.get(short) or ev_sk_all.get(mode) or ev_sk_all

    sraw = load(os.path.join(out, "sid_raw.stats.json"), {})
    ssk = load(os.path.join(out, "sid_sk.stats.json"), {})

    def row(tag, ev, st):
        u = ev.get("uniqueness", {}) or {}
        s = ev.get("structure", {}) or {}
        f = ev.get("fidelity", {}) or {}
        lcp = s.get("lcp", {}) or {}
        coh = (s.get("cohesion") or [{}])[0]
        return {
            "icr": u.get("icr") if u.get("icr") is not None else st.get("icr"),
            "dead_code_L1": (u.get("per_layer") or [{}])[0].get("dead_code_rate"),
            "lcp_nn": lcp.get("nn_mean"),
            "lcp_random": lcp.get("random_mean"),
            "lcp_ratio": lcp.get("lcp_ratio"),
            "cohesion_L1": coh.get("cohesion_ratio"),
            # A) 实际交付 codes / B) 纯 argmin 理论上限 / 二者之差 = 碰撞消解代价
            "fidelity_mse": (f.get("final") or {}).get("recon_mse"),
            "fidelity_mse_argmin": (f.get("final_argmin") or {}).get("recon_mse"),
            "fidelity_cost": f.get("fidelity_cost_of_collision_resolution"),
            "recon_r2": (f.get("final") or {}).get("r2"),
        }

    summary = {
        "short": short,
        "mode": mode,
        "emb": (tm.get("args") or {}).get("emb"),
        "out_dir": out,
        "train": {
            "epochs": (tm.get("args") or {}).get("epochs"),
            "batch_size": (tm.get("args") or {}).get("batch_size"),
            "lr": (tm.get("args") or {}).get("lr"),
            "max_steps": tm.get("max_steps"),
            "n_params": tm.get("n_params"),
            "best_loss": (tm.get("best") or {}).get("loss"),
            "best_collision": (tm.get("best") or {}).get("collision_rate"),
            "best_collision_epoch": (tm.get("best") or {}).get("collision_epoch"),
            "elapsed_sec": tm.get("elapsed_sec"),
        },
        "sid_raw": row("raw", raw, sraw),
        "sid_sk": row("sk", sk, ssk),
        "sinkhorn_cost": {
            "n_items_changed": ssk.get("n_items_changed_vs_raw"),
            "changed_ratio": ssk.get("changed_ratio"),
            "icr_gain": ssk.get("icr_gain"),
        },
    }

    dst = os.path.join(out, "summary.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    def fmt(v):
        if v is None:
            return "  n/a "
        if isinstance(v, (int, float)):
            return f"{v:6.4f}"
        return f"{v:>6}"

    print(f"\n--- {short}/{mode} ---")
    print(f"{'指标':<18}{'raw':>10}{'sinkhorn':>12}")
    keys = [("icr", "ICR"), ("dead_code_L1", "dead码 L1"), ("lcp_nn", "LCP(nn)"),
            ("lcp_random", "LCP(random)"), ("lcp_ratio", "LCP ratio"),
            ("cohesion_L1", "cohesion L1"), ("fidelity_mse", "fidelity MSE"),
            ("fidelity_mse_argmin", "fidelity(argmin)"), ("fidelity_cost", "fidelity代价")]
    for k, label in keys:
        print(f"{label:<18}{fmt(summary['sid_raw'].get(k)):>10}{fmt(summary['sid_sk'].get(k)):>12}")
    print(f"[saved] {dst}")


if __name__ == "__main__":
    main()
