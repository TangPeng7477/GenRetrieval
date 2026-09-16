# -*- coding: utf-8 -*-
"""SFT 评估的元数据与指标落盘（供 evaluate_run0.sh 调用）

为什么单独写一个脚本：
  - **meta**：要记录"这次评估到底是哪个版本"。**不能写进 result json** ——
    calc.py 假设它是 `list[dict]`（`json.load` 后直接 `for sample in test_data`），
    塞元数据会破坏它。所以写到旁边的 `<name>.meta.json`。
  - **metrics**：要从 calc.py 的 stdout 里把 HR/NDCG 抽出来存 json。calc.py 是
    MiniOneRec 原版，**不改它**，只在外面解析 —— 指标口径保持单一实现，不引入第二套。

口径声明（与 calc.py 完全一致）：
  HR@K   = 目标 SID 在 beam 内的排名 < K              （即 beam_ceiling@K）
  NDCG@K = mean(1/log(rank+2)) / (1/log 2)，单目标 IDCG=1
  未生成的物品得分 = -inf ⟹ 全库排序下 top-K 等价 beam 内 top-K
  ⚠️ MRR 不报告：对生成式无意义（≈ 1/beam 是结构常数，EVAL_PROTOCOL §3.4 标 n/a）

用法：
  python scripts/sft/eval_report.py meta --out <meta.json> --set k=v [--set k2=v2 ...]
  python scripts/sft/eval_report.py metrics --log <calc.log> --out <metrics.json>
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime


def _coerce(v: str):
    """把 shell 传来的字符串还原成合适类型（便于 json 里出现真正的数字/布尔）。"""
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def cmd_meta(a) -> int:
    meta = {}
    for kv in a.set:
        k, sep, v = kv.partition("=")
        if not sep:
            print(f"[WARN] 忽略非 K=V 项: {kv}", file=sys.stderr)
            continue
        meta[k] = _coerce(v)
    meta["written_at"] = datetime.now().isoformat(timespec="seconds")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[meta] {a.out}")
    for k in sorted(meta):
        print(f"        {k:30s} = {meta[k]}")
    return 0


def _floats(s: str):
    return [float(x) for x in s.replace(",", " ").split() if x.strip()]


def _count_result_json(path: str):
    """实际评估条数 = result json 的长度。读不到就返回 None（不猜）。"""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return len(json.load(f))
    except Exception as e:
        print(f"[WARN] 读 result json 失败: {e}", file=sys.stderr)
        return None


def cmd_metrics(a) -> int:
    txt = open(a.log, encoding="utf-8", errors="replace").read()

    m_topk = re.search(r"^\[([\d,\s]+)\]\s*$", txt, re.M)
    topk = [int(x) for x in m_topk.group(1).split(",")] if m_topk else []

    def grab(pat):
        # 行首锚定 + 只在行内找 '['；对 `NDCG:` 后面是 tab / 空格 / 意外字符都稳
        mm = re.search(pat, txt, re.M)
        return _floats(mm.group(1)) if mm else []

    hr_vals = grab(r"^HR[^\n\[]*\[([^\]]*)\]")
    ndcg_vals = grab(r"^NDCG:?[^\n\[]*\[([^\]]*)\]")

    # calc.py 的裸数字行：第一处是 n_beam，最后一处是 CC
    m_beam = re.search(r"^\s*(\d+)\s*$", txt, re.M)
    lines = [ln for ln in txt.strip().splitlines() if ln.strip()]
    m_cc = re.fullmatch(r"\s*(\d+)\s*", lines[-1]) if lines else None

    out = {
        "source_log": os.path.basename(a.log),
        "k": topk,
        "HR": {k: v for k, v in zip(topk, hr_vals)},
        "NDCG": {k: v for k, v in zip(topk, ndcg_vals)},
        "n_beam": int(m_beam.group(1)) if m_beam else None,
        "n_evaluated": _count_result_json(a.result_json),
        "n_generated_not_in_item_dict": int(m_cc.group(1)) if m_cc else None,
        "metric_scope": (
            "beam 内排名（calc.py 口径：minID < K）。未生成物品得分 = -inf，"
            "故全库排序下 top-K 等价 beam 内 top-K。"
            "MRR 不报告 —— 对生成式无意义（≈1/beam 结构常数）。"
        ),
        "parsed_at": datetime.now().isoformat(timespec="seconds"),
    }

    rc = 0
    if not hr_vals or not ndcg_vals or len(hr_vals) != len(ndcg_vals):
        out["error"] = "未能从 log 解析出 HR/NDCG —— 原始 log 已保留，请人工核对"
        rc = 1

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[metrics] {a.out}")
    if rc == 0:
        for k in topk:
            print(f"        HR@{k:<3d} = {out['HR'][k]:.4f}     NDCG@{k:<3d} = {out['NDCG'][k]:.4f}")
    else:
        print(f"        [WARN] {out['error']}")
    if out["n_generated_not_in_item_dict"]:
        print(f"        [WARN] 有 {out['n_generated_not_in_item_dict']} 条生成结果不属于 item_dict"
              f"（SID 合法但映射不到物品？）")
    return rc


def main():
    ap = argparse.ArgumentParser(description="SFT 评估元数据 / 指标落盘")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("meta", help="写 <name>.meta.json（记录这次评估是哪个版本）")
    p1.add_argument("--out", required=True)
    p1.add_argument("--set", action="append", default=[], metavar="K=V")
    p1.set_defaults(fn=cmd_meta)

    p2 = sub.add_parser("metrics", help="解析 calc.py 输出 -> metrics.json")
    p2.add_argument("--log", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--result-json", default="", help="可选：读它统计实际评估条数")
    p2.set_defaults(fn=cmd_metrics)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
