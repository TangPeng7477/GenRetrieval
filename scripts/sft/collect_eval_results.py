# -*- coding: utf-8 -*-
"""汇总 SFT 历次评估 -> docs/SFT_EVAL_RESULTS.md（入 git，供跨会话/跨环境对照）

设计（对齐 baseline/RESULTS.md 的既有惯例）：
  - **自动生成，不要手改**。数据源 = `results/sft/<EXP_ID>/eval_*.{meta,metrics}.json`
    （那两个目录被 .gitignore 忽略，明细不入仓；只有本汇总表入仓）。
  - 每次评估由 `evaluate_run0.sh` 落三个文件：`.json`（逐条预测）、`.meta.json`（版本/配置/耗时）、
    `.metrics.json`（HR / NDCG）。本脚本按同名主干把 meta 与 metrics 配对。
  - 记录字段：模型版本、HR、NDCG、**推理耗时**（+ 每样本耗时，便于跨 batch/beam 比较）。

用法：
  ./.venv/Scripts/python.exe scripts/sft/collect_eval_results.py            # -> docs/SFT_EVAL_RESULTS.md
  ./.venv/Scripts/python.exe scripts/sft/collect_eval_results.py --out X.md
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(ROOT, "results", "sft")
DEFAULT_OUT = os.path.join(ROOT, "docs", "SFT_EVAL_RESULTS.md")


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect(root):
    """返回 [(exp_id, meta, metrics)]，按 (域, EXP_ID, beam) 排序。"""
    rows = []
    for exp_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(exp_dir):
            continue
        exp_id = os.path.basename(exp_dir)
        for mp in sorted(glob.glob(os.path.join(exp_dir, "eval_*.meta.json"))):
            stem = mp[: -len(".meta.json")]
            meta = _load(mp) or {}
            metrics = _load(stem + ".metrics.json") or {}
            rows.append((exp_id, meta, metrics))
    rows.sort(key=lambda r: (
        str(r[1].get("domain", "")), r[0], r[1].get("num_beams") or 0,
        r[1].get("max_samples") or 0))
    return rows


def fmt_secs(v):
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v < 60:
        return f"{v:.0f}s"
    return f"{int(v // 60)}m{int(v % 60):02d}s"


def fmt_per_sample(secs, n):
    if not secs or not n:
        return "—"
    try:
        return f"{float(secs) / float(n) * 1000:.0f}ms"
    except (TypeError, ValueError, ZeroDivisionError):
        return "—"


def fmt_num(v, nd=4):
    return "—" if v is None else f"{float(v):.{nd}f}"


def fmt_time(iso):
    """'2026-09-16T20:41:11+08:00' -> '2026-09-16 20:41'（推理发生的时间点）"""
    if not iso:
        return "—"
    return str(iso).replace("T", " ")[:16]


def fmt_dur(secs, n):
    """'7m10s (24ms/条)'；无数据时 '—'。"""
    if not secs:
        return "—"
    total = fmt_secs(secs)
    per = fmt_per_sample(secs, n)
    return f"{total} ({per})" if per != "—" else total


def note_of(meta):
    tags = []
    if meta.get("registered_at_eval"):
        tags.append("现场注册(dry-run)")
    if meta.get("base_model_has_sid_token_map") is False:
        tags.append("非训练产物")
    if meta.get("max_samples"):
        tags.append("抽样")
    # 解码口径：do_sample 缺失（None）表示该次跑在本次改动之前，
    # 当时是**静默继承基座**（Qwen3/Qwen2.5 均为 true）⟹ 实际是束采样。
    ds = meta.get("do_sample")
    if ds is None:
        tags.append("采样:隐式(旧记录)")
    elif str(ds).lower() in ("true", "1", "yes"):
        tp = meta.get("top_p")
        tags.append(f"束采样(T={meta.get('temperature', '?')},p={tp if tp is not None else '?'})")
    return " / ".join(tags) if tags else "—"


def build_md(rows, generated_at):
    out = []
    out.append("# SFT 评估结果表")
    out.append("")
    out.append(f"> 自动生成（`./.venv/Scripts/python.exe scripts/sft/collect_eval_results.py`），"
               f"共 **{len(rows)}** 条记录，生成于 {generated_at}。**请勿手改。**")
    out.append(">")
    out.append("> 数据源：`results/sft/<EXP_ID>/eval_*.{meta,metrics}.json`（该目录被 `.gitignore` "
               "忽略，明细不入仓；**只有本表入仓**，供跨会话 / 跨环境对照）。")
    out.append("")
    out.append("## 读表前必看（口径边界）")
    out.append("")
    out.append("> 1. 🔴 **候选集 = 模型自己生成的 beam，不是全库。** "
               "`HR@K` / `NDCG@K` 都是 **beam 内排名**（`calc.py` 的 `minID < K`），"
               "**本质是 `beam_ceiling@K`** —— 与 [`baseline/RESULTS.md`](../baseline/RESULTS.md) "
               "的「全库排序、无负采样」口径**不可直接横比**（`EVAL_PROTOCOL` §3.4 / §5）。")
    out.append("> 2. 🔴 **`HR@K` 的硬上限 = beam 宽度。** 读表必须带 `beam` 列："
               "`beam=20` 的行天然被锁在 20 个候选内，**不能与 `beam=50` 的行比强弱**。")
    out.append("> 3. **不报告 MRR** —— 生成式下 `MRR ≈ 1/beam` 是结构常数，不携带排序质量信息。")
    out.append("> 4. 🔴 **`格式` 列是会改数字的变量**（`chatml` / `alpaca` / `verbatim`）："
               "同一模型换骨架后 `HR@K` 的变化里混着「格式效应」，"
               "**跨行比较前先确认 `格式` 列相同**（2026-09-18 起默认 `chatml`；`—` = 该次运行没记录）。")
    out.append("> 5. **抽样行（`样本` 列 < 全量）与全量行不可混比**，样本量不同。")
    out.append("> 6. **推理时间** = `evaluate.py` 开始生成的那一刻（`started_at`）；"
               "**耗时** = 该步墙钟时长（不含指标计算），括号内是每样本值 —— "
               "跨行比速度看括号，它已归一化掉 batch / beam / 样本数差异。")
    out.append("")
    out.append("---")
    out.append("")

    if not rows:
        out.append("_（暂无记录。跑一次 `bash evaluate_run0.sh` 后重跑本脚本。）_")
        return "\n".join(out) + "\n"

    # 按域分组
    by_domain = {}
    for exp_id, meta, metrics in rows:
        by_domain.setdefault(str(meta.get("domain") or "未知域"), []).append((exp_id, meta, metrics))

    for domain, group in by_domain.items():
        n_items = next((m.get("n_items") for _, m, _ in group if m.get("n_items")), "?")
        out.append(f"## {domain}（n_items={n_items}）")
        out.append("")
        out.append("| EXP_ID | 模型版本 | 格式 | 推理时间 | 样本 | beam | HR@1 | HR@5 | HR@10 | "
                   "HR@20 | NDCG@10 | 耗时 | commit | 备注 |")
        out.append("|---|---|:--:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")

        for exp_id, meta, metrics in group:
            hr = metrics.get("HR") or {}
            ndcg = metrics.get("NDCG") or {}
            n_eval = metrics.get("n_evaluated") or meta.get("max_samples")
            secs = meta.get("eval_seconds")
            beam = meta.get("num_beams")
            bold = f"**{exp_id}**"
            out.append(
                f"| {bold} "
                f"| `{meta.get('base_model', '—')}` "
                f"| {meta.get('prompt_format', '—')} "
                f"| {fmt_time(meta.get('started_at') or meta.get('written_at'))} "
                f"| {n_eval if n_eval else '—'} "
                f"| {beam if beam is not None else '—'} "
                f"| {fmt_num(hr.get('1'))} "
                f"| {fmt_num(hr.get('5'))} "
                f"| {fmt_num(hr.get('10'))} "
                f"| {fmt_num(hr.get('20'))} "
                f"| {fmt_num(ndcg.get('10'))} "
                f"| {fmt_dur(secs, n_eval)} "
                f"| `{meta.get('git_commit', '—')}` "
                f"| {note_of(meta)} |")
        out.append("")

    out.append("---")
    out.append("")
    out.append("## 参考锚点（非本表口径，仅供理解量级）")
    out.append("")
    out.append("| 参照 | HR@10 | 口径 |")
    out.append("|---|---:|---|")
    out.append("| 未训练 Qwen3-0.6B | 0.0000 | 本表口径（beam 内），见上表 |")
    out.append("| `sid_gr`（零 LLM，训练了自己的解码器） | 0.0168 | beam 内，`EVAL_PROTOCOL` §3.4 |")
    out.append("| `sasrec` | 0.0395 | **全库排序**，`baseline/RESULTS.md` |")
    out.append("| `content_ann` | 0.0288 | **全库排序**，`baseline/RESULTS.md` |")
    out.append("")
    out.append("🔴 `EVAL_PROTOCOL` §5 的 SFT 达标线是 **HR@10 > `sasrec`（I&S 0.0395）**，"
               "那是**全库口径**；本表的 beam 内数字要往上打赢它，需要 `beam` 足够宽"
               "（`beam_ceiling` 就是天花板）—— 比较前先确认两边的口径。")
    out.append("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="汇总 SFT 评估结果 -> Markdown")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="results/sft 目录")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        print(f"[WARN] 目录不存在: {a.root}（还没有任何评估产物）")

    rows = collect(a.root) if os.path.isdir(a.root) else []
    md = build_md(rows, datetime.now().isoformat(timespec="seconds"))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write(md)

    print(f"[ok] {a.out}   共 {len(rows)} 条记录")
    for exp_id, meta, metrics in rows:
        hr = metrics.get("HR") or {}
        print(f"     {exp_id:22s} beam={meta.get('num_beams')} "
              f"n={metrics.get('n_evaluated') or meta.get('max_samples')} "
              f"HR@10={fmt_num(hr.get('10'))} "
              f"推理={fmt_secs(meta.get('eval_seconds'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
