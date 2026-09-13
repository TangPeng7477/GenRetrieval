"""把 baseline/results/**/metrics.json 汇成 Markdown 对比表。

用法：
    python -m baseline.scripts.summarize                 # 打印到 stdout
    python -m baseline.scripts.summarize --out baseline   # 同时写 <out>/RESULTS.md（即 baseline/RESULTS.md）
"""

from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_ROOT = os.path.join(ROOT, "baseline", "results")

# 对比矩阵的模型顺序 = RESULTS.md 的行顺序。改这里等于改对外结果表的行序。
# bert4rec 已移出对比矩阵（2026-09-13，理由见 baseline/run.py 的 registry 注释）；
# DROPPED 是防御性过滤：万一磁盘上还留着它的 metrics.json，也不会漏进结果表。
# 2026-09-14 定版：gru4rec / sasrec（经典序列）+ twotower_id（双塔）+ content_ann（内容检索探针）。
# twotower_mm 已移出主榜（实现类保留，registry 加回一行即可），产物见 _dropped/2026-09-14_cut_twotower_mm/。
ORDER = ["gru4rec", "sasrec", "twotower_id", "content_ann"]
DROPPED = {"bert4rec", "pop", "itemknn", "bprmf", "sid_prefix", "sid_gr", "twotower_mm"}

# **候选集不是全库**的模型：这些模型的 HR/NDCG/MRR 只在"自己生成出来的候选"里排序，
# 与其余行的 25k 全库排序**不可横比**。见 baseline/README.md 踩坑 B-08。
#
# 🔴 **为什么 MRR 必须置 n/a（2026-09-13 实测钉死，别再改回去）**
# `common/metrics.py::ranks_from_scores` 的 rank = `#{score > score_target} + 1`。
# beam-restricted 模型未生成的位置 score = `-inf`，于是**目标未命中时 tgt_score = -inf**，
# `rank` 退化成 ≈「实际生成的候选数 + 1」≈ **20 左右**，而不是全库量级的几千。
# 于是 `MRR = mean(1/rank)` 被结构性抬高到 ≈ 1/候选数。
# 实测证据（v2 LOO）：MRR / HR@10 比值 —— sasrec 0.54、sid_prefix 0.33、**sid_gr 3.01**。
# 前两者的 MRR 全部来自真实命中；sid_gr 的 MRR 是 HR@10 的 3 倍，只能解释为
# "未命中样本的 rank 也很小"。反推：命中样本只贡献 0.0034，其余 0.0472 来自
# 98.3% 的未命中 → 未命中时平均 rank ≈ 20.8（≈ beam 宽度）。
# ⚠️ 曾误以为 "VG sid_gr MRR=0.0555 > sasrec 0.0493 ⇒ MRR 有意义、推翻了本结论"。
#    那是错觉：VG 上 sasrec 本身更强，两者数值接近纯属巧合；比值 1.57 仍远高于 0.5。
# 判据看 **MRR/HR@10 比值**（正常 ~0.5，异常 >1），不要看绝对值。
BEAM_RESTRICTED = {"sid_gr"}
NA_FOR_BEAM = {"MRR"}          # 这些指标对 beam-restricted 模型无意义，表中直接置 n/a


def load_all(include_quick: bool = False) -> list[dict]:
    out = []
    for p in glob.glob(os.path.join(RESULT_ROOT, "*", "*", "metrics.json")):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("quick") and not include_quick:
            continue          # 冒烟结果默认不进对比表
        if d.get("model") in DROPPED:
            continue          # 已砍掉的对比项
        out.append(d)
    return out


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def table(rows: list[dict], metric_keys: list[str]) -> str:
    head = "| 组 | 模型 | " + " | ".join(metric_keys) + " | 用时 |"
    sep = "|---|---|" + "---|" * (len(metric_keys) + 1)
    lines = [head, sep]
    for r in rows:
        t = r.get("test", {})
        beam_restricted = r.get("model") in BEAM_RESTRICTED
        cells = []
        for k in metric_keys:
            if beam_restricted and k in NA_FOR_BEAM:
                cells.append("n/a ⚠️")      # 退化指标，不报
            else:
                cells.append(fmt(t.get(k)))
        name = f"**{r.get('model','')}**" + (" ⚠️候选集=beam" if beam_restricted else "")
        lines.append(f"| {r.get('group','')} | {name} | "
                     + " | ".join(cells) + f" | {r.get('total_sec','—')}s |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出 markdown 文件路径（默认只打印）")
    ap.add_argument("--include-quick", action="store_true", help="把 --quick 冒烟结果也算进来")
    args = ap.parse_args()

    rows = load_all(include_quick=args.include_quick)
    if not rows:
        raise SystemExit("还没有任何 metrics.json —— 先跑 python -m baseline.run")

    caveats = "\n".join([
        "> **读表前必看（口径边界）**",
        ">",
        "> 1. **候选集是全库、无负采样**，且 test 的 history 含 train+valid 段全部物品，",
        ">    并额外屏蔽了**重建出来的完整训练序列**（`baseline/README.md §2.2`）。",
        ">    这比常见的 LOO + 1 负样本口径**严格得多**，因此绝对值天然低于论文表格。",
        "> 2. **不要拿本表的绝对值和 `SURVEY.md` 的公开数字直接横向比**。",
        ">    文献锚点（CCFRec / MTGRec 同类目）只用于确认「量级是否在同一梯队」；",
        ">    唯一合法的比较是**本表内部各模型之间**——它们共用同一份切分、屏蔽集与指标实现。",
        "> 3. **本表为 2026-09-13 精简定版**：只保留 `gru4rec` / `sasrec`（经典序列模型）"
        " 与 `content_ann`（内容检索探针，用于回答「多模态融合向量本身值多少」）。",
        "> 4. `--quick` 冒烟结果默认不列入（结果 JSON 里带 `quick` 字段）。",
        "> 5. `sid_prefix` / `sid_gr` / `pop` / `itemknn` / `bprmf` 已移出主榜，"
        " 产物归档在 `baseline/_dropped/2026-09-13_cut_models/`（可回溯），理由见 `run.py` registry 注释。",
        ">    ⚠️ 因此**本表不再含生成式召回行**；SFT/RL 的达标线见 `docs/EVAL_PROTOCOL.md §5`（已按新矩阵重设）。",
    ])
    parts = ["# baseline 实测结果表",
             f"> 自动生成（`python -m baseline.scripts.summarize --out baseline`），共 {len(rows)} 条记录。",
             "",
             caveats]

    for dom in ["IandS", "VG"]:
        sub = [r for r in rows if r["domain"] == dom]
        if not sub:
            continue
        sub.sort(key=lambda r: ORDER.index(r["model"]) if r["model"] in ORDER else 99)
        parts.append(f"\n## {dom}（n_items={sub[0]['data']['n_items']}，"
                     f"n_users={sub[0]['data']['n_users']}）\n")
        parts.append(table(sub, ["HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10",
                                 "MRR", "coverage@10"]))
        # 缺项必须**按域**报（全局 seen 会把"另一域已跑"误判成"本域已跑"）
        dom_missing = [m for m in ORDER if m not in {r["model"] for r in sub}]
        if dom_missing:
            parts.append("")
            parts.append(f"*本域尚未产出*：{', '.join('`%s`' % m for m in dom_missing)}"
                         f"（未跑或仍在训练中）。")
        if any("beam_ceiling@20" in r.get("test", {}) for r in sub):
            parts.append("\n生成式召回的 beam 覆盖上限（目标是否出现在生成出的 SID 里）：\n")
            parts.append(table([r for r in sub if "beam_ceiling@20" in r.get("test", {})],
                               ["beam_ceiling@20", "HR@10", "hit@10_rank_mean"]))
    md = "\n".join(parts) + "\n"
    print(md)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "RESULTS.md"), "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n已写入 {os.path.join(args.out, 'RESULTS.md')}")


if __name__ == "__main__":
    main()
