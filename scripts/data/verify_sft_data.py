"""SFT 数据集体检：用真实 tokenizer 量 token 长度、校验 CSV ↔ index.json 一致性。

为什么不只看行数：SID 方案的全部前提都押在「`<a_147>` 是**一个** token」上。
若 tokenizer 把它切成 5 个子词，那么 3 层 SID = 15 个 token、约束解码的 Trie 也要按子词建，
整个解码口径就变了。所以这条必须实测，不能假设。

用法
----
```bash
./.venv/Scripts/python.exe scripts/data/verify_sft_data.py --domain IandS
./.venv/Scripts/python.exe scripts/data/verify_sft_data.py --domain all
```
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


def pct(x, q):
    return float(np.percentile(x, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all")
    ap.add_argument("--data_root", default="data/Amazon23")
    ap.add_argument("--tokenizer", default="models/Qwen3-0.6B")
    ap.add_argument("--n_probe", type=int, default=3000, help="每个任务抽多少条算长度分位")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    base_vocab = len(tk)
    print(f"tokenizer: {args.tokenizer}  base vocab = {base_vocab:,}")

    domains = ["IandS", "VG"] if args.domain == "all" else [args.domain]
    report = {}

    for dom in domains:
        root = os.path.join(args.data_root, dom, "sft")
        print("\n" + "=" * 74)
        print(f"[{dom}]  {root}")
        print("=" * 74)

        with open(os.path.join(root, "stats.json"), encoding="utf-8") as f:
            stats = json.load(f)
        with open(os.path.join(root, "index", f"{dom}.index.json"), encoding="utf-8") as f:
            index = json.load(f)
        with open(os.path.join(root, "info", "prompt_templates.json"), encoding="utf-8") as f:
            tpl = json.load(f)
        with open(os.path.join(root, "info", "sid2items.json"), encoding="utf-8") as f:
            sid2items = json.load(f)

        # ── ① 加 SID token，验证「一个 SID 码 = 一个 token」────────
        with open(os.path.join(root, "info", "sid_vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        # 每个域独立 tokenizer 副本，避免跨域互相污染
        from transformers import AutoTokenizer as _AT
        t = _AT.from_pretrained(args.tokenizer, trust_remote_code=True)
        n_added = t.add_tokens(vocab)
        bad = []
        for tok in vocab[::37]:  # 抽样验证（全量 768 个也很快，抽 21 个足够抓格式问题）
            ids = t.encode(tok, add_special_tokens=False)
            if len(ids) != 1 or ids[0] < base_vocab:
                bad.append((tok, ids))
        print(f"  ① 新增 token {n_added}/{len(vocab)}  抽样 {len(vocab[::37])} 个："
              f"{'全部为单词元 ✅' if not bad else '❌ ' + str(bad[:5])}")

        sid_of = {str(i): "".join(v) for i, v in index.items()}

        # ── ② CSV ↔ index 一致性 ───────────────────────────────────
        row_counts, rt_bad, amb = {}, 0, {}
        for split in ("train", "valid", "test"):
            p = os.path.join(root, split, f"{dom}_5_{split}.csv")
            n, n_amb = 0, 0
            with open(p, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    n += 1
                    iid = row["item_id"]
                    if sid_of.get(iid) != row["item_sid"]:
                        rt_bad += 1
                    if len(sid2items[row["item_sid"]]["items"]) > 1:
                        n_amb += 1
            row_counts[split] = n
            amb[split] = n_amb
        ok_rt = (rt_bad == 0) and all(row_counts[s] == stats["splits"][s]["n_written"] for s in row_counts)
        print(f"  ② CSV 行数 {row_counts}  与 index.json 不一致 {rt_bad} 条  "
              f"{'✅' if ok_rt else '❌'}")

        # ── ③ 各任务 token 长度（真实 tokenizer）───────────────────
        HEAD = tpl["alpaca_header"]
        T = tpl["tasks"]

        def render(task, d):
            s = T[task]
            body = s["instruction"] + "\n\n" + f"### User Input: \n{s['input'].format(**d)}\n\n### Response:\n"
            target = s["output"].format(**d) + "\n"
            return HEAD + body, target

        def measure(task, records):
            lens_p, lens_t = [], []
            for d in records:
                pr, tg = render(task, d)
                lens_p.append(len(t.encode(pr, add_special_tokens=False)))
                lens_t.append(len(t.encode(tg, add_special_tokens=False)))
            lens_p = np.array(lens_p)
            lens_t = np.array(lens_t)
            return {
                "n": len(records),
                "prompt_mean": round(float(lens_p.mean()), 1),
                "prompt_p95": pct(lens_p, 95),
                "prompt_max": int(lens_p.max()),
                "target_mean": round(float(lens_t.mean()), 1),
                "target_max": int(lens_t.max()),
                "total_p95": pct(lens_p + lens_t, 95),
                "total_max": int((lens_p + lens_t).max()),
            }

        out = {}
        # T1
        for split in ("train", "valid", "test"):
            p = os.path.join(root, split, f"{dom}_5_{split}.csv")
            recs = []
            with open(p, encoding="utf-8") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    if i >= args.n_probe:
                        break
                    recs.append({"hist": T["seq2sid"]["hist_sep"].join(eval(row["history_item_sid"])),  # noqa: S307
                                 "target": row["item_sid"]})
            out[f"T1_seq2sid/{split}"] = measure("seq2sid", recs)

        # T2 / T4（全量抽样）
        recs2, recs4 = [], []
        with open(os.path.join(root, "tasks", "itemfeat.jsonl"), encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.n_probe * 2:
                    break
                d = json.loads(line)
                recs2.append({"sid": d["sid"], "title": d["title"]} if d["task"] == "sid2title"
                             else {"title": d["title"], "sid": d["sid"]})
        out["T2_itemfeat"] = measure("sid2title", recs2[: args.n_probe])
        with open(os.path.join(root, "tasks", "text2sid.jsonl"), encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.n_probe:
                    break
                d = json.loads(line)
                recs4.append({"text": d["text"], "sid": d["sid"]})
        out["T4_text2sid"] = measure("text2sid", recs4)

        print(f"  ③ token 长度（抽样 ≤{args.n_probe} 条/任务，含新增 SID 词表）")
        print(f"     {'任务':<20}{'n':>7}{'prompt均值':>11}{'prompt p95':>11}{'prompt max':>11}"
              f"{'target均值':>11}{'total p95':>10}{'total max':>10}")
        for k, v in out.items():
            print(f"     {k:<20}{v['n']:>7}{v['prompt_mean']:>11.1f}{v['prompt_p95']:>11.0f}"
                  f"{v['prompt_max']:>11}{v['target_mean']:>11.1f}{v['total_p95']:>10.0f}{v['total_max']:>10}")

        # ── ④ 解码一条真样本，确认 SID token 没被切碎 ───────────────
        with open(os.path.join(root, "train", f"{dom}_5_train.csv"), encoding="utf-8") as f:
            r0 = next(csv.DictReader(f))
        hist = eval(r0["history_item_sid"])  # noqa: S307
        pr, tg = render("seq2sid", {"hist": T["seq2sid"]["hist_sep"].join(hist), "target": r0["item_sid"]})
        ids = t.encode(pr, add_special_tokens=False) + t.encode(tg, add_special_tokens=False)
        toks = [t.decode([i]) for i in ids]
        n_sid_tok = sum(1 for x in toks if x.startswith("<") and x.endswith(">") and "_" in x)
        print(f"  ④ 首条样本 {len(ids)} token，其中 SID token {n_sid_tok} 个 "
              f"（期望 {3 * len(hist)} 历史 + 3 目标 = {3 * (len(hist) + 1)}）"
              f"  {'✅' if n_sid_tok == 3 * (len(hist) + 1) else '❌'}")
        print(f"     目标段 token 序列 = {[x for x in toks[-4:]]}")

        report[dom] = {
            "base_vocab": base_vocab,
            "n_added_tokens": n_added,
            "single_token_ok": not bad,
            "roundtrip_ok": bool(ok_rt),
            "rows": row_counts,
            "ambiguous_target": {k: round(v / row_counts[k], 4) for k, v in amb.items()},
            "lengths": out,
            "collision": stats["sid_quality"],
        }

    outp = os.path.join(args.data_root, "sft_verify.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n  report -> {outp}")


if __name__ == "__main__":
    main()
