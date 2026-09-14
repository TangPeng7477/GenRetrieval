#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 SFT 四类任务产物渲染成明文 prompt（Alpaca + ChatML 双版）。

输入（scripts/data/prepare_sft_data.py 的产物）：
    data/Amazon23/<域>/sft/train/<域>_5_{train,valid,test}.csv     T1
    data/Amazon23/<域>/sft/tasks/itemfeat.jsonl                    T2 (sid2title + title2sid)
    data/Amazon23/<域>/sft/tasks/seq2title_<split>.jsonl           T3
    data/Amazon23/<域>/sft/tasks/text2sid.jsonl                    T4

输出：
    data/Amazon23/<域>/sft/prompts/<fmt>/T1_seq2sid.<split>.jsonl
                                        T2_itemfeat.jsonl
                                        T3_seq2title.<split>.jsonl
                                        T4_text2sid.jsonl
                                        stats.json
    data/Amazon23/sft_prompts_verify.json

每行：{"task","split","prompt","completion","n_prompt_tok","n_compl_tok","meta"}

口径（2026-09-14 定）：
  1. instruction 句在 T1 的 input 里重复一遍 —— 保留（MiniOneRec 原文，去掉后 input 指令不明确）
  2. history 一律用 SID 序列，不用 title
  3. 所有引号一律去掉：SID 裸写（<a_43><b_123><c_242>），title / text 也裸写
     —— 尖括号本身是天然定界符；且 completion 必须裸写，否则 Trie 约束解码的
        首 token 会变成引号而不是 <a_*>，需要改 LogitProcessor 的起点
  4. completion 末尾**不加** "\\n"（MiniOneRec 原文有，但在 Trie 下必被 -inf 屏蔽，
     永远生成不出来，是死权重）。EOS 由训练端 encode(eos=True) 追加
  5. T3 只用 title 版（MiniOneRec 的 description 分支本身是注释掉的）
  6. completion 文本里**不含 EOS**，n_compl_tok 也不计 EOS

用法：
    ./.venv/Scripts/python.exe scripts/data/build_sft_prompts.py --domain IandS --max_rows 2000
    ./.venv/Scripts/python.exe scripts/data/build_sft_prompts.py --domain all
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "Amazon23")
DOMAINS = {
    "IandS": "industrial and scientific items",
    "VG": "video games",
}

# ──────────────────────────────────────────────────────────────────────
# 提示词模板
# ──────────────────────────────────────────────────────────────────────
# [A] MiniOneRec 逐字版 —— 只用于校验（证明我们没抄错），不落盘
V_HEADER = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request. \n\n"
    "### Instruction:\n"
)
V_TAILS = {
    "seq2sid": "Can you predict the next possible item that the user may expect?\n\n",
    "sid2title": "Answer the question about item identification.\n\n",
    "title2sid": "Answer the question about item identification.\n\n",
    "seq2title": "Can you recommend the next item for the user based on their interaction history?\n\n",
}
V_INPUT = {
    # 注意：T1 原文把 instruction 句又重复了一遍（口径 1，保留）
    "seq2sid": "The user has interacted with items {hist} in chronological order. "
               "Can you predict the next possible item that the user may expect?",
    "sid2title": 'What is the title of item "{sid}"?',
    "title2sid": "Which item has the title: {title}?",
    "seq2title": "The user has sequentially interacted with items {hist}. "
                 "Can you recommend the next item for him? Tell me the title of the item",
}

# [B] 本项目定版 —— 落盘用。与 [A] 的差异只有两处：引号去掉、completion 末尾的 \n 去掉
H_INSTRUCTION = {
    "seq2sid": "Can you predict the next possible item that the user may expect?",
    "sid2title": "Answer the question about item identification.",
    "title2sid": "Answer the question about item identification.",
    "seq2title": "Can you recommend the next item for the user based on their interaction history?",
    # T4 复用 T2 的 instruction（同为「按描述反查物品」）
    "text2sid": "Answer the question about item identification.",
}
H_INPUT = {
    "seq2sid": "The user has interacted with items {hist} in chronological order. "
               "Can you predict the next possible item that the user may expect?",
    "sid2title": "What is the title of item {sid}?",
    "title2sid": "Which item has the title: {title}?",
    "seq2title": "The user has sequentially interacted with items {hist}. "
                 "Can you recommend the next item for him? Tell me the title of the item",
    "text2sid": "An item can be described as follows: {text}. Which item is it describing?",
}
ALPACA_TAIL = "### User Input: \n{input}\n\n### Response:\n"

SYSTEM_MSG = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request."
)
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def alpaca_prompt(task: str, input_text: str) -> str:
    return V_HEADER + H_INSTRUCTION[task] + "\n\n" + ALPACA_TAIL.format(input=input_text)


def chatml_prompt(task: str, input_text: str) -> str:
    return (
        f"{IM_START}system\n{SYSTEM_MSG}{IM_END}\n"
        f"{IM_START}user\n{H_INSTRUCTION[task]}\n\n{input_text}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def verbatim_prompt(task: str, input_text: str) -> str:
    return V_HEADER + V_TAILS[task] + ALPACA_TAIL.format(input=input_text)


# ──────────────────────────────────────────────────────────────────────
# 数据读取
# ──────────────────────────────────────────────────────────────────────
def read_jsonl(p, limit=-1):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if 0 < limit <= len(out):
                break
    return out


def read_csv_rows(p, limit=-1):
    out = []
    with open(p, encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if 0 <= limit <= i:
                break
            out.append(r)
    return out


def build_records(dom: str, limit: int):
    """返回 [(task, split, input_text, completion, meta), ...]，不含格式信息。"""
    root = os.path.join(DATA, dom, "sft")
    recs = []

    # ── T1 seq2sid ──
    for split in ("train", "valid", "test"):
        p = os.path.join(root, "train", f"{dom}_5_{split}.csv")
        if not os.path.exists(p):
            continue
        for r in read_csv_rows(p, limit):
            hist = ast.literal_eval(r["history_item_sid"])
            recs.append((
                "seq2sid", split,
                H_INPUT["seq2sid"].format(hist=", ".join(hist)),
                r["item_sid"],
                {"user_id": r["user_id"], "target_item_id": int(r["item_id"]),
                 "n_hist": len(hist)},
            ))

    # ── T2 sid2title / title2sid ──
    p = os.path.join(root, "tasks", "itemfeat.jsonl")
    if os.path.exists(p):
        for r in read_jsonl(p, limit):
            if r["task"] == "sid2title":
                recs.append(("sid2title", "index",
                             H_INPUT["sid2title"].format(sid=r["sid"]),
                             r["title"], {"item_id": r["item_id"]}))
            else:
                recs.append(("title2sid", "index",
                             H_INPUT["title2sid"].format(title=r["title"]),
                             r["sid"], {"item_id": r["item_id"]}))

    # ── T3 seq2title ──
    for split in ("train", "valid", "test"):
        p = os.path.join(root, "tasks", f"seq2title_{split}.jsonl")
        if not os.path.exists(p):
            continue
        for r in read_jsonl(p, limit):
            recs.append((
                "seq2title", split,
                H_INPUT["seq2title"].format(hist=", ".join(r["history_item_sid"])),
                r["target_title"],
                {"user_id": r["user_id"], "target_item_id": r["target_item_id"]},
            ))

    # ── T4 text2sid ──
    p = os.path.join(root, "tasks", "text2sid.jsonl")
    if os.path.exists(p):
        for r in read_jsonl(p, limit):
            recs.append(("text2sid", "index",
                         H_INPUT["text2sid"].format(text=r["text"]),
                         r["sid"], {"item_id": r["item_id"]}))
    return recs


# ──────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────
def run_domain(dom: str, formats, limit, tok, max_len_warn):
    t0 = time.time()
    recs = build_records(dom, limit)
    print(f"[{dom}] 记录数 {len(recs):,}")

    out_root = os.path.join(DATA, dom, "sft", "prompts")
    stats = {"domain": dom, "n_records": len(recs), "limit": limit,
             "formats": {}, "tasks": {}}

    # 按 (task, split) 聚合
    groups = {}
    for task, split, inp, comp, meta in recs:
        groups.setdefault((task, split), []).append((inp, comp, meta))

    for fmt in formats:
        fdir = os.path.join(out_root, fmt)
        os.makedirs(fdir, exist_ok=True)
        render = alpaca_prompt if fmt == "alpaca" else chatml_prompt
        fmt_stat = {"files": {}, "total_rows": 0, "tok_len": {}}

        for (task, split), rows in groups.items():
            fname = f"{task}.{split}.jsonl" if split != "index" else f"{task}.jsonl"
            # T2 两个子任务合在一个文件里（与 tasks/itemfeat.jsonl 对应）
            if task in ("sid2title", "title2sid"):
                fname = "T2_itemfeat.jsonl"
            else:
                prefix = {"seq2sid": "T1", "seq2title": "T3", "text2sid": "T4"}[task]
                fname = f"{prefix}_{task}.{split}.jsonl" if split != "index" else f"{prefix}_{task}.jsonl"

            path = os.path.join(fdir, fname)
            prompts = [render(task, inp) for inp, _, _ in rows]
            comps = [c for _, c, _ in rows]

            # 批量 tokenize（只算长度）
            p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
            c_ids = tok(comps, add_special_tokens=False)["input_ids"]

            n_p = np.array([len(x) for x in p_ids])
            n_c = np.array([len(x) for x in c_ids])
            total = n_p + n_c + 1  # +1 = EOS

            mode = "a" if fname in fmt_stat["files"] else "w"
            with open(path, mode, encoding="utf-8") as f:
                for (inp, comp, meta), pr, a, b in zip(rows, prompts, n_p, n_c):
                    f.write(json.dumps({
                        "task": task, "split": split,
                        "prompt": pr, "completion": comp,
                        "n_prompt_tok": int(a), "n_compl_tok": int(b),
                        "meta": meta,
                    }, ensure_ascii=False) + "\n")

            key = f"{task}.{split}"
            fmt_stat["files"][fname] = fmt_stat["files"].get(fname, 0) + len(rows)
            fmt_stat["tok_len"][key] = {
                "n": int(len(rows)),
                "prompt_mean": float(n_p.mean()), "prompt_max": int(n_p.max()),
                "compl_mean": float(n_c.mean()), "compl_max": int(n_c.max()),
                "total_max": int(total.max()),
                "over_320": int((total > 320).sum()),
            }
            fmt_stat["total_rows"] += len(rows)
            if total.max() > 320:
                print(f"  ⚠ {fmt}/{fname} 有 {int((total>320).sum())} 条 > 320 token"
                      f" (max {total.max()})")

        with open(os.path.join(fdir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(fmt_stat, f, ensure_ascii=False, indent=2)
        stats["formats"][fmt] = fmt_stat

    stats["tasks"] = {f"{t}.{s}": len(v) for (t, s), v in groups.items()}
    stats["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_root, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[{dom}] 完成 {stats['seconds']}s -> {out_root}")
    return stats


# ──────────────────────────────────────────────────────────────────────
# 逐 token 校验：与 MiniOneRec 三个 Dataset 类产出的 input_ids 比对
# ──────────────────────────────────────────────────────────────────────
def verify(dom: str, n_sample: int):
    """逐 token 校验：用 MiniOneRec 三个 Dataset 类自己加载的数据构造 ground truth。

    已踩的坑：CSVBaseDataset 的 sample>0 走的是 pandas 随机采样
    （data.py:94 `self.data.sample(sample, random_state=seed)`），不是取前 N 条。
    所以必须用 ds.data（采样后的）逐行构造 mine —— 拿自己 jsonl 的前 N 条去对会全错。
    """
    sys.path.insert(0, ROOT)
    from transformers import AutoTokenizer
    from data import SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset

    tk = AutoTokenizer.from_pretrained("models/Qwen3-0.6B", trust_remote_code=True)
    root = os.path.join(DATA, dom, "sft")
    vocab = json.load(open(os.path.join(root, "info", "sid_vocab.json"), encoding="utf-8"))
    tk.add_tokens(vocab)
    category = DOMAINS[dom]
    EOS = tk.eos_token_id

    def enc(s):
        return tk.encode(s, add_special_tokens=False)

    report = {"domain": dom, "checks": [], "ok": True}

    def batch(name, pairs, note=""):
        bad, first = 0, None
        for gt, mine in pairs:
            if gt != mine:
                bad += 1
                if first is None:
                    for i, (a, b) in enumerate(zip(gt, mine)):
                        if a != b:
                            first = {"idx": i, "gt_tok": tk.decode([a]),
                                     "mine_tok": tk.decode([b])}
                            break
                    else:
                        first = {"idx": -1, "note": "仅长度不同"}
        c = {"name": name, "n": len(pairs), "n_mismatch": bad, "identical": bad == 0}
        if note:
            c["note"] = note
        if first:
            c["first_mismatch"] = first
        report["checks"].append(c)
        print(f"  {'[OK]' if bad == 0 else '[FAIL]'} {name}: {len(pairs) - bad}/{len(pairs)}"
              + (f"  首个差异 {first}" if first else ""))
        return bad == 0

    # ── T1 ──（verbatim 版：带引号 + completion 末尾带 \n，应与 MiniOneRec 逐 token 一致）
    ds = SidSFTDataset(train_file=os.path.join(root, "train", f"{dom}_5_train.csv"),
                       tokenizer=tk, max_len=512, sample=n_sample, category=category)
    pairs = []
    for i in range(min(n_sample, len(ds.inputs))):
        row = ds.data.iloc[i]
        hist = ast.literal_eval(str(row["history_item_sid"]))
        pr = verbatim_prompt("seq2sid", V_INPUT["seq2sid"].format(hist=", ".join(hist)))
        pairs.append((ds.inputs[i]["input_ids"],
                      enc(pr) + enc(str(row["item_sid"]) + "\n") + [EOS]))
    batch("T1.seq2sid", pairs, "verbatim 版对齐 MiniOneRec SidSFTDataset")

    # ── T2 ──（SidItemFeatDataset 内部顺序：先全 sid2title 再全 title2sid）
    ds2 = SidItemFeatDataset(item_file=os.path.join(root, "index", f"{dom}.item.json"),
                             index_file=os.path.join(root, "index", f"{dom}.index.json"),
                             tokenizer=tk, max_len=512, sample=n_sample, category=category)
    # 注意：SidItemFeatDataset 的 sample 是 random.sample（data.py:723），会打乱顺序，
    # 所以按 r["task"] 分组，不能按索引切一半
    p_s2t, p_t2s = [], []
    for i, r in enumerate(ds2.data):
        if r["task"] == "sid2title":
            pr = verbatim_prompt("sid2title", V_INPUT["sid2title"].format(sid=r["input"]))
            p_s2t.append((ds2.inputs[i]["input_ids"],
                          enc(pr) + enc(r["output"] + "\n") + [EOS]))
        else:
            pr = verbatim_prompt("title2sid", V_INPUT["title2sid"].format(title=r["input"]))
            p_t2s.append((ds2.inputs[i]["input_ids"],
                          enc(pr) + enc(r["output"] + "\n") + [EOS]))
    batch("T2.sid2title", p_s2t)
    batch("T2.title2sid", p_t2s)

    # ── T3 ──（FusionSeqRecDataset；history/target 都从它自己的 data 取，规避随机采样错位）
    ds3 = FusionSeqRecDataset(
        train_file=os.path.join(root, "train", f"{dom}_5_train.csv"),
        item_file=os.path.join(root, "index", f"{dom}.item.json"),
        index_file=os.path.join(root, "index", f"{dom}.index.json"),
        tokenizer=tk, max_len=512, sample=n_sample, category=category)
    pairs = []
    for i in range(min(n_sample, len(ds3.inputs))):
        h = ds3.get_history(ds3.data.iloc[i])
        pr = verbatim_prompt("seq2title", V_INPUT["seq2title"].format(hist=h["history_str"]))
        pairs.append((ds3.inputs[i]["input_ids"],
                      enc(pr) + enc(h["target_title"] + "\n") + [EOS]))
    batch("T3.seq2title", pairs)

    report["ok"] = all(c["identical"] for c in report["checks"])
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all", choices=["IandS", "VG", "all"])
    ap.add_argument("--formats", default="alpaca,chatml")
    ap.add_argument("--max_rows", type=int, default=-1, help="每文件最多取多少条（-1=全量）")
    ap.add_argument("--verify", action="store_true", help="跑逐 token 对齐校验")
    ap.add_argument("--n_sample", type=int, default=200)
    args = ap.parse_args()

    doms = ["IandS", "VG"] if args.domain == "all" else [args.domain]
    formats = [f for f in args.formats.split(",") if f]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.verify:
        reports = []
        for dom in doms:
            print(f"=== verify {dom} ===")
            reports.append(verify(dom, args.n_sample))
        out = os.path.join(DATA, "sft_prompts_verify.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"reports": reports,
                       "ok": all(r["ok"] for r in reports)}, f, ensure_ascii=False, indent=2)
        print(f"\n校验报告 -> {out}   全部通过: {all(r['ok'] for r in reports)}")
        return

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("models/Qwen3-0.6B", trust_remote_code=True)
    vocab = json.load(open(os.path.join(DATA, doms[0], "sft", "info", "sid_vocab.json"),
                           encoding="utf-8"))
    tk.add_tokens(vocab)
    for dom in doms:
        run_domain(dom, formats, args.max_rows, tk, 320)


if __name__ == "__main__":
    main()
