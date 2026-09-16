"""--tasks 开关探针：验证"分任务训练"的路由正确性。

为什么需要它
------------
sft.py 原本把 3 路 Dataset 无条件 ConcatDataset 拼成一份（MiniOneRec 原样）。
改成 --tasks 开关后，"选了某几路" ≠ "实际训了那几个任务" —— 最典型的坑是
T2a/T2b 由同一个 SidItemFeatDataset 同时产出（data.py:711-725 两个循环），
类外拆不开。本脚本用真实 tokenizer + 真实产物把路由验穿。

检查项
------
A. 解析  : resolve_tasks 对合法/非法/边界输入的返回与 WARN
B. 路由  : 三路 Dataset 实际产出什么任务（prompt 模板 + 目标类型 SID/TEXT）
C. 规模  : 全量（--tasks 全开）时每路条数与占比

用法
----
    ./.venv/Scripts/python.exe scripts/sft/probe_task_switch.py --domain IandS
    ./.venv/Scripts/python.exe scripts/sft/probe_task_switch.py --domain IandS --n-sample 500
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CATEGORY_MAP = {"IandS": "industrial and scientific items", "VG": "video games"}


def hr(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="IandS", help="IandS | VG")
    p.add_argument("--base-model", default=os.path.join(ROOT, "models", "Qwen3-0.6B"))
    p.add_argument("--data-root", default=os.path.join(ROOT, "data", "Amazon23"))
    p.add_argument("--n-sample", type=int, default=2000, help="每路取样条数；-1 为全量（很慢）")
    p.add_argument("--cutoff-len", type=int, default=400)
    return p.parse_args()


def main():
    args = parse_args()
    dom = args.domain
    sft_dir = os.path.join(args.data_root, dom, "sft")
    vocab_path = os.path.join(sft_dir, "info", "sid_vocab.json")
    index_path = os.path.join(sft_dir, "index", f"{dom}.index.json")
    item_path = os.path.join(sft_dir, "index", f"{dom}.item.json")
    train_csv = os.path.join(sft_dir, "train", f"{dom}_5_train.csv")
    text2sid = os.path.join(sft_dir, "tasks", "text2sid.jsonl")
    category = CATEGORY_MAP.get(dom, "industrial and scientific items")

    for p in (vocab_path, index_path, item_path, train_csv):
        if not os.path.exists(p):
            print(f"[FATAL] 缺文件: {p}")
            return 2

    # ---------------------------------------------------------------- A
    hr("A. resolve_tasks 解析（sft.py 的真实函数）")
    from sft import resolve_tasks, TASK_REGISTRY

    print(f"  注册表: {list(TASK_REGISTRY)}")
    cases = [
        ("T1,T2a,T2b,T3", True, 0),   # 默认 = Run-0 锚点
        ("T1",            True, 0),
        ("T1,T3",         True, 0),
        ("T1,T2a",        True, 1),   # 只给一半 -> 应 WARN
        ("T2a",           True, 1),
        ("T1,,T2b",       True, 1),   # 空项被忽略 -> ['T1','T2b']；只选 T2b 不选 T2a，应 WARN
        ("T9",            False, 0),
        ("",              False, 0),
    ]
    bad = 0
    for s, should_pass, want_warns in cases:
        try:
            sel, warns = resolve_tasks(s)
            got, detail = True, f"selected={sel} warns={len(warns)}"
        except ValueError as e:
            got, detail = False, f"ValueError: {str(e)[:52]}"
            sel, warns = [], []
        flag = "OK  "
        if got != should_pass or (got and len(warns) != want_warns):
            flag = "FAIL"
            bad += 1
        print(f"  [{flag}] tasks={s!r:16s} -> {detail}")
    print(f"  -> {'PASS' if not bad else f'FAIL ({bad})'}")
    if bad:
        return 1

    # ---------------------------------------------------------------- B
    hr(f"B. 三路 Dataset 实际任务身份  [{dom}]")
    from transformers import AutoTokenizer
    from sft import SidVocabLoader
    from data import SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset

    loader = SidVocabLoader(vocab_path)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    added = tok.add_tokens(loader.tokens)
    sid_ids = {tok.convert_tokens_to_ids(t) for t in loader.tokens}
    print(f"  tokenizer +{added} SID tokens -> len={len(tok)}")

    n = args.n_sample
    plan = [
        ("T1",  "SidSFTDataset",       lambda: SidSFTDataset(
            train_file=train_csv, tokenizer=tok, max_len=args.cutoff_len,
            sample=n, seed=42, category=category)),
        ("T2a/T2b", "SidItemFeatDataset", lambda: SidItemFeatDataset(
            item_file=item_path, index_file=index_path, tokenizer=tok,
            max_len=args.cutoff_len, sample=n, seed=42, category=category)),
        ("T3",  "FusionSeqRecDataset", lambda: FusionSeqRecDataset(
            train_file=train_csv, item_file=item_path, index_file=index_path,
            tokenizer=tok, max_len=args.cutoff_len, sample=n, seed=42, category=category)),
    ]

    failures = 0
    for tag, cls_name, build in plan:
        t0 = time.time()
        ds = build()
        dt = time.time() - t0
        if type(ds).__name__ != cls_name:
            print(f"  [FAIL] {tag}: 期望 {cls_name}，实得 {type(ds).__name__}")
            failures += 1
            continue
        row = ds[0]
        lab = [x for x in row["labels"] if x != -100]
        n_sid_in_target = sum(1 for x in lab if x in sid_ids)
        tgt_text = tok.decode(lab).replace("\n", "\\n")
        prompt_tail = tok.decode(row["input_ids"])[-110:].replace("\n", "\\n")
        target_kind = "SID" if n_sid_in_target >= 3 else "TEXT"
        print(f"\n  --- {tag}  ({cls_name})  {dt:.1f}s  len={len(ds):,} ---")
        print(f"      prompt 尾部 : ...{prompt_tail}")
        print(f"      目标        : {tgt_text[:96]}")
        print(f"      目标类型    : {target_kind}  (3-SID 命中 {n_sid_in_target})")
        if tag == "T1" and target_kind != "SID":
            print("      [FAIL] T1 目标必须是 SID"); failures += 1
        if tag == "T3" and target_kind != "TEXT":
            print("      [FAIL] T3 目标必须是 title 文本"); failures += 1

    # ---------------------------------------------------------------- C
    hr("C. 全量规模（--tasks T1,T2a,T2b,T3）")
    n_rows = 0
    with open(train_csv, encoding="utf-8") as f:
        next(f)
        for _ in f:
            n_rows += 1
    n_items = len(json.load(open(index_path, encoding="utf-8")))
    n_t4 = sum(1 for _ in open(text2sid, encoding="utf-8")) if os.path.exists(text2sid) else 0

    rows = [
        ("T1",  "SidSFTDataset",       n_rows,        "train CSV  -> 目标 SID"),
        ("T2a", "SidItemFeatDataset",  n_items,       "index.item.json  -> title"),
        ("T2b", "SidItemFeatDataset",  n_items,       "index.item.json  -> SID（同一实例的另一个循环）"),
        ("T3",  "FusionSeqRecDataset", n_rows,        "同一 train CSV，目标换成 title"),
    ]
    total = sum(r[2] for r in rows)
    for t, cls, cnt, note in rows:
        print(f"  {t:4s} {cls:22s} {cnt:>9,}  {cnt/total*100:5.1f}%   {note}")
    print(f"  {'':4s} {'合计':22s} {total:>9,}  100.0%")
    print(f"\n  [未接线] T4 text2sid: {n_t4:,} 条（tasks/text2sid.jsonl）"
          f" —— data.py 无对应 Dataset 类，要跑需新写")

    hr("结论")
    print(f"  A 解析 : {'PASS' if not bad else 'FAIL'}")
    print(f"  B 路由 : {'PASS' if not failures else f'FAIL ({failures})'}")
    print(f"  默认 --tasks=T1,T2a,T2b,T3 合计 {total:,} 条（与 MiniOneRec ConcatDataset 等价）")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
