#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估链路自检（离线，不需要 GPU，不需要真跑一遍 eval）。

为什么需要它 —— **恒 0 是不可证伪的**：
未训练基座的 HR/NDCG 全 0 有三种可能并存的原因，光看结果分不出是哪一种：
    ① 模型没学到（预期，见 SFT_PIPELINE §3.5）
    ② `calc.py` 的指标算错了
    ③ 结果 json 里的真值 `output` 写错了
本脚本用**已知答案**把 ②③ 排除掉，于是"全 0"只能归因于 ①。

两段检查：
  A. 指标链（`calc.py`）—— 造一份"已知名次"的合成结果，看它是否复现独立推出的期望值。
     不需任何前置产物。
  B. 真值闭环 —— 给定一份真实结果 json，按 evaluate.py 的参数原样实例化 `EvalSidDataset`，
     逐条核对 `output` / `history_str` 是否等于数据集真值（规矩：探针必须实例化真实类，
     不准手抄模板）。需要 `--result-json`。

用法：
    ./.venv/Scripts/python.exe scripts/sft/verify_eval_chain.py
    ./.venv/Scripts/python.exe scripts/sft/verify_eval_chain.py \
        --result-json results/sft/<EXP_ID>/eval_IandS_beam20_n32.json
"""
import argparse
import contextlib
import io
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)          # 否则 import calc / data 失败

PASS, FAIL = "PASS", "FAIL"
_fail = 0


def check(tag, msg, ok):
    global _fail
    if not ok:
        _fail += 1
    print(f"  [{PASS if ok else FAIL}] {tag:34s} {msg}")
    return ok


# =====================================================================  A. 指标链
def run_calc(path, item_path):
    """同进程调 calc.gao。

    ⚠️ 不要用 subprocess 解析 stdout：calc.py 用 numpy 默认 8 位小数打印，
    解析回来会丢精度、把"数值一致"误判成 FAIL（第一版就撞了这个坑）。
    """
    np.set_printoptions(precision=17)
    import calc
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        calc.gao(path=path, item_path=item_path)
    txt = buf.getvalue()
    nd = re.search(r"NDCG:\s*\[([^\]]*)\]", txt)
    hr = re.search(r"HR\s*\[([^\]]*)\]", txt)
    assert nd and hr, f"解析 calc.py 输出失败：\n{txt[-600:]}"
    f = lambda s: [float(x) for x in s.replace("\n", " ").split()]
    cc = re.findall(r"^(\d+)\s*$", txt, re.M)
    return f(nd.group(1)), f(hr.group(1)), (int(cc[-1]) if cc else None)


def expect_metrics(ranks, n, ks):
    """按 calc.py 的定义**独立**算期望（这一步是"独立的尺子"）。

        minID = 首次命中的 0 基位置；HR@K = P(minID < K)
        NDCG@K = mean( 1/ln(minID+2) · [minID < K] ) / (1/ln2)
    """
    hr = {k: sum(1 for r in ranks if r < k) / n for k in ks}
    nd = {k: (sum(1.0 / math.log(r + 2) for r in ranks if r < k) / n)
          / (1.0 / math.log(2)) for k in ks}
    return hr, nd


def part_a(item_path, n=32, n_beam=20, ks=(1, 3, 5, 10, 20), tmp=".workbuddy/_trash"):
    print("=" * 78)
    print("A. 指标链：把已知名次的预测喂给 calc.py，看是否复现解析解")
    print("=" * 78)
    sids = [l.split("\t")[0].strip() for l in io.open(item_path, encoding="utf-8")]
    n_items = len(sids)
    os.makedirs(tmp, exist_ok=True)

    variants = [("全 rank-0", lambda i: 0),
                ("全 rank-2", lambda i: 2),
                ("混合 0/1/2/3", lambda i: i % 4),
                ("全未命中", lambda i: None)]

    for name, fn in variants:
        rows = []
        for i in range(n):
            r = fn(i)
            tgt = sids[i % n_items]                      # 该样本的"真值"
            fills = [s for s in sids[:40] if s != tgt]
            pred = [tgt if r is not None and j == r else fills[j % len(fills)]
                    for j in range(n_beam)]
            rows.append({"input": "x", "output": tgt, "history_str": f"h{i}", "predict": pred})
        p = os.path.join(tmp, f"_vac_{abs(hash(name)) % 10**8}.json")
        io.open(p, "w", encoding="utf-8", newline="").write(json.dumps(rows, ensure_ascii=False))

        got_nd, got_hr, cc = run_calc(p, item_path)
        ranks = [fn(i) for i in range(n)]
        # 未命中的样本对 HR 与 NDCG 都贡献 0，故只把命中的名次交给期望函数、分母仍取 n
        e_hr, e_nd = expect_metrics([r for r in ranks if r is not None], n, ks)
        for j, k in enumerate(ks):
            ok = abs(got_hr[j] - e_hr[k]) < 1e-12 and abs(got_nd[j] - e_nd[k]) < 1e-12
            check(f"{name} @K={k}", f"HR {got_hr[j]:.6f}/{e_hr[k]:.6f}  "
                                    f"NDCG {got_nd[j]:.6f}/{e_nd[k]:.6f}", ok)
        check(f"{name} CC(全在 item_dict)", f"{cc}", cc == 0)

    # CC 的正对照：塞一个不存在的 SID，CC 必须精确等于塞入个数
    rows = []
    foreign = "<a_999><b_999><c_999>"
    for i in range(n):
        tgt = sids[i]
        pred = [foreign] * 3 + [s for s in sids[:40] if s != tgt][:n_beam - 3]
        rows.append({"input": "x", "output": tgt, "history_str": f"h{i}", "predict": pred})
    p = os.path.join(tmp, "_vac_cc.json")
    io.open(p, "w", encoding="utf-8", newline="").write(json.dumps(rows, ensure_ascii=False))
    _, _, cc = run_calc(p, item_path)
    check("CC 正对照（每样本塞 3 个异物）", f"CC={cc}，期望 {3 * n}", cc == 3 * n)


# ==================================================================  B. 真值闭环
def part_b(result_json, test_csv, item_file, tokenizer_dir, sid_vocab, category,
           max_len=2560, seed=42):
    print()
    print("=" * 78)
    print("B. 真值闭环：结果 json 的 output / history_str 是否等于数据集真值")
    print("=" * 78)
    from transformers import AutoTokenizer
    import data as D

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    tok.add_tokens(json.load(io.open(sid_vocab, encoding="utf-8")))

    recs = json.load(io.open(result_json, encoding="utf-8"))
    n = len(recs)
    print(f"  结果 json : {result_json}  ({n} 条)")
    if n > 4000:
        print("  [SKIP] 条数过多（>4000），本自检面向 dry-run / 小样本结果")
        return

    # 与 evaluate.py:177 完全同参（max_len=2560 / test=True / K=0 / seed / sample=n）
    ds = D.EvalSidDataset(train_file=test_csv, tokenizer=tok, max_len=max_len,
                          category=category, test=True, K=0, seed=seed, sample=n)
    rows = ds.get_all()
    print(f"  数据集    : EvalSidDataset(... sample={n}) -> {len(rows)} 条")

    idx = {}
    for r in rows:
        idx.setdefault(str(r.get("history_str", "")).strip(), set()).add(
            str(r.get("output", "")).strip().strip('"').strip())

    bad_hist = bad_out = 0
    for rec in recs:
        h = str(rec.get("history_str", "")).strip()
        tgt = str(rec.get("output", "")).strip().strip('"').strip()
        if h not in idx:
            bad_hist += 1
            if bad_hist <= 3:
                print(f"    [历史对不上] hist={h[:60]!r}")
        elif tgt not in idx[h]:
            bad_out += 1
            if bad_out <= 3:
                print(f"    [真值对不上] json={tgt!r}  ds={sorted(idx[h])!r}")

    check("history_str 能在数据集中找到", f"找不到 {bad_hist}/{n}", bad_hist == 0)
    check("output 与数据集真值一致", f"不一致 {bad_out}/{n}", bad_out == 0)
    cov = len({str(r.get('history_str', '')).strip() for r in recs} - set(idx))
    check("数据集被结果覆盖（反向）", f"未覆盖 {cov}/{len(idx)}", cov == 0)

    sids = {l.split("\t")[0].strip() for l in io.open(item_file, encoding="utf-8")}
    not_in = sum(1 for r in recs
                 if str(r.get("output", "")).strip().strip('"').strip() not in sids)
    check("真值都在 item_info.txt 里", f"不在 {not_in}/{n}（否则 calc.py 必然 match 不到）",
          not_in == 0)


def main():
    ap = argparse.ArgumentParser(description="评估链路自检（离线）")
    ap.add_argument("--domain", default="IandS")
    ap.add_argument("--sft-dir", default="")
    ap.add_argument("--tokenizer-dir", default="models/Qwen3-0.6B")
    ap.add_argument("--category", default="industrial and scientific items")
    ap.add_argument("--result-json", default="", help="给了就额外跑 B 段真值闭环")
    ap.add_argument("--skip-a", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    sft = args.sft_dir or f"data/Amazon23/{args.domain}/sft"
    item_file = f"{sft}/info/{args.domain}.item_info.txt"

    if not args.skip_a:
        part_a(item_file)

    if args.result_json:
        part_b(args.result_json, f"{sft}/test/{args.domain}_5_test.csv", item_file,
               args.tokenizer_dir, f"{sft}/info/sid_vocab.json", args.category)

    print()
    print("=" * 78)
    print(f"结果: {'全部通过' if _fail == 0 else f'{_fail} 项失败'}")
    print("=" * 78)
    sys.exit(0 if _fail == 0 else 1)


if __name__ == "__main__":
    main()
