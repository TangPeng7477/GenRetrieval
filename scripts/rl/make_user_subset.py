#!/usr/bin/env python
"""按**同一批用户**从 train/test CSV 抽子集，产出可直接喂给 RL 训练与评估的两个 CSV。

为什么需要它
------------
全量 run 里「训练用户 ⊇ 评估用户」= **100%**（每个用户的最后一条是 test，其余是 train）。
而 `MAX_STEPS` 前缀方案只看得到约 **7%** 交集（470 步≈3,700 用户 vs 评估 5,000 用户）
⟹ 小规模迭代不是全量 run 的忠实缩放版。本脚本让**训练用户 ≡ 评估用户**。

🔴 为什么只切 T1/T3（配合 `RL_TASKS=T1,T3` 使用）
------------------------------------------------
RL 侧三路数据源不同：
  · T1 = `SidDataset(train_file)`              —— 从 train CSV 派生 ⟹ **可按用户切**
  · T3 = `RLSeqTitle2SidDataset(train_file)`   —— 同上 ⟹ **可按用户切**
  · T2 = `RLTitle2SidDataset(item_info, ...)`  —— **item-level，无用户维度** ⟹ 切不了
所以做「同一批用户」时必须把 T2 去掉，否则它的行数（51,433，与用户无关）会把配比从
19% 顶到 63%、单次迭代从 38 min 涨到 3.5 h。

用法
----
    python scripts/rl/make_user_subset.py --domain IandS --n-users 5000 --seed 42

产出（与源文件同目录；文件名一律 ASCII）
    train/<DOMAIN>_5_train.<TAG>.csv
    test/<DOMAIN>_5_test.<TAG>.csv
    info/<DOMAIN>.user_subset.<TAG>.json    清单（seed / 用户数 / 用户列表 / 行数，供 provenance）

然后这样跑（训练用户 ≡ 评估用户）：
    TRAIN_FILE=<那支 train csv> RL_TASKS=T1,T3 RL_T3_SAMPLE=<按比例算> ... bash rl_run0.sh
    TEST_FILE=<那支 test csv> ... bash evaluate_run0.sh

实现说明
--------
用 `csv` 模块流式读写、不用 pandas 整表进出内存（IandS train CSV 222 MB）；
也刻意保留字段的原始文本，避免 pandas 往返改写（下游有 `eval(row['history_item_id'])`）。

⚠️ 选的是**两个文件都出现过的用户**（交集）：某用户在 train 有行但 test 无行时，
   只出现在训练侧会白占预算，且无法评估。
"""

import argparse
import csv
import json
import os
import random
import sys


def _find_uid_col(header):
    for i, name in enumerate(header):
        if name.strip() == "user_id":
            return i
    raise SystemExit(f"[BAD] 表头里找不到 user_id 列：{header}")


def scan_users(path):
    """流式扫一遍，只取 user_id 列。返回 (header, uid_idx, set(users), n_rows)。"""
    users = set()
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        uid_idx = _find_uid_col(header)
        for row in r:
            if not row:
                continue
            n += 1
            users.add(row[uid_idx])
    return header, uid_idx, users, n


def write_filtered(src, dst, keep):
    """流式过滤写出（按 user_id 列名自动定位，不写死列序号）。"""
    n_in = n_out = 0
    with open(src, newline="", encoding="utf-8") as fi, \
         open(dst, "w", newline="", encoding="utf-8") as fo:
        r = csv.reader(fi)
        w = csv.writer(fo, lineterminator="\n")  # 显式 LF：csv.writer 默认在 Windows 上写 CRLF
        header = next(r)
        uid_idx = _find_uid_col(header)
        w.writerow(header)
        for row in r:
            if not row:
                continue
            n_in += 1
            if row[uid_idx] in keep:
                w.writerow(row)
                n_out += 1
    return n_in, n_out


def main():
    ap = argparse.ArgumentParser(description="按同一批用户抽 RL 训练/评估子集")
    ap.add_argument("--domain", default="IandS", help="类别名，默认 IandS")
    ap.add_argument("--n-users", type=int, default=5000, help="抽多少用户，默认 5000")
    ap.add_argument("--seed", type=int, default=42, help="抽样种子（决定抽到哪批用户），默认 42")
    ap.add_argument("--data-root", default="data/Amazon23", help="默认 data/Amazon23")
    ap.add_argument("--train-csv", default="", help="覆盖默认的 train CSV 路径")
    ap.add_argument("--test-csv", default="", help="覆盖默认的 test CSV 路径")
    ap.add_argument("--tag", default="", help="文件名后缀，默认 u<N>k 例 u5k")
    args = ap.parse_args()

    d = args.domain
    root = os.path.join(args.data_root, d, "sft")
    train_csv = args.train_csv or os.path.join(root, "train", f"{d}_5_train.csv")
    test_csv = args.test_csv or os.path.join(root, "test", f"{d}_5_test.csv")
    tag = args.tag
    if not tag:
        tag = f"u{args.n_users // 1000}k" if args.n_users % 1000 == 0 else f"u{args.n_users}"

    for p in (train_csv, test_csv):
        if not os.path.isfile(p):
            raise SystemExit(f"[MISSING] {p}")

    print(f"[1/4] 扫 train: {train_csv}")
    _, _, tr_users, tr_rows = scan_users(train_csv)
    print(f"      行={tr_rows:,}  用户={len(tr_users):,}  行/用户={tr_rows / max(len(tr_users), 1):.2f}")

    print(f"[2/4] 扫 test : {test_csv}")
    _, _, te_users, te_rows = scan_users(test_csv)
    print(f"      行={te_rows:,}  用户={len(te_users):,}  行/用户={te_rows / max(len(te_users), 1):.2f}")

    cand = sorted(tr_users & te_users)
    print(f"      train ∩ test 用户 = {len(cand):,}")
    if len(cand) < args.n_users:
        raise SystemExit(f"[BAD] 交集用户 {len(cand):,} < 请求的 {args.n_users:,}，无法抽样")

    picked = set(random.Random(args.seed).sample(cand, args.n_users))
    print(f"[3/4] seed={args.seed} 抽 {args.n_users:,} 个用户（确定性：换 seed 换一批）")

    out_train = os.path.join(root, "train", f"{d}_5_train.{tag}.csv")
    out_test = os.path.join(root, "test", f"{d}_5_test.{tag}.csv")
    out_man = os.path.join(root, "info", f"{d}.user_subset.{tag}.json")

    tr_in, tr_out = write_filtered(train_csv, out_train, picked)
    te_in, te_out = write_filtered(test_csv, out_test, picked)
    print(f"[4/4] 写出：")
    print(f"      {out_train}   ({tr_in:,} -> {tr_out:,} 行, {os.path.getsize(out_train) / 1e6:.1f} MB)")
    print(f"      {out_test}   ({te_in:,} -> {te_out:,} 行, {os.path.getsize(out_test) / 1e6:.1f} MB)")

    # 按比例算 T3 上限，保住全量 run 的 T1:T3 配比（T3 原为硬编码绝对上限 10000）
    t3_full, t1_full = 10_000, 208_999
    t3_cap = max(1, round(tr_out * t3_full / t1_full))

    manifest = {
        "script": "scripts/rl/make_user_subset.py",
        "domain": d,
        "seed": args.seed,
        "n_users": args.n_users,
        "tag": tag,
        "source_train_csv": train_csv,
        "source_test_csv": test_csv,
        "out_train_csv": out_train,
        "out_test_csv": out_test,
        "train_rows_in": tr_in, "train_rows_out": tr_out,
        "test_rows_in": te_in, "test_rows_out": te_out,
        "rows_per_user_train": round(tr_out / args.n_users, 3),
        "recommend_RL_TASKS": "T1,T3",
        "recommend_RL_T3_SAMPLE": t3_cap,
        "recommend_env": {
            "TRAIN_FILE": out_train,
            "TEST_FILE": out_test,
            "RL_TASKS": "T1,T3",
            "RL_T3_SAMPLE": t3_cap,
        },
        "users": sorted(picked),
    }
    with open(out_man, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"      {out_man}   (清单，含用户列表与建议参数)")

    steps = (tr_out + t3_cap) // 32
    print()
    print("建议参数（训练用户 ≡ 评估用户）：")
    print(f"  TRAIN_FILE={out_train}")
    print(f"  RL_TASKS=T1,T3   RL_T2_SAMPLE=-1   RL_T3_SAMPLE={t3_cap}")
    print(f"  训练集 = {tr_out:,}(T1) + {t3_cap:,}(T3) = {tr_out + t3_cap:,} 行"
          f" ⟹ {steps:,} 步/epoch ≈ {steps * 4.9 / 60:.0f} min @4.9 s/step")
    print(f"  TEST_FILE={out_test}   (评估 {te_out:,} 行 = {args.n_users:,} 个用户)")
    print()
    print(f"⚠️ 旧锚点作废：IandS-all 在全量 test 上的 HR@10=0.0356 与这套评估集**不可比**，")
    print(f"   必须先补锚点：TEST_FILE={out_test} MODEL_PATH=outputs/IandS-all/final_checkpoint \\")
    print(f"                 MAX_SAMPLES=0 BATCH_SIZE=12 bash evaluate_run0.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
