"""LOO 切分的 prepare 脚本（plain leave-one-out + sliding-window train）。

与 `prepare_amazon23.py`（timestamp 切分）的差别：
- timestamp 版：valid/test 每用户只取段末 1 条；
- LOO 版（这里）：合并 train/valid/test 三个段、按 timestamp 排序去重后，对整条用户序列做 sliding window；
  test 取最后 1 条（标准 LOO），valid 取倒数第 2 条；
  train 是 3..K-2 位置的 sliding window 样本。

切分依据：`docs/EVAL_PROTOCOL.md`（定版协议）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np


def read_split_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "user": row["user_id"],
                    "asin": row["parent_asin"],
                    "ts":   int(row["timestamp"]),
                    "rating": float(row.get("rating", 0) or 0),
                })
            except (ValueError, KeyError):
                continue
    return rows


def load_splits(raw_dir, categories):
    splits = {"train": [], "valid": [], "test": []}
    for cat in categories:
        for name in ("train", "valid", "test"):
            p = os.path.join(raw_dir, f"{cat}_5core_timestamp.{name}.csv")
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            splits[name].extend(read_split_csv(p))
    return splits


def build_user_histories(splits):
    per_user = defaultdict(lambda: {"train": [], "valid": [], "test": []})
    for name, rows in splits.items():
        for r in rows:
            per_user[r["user"]][name].append(r)
    for u, d in per_user.items():
        for name in ("train", "valid", "test"):
            seen = set(); uniq = []
            for r in sorted(d[name], key=lambda x: x["ts"]):
                if r["asin"] in seen:
                    continue
                seen.add(r["asin"])
                uniq.append(r)
            d[name] = uniq
    return per_user


def build_samples_loo(per_user, history_len):
    """plain LOO + sliding-window train。"""
    train, valid, test = [], [], []
    for u, d in per_user.items():
        all_items = d["train"] + d["valid"] + d["test"]
        all_items.sort(key=lambda x: x["ts"])
        seen = set(); uniq = []
        for r in all_items:
            if r["asin"] in seen:
                continue
            seen.add(r["asin"])
            uniq.append(r)

        K = len(uniq)
        if K < 3:
            continue

        for k in range(3, K - 1):                # 0-based target 索引 hi = k-1 ∈ [2, K-3]
                                              # ⚠️ 必须 K-1 而非 K：保留 uniq[K-2] 给 valid、
                                              # uniq[K-1] 给 test，否则 valid target = train 最后 target
                                              # → valid HR@10 trivial（实测 60%+，数据泄漏放大）
            hi = k - 1
            lo = max(0, hi - history_len)
            hist = [r["asin"] for r in uniq[lo:hi]]
            train.append((u, hist, uniq[hi]["asin"], uniq[hi]["ts"]))

        hi = K - 2                              # valid target = uniq[K-2]，差 train 末尾 1 项
        lo = max(0, hi - history_len)
        valid.append((u, [r["asin"] for r in uniq[lo:hi]], uniq[hi]["asin"], uniq[hi]["ts"]))

        hi = K - 1                              # test target = uniq[K-1]，差 train 末尾 2 项
        lo = max(0, hi - history_len)
        test.append((u, [r["asin"] for r in uniq[lo:hi]], uniq[hi]["asin"], uniq[hi]["ts"]))

    return train, valid, test


def write_inter(path, samples, user2idx, item2idx):
    n_drop = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for u, hist, tgt, _ in samples:
            h = " ".join(str(item2idx[a]) for a in hist if a in item2idx)
            if u not in user2idx or tgt not in item2idx or not h:
                n_drop += 1
                continue
            f.write(f"{user2idx[u]}\t{h}\t{item2idx[tgt]}\n")
    if n_drop:
        print(f"  [drop {n_drop} 条因 user/item 不在表中]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Industrial_and_Scientific")
    ap.add_argument("--short", default="IandS")
    ap.add_argument("--raw_dir", default="data/Amazon23/raw")
    ap.add_argument("--out_dir", default="data/Amazon23")
    ap.add_argument("--history_len", type=int, default=20)
    ap.add_argument("--min_rating", type=float, default=0.0)
    ap.add_argument("--sid_dir", default=None,
                    help="如指定，则丢弃 sid 表外的 item（让 .inter 与 sid_raw.npy 行数对齐，"
                         "v2 LOO 与 v1 timestamp 训练出来的 sid 之间的兼容层）")
    args = ap.parse_args()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    out_dir = os.path.join(args.out_dir, args.short)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"[LOO] categories={cats}  history_len={args.history_len}")
    print("=" * 70)

    splits = load_splits(args.raw_dir, cats)
    per_user = build_user_histories(splits)
    print(f"  users with any data: {len(per_user):,}")

    train, valid, test = build_samples_loo(per_user, args.history_len)
    print(f"  train samples: {len(train):,}  valid: {len(valid):,}  test: {len(test):,}")

    # ── sid 表外 item 过滤（v2 LOO + v1 timestamp sid 产物兼容层）──
    # LOO 协议合并三段去重后 item 集合可能比 v1 timestamp 多 1~几个冷启动 item；
    # sid_raw 是 RQ-VAE 在 v1 timestamp 数据上训的（行号 = item_id = 字母序位置），
    # 所以 .inter 里这些"超出 sid 表"的 item 没有 SID，会让 sid_prefix/sid_gr 报错。
    # 修复：按字母序保留前 n_sid_keep 个 asin，让 .inter 的 item_id 0..N-1 与 sid_raw 行号严格对齐。
    n_sid_keep = None
    if args.sid_dir:
        sid_path = os.path.join(args.sid_dir, "sid_raw.npy")
        if os.path.exists(sid_path):
            sid = np.load(sid_path)
            n_sid_keep = int(sid.shape[0])
            print(f"  [sid] sid_raw.shape[0] = {n_sid_keep:,} → 按字母序保留前 {n_sid_keep} 个 item")

    item_counter = Counter()
    for u, hist, tgt, _ in train + valid + test:
        for a in hist: item_counter[a] += 1
        item_counter[tgt] += 1

    kept_items = None
    if n_sid_keep is not None and len(item_counter) > n_sid_keep:
        sorted_asins = sorted(item_counter)
        kept_items = set(sorted_asins[:n_sid_keep])
        dropped = set(sorted_asins[n_sid_keep:])
        print(f"  [sid] 丢弃 {len(dropped):,} 个 sid 表外 item: {sorted(dropped)[:5]}...")

        def _filter(samples):
            out = []
            for u, h, t, ts in samples:
                if t not in kept_items:
                    continue
                h2 = [a for a in h if a in kept_items]
                if not h2:
                    continue
                out.append((u, h2, t, ts))
            return out

        before = (len(train), len(valid), len(test))
        train = _filter(train)
        valid = _filter(valid)
        test  = _filter(test)
        print(f"  [sid] 丢弃样本: train {before[0]-len(train)} / "
              f"valid {before[1]-len(valid)} / test {before[2]-len(test)}")
        item_counter = Counter({a: c for a, c in item_counter.items() if a in kept_items})

    item2idx = {a: i for i, a in enumerate(sorted(item_counter))}
    user2idx = {u: i for i, u in enumerate(sorted({u for u, *_ in train + valid + test}))}
    print(f"  n_items={len(item2idx):,}  n_users={len(user2idx):,}  "
          f"n_items_with_sid={sid.shape[0] if (args.sid_dir and (sid_path := os.path.join(args.sid_dir, 'sid_raw.npy')) and os.path.exists(sid_path)) else 'n/a'}")

    write_inter(os.path.join(out_dir, f"{args.short}.train.inter"), train, user2idx, item2idx)
    write_inter(os.path.join(out_dir, f"{args.short}.valid.inter"), valid, user2idx, item2idx)
    write_inter(os.path.join(out_dir, f"{args.short}.test.inter"),  test,  user2idx, item2idx)

    stats = {
        "split_protocol": "loo",
        "history_len": args.history_len,
        "categories": cats,
        "n_items": len(item2idx),
        "n_users": len(user2idx),
        "n_sid_keep": n_sid_keep,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test":  len(test),
        "valid_per_user_avg": round(len(valid) / max(1, len(user2idx)), 4),
        "test_per_user_avg":  round(len(test)  / max(1, len(user2idx)), 4),
    }
    with open(os.path.join(out_dir, f"{args.short}.stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"  stats -> {os.path.join(out_dir, args.short + '.stats.json')}")


if __name__ == "__main__":
    main()