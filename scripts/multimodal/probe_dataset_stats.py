"""双域同口径数据统计：metadata 字段覆盖率 / 评分分布 / 长尾分位数 / 冷启动比例。

零依赖，系统 python 直接跑（从仓库根目录执行）：
    python scripts/multimodal/probe_dataset_stats.py
输出 markdown 表格片段，供 docs/DATASET.md 使用。
"""
import csv
import json
import os
import statistics
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RAW = os.path.join(ROOT, "data", "Amazon23", "raw")
DATA = os.path.join(ROOT, "data", "Amazon23")


def pct(values, q):
    """q in [0,1]，线性插值分位数。"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def fmt(x):
    return f"{x:.1f}" if isinstance(x, float) else str(x)


def field_coverage(short):
    path = os.path.join(DATA, short, f"{short}.item.json")
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if isinstance(items, dict):
        items = list(items.values())
    n = len(items)
    keys = ["asin", "title", "images", "has_image", "item_id", "brand",
            "main_category", "categories", "features", "price", "description"]
    out = {}
    for k in keys:
        c = 0
        for it in items:
            v = it.get(k, None)
            if v in (None, "", [], {}):
                continue
            c += 1
        out[k] = c / n
    return n, out


def rating_dist(cat):
    """全量（train+valid+test）评分分布，与 DATASET.md §8.1 口径一致。"""
    cnt = Counter()
    total = 0
    for split in ("train", "valid", "test"):
        path = os.path.join(RAW, f"{cat}_5core_timestamp.{split}.csv")
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                cnt[int(round(float(row["rating"])))] += 1
                total += 1
    return total, cnt


def long_tail(cat):
    """商品侧 / 用户侧交互数分位数（全量 = train+valid+test）。"""
    item_cnt, user_cnt = Counter(), Counter()
    for split in ("train", "valid", "test"):
        path = os.path.join(RAW, f"{cat}_5core_timestamp.{split}.csv")
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                item_cnt[row["parent_asin"]] += 1
                user_cnt[row["user_id"]] += 1
    iv, uv = list(item_cnt.values()), list(user_cnt.values())
    return {
        "n_items": len(iv), "n_users": len(uv),
        "item": {q: pct(iv, q) for q in (0, 0.5, 0.9, 0.99)},
        "item_mean": statistics.mean(iv), "item_max": max(iv),
        "user": {q: pct(uv, q) for q in (0, 0.5, 0.9, 0.99)},
        "user_mean": statistics.mean(uv), "user_max": max(uv),
    }


def cold_start(cat):
    """test 段目标商品里，有多少比例在 train 段从未出现。"""
    def items(split):
        s = set()
        path = os.path.join(RAW, f"{cat}_5core_timestamp.{split}.csv")
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                s.add(row["parent_asin"])
        return s

    tr, te = items("train"), items("test")
    cold = te - tr
    return len(te), len(cold), len(cold) / len(te)


for short, cat in (("IandS", "Industrial_and_Scientific"), ("VG", "Video_Games")):
    print(f"\n{'=' * 60}\n## {short}（{cat}）\n{'=' * 60}")

    n, cov = field_coverage(short)
    print(f"\n### metadata 字段覆盖率（n={n}）\n")
    print("| 字段 | 覆盖率 |")
    print("|---|---:|")
    for k, v in cov.items():
        print(f"| `{k}` | {v * 100:.1f}% |")

    total, dist = rating_dist(cat)
    print(f"\n### 评分分布（train 段，n={total:,}）\n")
    print("| 评分 | 数量 | 占比 |")
    print("|---:|---:|---:|")
    for star in (1, 2, 3, 4, 5):
        c = dist.get(star, 0)
        print(f"| {star}★ | {c:,} | {c / total * 100:.2f}% |")

    lt = long_tail(cat)
    print(f"\n### 长尾分位数（全量，items={lt['n_items']:,} users={lt['n_users']:,}）\n")
    print("| 维度 | min | p50 | mean | p90 | p99 | max |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    print(f"| 每个商品的交互数 | {fmt(lt['item'][0])} | {fmt(lt['item'][0.5])} | "
          f"{lt['item_mean']:.1f} | {fmt(lt['item'][0.9])} | {fmt(lt['item'][0.99])} | {lt['item_max']:,} |")
    print(f"| 每个用户的交互数 | {fmt(lt['user'][0])} | {fmt(lt['user'][0.5])} | "
          f"{lt['user_mean']:.1f} | {fmt(lt['user'][0.9])} | {fmt(lt['user'][0.99])} | {lt['user_max']:,} |")

    nte, ncold, ratio = cold_start(cat)
    print(f"\n### 冷启动（test 目标 {nte:,}，未在 train 出现 {ncold:,}，{ratio * 100:.2f}%）")
