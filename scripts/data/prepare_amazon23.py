#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1-2: Amazon Reviews 2023 -> MiniOneRec 兼容数据格式（支持多类目合并）
=======================================================================
输入（scripts/data/download_amazon23.sh 下载）:
    data/Amazon23/raw/{Category}_5core_timestamp.{train,valid,test}.csv
    data/Amazon23/raw/meta_{Category}.jsonl

输出（data/Amazon23/<short>/）:
    <short>.item.json         item_idx -> {asin,title,features,categories,brand,images,domain,...}
    <short>.item2id           asin -> item_idx
    <short>.item_domain.tsv   item_idx \t domain            （跨域评测切片用）
    <short>.domain2id         domain \t domain_id
    <short>.user_domains.tsv  user_idx \t 逗号分隔的 domain 集合
    <short>.{train,valid,test}.inter   user_id \t "i1 i2 ..." \t target   (V0 兼容)
    <short>.stats.json        统计摘要（含 by_domain 分组）

切分设计（与官方 5core/timestamp 对齐，无泄漏）:
    * train: 每用户训练段内的**滑窗**样本 —— 每个 target 只依赖它之前的 train 物品
    * valid: 该用户 train 全部物品 -> 预测 valid 物品（每用户 1 条）
    * test : 该用户 train+valid 全部物品 -> 预测 test 物品（每用户 1 条）
    * 历史长度统一截断到最近 --history_len 个（默认 20）
    * 实测官方切分是全局时间切分（max(train_ts) < min(test_ts)），因此不存在跨期泄漏

多类目（提升泛化性）:
    * --categories 逗号分隔，如 Industrial_and_Scientific,Video_Games
    * 每个物品打 domain 标签，item_idx 按 (domain_id, asin) 排序 ——
      单类目时退化为按 asin 排序，与历史产物完全一致
    * 跨类目用户（同一 user_id 出现在多个类目）会自然形成跨域序列，
      统计里给出 cross_domain_users，可用于"跨域用户迁移"分析

用法:
    # 单类目（与历史一致）
    python scripts/data/prepare_amazon23.py \
        --categories Industrial_and_Scientific --short IandS --history_len 20

    # 多类目合并（2.0 主实验集）
    python scripts/data/prepare_amazon23.py \
        --categories Industrial_and_Scientific,Video_Games --short Multi2 --history_len 20
"""

import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict

# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def clean_text(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(str(i) for i in x if i)
    return str(x)


def squeeze(s, limit):
    s = " ".join(str(s).split())
    return s[:limit]


def read_split_csv(path):
    """官方 5core 分片: user_id,parent_asin,rating,timestamp(ms)"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "user": r["user_id"],
                    "asin": r["parent_asin"],
                    "rating": float(r["rating"]),
                    "ts": int(r["timestamp"]) // 1000,   # ms -> s
                }
            )
    return rows


def default_short(category):
    """Industrial_and_Scientific -> IandS；其余取各词首字母，重名时会在 main 里加后缀"""
    parts = [p for p in category.split("_") if p]
    if category == "Industrial_and_Scientific":
        return "IandS"
    return "".join(p[0] for p in parts).upper() if len(parts) > 1 else category[:5]


# --------------------------------------------------------------------------- #
# 1) 物品表：从 metadata 里抽取我们需要的字段
# --------------------------------------------------------------------------- #

def build_item_table(meta_path, item_set, max_features=8, max_desc=1000):
    """流式读 1GB metadata，只保留 item_set 里的物品。"""
    table = {}
    total_lines = 0
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            if total_lines % 500000 == 0:
                print(f"    ... metadata lines={total_lines:,} kept={len(table):,}")
            if '"parent_asin"' not in line:
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            asin = m.get("parent_asin")
            if not asin or asin not in item_set or asin in table:
                continue

            feats = m.get("features") or []
            if not isinstance(feats, list):
                feats = [feats]
            feats = [squeeze(x, 200) for x in feats if squeeze(x, 200)][:max_features]

            cats = m.get("categories") or []
            if not isinstance(cats, list):
                cats = [cats]
            cats = [squeeze(x, 80) for x in cats if squeeze(x, 80)]

            desc = m.get("description") or []
            if not isinstance(desc, list):
                desc = [desc]
            desc = squeeze(" ".join(str(d) for d in desc), max_desc)

            imgs = m.get("images") or []
            img_urls = []
            for im in imgs:
                if isinstance(im, dict):
                    u = im.get("hi_res") or im.get("large") or im.get("thumb")
                    if u:
                        img_urls.append(u)
            # 主图 + 副图（副图最多存 4 张，供后续消融多图融合）
            img_urls = img_urls[:5]

            store = squeeze(m.get("store") or "", 120)
            table[asin] = {
                "asin": asin,
                "title": squeeze(m.get("title") or "", 300),
                "features": feats,
                "description": desc,
                "categories": cats,
                "brand": store,
                "main_category": squeeze(m.get("main_category") or "", 80),
                "price": m.get("price"),
                "images": img_urls,
                "has_image": bool(img_urls),
            }
    print(f"  metadata 行数={total_lines:,}  命中物品={len(table):,}/{len(item_set):,}")
    return table


# --------------------------------------------------------------------------- #
# 2) 序列切分
# --------------------------------------------------------------------------- #

def build_user_histories(splits):
    """user -> {'train': [...], 'valid': [...], 'test': [...]}（按时间排序、去重）

    多类目时同一 user 的跨域交互会合并进同一条序列（这是跨域序列推荐的设定）。
    """
    per_user = defaultdict(lambda: {"train": [], "valid": [], "test": []})
    for name, rows in splits.items():
        for r in rows:
            per_user[r["user"]][name].append(r)
    for u, d in per_user.items():
        for name in ("train", "valid", "test"):
            seen = set()
            uniq = []
            for r in sorted(d[name], key=lambda x: x["ts"]):
                if r["asin"] in seen:
                    continue
                seen.add(r["asin"])
                uniq.append(r)
            d[name] = uniq
    return per_user


def build_samples(per_user, history_len, min_rating, min_train_len=1):
    """
    返回 (train_samples, valid_samples, test_samples)
    sample = (user, [hist_asin], target_asin, target_ts)
    """
    train, valid, test = [], [], []

    def keep(r):
        return r["rating"] >= min_rating

    for u, d in per_user.items():
        tr = [r for r in d["train"] if keep(r)]
        va = [r for r in d["valid"] if keep(r)]
        te = [r for r in d["test"] if keep(r)]
        if not tr:
            continue

        # train: 滑窗（第 i 个 target 只看前 i 个 train 物品）
        for i in range(1, len(tr)):
            hist = [r["asin"] for r in tr[max(0, i - history_len):i]]
            if len(hist) < min_train_len:
                continue
            train.append((u, hist, tr[i]["asin"], tr[i]["ts"]))

        # valid: 全部 train -> valid 目标
        if va and len(tr) >= 1:
            hist = [r["asin"] for r in tr[-history_len:]]
            valid.append((u, hist, va[-1]["asin"], va[-1]["ts"]))

        # test: 全部 train + valid -> test 目标
        if te:
            hist_rows = tr + va
            hist = [r["asin"] for r in hist_rows[-history_len:]]
            test.append((u, hist, te[-1]["asin"], te[-1]["ts"]))

    return train, valid, test


# --------------------------------------------------------------------------- #
# 3) 输出
# --------------------------------------------------------------------------- #

def write_inter(path, samples, user2idx, item2idx):
    with open(path, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for u, hist, tgt, _ in samples:
            h = " ".join(str(item2idx[a]) for a in hist if a in item2idx)
            if tgt not in item2idx:
                continue
            f.write(f"{user2idx[u]}\t{h}\t{item2idx[tgt]}\n")


def sample_domain_stats(samples, asin2domain):
    """按 target 所属 domain 统计样本数；同时给出历史跨域的样本比例。"""
    tgt_dom = Counter()
    hist_cross = 0
    for _, hist, tgt, _ in samples:
        tgt_dom[asin2domain.get(tgt, "?")] += 1
        doms = {asin2domain.get(a, "?") for a in hist}
        if len(doms) > 1:
            hist_cross += 1
    return {
        "by_target_domain": dict(tgt_dom),
        "samples_with_cross_domain_history": hist_cross,
        "cross_domain_history_ratio": round(hist_cross / max(1, len(samples)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="Industrial_and_Scientific",
                    help="逗号分隔的类目列表；多类目会合并成一份数据集")
    ap.add_argument("--category", default=None,
                    help="[兼容旧参数] 单个类目，等价于 --categories <它>")
    ap.add_argument("--short", default="IandS", help="输出文件名前缀")
    ap.add_argument("--raw_dir", default="data/Amazon23/raw")
    ap.add_argument("--out_dir", default="data/Amazon23")
    ap.add_argument("--history_len", type=int, default=20)
    ap.add_argument("--min_rating", type=float, default=0.0,
                    help=">0 则过滤低分交互（隐式反馈默认全留）")
    ap.add_argument("--max_items_per_domain", type=int, default=0,
                    help=">0 时每个 domain 按交互频次截断物品表（长尾消融 / 缓解类目不平衡）")
    ap.add_argument("--domain_names", default=None,
                    help="可选的 domain 短名，逗号分隔，需与 --categories 一一对应")
    args = ap.parse_args()

    cats = ([c.strip() for c in args.category.split(",") if c.strip()]
            if args.category else
            [c.strip() for c in args.categories.split(",") if c.strip()])
    if args.domain_names:
        dom_names = [d.strip() for d in args.domain_names.split(",") if d.strip()]
        assert len(dom_names) == len(cats), "--domain_names 数量必须与 --categories 一致"
    else:
        dom_names = [default_short(c) for c in cats]
    cat2dom = dict(zip(cats, dom_names))
    dom2id = {d: i for i, d in enumerate(dom_names)}

    raw = os.path.join(args.raw_dir)
    out_dir = os.path.join(args.out_dir, args.short)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"categories={cats}")
    print(f"domains   ={dom_names}  short={args.short}  history_len={args.history_len}")
    print("=" * 70)

    # ---- 读切分（每个类目一份，打 domain 标签）----
    splits = {}
    per_cat_rows = {}
    for cat in cats:
        rows_by_split = {}
        for name in ("train", "valid", "test"):
            p = os.path.join(raw, f"{cat}_5core_timestamp.{name}.csv")
            rows = read_split_csv(p)
            for r in rows:
                r["domain"] = cat2dom[cat]
                r["category"] = cat
            rows_by_split[name] = rows
            print(f"[split] {cat:28} {name:5} rows={len(rows):,}")
        per_cat_rows[cat] = rows_by_split
        for name in ("train", "valid", "test"):
            splits.setdefault(name, []).extend(rows_by_split[name])

    for name in ("train", "valid", "test"):
        print(f"[split] {'ALL（合并）':28} {name:5} rows={len(splits[name]):,}")

    # 全局时间切分检查（防泄漏）
    max_tr = max(r["ts"] for r in splits["train"])
    min_te = min(r["ts"] for r in splits["test"])
    print(f"[leak-check] max(train_ts)={max_tr}  min(test_ts)={min_te}  "
          f"{'OK 无重叠' if max_tr < min_te else '!! 存在时间重叠，检查切分'}")

    # ---- 物品表（逐类目解析 metadata）----
    cat_item_set = {
        cat: {r["asin"] for s in per_cat_rows[cat].values() for r in s} for cat in cats
    }
    item_table = {}
    for cat in cats:
        item_set_c = set(cat_item_set[cat])
        if args.max_items_per_domain and len(item_set_c) > args.max_items_per_domain:
            cnt = Counter(r["asin"] for s in per_cat_rows[cat].values() for r in s)
            item_set_c &= {a for a, _ in cnt.most_common(args.max_items_per_domain)}
            print(f"[truncate] {cat} 物品表截断至 {len(item_set_c):,}（按交互频次）")
            cat_item_set[cat] = item_set_c
        meta_path = os.path.join(raw, f"meta_{cat}.jsonl")
        print(f"[metadata] {cat} 解析中 ...")
        t = build_item_table(meta_path, item_set_c)
        for asin, rec in t.items():
            rec["domain"] = cat2dom[cat]
            rec["category"] = cat
            item_table[asin] = rec

    # 跨类目 ASIN 碰撞检查（正常应为 0）
    all_asins = [a for cat in cats for a in cat_item_set[cat]]
    dup = [a for a, n in Counter(all_asins).items() if n > 1]
    print(f"[items] 跨类目 ASIN 碰撞={len(dup)}")

    # 只保留「有 title」的物品（无 title 无法建文本向量/进 SFT 语料）
    item_set = {a for a in item_table if item_table[a].get("title")}
    asin2domain = {a: item_table[a]["domain"] for a in item_set}
    print(f"[items] 有 title 的物品 {len(item_set):,}")

    # ---- 确定性索引：按 (domain_id, asin) 排序；单类目等价于按 asin 排序 ----
    sorted_items = sorted(item_set, key=lambda a: (dom2id[asin2domain[a]], a))
    item2idx = {a: i for i, a in enumerate(sorted_items)}

    # ---- 用户序列 & 样本 ----
    per_user = build_user_histories(splits)
    tr_users = {r["user"] for r in splits["train"]}
    va_users = {r["user"] for r in splits["valid"]}
    te_users = {r["user"] for r in splits["test"]}
    dropped_va = len(va_users - tr_users)
    dropped_te = len(te_users - tr_users)

    train, valid, test = build_samples(per_user, args.history_len, args.min_rating)

    def filt(s):
        return [x for x in s if x[2] in item2idx and all(a in item2idx for a in x[1])]
    train, valid, test = filt(train), filt(valid), filt(test)
    print(f"[samples] train={len(train):,} valid={len(valid):,} test={len(test):,}")

    users = sorted({s[0] for s in train + valid + test})
    user2idx = {u: i for i, u in enumerate(users)}

    # ---- 写文件 ----
    write_inter(os.path.join(out_dir, f"{args.short}.train.inter"), train, user2idx, item2idx)
    write_inter(os.path.join(out_dir, f"{args.short}.valid.inter"), valid, user2idx, item2idx)
    write_inter(os.path.join(out_dir, f"{args.short}.test.inter"), test, user2idx, item2idx)

    item_idx_map = {}
    for asin, idx in item2idx.items():
        rec = dict(item_table[asin])
        rec["item_id"] = idx
        item_idx_map[str(idx)] = rec
    with open(os.path.join(out_dir, f"{args.short}.item.json"), "w", encoding="utf-8") as f:
        json.dump(item_idx_map, f, ensure_ascii=False)

    with open(os.path.join(out_dir, f"{args.short}.item2id"), "w", encoding="utf-8") as f:
        for asin, idx in item2idx.items():
            f.write(f"{asin}\t{idx}\n")

    # 物品 domain 映射（跨域评测切片的关键文件）
    with open(os.path.join(out_dir, f"{args.short}.item_domain.tsv"), "w", encoding="utf-8") as f:
        for asin, idx in item2idx.items():
            f.write(f"{idx}\t{asin2domain[asin]}\n")
    with open(os.path.join(out_dir, f"{args.short}.domain2id"), "w", encoding="utf-8") as f:
        for d, i in dom2id.items():
            f.write(f"{d}\t{i}\n")

    # 用户 domain 组成（是否有跨域用户）
    user_doms = defaultdict(set)
    for name in ("train", "valid", "test"):
        for r in splits[name]:
            if r["asin"] in item2idx:
                user_doms[r["user"]].add(r["domain"])
    with open(os.path.join(out_dir, f"{args.short}.user_domains.tsv"), "w", encoding="utf-8") as f:
        for u in users:
            f.write(f"{user2idx[u]}\t{','.join(sorted(user_doms.get(u, set())))}\n")
    n_cross_users = sum(1 for u in users if len(user_doms.get(u, set())) > 1)

    # ---- 统计 ----
    n_img = sum(1 for v in item_idx_map.values() if v["has_image"])
    n_feat = sum(1 for v in item_idx_map.values() if v["features"])
    hl_tr = [len(h) for _, h, _, _ in train]
    hl_te = [len(h) for _, h, _, _ in test]

    by_domain = {}
    for d in dom_names:
        dom_items = [a for a in item2idx if asin2domain[a] == d]
        tr_d = sum(1 for _, _, t, _ in train if asin2domain[t] == d)
        va_d = sum(1 for _, _, t, _ in valid if asin2domain[t] == d)
        te_d = sum(1 for _, _, t, _ in test if asin2domain[t] == d)
        img_d = sum(1 for a in dom_items if item_table[a]["has_image"])
        by_domain[d] = {
            "category": cats[dom2id[d]],
            "n_items": len(dom_items),
            "n_train": tr_d,
            "n_valid": va_d,
            "n_test": te_d,
            "image_coverage": round(img_d / max(1, len(dom_items)), 4),
        }

    stats = {
        "categories": cats,
        "domains": dom_names,
        "short": args.short,
        "history_len_cap": args.history_len,
        "min_rating": args.min_rating,
        "n_items": len(item2idx),
        "n_users": len(user2idx),
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "n_cross_domain_users": n_cross_users,
        "valid_users_total": len(va_users),
        "test_users_total": len(te_users),
        "dropped_valid_users_no_train_history": dropped_va,
        "dropped_test_users_no_train_history": dropped_te,
        "image_coverage": round(n_img / max(1, len(item_idx_map)), 4),
        "features_coverage": round(n_feat / max(1, len(item_idx_map)), 4),
        "train_hist_len": {"min": min(hl_tr), "p50": statistics.median(hl_tr),
                           "mean": round(statistics.mean(hl_tr), 2), "max": max(hl_tr)},
        "test_hist_len": {"min": min(hl_te), "p50": statistics.median(hl_te),
                          "mean": round(statistics.mean(hl_te), 2), "max": max(hl_te)},
        "max_train_ts": max_tr,
        "min_test_ts": min_te,
        "time_leak": max_tr >= min_te,
        "by_domain": by_domain,
        "train_domain_mix": sample_domain_stats(train, asin2domain),
        "valid_domain_mix": sample_domain_stats(valid, asin2domain),
        "test_domain_mix": sample_domain_stats(test, asin2domain),
    }
    with open(os.path.join(out_dir, f"{args.short}.stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[stats]")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[done] -> {out_dir}")


if __name__ == "__main__":
    main()
