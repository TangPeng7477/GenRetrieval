"""构造 SFT 数据集（SID 版）：把 LOO 的 `*.inter` + 定版 `sid_raw` 转成 MiniOneRec 可消费的格式。

设计依据
--------
* **主任务（T1 seq2sid）** 提示词与 MiniOneRec（`data.py::SidSFTDataset`）**逐字一致**，
  保证 M3 的第一次 SFT 能和 V0 锚点（README §3.4）在同口径下比。
* **辅助任务** 对齐 LC-Rec（ICDE'24, arXiv:2311.09049）的 alignment tuning 思想：
  T2 `sid↔title`（MiniOneRec `SidItemFeatDataset` 已有，LC-Rec 叫 index2item / item2index）、
  T3 `seq2title`（MiniOneRec `FusionSeqRecDataset`）、
  T4 `text2sid`（本项目新增：用 title+brand+categories+features 反查 SID，LC-Rec item2index 的富文本变体；
  MiniOneRec 只有 title，本仓 Amazon23 有 `features` 字段，信息密度更高）。
* 四个任务**分别落盘成独立文件**，配比在训练端按 M4 的消融矩阵组合（`docs/UPGRADE_PLAN.md §5.3`），
  不在这里写死。

产物布局（`data/Amazon23/<域>/sft/`）
------------------------------------
```
index/<域>.index.json        {item_id(str): ["<a_12>","<b_34>","<c_56>"]}   ← sft.py 的 sid_index_path
index/<域>.item.json         {item_id(str): {"title","description","text"}} ← sft.py 的 item_meta_path
train|valid|test/<域>_5_<split>.csv                       主任务 T1（列与 MiniOneRec 完全一致）
tasks/itemfeat.jsonl          T2: sid2title + title2sid
tasks/seq2title_<split>.jsonl T3
tasks/text2sid.jsonl          T4
info/<域>.item_info.txt       sid \t title \t item_id（convert_dataset.py 同格式）
info/sid2items.json           sid 字符串 → [item_id,...]（碰撞桶）+ 严格口径用的 representative
info/sid_vocab.json           有序 SID token 列表（3×256）
info/codebook.npy             (L,K,D) = (3,256,32)，供 M4 语义初始化
info/prompt_templates.json    四任务模板（训练端直接读，避免两端漂移）
stats.json                    全部计数与校验数字
```

用法
----
```bash
./.venv/Scripts/python.exe scripts/data/prepare_sft_data.py --domain IandS
./.venv/Scripts/python.exe scripts/data/prepare_sft_data.py --domain VG
```
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
import time

import numpy as np

PREFIXES = ["a", "b", "c"]

# ── 提示词模板（与 MiniOneRec data.py 逐字一致的部分已标注；改动会让 V0 锚点失效）──
ALPACA_HEADER = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request. \n\n### Instruction:\n"
)

PROMPT_TEMPLATES = {
    # ── T1：MiniOneRec SidSFTDataset / SidDataset 原样 ──
    "seq2sid": {
        "instruction": "Can you predict the next possible item that the user may expect?",
        "input": (
            "The user has interacted with items {hist} in chronological order. "
            "Can you predict the next possible item that the user may expect?"
        ),
        "output": "{target}",
        "hist_sep": ", ",
    },
    # ── T2a：MiniOneRec SidItemFeatDataset (sid2title) ──
    "sid2title": {
        "instruction": "Answer the question about item identification.",
        "input": 'What is the title of item "{sid}"?',
        "output": "{title}",
    },
    # ── T2b：MiniOneRec SidItemFeatDataset (title2sid) ──
    "title2sid": {
        "instruction": "Answer the question about item identification.",
        "input": "Which item has the title: {title}?",
        "output": "{sid}",
    },
    # ── T3：MiniOneRec FusionSeqRecDataset ──
    "seq2title": {
        "instruction": "Can you recommend the next item for the user based on their interaction history?",
        "input": (
            "The user has sequentially interacted with items {hist}. "
            "Can you recommend the next item for him? Tell me the title of the item"
        ),
        "output": "{title}",
        "hist_sep": ", ",
    },
    # ── T4：本项目新增（LC-Rec item2index 的富文本变体）──
    "text2sid": {
        "instruction": "Answer the question about item identification.",
        "input": "An item can be described as follows: \"{text}\". Which item is it describing?",
        "output": "{sid}",
    },
}

CSV_COLUMNS = [
    "user_id",
    "history_item_title",
    "item_title",
    "history_item_id",
    "item_id",
    "history_item_sid",
    "item_sid",
]

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\r\n\t]+")


def clean_text(s, max_chars):
    """归一化空白、去控制字符、截断。返回 (文本, 是否被截断)。"""
    if s is None:
        return "", False
    if isinstance(s, (list, tuple)):
        s = " ".join(str(x) for x in s)
    s = _CTRL.sub(" ", str(s))
    s = _WS.sub(" ", s).strip()
    if max_chars and len(s) > max_chars:
        return s[: max_chars - 1].rstrip() + "…", True
    return s, False


def maybe_list(v):
    """item.json 里 features/categories 存的是 list 的 str(repr)，兼容两种。"""
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if not v:
        return []
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            out = eval(s)  # noqa: S307 - 本仓自有产物，格式已知
            if isinstance(out, (list, tuple)):
                return [str(x) for x in out]
        except Exception:
            pass
    return [s] if s else []


def read_inter(path):
    """读 `*.inter`（RecBole 风格 tsv，首行是表头）。返回 [(user, [hist_ids], target_id), ...]"""
    rows = []
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        assert header[0].startswith("user_id"), f"unexpected header: {header}"
        for line in r:
            if len(line) < 3 or not line[0].strip():
                continue
            hist = [int(x) for x in line[1].split()] if line[1].strip() else []
            rows.append((line[0].strip(), hist, int(line[2])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="IandS / VG")
    ap.add_argument("--data_root", default="data/Amazon23")
    ap.add_argument("--sid_npy", default="", help="默认 results/sid_e5000/<域>/gate__init8192/sid_raw.npy")
    ap.add_argument("--sid_ckpt", default="", help="默认同上目录的 ckpt/selected_model.pth（取 codebook）")
    ap.add_argument("--out_dir", default="", help="默认 <data_root>/<域>/sft")
    ap.add_argument("--max_title_chars", type=int, default=128)
    ap.add_argument("--max_text_chars", type=int, default=512)
    ap.add_argument("--max_features", type=int, default=6)
    args = ap.parse_args()

    dom = args.domain
    base = os.path.join(args.data_root, dom)
    out_dir = args.out_dir or os.path.join(base, "sft")
    sid_npy = args.sid_npy or os.path.join("results", "sid_e5000", dom, "gate__init8192", "sid_raw.npy")
    sid_ckpt = args.sid_ckpt or os.path.join(
        "results", "sid_e5000", dom, "gate__init8192", "ckpt", "selected_model.pth"
    )
    t0 = time.time()

    print("=" * 74)
    print(f"[SFT-DATA] domain={dom}  sid={sid_npy}")
    print("=" * 74)

    # ── 1. 载入 SID ────────────────────────────────────────────────
    codes = np.load(sid_npy)                       # (N, L) int
    N, L = codes.shape
    assert L == len(PREFIXES), f"SID 层数 {L} 与 PREFIXES {PREFIXES} 不匹配"
    print(f"  sid_raw: {codes.shape}  dtype={codes.dtype}")

    # item_id → 三层 token 列表 / 拼接串
    sid_tokens = {i: [f"<{PREFIXES[l]}_{int(codes[i, l])}>" for l in range(L)] for i in range(N)}
    sid_str = {i: "".join(sid_tokens[i]) for i in range(N)}

    # ── 2. 载入 item 元数据 ────────────────────────────────────────
    with open(os.path.join(base, f"{dom}.item.json"), encoding="utf-8") as f:
        raw_items = json.load(f)
    n_meta = len(raw_items)
    assert n_meta >= N, f"item.json 只有 {n_meta} 条，少于 SID 的 {N} 行"
    print(f"  item.json: {n_meta:,} 条")

    titles, texts, n_trunc_title, n_trunc_text, n_empty_title = {}, {}, 0, 0, 0
    for k, v in raw_items.items():
        iid = int(k)
        if iid >= N:
            continue
        t, tr = clean_text(v.get("title", ""), args.max_title_chars)
        n_trunc_title += int(tr)
        if not t:
            n_empty_title += 1
            t = f"Item_{iid}"
        titles[iid] = t

        parts = [t]
        brand = clean_text(v.get("brand", ""), 64)[0]
        if brand:
            parts.append(f"Brand: {brand}")
        cats = [c for c in maybe_list(v.get("categories", "")) if c and not c.startswith("[")]
        if cats:
            parts.append("Categories: " + ", ".join(cats[:4]))
        feats = maybe_list(v.get("features", ""))[: args.max_features]
        feats = [clean_text(x, 80)[0] for x in feats]
        feats = [x for x in feats if x]
        if feats:
            parts.append("Features: " + "; ".join(feats))
        desc = clean_text(v.get("description", ""), 200)[0]
        if desc:
            parts.append(desc)
        full, tr = clean_text(" | ".join(parts), args.max_text_chars)
        n_trunc_text += int(tr)
        texts[iid] = full

    print(f"  title 截断 {n_trunc_title:,} 条 / text 截断 {n_trunc_text:,} 条 / 空 title {n_empty_title:,} 条")

    # ── 3. 碰撞桶（语义桶口径）与严格口径的 representative ─────────
    bucket = collections.defaultdict(list)
    for i in range(N):
        bucket[sid_str[i]].append(i)
    n_unique = len(bucket)
    collided_items = sum(len(v) for v in bucket.values() if len(v) > 1)
    max_bucket = max(len(v) for v in bucket.values())
    print(f"  SID 唯一率 ICR = {n_unique / N:.4f}  碰撞物品 {collided_items:,} ({collided_items/N:.2%})  "
          f"最大桶 {max_bucket}")

    # ── 4. 目录 ────────────────────────────────────────────────────
    for sub in ("index", "train", "valid", "test", "tasks", "info"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # ── 5. index/（sft.py 直接消费）────────────────────────────────
    idx_path = os.path.join(out_dir, "index", f"{dom}.index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({str(i): sid_tokens[i] for i in range(N)}, f, ensure_ascii=False)

    meta_path = os.path.join(out_dir, "index", f"{dom}.item.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {str(i): {"title": titles[i], "description": texts[i], "text": texts[i]} for i in range(N)},
            f, ensure_ascii=False,
        )
    print(f"  index -> {idx_path}")

    # ── 6. 主任务 CSV ──────────────────────────────────────────────
    split_stats = {}
    for split in ("train", "valid", "test"):
        rows = read_inter(os.path.join(base, f"{dom}.{split}.inter"))
        out_csv = os.path.join(out_dir, split, f"{dom}_5_{split}.csv")
        n_drop_hist, n_drop_oob = 0, 0
        hist_lens, n_amb_target = [], 0
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for u, hist, tgt in rows:
                if not hist:
                    n_drop_hist += 1
                    continue
                if tgt >= N or any(h >= N for h in hist):
                    n_drop_oob += 1
                    continue
                if len(bucket[sid_str[tgt]]) > 1:
                    n_amb_target += 1
                hist_lens.append(len(hist))
                w.writerow({
                    "user_id": f"A{u}",
                    "history_item_title": str([titles[h] for h in hist]),
                    "item_title": titles[tgt],
                    "history_item_id": str(hist),
                    "item_id": tgt,
                    "history_item_sid": str([sid_str[h] for h in hist]),
                    "item_sid": sid_str[tgt],
                })
        split_stats[split] = {
            "n_raw": len(rows),
            "n_written": len(hist_lens),
            "n_drop_empty_hist": n_drop_hist,
            "n_drop_out_of_range": n_drop_oob,
            "hist_len_mean": round(float(np.mean(hist_lens)), 3) if hist_lens else 0.0,
            "hist_len_max": int(max(hist_lens)) if hist_lens else 0,
            "ambiguous_target_ratio": round(n_amb_target / max(1, len(hist_lens)), 6),
        }
        s = split_stats[split]
        print(f"  {split:5s}: {s['n_written']:,} 条 (drop hist={s['n_drop_empty_hist']}, oob={s['n_drop_out_of_range']})"
              f"  hist 均长 {s['hist_len_mean']}  目标 SID 歧义 {s['ambiguous_target_ratio']:.2%}")

        # T3 seq2title
        with open(os.path.join(out_dir, "tasks", f"seq2title_{split}.jsonl"), "w", encoding="utf-8") as f:
            for u, hist, tgt in rows:
                if not hist or tgt >= N or any(h >= N for h in hist):
                    continue
                f.write(json.dumps({
                    "task": "seq2title",
                    "user_id": f"A{u}",
                    "history_item_sid": [sid_str[h] for h in hist],
                    "target_title": titles[tgt],
                    "target_item_id": tgt,
                }, ensure_ascii=False) + "\n")

    # ── 7. T2 sid↔title（每物品正反两条）──────────────────────────
    p = os.path.join(out_dir, "tasks", "itemfeat.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for i in range(N):
            f.write(json.dumps({"task": "sid2title", "sid": sid_str[i], "title": titles[i],
                                "item_id": i}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"task": "title2sid", "title": titles[i], "sid": sid_str[i],
                                "item_id": i}, ensure_ascii=False) + "\n")
    print(f"  T2 itemfeat: {2 * N:,} 条")

    # ── 8. T4 text2sid ─────────────────────────────────────────────
    p = os.path.join(out_dir, "tasks", "text2sid.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for i in range(N):
            f.write(json.dumps({"task": "text2sid", "text": texts[i], "sid": sid_str[i],
                                "item_id": i}, ensure_ascii=False) + "\n")
    print(f"  T4 text2sid: {N:,} 条")

    # ── 9. info/ ───────────────────────────────────────────────────
    # 9.1 item_info（MiniOneRec convert_dataset.py 同格式：sid \t title \t item_id）
    with open(os.path.join(out_dir, "info", f"{dom}.item_info.txt"), "w", encoding="utf-8") as f:
        for i in range(N):
            f.write(f"{sid_str[i]}\t{titles[i]}\t{i}\n")

    # 9.2 训练频次（给严格口径选 representative）
    freq = collections.Counter()
    for split in ("train", "valid", "test"):
        for _u, hist, tgt in read_inter(os.path.join(base, f"{dom}.{split}.inter")):
            freq[tgt] += 1
            for h in hist:
                freq[h] += 1

    sid2items = {}
    for sid_s, ids in bucket.items():
        # representative：训练频次最高，平局取最小 item_id（确定性，且与 target 无关，无泄漏）
        rep = min(ids, key=lambda i: (-freq.get(i, 0), i))
        sid2items[sid_s] = {"items": sorted(ids), "representative": rep, "size": len(ids)}
    with open(os.path.join(out_dir, "info", "sid2items.json"), "w", encoding="utf-8") as f:
        json.dump(sid2items, f, ensure_ascii=False)

    # 9.3 SID 词表（有序，供 tokenizer.add_tokens 与语义初始化对齐）
    vocab = [f"<{PREFIXES[l]}_{k}>" for l in range(L) for k in range(int(codes[:, l].max()) + 1)]
    with open(os.path.join(out_dir, "info", "sid_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"  SID 词表: {len(vocab)} 个 token（{L} 层 × 各 {len(vocab)//L} 码）")

    # 9.4 codebook（M4 语义初始化用，来自 RQ-VAE ckpt）
    codebook_info = {"path": None, "shape": None}
    if os.path.exists(sid_ckpt):
        try:
            import torch
            sd = torch.load(sid_ckpt, map_location="cpu", weights_only=False)
            st = sd["state_dict"]
            cb = np.stack([st[f"rq.vq_layers.{l}.embedding.weight"].numpy() for l in range(L)])
            cb_path = os.path.join(out_dir, "info", "codebook.npy")
            np.save(cb_path, cb)
            codebook_info = {"path": cb_path, "shape": list(cb.shape), "source": sid_ckpt}
            print(f"  codebook: {tuple(cb.shape)} -> {cb_path}")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] codebook 导出失败（不影响主流程）: {e}")
    else:
        print(f"  [warn] 找不到 ckpt {sid_ckpt}，跳过 codebook 导出")

    # 9.5 模板落盘
    with open(os.path.join(out_dir, "info", "prompt_templates.json"), "w", encoding="utf-8") as f:
        json.dump({"alpaca_header": ALPACA_HEADER, "tasks": PROMPT_TEMPLATES}, f,
                  ensure_ascii=False, indent=1)

    # ── 10. stats ──────────────────────────────────────────────────
    stats = {
        "domain": dom,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sid_npy": sid_npy,
        "n_items": int(N),
        "n_layers": int(L),
        "codebook": codebook_info,
        "sid_quality": {
            "n_unique_sid": int(n_unique),
            "icr": round(n_unique / N, 6),
            "collision_rate": round(1 - n_unique / N, 6),
            "n_collided_items": int(collided_items),
            "max_bucket_size": int(max_bucket),
        },
        "title": {
            "max_chars": args.max_title_chars,
            "n_truncated": int(n_trunc_title),
            "n_empty_fallback": int(n_empty_title),
        },
        "splits": split_stats,
        "tasks": {
            "T1_seq2sid": {s: split_stats[s]["n_written"] for s in ("train", "valid", "test")},
            "T2_itemfeat": 2 * int(N),
            "T3_seq2title": {s: split_stats[s]["n_written"] for s in ("train", "valid", "test")},
            "T4_text2sid": int(N),
        },
        "output_layout": {
            "sid_index_path": idx_path.replace("\\", "/"),
            "item_meta_path": meta_path.replace("\\", "/"),
            "train_csv": os.path.join(out_dir, "train", f"{dom}_5_train.csv").replace("\\", "/"),
            "valid_csv": os.path.join(out_dir, "valid", f"{dom}_5_valid.csv").replace("\\", "/"),
            "test_csv": os.path.join(out_dir, "test", f"{dom}_5_test.csv").replace("\\", "/"),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"  stats -> {os.path.join(out_dir, 'stats.json')}   ({stats['elapsed_sec']}s)")

    # ── 11. 自检：打印两条完整样本 ─────────────────────────────────
    t1 = PROMPT_TEMPLATES["seq2sid"]
    with open(os.path.join(out_dir, "train", f"{dom}_5_train.csv"), encoding="utf-8") as f:
        rr = list(csv.DictReader(f))
    print("\n  ── 样本自检（T1 seq2sid，真实产物）──")
    for row in rr[:2]:
        hist = eval(row["history_item_sid"])  # noqa: S307
        prompt = ALPACA_HEADER + t1["instruction"] + "\n\n" + \
            f"### User Input: \n{t1['input'].format(hist=t1['hist_sep'].join(hist))}\n\n### Response:\n"
        print("  " + "-" * 66)
        print("  " + prompt.replace("\n", "\n  ") + row["item_sid"] + "\n")
    print(f"  提示词字符数（第 1 条）= {len(prompt)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
