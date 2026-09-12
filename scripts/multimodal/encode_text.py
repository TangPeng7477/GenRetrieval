#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1-4a: 商品文本向量（Qwen3-Embedding-0.6B）
============================================
输入: data/Amazon23/<short>/<short>.item.json
输出: data/Amazon23/<short>/emb/emb_text_<fields>.npy      (N, 1024) float32, L2 归一化
      data/Amazon23/<short>/emb/emb_text_<fields>.meta.json

为什么用 Qwen3-Embedding-0.6B
-----------------------------
* esci-ai-search 项目实测：换掉 bge-base-en-v1.5（MTEB-en 63.5）后 LCP +20%、
  prefix cohesion ×1.58~2.29（MTEB-en 70.7）—— SID 质量对上游 embedding 极敏感
* 1024 维，fp16 显存 ~1.2GB，4GB 卡可跑

文本字段组合（消融用，--fields）:
    title                       : 只标题
    title+features              : 标题 + 卖点列表（2023 metadata 独有，信息密度高于 description）
    title+features+category     : 再加品牌/类目（默认）

用法:
    HF_ENDPOINT=https://hf-mirror.com python scripts/multimodal/encode_text.py \
        --short IandS --fields title+features+category --batch_size 16
"""

import argparse
import json
import os
import time

# 必须在 import torch / 初始化 CUDA 之前设置。
# Windows 不支持 expandable_segments（日志里那条 warning 就是它），
# 缓存分配器遇到大量不同 shape 时会不断申请新块、且无法合并，
# 在 4GB 卡上会把专用显存顶满并触发 WDDM 换页到系统内存（实测溢出 6.5GB）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def build_text(rec, fields):
    """按字段组合拼文本；顺序固定，保证可复现。"""
    parts = []
    if "title" in fields:
        parts.append(rec.get("title") or "")
    if "features" in fields:
        parts.extend(rec.get("features") or [])
    if "category" in fields:
        cats = rec.get("categories") or []
        if cats:
            parts.append(" > ".join(cats[-3:]))
        if rec.get("brand"):
            parts.append(rec["brand"])
    text = " | ".join(p.strip() for p in parts if p and p.strip())
    return text if text else "unknown item"


def last_token_pool(last_hidden, attention_mask):
    """Qwen3-Embedding 官方 pooling：有左 padding 取最后一个 token，否则按 mask 取真实末位。"""
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[idx, seq_lens]


def rss_gb():
    """进程常驻内存（Windows 用 K32GetProcessMemoryInfo，其它平台读 /proc）。失败返回 nan。"""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            # 关键：HANDLE 必须按指针宽度声明。若不声明，ctypes 默认 restype=c_int，
            # GetCurrentProcess() 的伪句柄 (-1) 会被截成 32 位；再传回
            # K32GetProcessMemoryInfo 时寄存器高位是脏数据 → 调用失败 → 探针恒返回 nan
            # （实测就是这样把内存探针跑成了摆设，而它恰恰是抓"显存溢出到系统内存"的唯一线索）。
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            fn = getattr(kernel32, "K32GetProcessMemoryInfo", None)
            if fn is None:                                          # 很旧的系统才走这条路
                fn = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            fn.restype = wintypes.BOOL

            if not fn(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return float("nan")
            return pmc.WorkingSetSize / 1e9
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:                                            # noqa: BLE001
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="IandS")
    ap.add_argument("--data_dir", default="data/Amazon23")
    ap.add_argument("--fields", default="title+features+category")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch_size", type=int, default=64, help="单个 batch 的最大物品数（上限）")
    ap.add_argument("--token_budget", type=int, default=4096,
                    help="单个 batch 的 token 预算（长度分桶后按此装箱）。"
                         "4GB 卡用 4096：实测 8192 会把 Qwen3-Embedding-0.6B 顶到显存溢出换页。"
                         "注意与 batch_size 是『两者取更严』的关系，不是二选一")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=20, help="每 N 个 batch 落一次断点（0=关闭）")
    ap.add_argument("--empty_cache_every", type=int, default=25,
                    help="每 N 个 batch 调一次 torch.cuda.empty_cache()（仅当碎片空闲块 >0.5GB 才真正执行）")
    ap.add_argument("--no_resume", action="store_true", help="忽略已有断点，从头跑")
    ap.add_argument("--no_bucket", action="store_true",
                    help="关闭长度分桶（退化为原始顺序 + 定长 batch，仅用于对照）")
    args = ap.parse_args()

    fields = [f.strip() for f in args.fields.split("+") if f.strip()]
    root = os.path.join(args.data_dir, args.short)
    out_dir = os.path.join(root, "emb")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.fields.replace("+", "-")
    if args.limit:
        # 冒烟测试产物放到 _smoke/ 子目录，绝不污染正式产物
        # （否则 encode_all.sh 的 `emb_text_*.npy` 判定会误以为"已完成"而跳过）
        out_dir = os.path.join(out_dir, "_smoke")
        os.makedirs(out_dir, exist_ok=True)
    out_npy = os.path.join(out_dir, f"emb_text_{tag}.npy")

    with open(os.path.join(root, f"{args.short}.item.json"), "r", encoding="utf-8") as f:
        items = json.load(f)
    idxs = sorted(items, key=lambda k: int(k))
    if args.limit:
        idxs = idxs[: args.limit]
    texts = [build_text(items[i], fields) for i in idxs]
    lens = [len(t) for t in texts]
    print(f"[text] 物品={len(idxs):,} 字段={'+'.join(fields)} 文本均长={np.mean(lens):.0f} 字符 "
          f"(min={min(lens)}, max={max(lens)})")
    print(f"[text] 示例: {texts[0][:160]}")

    print(f"[model] 加载 {args.model} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    # --- 真实 token 数（用于长度分桶的装箱预算）---
    ntok = []
    for s in range(0, len(texts), 4096):
        enc = tok(texts[s: s + 4096], padding=False, truncation=True,
                  max_length=args.max_length)
        ntok.extend(len(x) for x in enc["input_ids"])
    ntok = np.asarray(ntok, dtype=np.int64)
    print(f"[text] token 统计: 均 {ntok.mean():.0f}  中位 {np.median(ntok):.0f}  "
          f"p95 {np.percentile(ntok, 95):.0f}  max {ntok.max()}  "
          f"(截断上限 {args.max_length})")

    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    print(f"[model] 就绪 ({time.time()-t0:.1f}s)  显存={torch.cuda.memory_allocated()/1e9:.2f}GB")

    embs = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)

    # --- 长度分桶：按 token 数升序排列，使同一 batch 内长度接近 ---
    # 目的：原来的"定长 batch + padding 到批内最长"在混合长度下浪费严重
    #      （均长 180 / 最长 490 token，padding 虚耗最高 2.7x），
    #      且 batch 越大越容易把 4GB 卡顶到显存碎片化。
    # 做法：按 token 预算装箱，每批 padding 到本批最长（分桶后≈该批自身长度）。
    order = np.argsort(ntok, kind="stable") if not args.no_bucket else np.arange(len(texts))

    # --- 断点续跑：脚本动辄跑数十分钟，被中断后不必从头再来 ---
    ckpt = out_npy + ".partial.npz"
    start_i = 0
    if os.path.exists(ckpt) and not args.no_resume:
        try:
            z = np.load(ckpt)
            if z["embs"].shape == embs.shape and np.array_equal(z["order"], order):
                embs = z["embs"]
                start_i = int(z["next_i"])
                print(f"[resume] 复用断点，从 {start_i:,}/{len(texts):,} 继续 "
                      f"(删掉 {ckpt} 可强制重跑)")
            else:
                print("[resume] 断点与当前配置不匹配，从头开始")
        except Exception as e:                                  # noqa: BLE001
            print(f"[resume] 断点不可用，从头开始: {e}")

    n = len(texts)
    with torch.inference_mode():
        # done = 本轮实际编码条数。注意不能用 i 来算速率：
        # 断点续跑时 i 从 start_i 起跳，i/elapsed 会把历史进度算进本轮，
        # 于是"item/s"虚高、ETA 虚低（实测虚高 2.7x）。
        i, bi, done = start_i, 0, 0
        while i < n:
            # 贪心装箱：长度已升序，加下一个物品后本批最大长度即 ntok[order[j]]
            # 两个条件是与(AND)关系 —— 谁先撞到就停谁，等价于本批条数 = min(batch_size, budget//本批最长)。实测：
            #   I&S(文本长, 均173) 只有 3% 的批次由 batch_size 卡住，97% 由 token_budget 卡住；
            #   VG (文本短, 均127) 有 13% 的批次由 batch_size 卡住。
            # 所以调 batch_size 对长文本域几乎无效，真正控速/控显存的是 token_budget。
            j = i + 1
            while (j < n and (j - i) < args.batch_size
                   and (j - i + 1) * int(ntok[order[j]]) <= args.token_budget):
                j += 1
            rows = order[i:j]
            enc = tok([texts[r] for r in rows], padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt").to("cuda")
            out = model(**enc)
            e = last_token_pool(out.last_hidden_state, enc["attention_mask"])
            e = torch.nn.functional.normalize(e.float(), p=2, dim=1)
            embs[rows] = e.cpu().numpy()
            i, bi, done = j, bi + 1, done + len(rows)
            if bi % 20 == 0 or i >= n:
                el = max(time.time() - t0, 1e-6)
                rate = done / el
                eta = max(n - i, 0) / max(rate, 1e-9)
                alloc = torch.cuda.memory_allocated() / 1e9
                resv = torch.cuda.memory_reserved() / 1e9
                print(f"  {i:,}/{n:,} ({i/n*100:.1f}%)  {rate:.1f} item/s  "
                      f"batch={len(rows)} tokens={int(ntok[rows].sum()):,}  "
                      f"cuda[alloc={alloc:.2f} resv={resv:.2f}] rss={rss_gb():.2f}GB  "
                      f"elapsed={el:.0f}s  eta={eta:.0f}s", flush=True)
            if args.ckpt_every and bi % args.ckpt_every == 0:
                np.savez(ckpt, embs=embs, next_i=i, order=order)
            # 释放分配器缓存中无法复用的空闲块（Windows 无法合并，会一路涨到换页）
            if args.empty_cache_every and bi % args.empty_cache_every == 0:
                gap = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                if gap > 0.5e9:
                    torch.cuda.empty_cache()

    np.save(out_npy, embs)
    meta = {
        "model": args.model,
        "fields": fields,
        "dim": int(embs.shape[1]),
        "n_items": int(embs.shape[0]),
        "normalized": True,
        "item_json": f"{args.short}.item.json",
        "row_order": "sorted(int(idx))",
        "max_length": args.max_length,
        "token_budget": args.token_budget,
        "length_bucketed": (not args.no_bucket),
        "text_mean_chars": float(np.mean(lens)),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    # meta 必须在清理 checkpoint 之前落盘：清理属于可有可无的收尾，
    # 一旦它抛异常（例如沙箱/杀软拦截删除），不能把已经跑完的成果的元信息一起带走。
    with open(os.path.join(out_dir, f"emb_text_{tag}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # checkpoint 清理一律 best-effort：主产物 .npy 已经落盘，删不掉只是留个垃圾文件，
    # 绝不能让整个脚本以非 0 退出（上游 encode_all.sh 会据此中止后续阶段）。
    if os.path.exists(ckpt):
        try:
            os.remove(ckpt)
        except OSError as e:                                    # noqa: PERF203
            print(f"[warn] 断点文件未能清理，忽略: {ckpt} ({e})")
    print(f"[done] {out_npy}  shape={embs.shape}")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
