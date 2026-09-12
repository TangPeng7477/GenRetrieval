#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1-3: 商品主图并发下载（仅标准库，不依赖 requests/PIL）
==========================================================
输入:  data/Amazon23/<short>/<short>.item.json      （prepare_amazon23.py 产出）
输出:  data/Amazon23/<short>/images/<item_idx>[_s<slot>].jpg
       data/Amazon23/<short>/image_manifest.tsv     逐条结果（可断点续传）
       data/Amazon23/<short>/image_stats.json       成功率统计

设计要点
--------
* 断点续传：manifest 中 status=ok 的条目跳过；重跑不重下
* 容错：超时/404/坏图都记 manifest 且不中断（多模态方案里"无图回退纯文本"是一等公民）
* 坏图判定：魔数校验（JPEG/PNG/GIF/WebP）+ 最小体积阈值，避免把错误页/占位像素存成图
* 默认只下主图（--num_images 1），副图留给后续多图融合消融
* 线程池并发（IO 密集，默认 16 线程）

用法:
    python scripts/multimodal/download_images.py --short IandS --workers 16
    python scripts/multimodal/download_images.py --short IandS --num_images 2 --limit 50  # 冒烟测试
    # 多类目合并集：复用已下好的单类目图片（按 asin 匹配，不重下）
    python scripts/multimodal/download_images.py --short Multi2 --reuse_from IandS
"""

import argparse
import json
import os
import random
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

MAGIC = [b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF8", b"RIFF"]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")


def build_reuse_index(data_dir, reuse_shorts):
    """image 文件名是 <item_idx>.jpg，而不同 short 的 idx 不同 —— 必须按 asin 桥接。

    返回 asin -> 绝对路径（仅在源 manifest 里 status=ok 的才可用）。
    """
    index = {}
    for sh in reuse_shorts:
        root = os.path.join(data_dir, sh)
        item_json = os.path.join(root, f"{sh}.item.json")
        manifest = os.path.join(root, "image_manifest.tsv")
        if not os.path.exists(item_json) or not os.path.exists(manifest):
            print(f"[reuse] 跳过 {sh}：缺少 item.json 或 manifest")
            continue
        with open(item_json, "r", encoding="utf-8") as f:
            old_items = json.load(f)
        ok = set()
        with open(manifest, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 5 and p[4] == "ok":
                    ok.add((p[0], p[1]))
        n = 0
        for old_idx, rec in old_items.items():
            if (old_idx, "0") not in ok:
                continue
            src = os.path.join(root, "images", f"{old_idx}.jpg")
            if os.path.exists(src) and src not in index.values():
                index[rec.get("asin")] = src
                n += 1
        print(f"[reuse] {sh}: 可用图片 {n:,} 张（asin 索引）")
    return index


def is_valid_image(data, min_bytes):
    if len(data) < min_bytes:
        return False
    return any(data.startswith(m) for m in MAGIC)


def fetch_to_file(url, dest, timeout, min_bytes, retries):
    """下载并落盘。返回 (status, nbytes, detail)。"""
    err = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "image/*,*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not is_valid_image(data, min_bytes):
                return "bad_image", len(data), "not an image / too small"
            with open(dest, "wb") as f:
                f.write(data)
            return "ok", len(data), ""
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                return f"http_{e.code}", 0, f"HTTP {e.code}"
            err = f"HTTP {e.code}"
        except Exception as e:                       # noqa: BLE001
            err = f"{type(e).__name__}:{str(e)[:70]}"
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1) + random.random() * 0.5)
    return "error", 0, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="IandS")
    ap.add_argument("--data_dir", default="data/Amazon23")
    ap.add_argument("--num_images", type=int, default=1, help="每物品取前 N 张（1=只主图）")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--min_bytes", type=int, default=2000)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help=">0 只处理前 N 个任务（冒烟测试）")
    ap.add_argument("--reuse_from", default="",
                    help="逗号分隔的已有 short，按 asin 复用其已下载图片（多类目合并集省流量）")
    args = ap.parse_args()

    root = os.path.join(args.data_dir, args.short)
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest_path = os.path.join(root, "image_manifest.tsv")

    with open(os.path.join(root, f"{args.short}.item.json"), "r", encoding="utf-8") as f:
        items = json.load(f)

    tasks = []
    for idx, rec in items.items():
        urls = rec.get("images") or []
        for slot in range(min(args.num_images, len(urls))):
            tasks.append((idx, slot, urls[slot]))
    if args.limit:
        tasks = tasks[: args.limit]

    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 5 and p[4] == "ok":
                    done.add((p[0], p[1]))
    todo = [(i, s, u) for (i, s, u) in tasks if (i, str(s)) not in done]

    # ---- 复用已有图片（多类目合并集：I&S 的图已经下好，不该重下）----
    reuse_index, reused = {}, []
    if args.reuse_from:
        reuse_index = build_reuse_index(
            args.data_dir, [s.strip() for s in args.reuse_from.split(",") if s.strip()]
        )
        still_todo = []
        for idx, slot, url in todo:
            asin = items.get(idx, {}).get("asin")
            src = reuse_index.get(asin)
            if src and slot == 0:
                reused.append((idx, slot, url, src))
            else:
                still_todo.append((idx, slot, url))
        todo = still_todo

    n_items_with_img = sum(1 for r in items.values() if r.get("images"))
    print(f"[images] 物品={len(items):,}  有图 URL={n_items_with_img:,} "
          f"({n_items_with_img/max(1,len(items))*100:.1f}%)")
    print(f"[images] 任务={len(tasks):,}  已 ok={len(done):,}  复用={len(reused):,}  本轮待下={len(todo):,}")

    results, t0 = [], time.time()
    new_file = not os.path.exists(manifest_path)
    with open(manifest_path, "a", encoding="utf-8") as mf:
        if new_file:
            mf.write("item_idx\tslot\turl\tbytes\tstatus\tdetail\n")
        for idx, slot, url, src in reused:
            dest = os.path.join(img_dir, f"{idx}.jpg" if slot == 0 else f"{idx}_s{slot}.jpg")
            try:
                shutil.copyfile(src, dest)
                nbytes = os.path.getsize(dest)
                mf.write(f"{idx}\t{slot}\t{url}\t{nbytes}\tok\treuse:{os.path.relpath(src)}\n")
                results.append("ok")
            except Exception as e:                      # noqa: BLE001
                mf.write(f"{idx}\t{slot}\t{url}\t0\terror\treuse_failed:{str(e)[:60]}\n")
                results.append("error")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {}
            for idx, slot, url in todo:
                dest = os.path.join(img_dir, f"{idx}.jpg" if slot == 0 else f"{idx}_s{slot}.jpg")
                futs[ex.submit(fetch_to_file, url, dest, args.timeout, args.min_bytes, args.retries)] = (idx, slot, url)
            for i, fut in enumerate(as_completed(futs), 1):
                idx, slot, url = futs[fut]
                status, nbytes, detail = fut.result()
                results.append(status)
                mf.write(f"{idx}\t{slot}\t{url}\t{nbytes}\t{status}\t{detail}\n")
                if i % 200 == 0 or i == len(todo):
                    el = max(time.time() - t0, 1e-6)
                    print(f"  {i}/{len(todo)}  {i/el:.1f} img/s  ok={results.count('ok')}  elapsed={el:.0f}s", flush=True)

    stats = {}
    for st in results:
        stats[st] = stats.get(st, 0) + 1
    ok_files = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    summary = {
        "items_total": len(items),
        "items_with_image_url": n_items_with_img,
        "url_coverage": round(n_items_with_img / max(1, len(items)), 4),
        "tasks_this_run": len(todo),
        "reused_this_run": len(reused),
        "downloaded_ok": stats.get("ok", 0),
        "status_counts": stats,
        "image_files_on_disk": ok_files,
        "effective_coverage": round(ok_files / max(1, len(items)), 4),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(root, "image_stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[images] " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
