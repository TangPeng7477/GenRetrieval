#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1-4b: 商品图像向量（SigLIP-base-patch16-224）
==============================================
输入: data/Amazon23/<short>/images/<item_idx>.jpg   （download_images.py 产出，有效率 99.91%）
输出: data/Amazon23/<short>/emb/emb_image_siglip.npy        (N, 768) float32, L2 归一化
      data/Amazon23/<short>/emb/emb_image_siglip.meta.json
      data/Amazon23/<short>/emb/emb_image_mask.npy           (N,) bool，False = 无图（26 个）

设计要点
--------
* 行序与 encode_text.py 严格一致（sorted(int(idx))），否则融合会错位——这是最容易踩的坑
* 无图/坏图 → 置零向量 + mask=False，融合时由门控自适应降权（不用丢弃该物品）
* SigLIP-base-patch16-224：768 维，fp16 显存 ~0.4GB，4GB 卡轻松跑

用法:
    HF_ENDPOINT=https://hf-mirror.com python scripts/multimodal/encode_image.py --short IandS --batch_size 32
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL = "google/siglip-base-patch16-224"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="IandS")
    ap.add_argument("--data_dir", default="data/Amazon23")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_images", type=int, default=1, help="每物品用前 N 张图取均值（>1 需先下副图）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=20, help="每 N 个 batch 落一次断点（0=关闭）")
    ap.add_argument("--no_resume", action="store_true", help="忽略已有断点，从头跑")
    args = ap.parse_args()

    root = os.path.join(args.data_dir, args.short)
    img_dir = os.path.join(root, "images")
    out_dir = os.path.join(root, "emb")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(root, f"{args.short}.item.json"), "r", encoding="utf-8") as f:
        items = json.load(f)
    idxs = sorted(items, key=lambda k: int(k))          # ← 与文本编码相同的行序
    if args.limit:
        idxs = idxs[: args.limit]

    # 收集待编码图片（N 行 × num_images 列，缺图记 None）
    paths, mask_rows = [], []
    for i in idxs:
        slots = []
        for s in range(args.num_images):
            p = os.path.join(img_dir, f"{i}.jpg" if s == 0 else f"{i}_s{s}.jpg")
            slots.append(p if os.path.exists(p) else None)
        paths.append(slots)
        mask_rows.append(any(slots))
    n_have = sum(mask_rows)
    print(f"[image] 物品={len(idxs):,}  有图={n_have:,} ({n_have/len(idxs)*100:.2f}%)  "
          f"每物品图数={args.num_images}")

    print(f"[model] 加载 {args.model} ...")
    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    dim = model.config.vision_config.hidden_size
    print(f"[model] 就绪 ({time.time()-t0:.1f}s)  dim={dim}  显存={torch.cuda.memory_allocated()/1e9:.2f}GB")

    embs = np.zeros((len(idxs), dim), dtype=np.float32)
    counts = np.array([sum(1 for p in slots if p) for slots in paths], dtype=np.float32)
    slot_done = np.zeros(len(idxs), dtype=np.int64)
    done = 0

    # --- 断点续跑：只在跑完才写盘的脚本，被中断一次就白跑几十分钟 ---
    ckpt = os.path.join(out_dir, "emb_image_siglip.partial.npz")
    if os.path.exists(ckpt) and not args.no_resume:
        try:
            z = np.load(ckpt)
            if z["embs"].shape == embs.shape:
                embs = z["embs"]
                slot_done = z["slot_done"]
                done = int(slot_done.sum())
                print(f"[resume] 复用断点，已完成 {int((slot_done >= counts).sum()):,} 个物品 "
                      f"(删掉 {ckpt} 可强制重跑)")
        except Exception as e:                                  # noqa: BLE001
            print(f"[resume] 断点不可用，从头开始: {e}")

    with torch.inference_mode():
        # 展平所有 (物品, slot) 任务，按 batch 处理；已完成的物品跳过
        pending = slot_done < counts
        flat = [(r, s, p) for r, slots in enumerate(paths)
                for s, p in enumerate(slots) if p and pending[r]]
        nb_total = (len(flat) + args.batch_size - 1) // args.batch_size
        for bi, st in enumerate(range(0, len(flat), args.batch_size)):
            chunk = flat[st: st + args.batch_size]
            imgs, rows = [], []
            for r, s, p in chunk:
                try:
                    im = Image.open(p).convert("RGB")
                    imgs.append(im)
                    rows.append(r)
                except Exception as e:                    # noqa: BLE001
                    print(f"  [warn] 读图失败 {p}: {e}")
            if not imgs:
                continue
            px = proc(images=imgs, return_tensors="pt").to("cuda")
            e = model.get_image_features(**px)
            e = torch.nn.functional.normalize(e.float(), p=2, dim=1).cpu().numpy()
            # 同一物品多张图 → 各自累加（最后按数量平均）
            for vec, r in zip(e, rows):
                embs[r] += vec
                slot_done[r] += 1
                done += 1
            if bi % 5 == 0 or st + len(chunk) >= len(flat):
                el = max(time.time() - t0, 1e-6)
                fin = int((slot_done >= counts).sum())
                eta = max(len(flat) - (st + len(chunk)), 0) / max(done / el, 1e-9)
                print(f"  {fin:,}/{len(idxs):,} 物品 ({fin/len(idxs)*100:.1f}%)  "
                      f"{done/el:.0f} img/s  elapsed={el:.0f}s  eta={eta:.0f}s", flush=True)
            if args.ckpt_every and (bi + 1) % args.ckpt_every == 0:
                np.savez(ckpt, embs=embs, slot_done=slot_done)

    # 多图平均 + 重新归一化
    nz = counts > 0
    embs[nz] /= counts[nz, None]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    nz2 = norms[:, 0] > 0
    embs[nz2] /= norms[nz2]

    mask = np.array(mask_rows, dtype=bool)
    np.save(os.path.join(out_dir, "emb_image_siglip.npy"), embs)
    if os.path.exists(ckpt):
        os.remove(ckpt)                                         # 跑完清掉断点
    np.save(os.path.join(out_dir, "emb_image_mask.npy"), mask)
    meta = {
        "model": args.model,
        "dim": int(dim),
        "n_items": int(len(idxs)),
        "n_have_image": int(n_have),
        "coverage": round(n_have / len(idxs), 4),
        "num_images_per_item": args.num_images,
        "normalized": True,
        "row_order": "sorted(int(idx))  (与 emb_text_* 一致)",
        "zero_vector_for_missing": True,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "emb_image_siglip.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] shape={embs.shape}  覆盖={meta['coverage']}")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
