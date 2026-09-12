# -*- coding: utf-8 -*-
"""诊断：文本重复组里，图像到底一不一样？门控有没有用上图像？

用法（从仓库根目录执行）：
    .venv/Scripts/python.exe scripts/multimodal/probe_gate_twins.py <IandS|VG>

输出：把"文本逐位重复"的组细分成 A/B/C 三类——
  A 文同 + 图同  → 融合必同（都是同一商品的多个 listing，不可分）
  B 文同 + 图不同 + 融合仍同 → 若 > 0 说明门控把图像分支关掉了（实测两域都是 0）
  C 文同 + 图不同 → 融合不同（被融合拆开）
反向也统计"图同 + 文不同"的组融合后是否仍相同（实测两域都是 0）。
"""
import json
import collections
import sys
import numpy as np

SHORT = sys.argv[1] if len(sys.argv) > 1 else "VG"
D = f"data/Amazon23/{SHORT}"

et = np.load(f"{D}/emb/long/emb_fused_text.npy").astype(np.float32)
ei = np.load(f"{D}/emb/emb_image_siglip.npy").astype(np.float32)
eg = np.load(f"{D}/emb/long/emb_fused_gate_e60.npy").astype(np.float32)
print(f"N={len(et)}  text{et.shape}  image{ei.shape}  gate{eg.shape}")


def groups_of(mat):
    inv = np.unique(mat, axis=0, return_inverse=True)[1]
    g = collections.defaultdict(list)
    for i, k in enumerate(inv):
        g[k].append(i)
    return [v for v in g.values() if len(v) > 1]


def all_same(mat, idx):
    return all((mat[idx[0]] == mat[j]).all() for j in idx[1:])


tg = groups_of(et)
ig = groups_of(ei)
gg = groups_of(eg)
print(f"bit-identical 组数：text={len(tg)}  image={len(ig)}  gate={len(gg)}")
print(f"单模态重复占比：text={sum(len(x) for x in tg)}/{len(et)}  "
      f"image={sum(len(x) for x in ig)}/{len(ei)}")

# 逐组分类
cat = collections.Counter()
ex = collections.defaultdict(list)
for idx in tg:
    img_same = all_same(ei, idx)
    gate_same = all_same(eg, idx)
    if img_same:
        k = "A_text同+图同→输入完全相同，gate必同" if gate_same else "A*_异常(图同但gate不同)"
    elif gate_same:
        k = "B_text同+图不同+gate同→门控把图像关掉了"
    else:
        k = "C_text同+图不同+gate不同→门控用上了图像"
    cat[k] += 1
    ex[k[0]].append(idx)

print("\n=== 文本重复组的细分（VG）===")
for k, v in sorted(cat.items()):
    print(f"  {v:4d}  {k}")

# 反向：image 相同但 text 不同
it_groups = [x for x in ig if not all_same(et, x)]
print(f"\n图像相同但文本不同的组：{len(it_groups)}"
      f"（其中 gate 仍相同 {sum(1 for x in it_groups if all_same(eg, x))} 组）")

# 元数据
man = {}
with open(f"{D}/image_manifest.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0].isdigit():
            man[int(p[0])] = p[2]

raw = json.load(open(f"{D}/{SHORT}.item.json", encoding="utf-8"))
if isinstance(raw, dict):
    items = {int(k): v for k, v in raw.items()}
else:
    items = {i: v for i, v in enumerate(raw)}


def title_of(i):
    v = items.get(i, {})
    t = v.get("title") or v.get("asin") or str(i)
    return t[:70]


print("\n=== 样例（每类 3 组）===")
for k in "ABC":
    for idx in ex.get(k, [])[:3]:
        print(f"\n[{k}] 组大小 {len(idx)}  item_idx={idx}")
        for i in idx:
            print(f"    {i:6d}  {title_of(i)}")
            print(f"           图: data/Amazon23/{SHORT}/images/{i}.jpg")
            print(f"           url: {man.get(i, '(无)')[:95]}")
