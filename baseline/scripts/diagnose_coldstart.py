"""把 test 的 HR@10 按「目标是否为冷启动物品」拆开 —— 判定 sid_prefix 的领先是否真来自内容泛化。

## 为什么需要这个脚本（2026-09-13）

I&S 上跑完第一批可训练基线后出现一个红旗：

    零训练 sid_prefix   HR@10 = 0.0449
    训练过的 sasrec     HR@10 = 0.0157
    训练过的 gru4rec    HR@10 = 0.0144
    训练过的 bprmf      HR@10 = 0.0084
    训练过的 sid_gr     HR@10 = 0.0081

**一个零训练的基线比所有训练过的模型高 3 倍**，在下结论之前必须先证伪。两个候选解释：

- **H1（真机理 · 内容泛化）**：本项目 test 有 **33%** 的目标商品在 train 里从未出现
  （`docs/DATASET.md §8.3`）。ID-based 方法对这类目标**结构上无法命中**（没有可查的 embedding），
  而 SID / 融合向量来自 item 的文本+图像内容，冷启动商品只要有内容就能被打分。
  若 H1 成立，`content_ann` / `sid_prefix` 在冷目标上应显著高于 `pop` / `itemknn`（后者应≈0）。
- **H0（口径或泄漏）**：若 `pop` 这类纯流行度基线在冷目标上也能命中，或 `sid_prefix` 在冷目标上
  的命中率高得不合理，说明评估有问题（mask 漏了、目标混进了历史、或冷/热标注写错）。

**这是一个可证伪的检查**：两种结果都直接决定 `RESULTS.md` 的结论能不能写。

## 口径（与 `RESULTS.md` 严格一致）

- 冷/热标注**样本级**：`cold = (target 不在全量 train 交互出现过的物品集合里)`。
  与 `DATASET.md §8.3` 的 **item 级**口径同源（它数的是"不同的 test 目标商品"，
  这里数的是"test 样本"），两者都打印，避免口径混淆。
- 评估完全复用 `common/nn.py::evaluate`（同一份 mask、同一个 rank 定义），
  只把得到的 ranks 按冷/热切开分别算指标。

用法：
    ./.venv/Scripts/python.exe -m baseline.scripts.diagnose_coldstart
    ./.venv/Scripts/python.exe -m baseline.scripts.diagnose_coldstart --domain IandS --models pop,itemknn,content_ann,sid_prefix
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ..common import nn as N
from ..common.data import load_domain
from ..generative.retrieval import ContentANN, SidPrefix
from ..models.heuristic import ItemKNN, PopRec

# 只做零训练组：它们秒级出结果，且正好横跨"ID-based vs 内容-based"两侧
ZERO_SHOT = {
    "pop":         (PopRec,     None),                       # 纯流行度（ID-based，无内容）
    "itemknn":     (ItemKNN,    dict(topk=100)),             # 协同过滤（ID-based，无内容）
    "content_ann": (ContentANN, dict(agg="mean")),           # 融合向量（纯内容，无协同）
    "sid_prefix":  (SidPrefix,  dict(weights=(1.0, 1.5, 2.0))),  # SID 前缀（内容量化后的结构）
}


def cold_flags(data, split: str = "test") -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (样本级 cold 布尔, 目标物品 id, 统计 dict)。"""
    train_items = np.zeros(data.n_items, dtype=bool)
    for seq in data.train_seq.values():
        for it in seq:
            train_items[it] = True

    ev = getattr(data, split)
    tgt = ev.targets
    cold = ~train_items[tgt]

    # item 级口径（对齐 DATASET.md §8.3）：不同的目标商品里有多少在 train 没出现过
    uniq = np.unique(tgt)
    uniq_cold = int((~train_items[uniq]).sum())
    stats = {
        "n_samples": int(len(tgt)),
        "n_cold_samples": int(cold.sum()),
        "cold_sample_ratio": float(cold.mean()),
        "n_distinct_targets": int(uniq.size),
        "n_distinct_cold_targets": uniq_cold,
        "cold_item_ratio": float(uniq_cold / max(uniq.size, 1)),
    }
    return cold, tgt, stats


def report(name: str, ranks: np.ndarray, cold: np.ndarray) -> dict:
    """把 ranks 按冷/热切开算 HR@1/HR@10。"""
    out = {}
    for tag, sel in (("all", np.ones_like(cold)), ("warm", ~cold), ("cold", cold)):
        r = ranks[sel]
        out[tag] = {
            "n": int(r.size),
            "HR@1": float((r <= 1).mean()) if r.size else float("nan"),
            "HR@10": float((r <= 10).mean()) if r.size else float("nan"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all", choices=["all", "IandS", "VG"])
    ap.add_argument("--models", default=",".join(ZERO_SHOT))
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    domains = ["IandS", "VG"] if args.domain == "all" else [args.domain]
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in wanted:
        if m not in ZERO_SHOT:
            raise SystemExit(f"未知模型 {m}；可选：{list(ZERO_SHOT)}")

    device = N.get_device()
    print(f"device={device}\n")

    for dom in domains:
        data = load_domain(dom, with_emb=True)
        cold, _, st = cold_flags(data)
        print(f"=== {dom} / test 冷热拆分 ===")
        print(f"  样本级：{st['n_cold_samples']:,} / {st['n_samples']:,} 冷 "
              f"= {st['cold_sample_ratio']*100:.2f}%")
        print(f"  商品级：{st['n_distinct_cold_targets']:,} / {st['n_distinct_targets']:,} 冷 "
              f"= {st['cold_item_ratio']*100:.2f}%"
              f"   ← 对齐 DATASET.md §8.3 的口径\n")

        rows = {}
        for name in wanted:
            cls, cfg = ZERO_SHOT[name]
            model = cls(n_items=data.n_items, n_users=data.n_users, maxlen=20, **(cfg or {}))
            model.fit(data, device=device, log=lambda *a: None)
            res = N.evaluate(model.score, data.test, data.n_items, model.pad_id,
                             device=device, batch_size=args.batch_size)
            rows[name] = report(name, res["ranks"], cold)

        hdr = f"{'模型':<12} | {'cold 占比':>9} | {'全部 HR@10':>10} | {'热 HR@10':>9} | {'冷 HR@10':>9} | {'热/冷 倍数':>10}"
        print(hdr)
        print("-" * len(hdr))
        for name in wanted:
            r = rows[name]
            ratio = (r["warm"]["HR@10"] / r["cold"]["HR@10"]) if r["cold"]["HR@10"] > 0 else float("inf")
            print(f"{name:<12} | {r['cold']['n']/r['all']['n']*100:8.2f}% |"
                  f" {r['all']['HR@10']:10.4f} | {r['warm']['HR@10']:9.4f} |"
                  f" {r['cold']['HR@10']:9.4f} | {ratio:10.2f}x")
        print()

        # 判据（先写死，避免事后找补）
        pop_cold = rows["pop"]["cold"]["HR@10"] if "pop" in rows else None
        sid_cold = rows["sid_prefix"]["cold"]["HR@10"] if "sid_prefix" in rows else None
        sid_warm = rows["sid_prefix"]["warm"]["HR@10"] if "sid_prefix" in rows else None
        if pop_cold is not None and sid_cold is not None:
            print("  判据：")
            if pop_cold <= 1e-6 and sid_cold > 10 * max(pop_cold, 1e-9):
                print(f"  ✅ H1 成立（内容泛化）：纯流行度在冷目标上 HR@10 = {pop_cold:.4f}（≈0），"
                      f"sid_prefix = {sid_cold:.4f}")
            elif pop_cold > 0.005:
                print(f"  ⚠️ H0 警报：纯流行度在冷目标上也拿到 {pop_cold:.4f} —— "
                      f"检查冷目标是否被误标、mask 是否漏了")
            else:
                print(f"  ⚠️ 不满足 H1 的强判据（pop 冷 {pop_cold:.4f}，sid_prefix 冷 {sid_cold:.4f}），需要看细节")
            if sid_warm is not None:
                print(f"     参考：sid_prefix 热目标 HR@10 = {sid_warm:.4f}")
        print()


if __name__ == "__main__":
    main()
