"""baseline 统一指标口径 —— 唯一实现，其他脚本一律 import 这里，不要各写一份。

与项目既有口径对齐（`utility.py:calculate_hit` / README §3.0）：

- **HR@K** = `1/U · Σ_u 1[ rank_u ≤ K ]`
- **NDCG@K** = `1/U · Σ_u 1[ rank_u ≤ K ] / log2(rank_u + 1)`（单正样本 ⇒ IDCG = 1）
- **MRR** = `1/U · Σ_u 1/rank_u`（未命中记 0）
- `rank_u` = 目标物品在**全库排序**里的位次（1-based）；同分时取"最优位次"
  （`rank = #{score > score_target} + 1`），与"随机打破平局"的期望一致，且可复现。
- 评估前会把用户**已交互过的物品**从候选中屏蔽（置 -inf），口径见 `common/data.py`。

另有覆盖率类指标（V0 pipeline 里没有，本项目新增，用于观察流行度偏置）：
- `coverage@10` = top-10 列表里出现过的不同物品数 / n_items
- `gini@10` = top-10 推荐频次的 Gini 系数（越大越集中，越偏头部）
"""

from __future__ import annotations

import json
import os

import numpy as np

KS = (1, 5, 10)


def ranks_from_scores(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """scores: (U, n_items)；targets: (U,)。返回 1-based rank（含屏蔽后的 -inf 不受影响）。"""
    tgt_score = scores[np.arange(scores.shape[0]), targets]
    rank = (scores > tgt_score[:, None]).sum(axis=1) + 1
    return rank.astype(np.int64)


def metrics_from_ranks(ranks: np.ndarray, ks=KS) -> dict:
    out: dict[str, float] = {}
    u = len(ranks)
    for k in ks:
        hit = ranks <= k
        out[f"HR@{k}"] = float(hit.mean())
        ndcg = np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)
        out[f"NDCG@{k}"] = float(ndcg.mean())
    out["MRR"] = float(np.where(ranks > 0, 1.0 / ranks, 0.0).mean())
    out["n_eval"] = int(u)
    return out


def coverage_metrics(topk: np.ndarray, n_items: int, k: int = 10) -> dict:
    """topk: (U, k) 推荐物品 id。"""
    flat = topk.reshape(-1)
    uniq, cnt = np.unique(flat, return_counts=True)
    p = cnt / cnt.sum()
    p = np.sort(p)
    n = len(p)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(p) / (n * 1.0))  # 归一化到 [-1,1] 后取正部
    return {
        f"coverage@{k}": float(len(uniq) / n_items),
        f"gini@{k}": float(abs(gini)),
    }


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
