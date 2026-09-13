"""baseline 共用的 torch 训练 / 全库排序评估骨架。

所有神经网络 baseline（BPR-MF / GRU4Rec / SASRec / BERT4Rec / SID-GR）共用这里的
训练循环与评估循环，保证**只有模型定义不同，训练与评估口径完全一致**。

评估约定：
- `score_fn(hist_tensor) -> (b, n_items)` 输出全库打分；
- 用户历史物品在打分后被置 `-inf`（屏蔽集见 `common/data.py`）；
- 分块跑（默认 512 用户/块），避免 (U, n_items) 全矩阵爆显存：
  I&S 13232×25847 fp32 ≈ 1.3GB，VG 同理，分块后每块 ≈ 53MB。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from . import metrics as M
from .data import EvalSet


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 全库排序评估
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    evalset: EvalSet,
    n_items: int,
    pad_id: int,
    device: torch.device,
    batch_size: int = 512,
    mask_seen: bool = True,
    topk_for_coverage: int = 10,
) -> dict:
    """返回 {'metrics': {...}, 'ranks': ndarray, 'topk': ndarray}。

    `score_fn(hist, users) -> (b, n_items)`：序列模型忽略 `users`，ID 类模型（BPR-MF）用它。
    """
    hist_all = evalset.padded(pad_id)
    n = len(evalset)
    ranks = np.zeros(n, dtype=np.int64)
    topk = np.zeros((n, topk_for_coverage), dtype=np.int64)

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        hb = torch.from_numpy(hist_all[s:e]).to(device)
        ub = torch.from_numpy(evalset.users[s:e]).to(device)
        scores = score_fn(hb, ub).float().cpu().numpy()
        # 安全检查：NaN 会让 (scores > target) 恒为 False → rank 恒为 1 → HR@K 假性接近 1.0。
        # 本项目实测踩过这个坑（左填充 + 因果掩码 ⇒ softmax 全屏蔽 ⇒ NaN），所以在这里硬拦。
        if not np.isfinite(scores).all():
            n_bad = int((~np.isfinite(scores)).any(axis=1).sum())
            raise RuntimeError(
                f"模型输出含 NaN/Inf（{n_bad}/{e - s} 条样本）。"
                f"常见原因：attention 的 key 被全部屏蔽、除零、log(0)。")
        # 退化解检测：候选分数几乎无差异 ⇒ `scores > tgt_score` 全 False ⇒ rank 恒为 1
        # ⇒ HR/NDCG/MRR 假性全部 = 1.0。分数本身是有限值，上面的 NaN 检查拦不住，
        # 必须单独拦。实测踩到：VG/twotower_mm 梯度爆炸(train_loss=731)后 ReLU 全部死亡，
        # 输出退化为常数，报出 HR@10=1.0000 / coverage@10=0.0048（2026-09-13）。
        if scores.shape[1] > 1:
            row_std = float(scores.std(axis=1).mean())
            if row_std < 1e-6:
                raise RuntimeError(
                    f"检测到退化解：候选分数几乎无差异（行内 std 均值={row_std:.3g}），"
                    f"rank 会恒为 1，HR/NDCG/MRR 会假性报 1.0。常见原因：梯度爆炸后"
                    f"ReLU 全部死亡。请查训练稳定性（学习率 / LayerNorm / 梯度裁剪）。")
        if mask_seen:
            rows, cols = evalset.flat_mask()
            lo = np.searchsorted(rows, s)
            hi = np.searchsorted(rows, e)
            if hi > lo:
                scores[rows[lo:hi] - s, cols[lo:hi]] = -np.inf
        tgt = evalset.targets[s:e]
        ranks[s:e] = M.ranks_from_scores(scores, tgt)
        if topk_for_coverage:
            # 取前 k 大（屏蔽后 -inf 不会被选中）
            idx = np.argpartition(-scores, topk_for_coverage - 1, axis=1)[:, :topk_for_coverage]
            for i in range(e - s):
                row = idx[i]
                topk[s + i] = row[np.argsort(-scores[i, row])]

    out = M.metrics_from_ranks(ranks)
    out.update(M.coverage_metrics(topk, n_items, topk_for_coverage))
    return {"metrics": out, "ranks": ranks, "topk": topk}


def summarize_ranks(ranks: np.ndarray) -> dict:
    """额外的分布刻画：命中时的位次分位数（面试可讲"是命中 top1 还是勉强进 top10"）。"""
    hit = ranks[ranks <= 10]
    if len(hit) == 0:
        return {"hit@10_rank_p50": None, "hit@10_rank_mean": None}
    return {
        "hit@10_rank_p50": float(np.percentile(hit, 50)),
        "hit@10_rank_mean": float(hit.mean()),
        "top1_share_of_hits": float((ranks == 1).sum() / max(len(hit), 1)),
    }


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 5.0
    patience: int = 5          # 早停：valid 主指标连续 patience 轮不升
    monitor: str = "NDCG@10"
    log_every: int = 50        # 每多少个 step 打一行日志（0 = 不打）
    seed: int = 42
    num_workers: int = 0


class SeqDataset(torch.utils.data.Dataset):
    """(history, target) 定长样本；history 左填充。"""

    def __init__(self, hist: np.ndarray, tgt: np.ndarray, pad_id: int):
        self.hist = torch.from_numpy(hist.astype(np.int64))
        self.tgt = torch.from_numpy(tgt.astype(np.int64))
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.tgt)

    def __getitem__(self, i):
        return self.hist[i], self.tgt[i]


def train_model(
    model: nn.Module,
    train_ds: torch.utils.data.Dataset,
    valid_eval_fn: Callable[[nn.Module], dict],
    cfg: TrainConfig,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> dict:
    """通用训练循环。`valid_eval_fn(model)` 返回 metrics dict（至少含 cfg.monitor）。

    返回 {'best_metric': float, 'best_epoch': int, 'history': [...], 'stopped_early': bool}
    """
    model.to(device)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=False, pin_memory=(device.type == "cuda"),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history, best, best_epoch, best_state, bad = [], -np.inf, -1, None, 0
    stopped = False

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        tot, cnt = 0.0, 0
        for step, (h, t) in enumerate(loader, 1):
            h = h.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(h, t)
            loss.backward()
            if cfg.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            tot += float(loss.detach()) * len(t)
            cnt += len(t)
            if cfg.log_every and step % cfg.log_every == 0:
                log(f"    ep{ep} step{step}/{len(loader)} loss={tot/max(cnt,1):.4f}")
        train_loss = tot / max(cnt, 1)

        m = valid_eval_fn(model)["metrics"]
        cur = float(m.get(cfg.monitor, m.get("NDCG@10", 0.0)))
        history.append({"epoch": ep, "train_loss": train_loss,
                        "valid": {k: v for k, v in m.items() if k != "n_eval"}, "sec": round(time.time() - t0, 1)})
        log(f"  epoch {ep}: train_loss={train_loss:.4f} valid {cfg.monitor}={cur:.4f} "
            f"HR@10={m.get('HR@10', float('nan')):.4f} ({time.time()-t0:.0f}s)")

        if cur > best + 1e-6:
            best, best_epoch, bad = cur, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"  早停：{cfg.patience} 轮未提升（best epoch {best_epoch}, {cfg.monitor}={best:.4f}）")
                stopped = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_metric": float(best), "best_epoch": best_epoch,
            "monitor": cfg.monitor, "history": history, "stopped_early": stopped}
