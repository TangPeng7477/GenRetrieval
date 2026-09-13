"""baseline 统一数据加载：把 `data/Amazon23/<domain>/*.inter` 读成序列样本。

口径（与 README §3.0 / 后续 SFT 阶段对齐，改动必须同步改这里）：

- **训练样本** = `<domain>.train.inter` 里已有的滑窗样本 `(history, target)`，原样使用。
  （滑窗由 `scripts/data/prepare_amazon23.py` 生成：第 i 个物品为 target，前 i 个最多 20 个为 history）
- **valid 评估** = 文件给定的 history（train 段，最多 20）→ 预测 valid 段最后一条。
- **test 评估**  = 文件给定的 history（**train 段 + valid 段全部物品**，最多 20）→ 预测 test 物品。
  这一点是本项目自有口径（不是标准 leave-one-out）：实测 3161/13232（I&S）条 test 的 history
  比 "train 段 + 仅 valid 目标" 多出若干条目，正是那些 **valid 段含 ≥2 个物品** 的用户 —— 见
  `probe_seq*.py` 的验证结论，复现见 `baseline/README.md` §口径。
- **屏蔽集**（seen mask）= 该条 history ∪ 从 train.inter 重建的该用户**完整**训练序列 ∪ valid 目标。
  目的：禁止"把用户已交互过的物品当成新推荐"。重建方法见 `reconstruct_train_sequences`。
  这条补屏蔽不是可选项：实测它额外屏蔽掉 I&S **3,722** / VG **10,186** 个"被 20 窗口挤出 history
  但仍属该用户历史"的物品，不补就会给所有 baseline 放水。
  **残余不可屏蔽项**：只可能出现在 `|valid 段| ≥ 20` 的用户上（valid 段物品排在拼接序列末尾，
  只要 `|valid 段| ≤ 20` 就全在窗口内）。用"test history 与 train 段零重叠"检测，实测这类样本
  I&S **33 / 13,232（0.249%）**、VG **12 / 11,276（0.106%）**。
  验证脚本：`baseline/scripts/probe_sequence_reconstruction.py`（三条前提逐条实测）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

DOMAIN_CATEGORY = {
    "IandS": "Industrial_and_Scientific",
    "VG": "Video_Games",
}

DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def domain_paths(domain: str, root: str = DEFAULT_ROOT) -> dict:
    """返回某个域的关键产物路径（缺失的键值为 None）。"""
    d = os.path.join(root, "data", "Amazon23", domain)
    sid_dir = os.path.join(root, "results", "sid_e5000", domain, "gate__init8192")
    emb = os.path.join(d, "emb", "long", "emb_fused_gate_e60.npy")
    if not os.path.exists(emb):  # 兼容早期融合产物落点
        emb = os.path.join(d, "emb", "emb_fused_gate.npy")
    return {
        "domain_dir": d,
        "train": os.path.join(d, f"{domain}.train.inter"),
        "valid": os.path.join(d, f"{domain}.valid.inter"),
        "test": os.path.join(d, f"{domain}.test.inter"),
        "stats": os.path.join(d, f"{domain}.stats.json"),
        "sid_raw": os.path.join(sid_dir, "sid_raw.npy"),
        "emb_fused": emb if os.path.exists(emb) else None,
    }


# ---------------------------------------------------------------------------
# 读 .inter
# ---------------------------------------------------------------------------


def load_inter(path: str) -> list[tuple[int, list[int], int]]:
    """读 RecBole 风格三列 .inter → [(user_id, history, target), ...]，保持文件顺序。"""
    rows: list[tuple[int, list[int], int]] = []
    with open(path, encoding="utf-8") as f:
        next(f)  # 表头 user_id:token / item_id_list:token_seq / item_id:token
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            u, h, t = parts[0], parts[1], parts[2]
            hist = [int(x) for x in h.split()] if h.strip() else []
            rows.append((int(u), hist, int(t)))
    return rows


def reconstruct_train_sequences(
    rows: list[tuple[int, list[int], int]],
    valid_rows: list[tuple[int, list[int], int]] | None = None,
    test_rows: list[tuple[int, list[int], int]] | None = None,
) -> dict[int, list[int]]:
    """从 train.inter 的滑窗样本重建每个用户的**完整**训练序列（未被 20 截断）。

    依据（已实测验证，复现脚本 `baseline/scripts/probe_sequence_reconstruction.py`）：
      滑窗对第 i 个物品（i 从 1 开始）取 `history = tr[max(0, i-20):i]`、`target = tr[i]`，
      所以 **第一个样本的 history 恒为 `[tr[0]]`**（实测两域首样本 history 长度 100% 为 1），
      后续每个样本的 target 依次是 `tr[1], tr[2], ...`。
      于是 `tr = [首样本.history[0]] + [各样本.target]`（按文件顺序）。
    校验：用它预测 valid 输入 `train_seq[-20:]`，`u ∈ train.inter` 的用户 **100% 命中**
    （I&S 14,853/14,853，VG 13,403/13,403）。

    ⚠️ **补齐一类"隐形用户"**（2026-09-13 实测发现）：训练段只有 **1 个**物品的用户
    产生不出任何滑窗样本（target 之前必须有 ≥1 个物品才算一条样本），因此**不在 train.inter 里**，
    但他们并未被丢弃、仍然要进 valid/test 评估——实测 I&S valid 1,221 人 / test 1,663 人、
    VG valid 828 人 / test 1,041 人。这类用户的 valid `history` **恰好就是那 1 个训练物品**
    （因为 train_seq 长度 1，`[-20:]` 取全部），所以可以用它精确补齐。
    不补齐的后果：这些用户的训练历史会算成空集，屏蔽集不完整（口径偏松）。
    """
    seqs: dict[int, list[int]] = {}
    for u, h, t in rows:
        if u not in seqs:
            seqs[u] = list(h) if h else []      # 取首样本全部 hist（v1 timestamp L=1 / v2 LOO L=2）
        seqs[u].append(t)

    n_patched = 0
    # ① 先补 valid 段：训练段长度 L ⇒ valid history 的前 L 项就是训练段全部物品
    # （v1 timestamp 路径：L=1，只对 K=2 的用户触发，K≥3 已被 prepare 过滤；
    #  v2 LOO 路径：L=2，K≥3 已强制，无 K=2 用户，本路径不触发）
    for src in (valid_rows, test_rows):
        if not src:
            continue
        for u, h, _ in src:
            if u not in seqs and h:
                seqs[u] = list(h[:1]) if h[:1] else []   # v1 timestamp L=1 路径专用
                n_patched += 1
    reconstruct_train_sequences.last_patched = n_patched  # 供日志/文档引用
    return seqs


# ---------------------------------------------------------------------------
# 数据集容器
# ---------------------------------------------------------------------------


@dataclass
class EvalSet:
    """一个 split 的评估集合：输入 history + 目标 + 需要屏蔽的物品。"""

    name: str
    users: np.ndarray           # (U,) int64
    hist: list[list[int]]       # 长度可变，≤ maxlen
    targets: np.ndarray         # (U,) int64
    masks: list[np.ndarray]     # 每条样本需要屏蔽的物品（含 history；不含 target）
    maxlen: int = 20
    _flat: tuple | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.targets)

    def flat_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """把 mask 列表摊平成 (rows, cols) 两个数组，用于批量置 -inf。

        逐用户 for 循环设 `scores[i, mask] = -inf` 在 1.3 万条测试样本上实测要 8 秒，
        且每个 epoch 的 valid 评估都要付一遍；摊平后一次花 0.1 秒（缓存）。
        """
        if self._flat is None:
            rows = [np.full(len(m), i, dtype=np.int64) for i, m in enumerate(self.masks) if len(m)]
            cols = [m for m in self.masks if len(m)]
            self._flat = (np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64),
                          np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64))
        return self._flat

    def padded(self, pad_id: int) -> np.ndarray:
        """左填充成 (U, maxlen) 定长矩阵（最新物品在最右）。"""
        out = np.full((len(self.hist), self.maxlen), pad_id, dtype=np.int64)
        for i, h in enumerate(self.hist):
            hh = h[-self.maxlen:]
            if hh:
                out[i, self.maxlen - len(hh):] = hh
        return out


@dataclass
class DomainData:
    domain: str
    n_items: int
    n_users: int
    paths: dict
    train_rows: list[tuple[int, list[int], int]]
    train_seq: dict[int, list[int]]
    valid: EvalSet
    test: EvalSet
    sid: np.ndarray | None = None
    emb: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def _build_eval_set(name, rows, train_seq, n_items, mask_extra_by_user=None) -> EvalSet:
    users, hist, targets, masks = [], [], [], []
    for u, h, t in rows:
        seen = set(h)
        seen.update(train_seq.get(u, ()))  # 完整训练序列（防 20 截断漏屏蔽）
        if mask_extra_by_user is not None:
            extra = mask_extra_by_user.get(u)
            if extra is not None:
                if isinstance(extra, int):
                    seen.add(extra)
                else:
                    seen.update(extra)
        seen.discard(t)  # 目标本身永不被屏蔽
        users.append(u)
        hist.append(h)
        targets.append(t)
        masks.append(np.fromiter(sorted(seen), dtype=np.int64, count=len(seen)))
    return EvalSet(
        name=name,
        users=np.asarray(users, dtype=np.int64),
        hist=hist,
        targets=np.asarray(targets, dtype=np.int64),
        masks=masks,
    )


def load_domain(domain: str, root: str = DEFAULT_ROOT, with_emb: bool = False) -> DomainData:
    paths = domain_paths(domain, root)
    train_rows = load_inter(paths["train"])
    valid_rows = load_inter(paths["valid"])
    test_rows = load_inter(paths["test"])
    train_seq = reconstruct_train_sequences(train_rows, valid_rows, test_rows)
    n_patched = getattr(reconstruct_train_sequences, "last_patched", 0)

    # 物品 id 是 0-based 连续整数，所以 n_items = max_id + 1（与 stats.json 的 n_items 对得上）
    max_item, max_user = -1, -1
    for rows in (train_rows, valid_rows, test_rows):
        for u, h, t in rows:
            if h and max(h) > max_item:
                max_item = max(h)
            if t > max_item:
                max_item = t
            if u > max_user:
                max_user = u
    for u in train_seq:
        if u > max_user:
            max_user = u
    n_items, n_users = max_item + 1, max_user + 1

    valid = _build_eval_set("valid", valid_rows, train_seq, n_items)
    # test 的 history 已含 valid 段全部物品，这里再显式补上 valid 目标（防御性）
    valid_target = {u: t for u, _, t in valid_rows}
    test = _build_eval_set("test", test_rows, train_seq, n_items, mask_extra_by_user=valid_target)

    sid = np.load(paths["sid_raw"]) if paths["sid_raw"] and os.path.exists(paths["sid_raw"]) else None
    emb = None
    if with_emb and paths["emb_fused"]:
        emb = np.load(paths["emb_fused"]).astype(np.float32)

    meta = {
        "n_train_samples": len(train_rows),
        "n_valid": len(valid_rows),
        "n_test": len(test_rows),
        "n_train_users_reconstructed": len(train_seq) - n_patched,
        "n_users_patched_from_valid": n_patched,
    }
    return DomainData(
        domain=domain,
        n_items=n_items,
        n_users=n_users,
        paths=paths,
        train_rows=train_rows,
        train_seq=train_seq,
        valid=valid,
        test=test,
        sid=sid,
        emb=emb,
        meta=meta,
    )


def train_arrays(rows: list[tuple[int, list[int], int]], maxlen: int = 20, pad_id: int = 0):
    """训练用 (user, history, target) —— history 左填充定长，pad 位置填 `pad_id`。

    注意 `pad_id` 必须取 `n_items`（即真实物品 id 之外的额外槽位），**不能用 0**——
    0 是真实物品 id，用 0 填充会让模型把"物品 0"当成 padding。
    """
    users = np.zeros((len(rows),), dtype=np.int64)
    hist = np.full((len(rows), maxlen), pad_id, dtype=np.int64)
    tgt = np.zeros((len(rows),), dtype=np.int64)
    for i, (u, h, t) in enumerate(rows):
        hh = h[-maxlen:]
        if hh:
            hist[i, maxlen - len(hh):] = hh
        tgt[i] = t
        users[i] = u
    return users, hist, tgt
