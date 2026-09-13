"""启发式经典 baseline：PopRec（流行度）与 ItemKNN（物品协同过滤）。

两者都**无需训练**、无需 GPU，成本近乎为零，但在论文基线表里是必备的"地板"——
如果生成式召回连它们都赢不了，问题一定不在模型。
"""

from __future__ import annotations

import numpy as np
import torch

from ..common.base import Baseline
from ..common.data import DomainData


def _build_ui_matrix(data: DomainData):
    """用重建的完整训练序列构造二值 user-item 矩阵（scipy 稀疏）。"""
    import scipy.sparse as sp

    u_idx, i_idx = [], []
    for u, seq in data.train_seq.items():
        if not seq:
            continue
        for it in set(seq):  # 同 split 内已去重，这里再保一次险
            u_idx.append(u)
            i_idx.append(it)
    shape = (data.n_users + 1, data.n_items)
    m = sp.csr_matrix((np.ones(len(u_idx), dtype=np.float32), (u_idx, i_idx)), shape=shape)
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


class PopRec(Baseline):
    """按训练集交互次数排序。零超参、零训练。"""

    name = "pop"

    def fit(self, data: DomainData, device=None, log=print) -> dict:
        cnt = np.zeros(self.n_items, dtype=np.float64)
        for seq in data.train_seq.values():
            for it in seq:
                cnt[it] += 1.0
        self.item_score = torch.from_numpy(cnt).float()
        self.extra = {
            "n_items_with_zero_interaction": int((cnt == 0).sum()),
            "top20_items": np.argsort(-cnt)[:20].tolist(),
        }
        log(f"  [pop] 已统计 {int(cnt.sum())} 次交互，零交互物品 {self.extra['n_items_with_zero_interaction']} 个")
        return self.extra

    def score(self, hist: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        return self.item_score.to(hist.device).unsqueeze(0).expand(hist.shape[0], -1)


class ItemKNN(Baseline):
    """物品协同过滤：余弦相似度 + 每行截断 top-k 近邻，分数 = 历史物品相似度之和。

    与 RecBole 的 `ItemKNN` 同口径（`similarity=cosine`、截断近邻数以控制泛化：
    `topk=100` 是 RecBole 在 Amazon 类数据上的默认档），差别只是这里用重建的完整
    训练序列构图（而不是滑窗样本），因为 CF 关心的是"共同被买"这条静态关系。
    """

    name = "itemknn"

    def __init__(self, n_items, n_users, maxlen=20, topk=100, **cfg):
        super().__init__(n_items, n_users, maxlen, topk=topk, **cfg)
        self.topk = topk

    def fit(self, data: DomainData, device=None, log=print) -> dict:
        import scipy.sparse as sp

        X = _build_ui_matrix(data)                     # (U, I) 二值
        col = np.asarray(X.sum(axis=0)).ravel()        # 物品被交互次数
        inv = np.zeros_like(col, dtype=np.float32)
        nz = col > 0
        inv[nz] = 1.0 / np.sqrt(col[nz])
        Xn = X.multiply(inv[None, :]).tocsr()          # 列归一化
        S = (Xn.T @ Xn).tocsr()                        # (I, I) 余弦相似度
        S.setdiag(0.0)
        S.eliminate_zeros()

        # 每行截断 top-k
        indptr, indices, values = [0], [], []
        for i in range(self.n_items):
            lo, hi = S.indptr[i], S.indptr[i + 1]
            if hi == lo:
                indptr.append(len(indices))
                continue
            v = S.data[lo:hi]
            c = S.indices[lo:hi]
            if len(v) > self.topk:
                sel = np.argpartition(-v, self.topk - 1)[: self.topk]
                v, c = v[sel], c[sel]
            order = np.argsort(-v)
            indices.extend(c[order].tolist())
            values.extend(v[order].tolist())
            indptr.append(len(indices))
        self.S = sp.csr_matrix(
            (np.asarray(values, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr)),
            shape=(self.n_items, self.n_items),
        )
        nnz = int(self.S.nnz)
        self.extra = {
            "topk": self.topk,
            "sim_nnz": nnz,
            "avg_neighbors": round(nnz / max(self.n_items, 1), 2),
            "cold_items_no_neighbor": int(np.sum(np.diff(self.S.indptr) == 0)),
        }
        log(f"  [itemknn] 相似度矩阵 nnz={nnz}（平均 {self.extra['avg_neighbors']} 邻居/物品，"
            f"无邻居物品 {self.extra['cold_items_no_neighbor']} 个）")
        return self.extra

    def score(self, hist: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        import scipy.sparse as sp

        h = hist.detach().cpu().numpy()
        b, L = h.shape
        mask = h != self.pad_id
        rows = np.repeat(np.arange(b), L)[mask.ravel()]
        cols = h.ravel()[mask.ravel()]
        H = sp.csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)),
                          shape=(b, self.n_items))
        H.sum_duplicates()
        H.data[:] = 1.0  # 同物品在历史里只计一次（与 RecBole 的 SUM 聚合一致，去重避免长序列放大）
        scores = (H @ self.S).toarray().astype(np.float32)
        return torch.from_numpy(scores)
