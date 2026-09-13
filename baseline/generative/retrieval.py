"""零训练的检索式对照：内容向量近邻（ANN）与 SID 前缀检索。

这两条是**回答"生成式召回到底需不需要"的关键对照**：
- `content_ann`：直接用本项目的**多模态融合向量**做近邻检索，不需要任何训练。
  如果生成式召回赢不了它，说明"先量化成 SID 再生成"这一步没赚到东西。
- `sid_prefix`：用已定版的 `sid_raw` 做**前缀匹配检索**（TIGER 提出但未实现的
  prefix matching 路线）。它和"自回归生成 SID"用同一份 SID，区别只在检索侧是
  "按前缀召回"还是"逐 token 生成"——用来把"SID 本身的信息量"与"生成器的能力"分开。

两者都不训练、不占显存，成本仅是若干次矩阵/集合运算。
"""

from __future__ import annotations

import numpy as np
import torch

from ..common.base import Baseline
from ..common.data import DomainData


class ContentANN(Baseline):
    """内容近邻：用户向量 = 历史物品融合向量的聚合（mean/last/max），余弦打分。

    - `agg="mean"`：用户兴趣的平均（最常用的双塔式内容召回）；
    - `agg="last"`：只用最近一个物品（"买了跟刚才看的最像的"），是极强的短序列 baseline；
    - `agg="max"`：逐维取最大（对多兴趣更宽容）。

    这一条**不使用任何协同信号**，纯粹"内容像不像"，所以它同时充当
    "多模态融合向量本身值多少"的探针。
    """

    name = "content_ann"

    def __init__(self, n_items, n_users, maxlen=20, agg="mean", **cfg):
        super().__init__(n_items, n_users, maxlen, agg=agg, **cfg)
        self.agg = agg

    def fit(self, data: DomainData, device=None, log=print) -> dict:
        emb = data.emb
        if emb is None:
            raise FileNotFoundError(
                "找不到融合向量 emb_fused_gate_e60.npy，content_ann 无法运行")
        assert emb.shape[0] == self.n_items, f"融合向量行数 {emb.shape[0]} != n_items {self.n_items}"
        e = torch.from_numpy(emb.astype(np.float32))
        e = torch.nn.functional.normalize(e, dim=-1)     # 余弦检索
        self.E = e
        self.extra = {"agg": self.agg, "dim": int(e.shape[1]),
                      "n_rows": int(e.shape[0]), "trainable": False}
        log(f"  [content_ann] 融合向量 {tuple(e.shape)}，聚合方式 = {self.agg}（零训练）")
        return self.extra

    def score(self, hist: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        E = self.E.to(hist.device)
        mask = hist.ne(self.pad_id)                       # (b, L)
        H = E[hist.clamp(max=self.n_items - 1)]           # (b, L, d)
        m = mask.unsqueeze(-1).float()
        if self.agg == "mean":
            u = (H * m).sum(1) / m.sum(1).clamp(min=1.0)
        elif self.agg == "max":
            u = (H * m - (1 - m) * 1e9).max(1).values
        elif self.agg == "last":
            # 历史左填充 ⇒ 最右侧的真实物品就是"最近一个"
            last_idx = mask.long().cumsum(1).argmax(1)
            u = H[torch.arange(H.shape[0], device=H.device), last_idx]
        else:
            raise ValueError(self.agg)
        u = torch.nn.functional.normalize(u, dim=-1)
        return u @ E.T


class SidPrefix(Baseline):
    """SID 前缀检索：候选物品得分 = `Σ_{j∈历史} Σ_l w_l · 1[前缀 l 层与物品 j 相同]`。

    设计要点（都写死在配置里，避免事后调参）：
    - 层级权重 `w = [1.0, 1.5, 2.0]`：越靠后的层匹配越具体，权重越高（与 RQ-VAE 的
      "粗到细"层次一致）；
    - **对历史求和，而不是取并集**：这一点很关键。第一版实现用 `np.isin`（布尔 OR），
      结果"与历史任一物品共享某层前缀"的物品全部同分，排序完全被流行度兜底项主导 ——
      实测 coverage@10 只有 0.0007（全部用户拿到几乎同一份 top-10）。
      改成按层**计数**后，与更多历史物品共享前缀的候选得分更高，排序才真正因人而异。
    - 用稀疏矩阵乘法算计数：`H (b×前缀空间) @ OneHot (前缀空间×物品)`，每批只需 b×L 次累加；
    - 流行度兜底项缩到 `1e-6`，**只用来打破平局**，不参与主排序。
    """

    name = "sid_prefix"

    def __init__(self, n_items, n_users, maxlen=20, weights=(1.0, 1.5, 2.0), **cfg):
        super().__init__(n_items, n_users, maxlen, weights=tuple(weights), **cfg)
        self.weights = tuple(weights)

    def fit(self, data: DomainData, device=None, log=print) -> dict:
        if data.sid is None:
            raise FileNotFoundError("找不到 sid_raw.npy")
        sid = np.asarray(data.sid, dtype=np.int64)
        assert sid.shape[0] == self.n_items
        self.sid = sid
        self.L = sid.shape[1]
        # 前缀编码：level l 的编码 = 前 l 层线性拼成一个整数（256 进制）
        pref = [np.zeros(self.n_items, dtype=np.int64)]
        for l in range(self.L):
            pref.append(pref[-1] * 256 + sid[:, l])
        self.pref = pref[1:]                     # pref[l-1] = 前 l 层编码

        # 倒排表：前缀码 → 该前缀下的物品列表。
        # 注意**不要**按 256^l 建全空间（level 3 是 1677 万，每批重建一次稀疏单位阵会白烧十几秒）；
        # 改为按"实际出现过的前缀"重编码（≤ n_items 个），再用 searchsorted 定位。
        self.inv_sorted, self.order_items, self.code_item = [], [], []
        for l in range(self.L):
            _, inv = np.unique(self.pref[l], return_inverse=True)   # 物品 → 前缀序号
            order = np.argsort(inv, kind="stable")
            self.inv_sorted.append(inv[order])
            self.order_items.append(order)
            self.code_item.append(inv)
        cnt = np.zeros(self.n_items, dtype=np.float64)
        for seq in data.train_seq.values():
            for it in seq:
                cnt[it] += 1.0
        self.pop = cnt
        self.extra = {"weights": list(self.weights), "n_levels": int(self.L),
                      "n_distinct_prefixes": [int(np.unique(p).size) for p in self.pref],
                      "trainable": False}
        log(f"  [sid_prefix] SID {tuple(sid.shape)}，层级权重 {self.weights}，"
            f"实际出现过的前缀数 {self.extra['n_distinct_prefixes']}，零训练")
        return self.extra

    def score(self, hist: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        h = hist.detach().cpu().numpy()
        b, L = h.shape
        hh = np.where(h == self.pad_id, 0, h)
        valid = h != self.pad_id                       # (b, L)
        out = np.zeros((b, self.n_items), dtype=np.float32)
        for l in range(self.L):
            code_item = self.code_item[l]                                  # 物品 → 前缀序号
            hist_codes = code_item[hh]                                     # (b, L)
            for i in range(b):
                cols = hist_codes[i][valid[i]]
                if cols.size == 0:
                    continue
                uniq, cts = np.unique(cols, return_counts=True)
                sorted_inv = self.inv_sorted[l]
                order = self.order_items[l]
                for c, k in zip(uniq, cts):
                    lo = np.searchsorted(sorted_inv, c, "left")
                    hi = np.searchsorted(sorted_inv, c, "right")
                    if hi > lo:
                        out[i, order[lo:hi]] += self.weights[l] * k
        out += (self.pop / max(self.pop.max(), 1.0)).astype(np.float32) * 1e-6
        return torch.from_numpy(out)
