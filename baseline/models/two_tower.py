"""双塔召回 baseline（Two-Tower / DSSM 范式）。

与序列模型（GRU4Rec / SASRec）的根本区别：**不对历史做时序建模**，
只把历史物品聚合成一个向量（平均池化），然后用户向量 · 物品向量 = 打分。
这是工业界（YouTube DNN / 百度 / 阿里）的主流召回形态，特点是
**用户向量与物品向量可分别离线预计算**，在线只需一次 ANN 检索。

本文件提供两个变体，差别只在「物品塔 / 用户塔的输入特征」：

| 变体 | 塔的输入 | 参数量 | 能否处理新物品 |
|---|---|---|---|
| `twotower_id` | 随机初始化的 ID embedding（纯协同） | ~1.67M | ❌ 新物品 = 随机噪声 |
| `twotower_mm` | 冻结的多模态融合向量 + 可学习投影 | ~82K | ✅ 有内容即可算出向量 |

两个变体**塔内特征空间自洽**（用户塔与物品塔用同一套特征），否则内积无意义。

与既有 baseline 的对照关系（这是加它的主要动机）：
- `twotower_id` vs `sasrec`       → 「历史池化」 vs 「时序建模」值多少；
- `twotower_mm` vs `twotower_id`  → 「多模态内容」 vs 「纯 ID 协同」值多少；
- `twotower_mm` vs `content_ann`  → 「学出来的投影」 vs 「原始向量直接检索」值多少。
  两者输入完全一致（同为归一化后的融合向量、同为 mean 聚合），差异只剩投影层。

有意的设计约束（与本项目其他 baseline 保持一致）：
1. **训练目标 = 全库 softmax**，不是工业界常见的 sampled softmax / BPR。
   理由同 `models/seq.py`：统一目标后，差异只剩检索范式，不掺损失函数的混杂。
2. **打分 = 内积**（不做余弦归一化、不加温度系数），与 `seq.py` 的 `h_t·Eᵀ` 同构。
3. **融合向量冻结**（不微调），与 `content_ann` 口径一致；只训练投影层与 MLP。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.data import DomainData
from .seq import TorchSeqBaseline


class TwoTowerNet(nn.Module):
    """用户塔 = 历史特征均值 → MLP；物品塔 = 物品特征 → MLP；打分 = 内积。"""

    def __init__(self, n_items, pad_id, maxlen, hidden=64, dropout=0.2,
                 feat="id", emb=None, **_):
        super().__init__()
        self.n_items, self.pad_id, self.maxlen = n_items, pad_id, maxlen
        self.hidden, self.feat = hidden, feat

        if feat == "id":
            self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=pad_id)
            nn.init.normal_(self.item_emb.weight, std=0.02)
            nn.init.zeros_(self.item_emb.weight[pad_id])
            in_dim = hidden
        elif feat == "mm":
            if emb is None:
                raise ValueError("feat='mm' 需要传入融合向量 emb")
            e = torch.as_tensor(emb, dtype=torch.float32)
            e = F.normalize(e, dim=-1)          # 与 content_ann 口径一致：余弦空间
            self.register_buffer("feat_mat", e)  # 冻结，不参与梯度
            # LayerNorm 不能省：1024→64 的裸线性层在 lr=1e-3 下会梯度爆炸
            # （实测 VG 第 11 轮 train_loss=731），爆炸后 ReLU 死亡、输出退化为常数。
            self.proj = nn.Sequential(nn.Linear(e.shape[1], hidden), nn.LayerNorm(hidden))
            in_dim = hidden
        else:
            raise ValueError(f"未知 feat={feat!r}，可选 'id' / 'mm'")

        # 两侧塔都做 LayerNorm 收尾：把向量限制在单位球面附近，
        # 内积因此天然落在有界区间，不会再出现分数爆炸 / 全库同分。
        self.user_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.item_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        # 两侧都归一化后内积 ∈ [-1,1]，直接喂 softmax 会因 logits 过小而学不动，
        # 故乘一个可学习温度（CLIP 式），由模型自己决定打分锐度。
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    # ---------------- 两侧塔 ----------------

    def _feat(self, idx: torch.Tensor) -> torch.Tensor:
        """物品 id → 特征向量。idx 可为任意形状。"""
        if self.feat == "id":
            return self.item_emb(idx)
        # pad_id = n_items 会越界，先 clamp；其贡献随后被 mask 乘 0，不影响结果
        return self.proj(self.feat_mat[idx.clamp(max=self.n_items - 1)])

    def item_vec(self) -> torch.Tensor:
        """物品塔：全库物品向量 (n_items, hidden)。"""
        return self.item_mlp(self._feat(torch.arange(self.n_items, device=self._device())))

    def user_vec(self, hist: torch.Tensor) -> torch.Tensor:
        """用户塔：历史平均池化 → MLP。"""
        H = self._feat(hist)                                   # (b, L, hidden)
        mask = hist.ne(self.pad_id).unsqueeze(-1).float()      # pad 位不计入均值
        pooled = (H * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.user_mlp(pooled)

    def _device(self) -> torch.device:
        return self.item_emb.weight.device if self.feat == "id" else self.feat_mat.device

    # ---------------- 打分 ----------------

    def logits(self, hist: torch.Tensor) -> torch.Tensor:
        return self.logit_scale * (self.user_vec(hist) @ self.item_vec().T)

    def loss(self, hist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.logits(hist), target)


class TwoTower(TorchSeqBaseline):
    """共用 fit/score，只有 `_build` 需要额外把融合向量塞进网络。"""

    NET_KEYS = ("hidden", "dropout", "feat")

    def fit(self, data: DomainData, device=None, log=print) -> dict:
        if self.cfg.get("feat") == "mm" and getattr(data, "emb", None) is None:
            raise FileNotFoundError(
                "twotower_mm 需要多模态融合向量（data.emb），"
                "请以 with_emb=True 加载数据（run.py 的 need_emb 已处理）")
        self._emb = getattr(data, "emb", None)
        return super().fit(data, device=device, log=log)

    def _build(self):
        return self.net_cls(self.n_items, self.pad_id, self.maxlen,
                            emb=getattr(self, "_emb", None), **self.net_kw)


class TwoTowerID(TwoTower):
    """纯 ID 双塔：与 gru4rec / sasrec 同为纯协同信号，归入 A 组。"""
    name = "twotower_id"
    net_cls = TwoTowerNet


class TwoTowerMM(TwoTower):
    """多模态双塔：物品塔吃融合向量，归入 B 组（与 content_ann 同组对照）。"""
    name = "twotower_mm"
    net_cls = TwoTowerNet
