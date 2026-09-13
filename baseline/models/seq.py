"""经典序列推荐 baseline：GRU4Rec / SASRec / BERT4Rec。

三者共用同一套训练循环与全库排序评估（`common/nn.py`），**只有网络结构不同**，
所以它们之间的差异可以干净地归因到"用什么结构建模序列"。

与原论文的差异（必须在文档里讲清楚，否则数字对不上）：
1. **训练目标统一为全库 softmax（full softmax）**，不是原版 SASRec 的 sampled softmax、
   也不是 GRU4Rec 的 BPR/pairwise。原因：生成式召回（TIGER/MiniOneRec）本来就是全词表 softmax，
   统一目标后，baseline 与生成式方法的差异只剩"检索范式"，不掺"损失函数"的混杂。
2. **序列长度 = 20**（本项目数据的历史截断口径），不是原论文的 50/100。
3. **BERT4Rec 的 next-item 化改编**：原版是双向 MLM（预测随机被 mask 的位置）。
   本项目在序列**尾部追加一个 [MASK] 作为查询位**来预测下一个物品，同时保留对随机位置
   的 mask 任务作为辅助监督 —— 这样它的推理接口与 SASRec/GRU4Rec 完全一致（都是"给历史、出下一个"），
   这是把 BERT4Rec 放进 next-item 协议的标准做法。
4. 输出层与 item embedding **共享权重**（原版 SASRec 不共享）。共享后参数量更小、
   也更贴近生成式模型"生成 token = 查同一个 embedding"的形态。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import nn as N
from ..common.base import Baseline
from ..common.data import DomainData, train_arrays


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------


class SASRecNet(nn.Module):
    """自注意力（因果掩码）+ 下一位置预测。Kang & McAuley, ICDM 2018, arXiv:1808.09781。"""

    def __init__(self, n_items, pad_id, maxlen, hidden=64, n_layers=2, n_heads=2, dropout=0.2, **_):
        super().__init__()
        self.n_items, self.pad_id, self.maxlen = n_items, pad_id, maxlen
        self.hidden, self.n_heads = hidden, n_heads
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.zeros_(self.item_emb.weight[pad_id])
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.emb_ln = nn.LayerNorm(hidden)
        self.emb_drop = nn.Dropout(dropout)
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(hidden * 4, hidden)) for _ in range(n_layers)])
        self.lns1 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.lns2 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.last_ln = nn.LayerNorm(hidden)
        self.register_buffer(
            "causal", torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1),
            persistent=False)

    def encode(self, hist: torch.Tensor) -> torch.Tensor:
        b, L = hist.shape
        pad = hist.eq(self.pad_id)
        pos = torch.arange(L, device=hist.device).unsqueeze(0).expand(b, L)
        x = self.item_emb(hist) * (self.hidden ** 0.5) + self.pos_emb(pos)
        x = self.emb_drop(self.emb_ln(x))
        # 因果掩码 ∪（key 是 pad 且不是自己）。
        # ⚠️ 必须保留"自注意力"这条通路：左填充下位置 0 的前缀全是 pad，
        # 若把该 query 的所有 key 都屏蔽，softmax 会得到 NaN，并在第二层沿注意力
        # 传播到末位（实测：15626/16074 条样本的分数变成 NaN，rank 恒为 1，
        # HR@10 假性报出 0.9721）。这是本项目实测踩到的坑，见 baseline/README.md。
        eye = torch.eye(L, dtype=torch.bool, device=hist.device)
        am = self.causal[:L, :L].unsqueeze(0) | (pad.unsqueeze(1) & ~eye)
        am = am.repeat_interleave(self.n_heads, dim=0)
        for attn, ffn, ln1, ln2 in zip(self.attns, self.ffns, self.lns1, self.lns2):
            y = ln1(x)
            a, _ = attn(y, y, y, attn_mask=am, need_weights=False)
            x = x + self.drop(a)
            x = x + self.drop(ffn(ln2(x)))
            x = x * (~pad).unsqueeze(-1)     # pad 位不携带任何信号
        return self.last_ln(x)

    def logits(self, hist: torch.Tensor) -> torch.Tensor:
        return self.encode(hist)[:, -1] @ self.item_emb.weight[: self.n_items].T

    def loss(self, hist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.logits(hist), target)


class GRU4RecNet(nn.Module):
    """GRU 序列编码 + 末位预测。Hidasi et al. 2016, arXiv:1511.06939。"""

    def __init__(self, n_items, pad_id, maxlen, hidden=64, n_layers=1, dropout=0.2, **_):
        super().__init__()
        self.n_items, self.pad_id, self.maxlen, self.hidden = n_items, pad_id, maxlen, hidden
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=pad_id)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.zeros_(self.item_emb.weight[pad_id])
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True,
                          dropout=(dropout if n_layers > 1 else 0.0))
        self.last_ln = nn.LayerNorm(hidden)

    def encode(self, hist: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.item_emb(hist))
        out, _ = self.gru(x)
        return self.last_ln(out)

    def logits(self, hist: torch.Tensor) -> torch.Tensor:
        # 历史左填充 ⇒ 最后一个位置恒为真实物品，取它即"最近行为"的表示
        return self.encode(hist)[:, -1] @ self.item_emb.weight[: self.n_items].T

    def loss(self, hist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.logits(hist), target)


class BERT4RecNet(nn.Module):
    """双向 Transformer + [MASK] 查询位。Sun et al. CIKM 2019, arXiv:1904.06690（next-item 改编见模块头注释）。"""

    def __init__(self, n_items, pad_id, maxlen, hidden=64, n_layers=2, n_heads=2,
                 dropout=0.2, mask_prob=0.2, **_):
        super().__init__()
        self.n_items, self.pad_id, self.maxlen, self.hidden = n_items, pad_id, maxlen, hidden
        self.mask_id = pad_id + 1
        self.mask_prob = mask_prob
        self.item_emb = nn.Embedding(n_items + 2, hidden, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden)   # +1 = 尾部查询位
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.zeros_(self.item_emb.weight[pad_id])
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, n_heads, hidden * 4, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.last_ln = nn.LayerNorm(hidden)

    def _forward_tokens(self, seq: torch.Tensor) -> torch.Tensor:
        """seq 已含尾部查询位，形状 (b, maxlen+1)。"""
        b, L = seq.shape
        pad = seq.eq(self.pad_id)
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(b, L)
        h = self.item_emb(seq) + self.pos_emb(pos)
        h = self.enc(h, src_key_padding_mask=pad)
        return self.last_ln(h)

    def _with_query(self, hist: torch.Tensor) -> torch.Tensor:
        q = torch.full((hist.shape[0], 1), self.mask_id, dtype=hist.dtype, device=hist.device)
        return torch.cat([hist, q], dim=1)

    def logits(self, hist: torch.Tensor) -> torch.Tensor:
        h = self._forward_tokens(self._with_query(hist))[:, -1]
        return h @ self.item_emb.weight[: self.n_items].T

    def loss(self, hist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pad = hist.eq(self.pad_id)
        pos_mask = (torch.rand_like(hist, dtype=torch.float) < self.mask_prob) & ~pad
        seq = hist.clone()
        seq[pos_mask] = self.mask_id
        h = self._forward_tokens(self._with_query(seq))
        logits_all = h @ self.item_emb.weight[: self.n_items].T

        # ① 尾部查询位 → 下一个物品（主监督）
        loss = F.cross_entropy(logits_all[:, -1], target)
        # ② 随机被 mask 的位置 → 还原自身（辅助监督，BERT4Rec 的 MLM 目标）
        if pos_mask.any():
            bi, li = pos_mask.nonzero(as_tuple=True)
            loss = loss + F.cross_entropy(logits_all[bi, li], hist[bi, li])
        return loss


# ---------------------------------------------------------------------------
# Baseline 包装（统一 fit / score 接口）
# ---------------------------------------------------------------------------


class TorchSeqBaseline(Baseline):
    trainable = True
    net_cls = None
    NET_KEYS: tuple[str, ...] = ()

    def __init__(self, n_items: int, n_users: int, maxlen: int = 20,
                 out_dir: str | None = None, **kw):
        # `out_dir` 必须在这里**显式接收**，否则它会留在 **kw 里并被下面的 train_kw 收走，
        # 最终传给 TrainConfig 报 "unexpected keyword argument 'out_dir'"。
        super().__init__(n_items, n_users, maxlen, out_dir=out_dir, **kw)
        self.net_kw = {k: kw[k] for k in self.NET_KEYS if k in kw}
        self.train_kw = {k: v for k, v in kw.items() if k not in self.NET_KEYS}
        self.net = None

    def _build(self):
        return self.net_cls(self.n_items, self.pad_id, self.maxlen, **self.net_kw)

    def fit(self, data: DomainData, device=None, log=print) -> dict:
        device = device or torch.device("cpu")
        cfg = N.TrainConfig(**self.train_kw)
        N.set_seed(cfg.seed)
        self.net = self._build().to(device)
        users, hist, tgt = train_arrays(data.train_rows, self.maxlen, self.pad_id)
        ds = N.SeqDataset(hist, tgt, self.pad_id)

        bs_eval = 1024
        def valid_fn(model):
            model.eval()
            return N.evaluate(lambda h, u: model.logits(h), data.valid, self.n_items,
                              self.pad_id, device, batch_size=bs_eval)

        log(f"  [{self.name}] 参数量 {N.count_params(self.net)/1e3:.1f}K  "
            f"训练样本 {len(ds)}  batch={cfg.batch_size} maxlen={self.maxlen} device={device}")
        info = N.train_model(self.net, ds, valid_fn, cfg, device, log=log)
        self.net.eval()
        self.extra = {
            "n_params": N.count_params(self.net),
            "best_valid_metric": round(info["best_metric"], 4),
            "best_epoch": info["best_epoch"],
            "stopped_early": info["stopped_early"],
            "train_history": info["history"],
        }
        return self.extra

    def score(self, hist: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        return self.net.logits(hist)

    def info(self) -> dict:
        d = super().info()
        d["n_params"] = self.extra.get("n_params", 0)
        return d


class SASRec(TorchSeqBaseline):
    name = "sasrec"
    net_cls = SASRecNet
    NET_KEYS = ("hidden", "n_layers", "n_heads", "dropout")


class GRU4Rec(TorchSeqBaseline):
    name = "gru4rec"
    net_cls = GRU4RecNet
    NET_KEYS = ("hidden", "n_layers", "dropout")


class BERT4Rec(TorchSeqBaseline):
    name = "bert4rec"
    net_cls = BERT4RecNet
    NET_KEYS = ("hidden", "n_layers", "n_heads", "dropout", "mask_prob")
