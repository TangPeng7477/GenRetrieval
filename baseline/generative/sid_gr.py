"""TIGER 形态的生成式召回 baseline（本项目称 **SID-GR**）——本地 4GB 卡可跑的"小号 TIGER"。

范式（Rajput et al., NeurIPS 2023, arXiv:2305.05065）：物品 → 层次化语义 ID（SID）→
序列模型**自回归生成**下一个物品的 SID；检索侧用**前缀树约束解码**保证生成的每个 token
都是合法前缀，从而一定能映射回真实物品。

与 TIGER 原版的**有意差异**（成本换可复现性，必须写清楚）：
1. 生成模型不是 T5，而是**从零训练的小 Transformer**（默认 2+2 层、d=128、4 头、≈1.6M 参数）。
   TIGER 用 T5-small（~13M）是因为要在多数据集上泛化；本项目单域单任务，小模型足够，
   且能在 4GB 显存里几分钟跑完一轮。
2. **RULER 式"更深 SID + 卷积"没上**，SID 直接复用本项目已定版的 `sid_raw`（RQ-VAE，3 层 × 256）。
3. 输出头用**整词表**（`3×256 + PAD + BOS`）而非按层分开的头，解码时**按层 + 前缀树双重约束**
   屏蔽非法 token（等价于 TIGER 的 trie 约束解码）。
4. 早停用 **valid 的 teacher-forcing 交叉熵**（生成式召回的标准做法），
   而不是 HR@K —— beam search 每轮都跑一遍太贵。

⚠️ **指标必须带 beam 宽度一起读**：生成式召回的 HR@10 天花板由 beam 决定
（beam=20 时最多只有 20 个候选 SID 参与排序）。所以这里额外报一个
**`beam_ceiling`** = 真实目标是否**出现在生成出来的 beam 里**（不限前 10），
用来把"生成能力不足"与"beam 太窄"分开。这是 TIGER 系列论文常见的坑，本项目显式暴露。
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import metrics as M
from ..common import nn as N
from ..common.base import Baseline
from ..common.data import DomainData, EvalSet, train_arrays


class SidGenNet(nn.Module):
    def __init__(self, n_items, pad_id, maxlen, sid, d_model=128, n_layers=2,
                 n_heads=4, dropout=0.1, n_dec_layers=None, **_):
        super().__init__()
        self.n_items, self.pad_id, self.maxlen = n_items, pad_id, maxlen
        sid_t = torch.as_tensor(np.asarray(sid, dtype=np.int64))
        self.register_buffer("sid_table", sid_t, persistent=False)
        self.n_levels = int(sid_t.shape[1])
        self.code_range = 256                       # 每层码本大小（本项目定版为 256）
        self.code_vocab = self.n_levels * self.code_range
        self.pad_token = self.code_vocab
        self.bos_token = self.code_vocab + 1
        vocab = self.code_vocab + 2
        self.vocab = vocab
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=self.pad_token)
        self.pos_enc = nn.Embedding(maxlen * self.n_levels, d_model)
        self.pos_dec = nn.Embedding(self.n_levels + 1, d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_enc.weight, std=0.02)
        nn.init.normal_(self.pos_dec.weight, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 4, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec_layers or n_layers)
        self.out = nn.Linear(d_model, vocab)
        self.register_buffer(
            "causal", torch.triu(torch.ones(self.n_levels + 1, self.n_levels + 1,
                                            dtype=torch.bool), diagonal=1), persistent=False)
        self.register_buffer("levels", torch.arange(self.n_levels), persistent=False)

    # ---------------- 编解码 ----------------

    def item_tokens(self, hist: torch.Tensor) -> torch.Tensor:
        """物品 id 序列 (b,L) → SID token 序列 (b, L*n_levels)，pad 物品用 PAD token。"""
        pad = hist.eq(self.pad_id)
        sids = self.sid_table[hist.clamp(max=self.n_items - 1)]     # (b, L, nlev)
        tok = sids + self.levels * self.code_range
        tok = torch.where(pad.unsqueeze(-1), torch.full_like(tok, self.pad_token), tok)
        return tok.flatten(1)

    def encode_memory(self, hist: torch.Tensor):
        tok = self.item_tokens(hist)
        pad = tok.eq(self.pad_token)
        L = tok.shape[1]
        x = self.tok_emb(tok) + self.pos_enc(torch.arange(L, device=tok.device))[None]
        return self.encoder(x, src_key_padding_mask=pad), pad

    def decode(self, tgt_in: torch.Tensor, mem: torch.Tensor, mem_pad: torch.Tensor) -> torch.Tensor:
        t = tgt_in.shape[1]
        x = self.tok_emb(tgt_in) + self.pos_dec(torch.arange(t, device=tgt_in.device))[None]
        h = self.decoder(x, mem, tgt_mask=self.causal[:t, :t], memory_key_padding_mask=mem_pad)
        return self.out(h)

    def target_tokens(self, target: torch.Tensor) -> torch.Tensor:
        return self.sid_table[target] + self.levels * self.code_range

    def loss(self, hist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mem, mem_pad = self.encode_memory(hist)
        tgt_tok = self.target_tokens(target)                                  # (b, nlev)
        bos = torch.full((tgt_tok.shape[0], 1), self.bos_token, dtype=tgt_tok.dtype,
                         device=tgt_tok.device)
        dec_in = torch.cat([bos, tgt_tok[:, :-1]], dim=1)
        logits = self.decode(dec_in, mem, mem_pad)                            # (b, nlev, vocab)
        return F.cross_entropy(logits.reshape(-1, self.vocab), tgt_tok.reshape(-1))

    # ---------------- 约束解码（前缀树） ----------------

    @torch.no_grad()
    def beam_search(self, hist: torch.Tensor, beam: int = 20, max_seqs: int = 2048):
        """返回 (scores (b,beam), codes (b,beam,nlev))。Trie 约束在这里体现为逐层 mask。

        `max_seqs` 控制单次前向的序列总数（用户数 × beam）。beam 展开会把 memory
        复制成 `(b*beam, S, d)`，默认 2048 条序列 ≈ 63MB，4GB 卡安全；
        不控的话 `512×20=10240` 条会直接爆显存（实测 RuntimeError）。
        """
        bs = max(1, max_seqs // beam)
        all_s, all_c = [], []
        for s in range(0, hist.shape[0], bs):
            sc, cd = self._beam_chunk(hist[s:s + bs], beam)
            all_s.append(sc)
            all_c.append(cd)
        return torch.cat(all_s, 0), torch.cat(all_c, 0)

    @torch.no_grad()
    def _beam_chunk(self, hist: torch.Tensor, beam: int):
        b = hist.shape[0]
        dev = hist.device
        mem, mem_pad = self.encode_memory(hist)
        mem_e = mem.repeat_interleave(beam, dim=0)          # 与 dec_in 的 user-major 排布对齐
        pad_e = mem_pad.repeat_interleave(beam, dim=0)

        def step(dec_in):
            logits = self.decode(dec_in, mem_e, pad_e)[:, -1]
            return F.log_softmax(logits.float(), dim=-1)

        # --- 第 0 层：只允许出现在码本里的首码
        # dec_in 也要展开成 b*beam（否则 tgt 批次 b 与 memory 批次 b*beam 对不上，实测直接 RuntimeError）；
        # 此时各 beam 输入完全相同，取第 0 个 beam 算一次即可（避免 topk 在 3 维张量上多出一维）。
        dec_in = torch.full((b * beam, 1), self.bos_token, dtype=torch.long, device=dev)
        lp0 = step(dec_in).view(b, beam, -1)[:, 0, :self.code_range]            # (b, 256)
        lp0 = lp0.masked_fill(~self.trie[0], -math.inf)
        scores, c0 = lp0.topk(beam, dim=-1)                                     # (b, beam)
        codes = c0.unsqueeze(-1)                                               # (b, beam, 1)

        for lv in range(1, self.n_levels):
            bos = torch.full((b, beam, 1), self.bos_token, dtype=torch.long, device=dev)
            dec_in = torch.cat([bos, codes], dim=-1).reshape(b * beam, lv + 1)
            lp = step(dec_in).view(b, beam, self.vocab)
            lp = lp[:, :, lv * self.code_range:(lv + 1) * self.code_range]      # 只留本层 token
            # 前 lv 层编码 = Σ c_t · 256^(lv-1-t)，必须与 _build_trie 里的编码方式一致
            prefix = (codes[:, :, :lv] * self.pow[self.n_levels - lv:]).sum(-1)
            allowed = self.prefix_allow[lv - 1][prefix.reshape(-1)].view(b, beam, -1)
            lp = lp.masked_fill(~allowed, -math.inf)
            cand = scores.unsqueeze(-1) + lp                                    # (b, beam, 256)
            scores, topi = cand.view(b, -1).topk(beam, dim=-1)
            parent, code = topi // self.code_range, topi % self.code_range
            codes = torch.cat([
                codes.gather(1, parent.unsqueeze(-1).expand(-1, -1, codes.shape[-1])),
                code.unsqueeze(-1)], dim=-1)
        return scores, codes


class SidGR(Baseline):
    name = "sid_gr"
    trainable = True

    def __init__(self, n_items, n_users, maxlen=20, d_model=128, n_layers=2,
                 n_dec_layers=None, n_heads=4, dropout=0.1, lr=1e-3, batch_size=256,
                 epochs=30, patience=4, beam=20, topk_eval=10, seed=42, out_dir=None, **kw):
        super().__init__(n_items, n_users, maxlen,
                         d_model=d_model, n_layers=n_layers, n_dec_layers=n_dec_layers,
                         n_heads=n_heads, dropout=dropout, lr=lr, batch_size=batch_size,
                         epochs=epochs, patience=patience, beam=beam, topk_eval=topk_eval,
                         seed=seed, **kw)
        self.beam = beam
        self.topk_eval = topk_eval
        self.net = None
        self._buckets = None
        self._out_dir = out_dir        # 落 best.pt 用；run.py 注入

    # ---------------- 前缀树 / 码本统计 ----------------

    def _ckpt_path(self) -> "str | None":
        if not self._out_dir:
            return None
        return os.path.join(self._out_dir, "best.pt")

    # ---------------- 前缀树 / 码本统计 ----------------

    def _build_trie(self, sid: np.ndarray):
        nlev = sid.shape[1]
        # trie[0]: 首码是否出现；(256,) bool
        trie0 = np.zeros(256, dtype=bool)
        trie0[np.unique(sid[:, 0])] = True
        # prefix_allow[l-1]: 形状 (256^l, 256)，给定前 l 层编码后允许的下一层码
        prefix_allow = []
        for lv in range(1, nlev):
            n_pref = 256 ** lv
            allow = np.zeros((n_pref, 256), dtype=bool)
            code = np.zeros(sid.shape[0], dtype=np.int64)
            for l in range(lv):
                code = code * 256 + sid[:, l]
            allow[code, sid[:, lv]] = True
            prefix_allow.append(allow)
        # 桶：SID 元组 → 物品列表（语义桶，见 README §3.3）
        buckets: dict[tuple, list[int]] = {}
        for i, row in enumerate(sid):
            buckets.setdefault(tuple(int(x) for x in row), []).append(i)
        return trie0, prefix_allow, buckets

    # ---------------- 训练 ----------------

    def fit(self, data: DomainData, device=None, log=print) -> dict:
        device = device or torch.device("cpu")
        cfg = self.cfg
        N.set_seed(cfg["seed"])
        if data.sid is None:
            raise FileNotFoundError("SID-GR 依赖 sid_raw.npy")
        sid = np.asarray(data.sid, dtype=np.int64)

        self.net = SidGenNet(self.n_items, self.pad_id, self.maxlen, sid,
                             d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                             n_dec_layers=cfg["n_dec_layers"], n_heads=cfg["n_heads"],
                             dropout=cfg["dropout"]).to(device)

        # 流行度（桶内排序 + 同分打破平局用）—— 必须在建桶之前算好
        pop = np.zeros(self.n_items, dtype=np.float64)
        for seq in data.train_seq.values():
            for it in seq:
                pop[it] += 1.0
        self._pop = pop

        trie0, prefix_allow, buckets = self._build_trie(sid)
        self._buckets = {k: sorted(v, key=lambda i: -self._pop_of(i)) for k, v in buckets.items()}
        self.net.trie = [torch.from_numpy(trie0).to(device)]
        self.net.prefix_allow = [torch.from_numpy(a).to(device) for a in prefix_allow]
        self.net.pow = torch.tensor(
            [256 ** (self.net.n_levels - 1 - i) for i in range(self.net.n_levels)],
            dtype=torch.long, device=device)

        users, hist, tgt = train_arrays(data.train_rows, self.maxlen, self.pad_id)
        ds = N.SeqDataset(hist, tgt, self.pad_id)
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                                             pin_memory=(device.type == "cuda"))
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg["lr"])
        vh, vt = data.valid.padded(self.pad_id), data.valid.targets
        # 前缀树节点 = 逐层"实际出现过的前缀"个数；`allow` 形状是 (前缀, 下一层码)，
        # 所以要对 **axis=1** 求 any 才知道哪些前缀真实存在（早期版本误用 axis=0，
        # 把 4.3 万个节点报成了 512，看起来像 trie 被截断——别再用那个数字）。
        nodes = [int(trie0.sum())] + [int(a.any(1).sum()) for a in prefix_allow]
        log(f"  [sid_gr] 参数量 {N.count_params(self.net)/1e6:.2f}M  SID {tuple(sid.shape)}  "
            f"桶数 {len(buckets)}（平均桶大小 {sid.shape[0]/len(buckets):.2f}）  "
            f"合法 SID 前缀树节点 {sum(nodes) + len(buckets)}（逐层 {nodes + [len(buckets)]}）")

        history, best, best_ep, best_state, bad = [], np.inf, -1, None, 0
        for ep in range(1, cfg["epochs"] + 1):
            self.net.train()
            t0, tot, cnt = time.time(), 0.0, 0
            for h, t in loader:
                h, t = h.to(device, non_blocking=True), t.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = self.net.loss(h, t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(t)
                cnt += len(t)
            self.net.eval()
            with torch.no_grad():
                vl, vn = 0.0, 0
                for s in range(0, len(vh), 1024):
                    hb = torch.from_numpy(vh[s:s + 1024]).to(device)
                    tb = torch.from_numpy(vt[s:s + 1024]).to(device)
                    vl += float(self.net.loss(hb, tb)) * len(tb)
                    vn += len(tb)
            vl /= max(vn, 1)
            history.append({"epoch": ep, "train_loss": tot / max(cnt, 1),
                            "valid_ce": vl, "sec": round(time.time() - t0, 1)})
            log(f"  epoch {ep}: train_ce={tot/max(cnt,1):.4f} valid_ce={vl:.4f} ({time.time()-t0:.0f}s)")
            # 每轮结束后把当前 best ckpt 落盘（覆盖），即便被任务打断也能从最近一次 best 恢复评估
            ckpt_path = self._ckpt_path()
            if ckpt_path is not None and best_state is not None:
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({"best_state": best_state, "best_epoch": best_ep,
                            "best_valid_ce": best, "epoch": ep, "history": history}, ckpt_path)
            if vl < best - 1e-4:
                best, best_ep, bad = vl, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    log(f"  早停：{cfg['patience']} 轮未提升（best epoch {best_ep}, valid_ce={best:.4f}）")
                    break

        if best_state:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.extra = {
            "n_params": N.count_params(self.net),
            "best_valid_ce": round(float(best), 4),
            "best_epoch": best_ep,
            "n_buckets": len(buckets),
            "train_history": history,
        }
        return self.extra

    def load_best(self) -> bool:
        """从 `_out_dir/best.pt` 恢复 best_state。如果文件不存在返回 False。"""
        ckpt_path = self._ckpt_path()
        if not ckpt_path or not os.path.exists(ckpt_path):
            return False
        # 先按 SID 维度重建 net 结构（需 self.net 已存在；fit 后已构造）
        if self.net is None:
            return False
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ckpt["best_state"])
        self.net.eval()
        self.extra = self.extra or {}
        self.extra.update({
            "best_valid_ce": ckpt.get("best_valid_ce"),
            "best_epoch": ckpt.get("best_epoch"),
            "history_restored_from_ckpt": True,
        })
        return True

    def _pop_of(self, i: int) -> float:
        return float(self._pop[i]) if hasattr(self, "_pop") else 0.0

    # ---------------- 推理 ----------------

    @torch.no_grad()
    def generate_items(self, hist: torch.Tensor, beam: int | None = None):
        """返回 (item_scores (b, n_items) float32, n_generated (b,))。未生成的位置为 -inf。"""
        beam = beam or self.beam
        scores, codes = self.net.beam_search(hist, beam=beam)
        scores = scores.detach().cpu().numpy()
        codes = codes.detach().cpu().numpy()
        b = hist.shape[0]
        out = np.full((b, self.n_items), -np.inf, dtype=np.float32)
        n_gen = np.zeros(b, dtype=np.int64)
        for i in range(b):
            order = np.argsort(-scores[i])
            filled = 0
            for bi in order:
                key = tuple(int(x) for x in codes[i, bi])
                items = self._buckets.get(key)
                if not items:
                    continue
                for pos, it in enumerate(items):
                    if not np.isfinite(out[i, it]):
                        out[i, it] = scores[i, bi] - 1e-3 * pos   # 桶内按流行度微调，保证确定序
                        filled += 1
            n_gen[i] = filled
        return out, n_gen

    def score(self, hist: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.generate_items(hist)[0])

    def evaluate_custom(self, evalset: EvalSet, device, batch_size: int = 256,
                        beam: int | None = None) -> dict:
        """带 `beam_ceiling` 的评估：目标是否出现在 beam 里（π 生成上限），与 HR@10 分开报。"""
        beam = beam or self.beam
        hist_all = evalset.padded(self.pad_id)
        ranks = np.zeros(len(evalset), dtype=np.int64)
        generated = np.zeros(len(evalset), dtype=bool)
        topk_all = np.zeros((len(evalset), self.topk_eval), dtype=np.int64)
        mrows, mcols = evalset.flat_mask()
        for s in range(0, len(evalset), batch_size):
            e = min(s + batch_size, len(evalset))
            hb = torch.from_numpy(hist_all[s:e]).to(device)
            sc, n_gen = self.generate_items(hb, beam=beam)
            lo = np.searchsorted(mrows, s)
            hi = np.searchsorted(mrows, e)
            if hi > lo:
                sc[mrows[lo:hi] - s, mcols[lo:hi]] = -np.inf
            tgt = evalset.targets[s:e]
            ranks[s:e] = M.ranks_from_scores(sc, tgt)
            generated[s:e] = np.isfinite(sc[np.arange(e - s), tgt])
            k = self.topk_eval
            idx = np.argpartition(-sc, k - 1, axis=1)[:, :k]
            for i in range(e - s):
                topk_all[s + i] = idx[i][np.argsort(-sc[i, idx[i]])]
        out = M.metrics_from_ranks(ranks)
        out.update(M.coverage_metrics(topk_all, self.n_items, self.topk_eval))
        out[f"beam_ceiling@{beam}"] = float(generated.mean())
        return {"metrics": out, "ranks": ranks, "generated": generated}
