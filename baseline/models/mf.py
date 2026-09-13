"""BPR-MF：非序列的经典协同过滤 baseline（Rendle et al. 2012, arXiv:1205.2618）。

用途：回答"序列建模到底有没有用"——它是**无序列**的强 CF 代表，
也是论文基线表里的常客。评估时用 user id 查表，不使用行为序列。

实现口径：
- BPR pairwise loss：`-log σ(u·i⁺ − u·i⁻)`，负样本**均匀采样**（不排已交互物品，
  与 BPR 原论文一致）；这会让绝对指标略偏乐观，但它对所有对比对象的影响方向一致。
- 用 valid 的 NDCG@10 早停选 ckpt（与其余 baseline 同一套早停逻辑）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..common import nn as N
from ..common.base import Baseline
from ..common.data import DomainData, train_arrays


class BPRMF(Baseline):
    name = "bprmf"
    trainable = True

    def __init__(
        self,
        n_items: int,
        n_users: int,
        maxlen: int = 20,
        dim: int = 64,
        lr: float = 3e-3,
        batch_size: int = 4096,
        epochs: int = 60,
        reg: float = 1e-6,
        patience: int = 5,
        seed: int = 42,
        **kw,
    ):
        super().__init__(n_items, n_users, maxlen,
                         dim=dim, lr=lr, batch_size=batch_size, epochs=epochs,
                         reg=reg, patience=patience, seed=seed, **kw)
        self.net = None

    def fit(self, data: DomainData, device=None, log=print) -> dict:
        device = device or torch.device("cpu")
        cfg = self.cfg
        N.set_seed(cfg["seed"])

        users, _, tgt = train_arrays(data.train_rows, self.maxlen, self.pad_id)
        u_t = torch.from_numpy(users).to(device)
        i_t = torch.from_numpy(tgt).to(device)
        n_pairs = len(u_t)

        class _MF(nn.Module):
            def __init__(self, nu, ni, d):
                super().__init__()
                self.eu = nn.Embedding(nu, d)
                self.ei = nn.Embedding(ni, d)
                nn.init.normal_(self.eu.weight, std=0.01)
                nn.init.normal_(self.ei.weight, std=0.01)

        net = _MF(self.n_users, self.n_items, cfg["dim"]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        g = torch.Generator(device="cpu").manual_seed(cfg["seed"])

        history, best, best_ep, best_state, bad = [], -np.inf, -1, None, 0
        bs = cfg["batch_size"]
        for ep in range(1, cfg["epochs"] + 1):
            net.train()
            perm = torch.randperm(n_pairs, generator=g)
            tot = 0.0
            for s in range(0, n_pairs, bs):
                idx = perm[s:s + bs]
                u = u_t[idx]
                pos = i_t[idx]
                neg = torch.randint(0, self.n_items, (len(idx),), generator=g).to(device)
                ue = net.eu(u)
                ps = (ue * net.ei(pos)).sum(-1)
                ns = (ue * net.ei(neg)).sum(-1)
                loss = -torch.nn.functional.logsigmoid(ps - ns).mean()
                if cfg["reg"]:
                    loss = loss + cfg["reg"] * (ue.pow(2).sum() + net.ei(pos).pow(2).sum()
                                                + net.ei(neg).pow(2).sum()) / len(idx)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)

            net.eval()
            with torch.no_grad():
                m = N.evaluate(lambda h, uu: net.eu(uu) @ net.ei.weight.T,
                               data.valid, self.n_items, self.pad_id, device)["metrics"]
            cur = m["NDCG@10"]
            history.append({"epoch": ep, "train_loss": tot / n_pairs,
                            "valid": {k: v for k, v in m.items() if k != "n_eval"}})
            log(f"  epoch {ep}: train_loss={tot/n_pairs:.4f} valid NDCG@10={cur:.4f} HR@10={m['HR@10']:.4f}")
            if cur > best + 1e-6:
                best, best_ep, bad = cur, ep, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    log(f"  早停：{cfg['patience']} 轮未提升（best epoch {best_ep}, NDCG@10={best:.4f}）")
                    break

        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        self.net = net
        self.extra = {
            "n_params": N.count_params(net),
            "best_valid_NDCG@10": round(float(best), 4),
            "best_epoch": best_ep,
            "train_history": history,
        }
        return self.extra

    def score(self, hist: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        return self.net.eu(users) @ self.net.ei.weight.T

    def info(self) -> dict:
        d = super().info()
        d["n_params"] = self.extra.get("n_params", 0)
        return d
