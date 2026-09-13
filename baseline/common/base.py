"""baseline 模型统一接口。

所有 baseline（无论是否可训练）都实现这三个方法，保证编排脚本只用一套调用：
    fit(data, device, log) -> info dict
    score(hist, users)     -> (b, n_items) 张量/数组，越大越靠前
    info()                 -> {"model", "n_params", "trainable", "cfg", ...}
"""

from __future__ import annotations

import torch

from .data import DomainData


class Baseline:
    name = "baseline"
    trainable = False

    def __init__(self, n_items: int, n_users: int, maxlen: int = 20,
                 out_dir: str | None = None, **cfg):
        self.n_items = n_items
        self.n_users = n_users
        self.maxlen = maxlen
        self.pad_id = n_items  # 真实 id 之外的额外槽位
        # `out_dir` 由 run.py 注入（落 ckpt 用）。必须在这里**显式消费**：
        # 否则它会漏进 torch 子类的 `train_kw`，被当成训练超参传给 TrainConfig 而报
        # "unexpected keyword argument 'out_dir'"（实测踩到，见 2026-09-13 双塔接入）。
        # 它不进 self.cfg，因此不影响超参留痕。
        self.out_dir = out_dir
        self.cfg = dict(cfg)
        self.extra: dict = {}

    # --- 子类实现 ---------------------------------------------------------
    def fit(self, data: DomainData, device: torch.device, log=print) -> dict:
        return {}

    def score(self, hist: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def info(self) -> dict:
        return {
            "model": self.name,
            "trainable": self.trainable,
            "n_params": 0,
            "cfg": self.cfg,
            **self.extra,
        }
