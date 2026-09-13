"""baseline 统一入口。

用法（在项目根目录执行）：

  python -m baseline.run --model pop,itemknn --domain IandS
  python -m baseline.run --model all --domain all
  python -m baseline.run --model sid_gr --domain VG --beam 20
  python -m baseline.run --model sasrec --domain IandS --quick      # 冒烟（1 轮 + 子采样）

产物：`baseline/results/<domain>/<model>/metrics.json` + `run.log`
汇总：`python -m baseline.scripts.summarize`

设计约束：
- 数据**每域只加载一次**，同域多个模型复用（加载 VG 的 645k 样本约需数秒）。
- 每个模型的 `valid` / `test` 都评估，`valid` 用于早停与选 ckpt（对可训练模型），
  `test` 是最终对外报数。
- 结果 JSON 自带 `git_commit` / `data.meta` / 模型超参，保证"数字能指到机器可读产物"。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch

from .common import nn as N
from .common.data import load_domain
from .common.metrics import save_json
from .generative.sid_gr import SidGR

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_ROOT = os.path.join(ROOT, "baseline", "results")

# ---------------------------------------------------------------------------
# 模型注册：名称 → (类, 分组, 默认超参)
# 超参取值说明见 baseline/README.md §复现设置；改这里等于改实验登记表。
# ---------------------------------------------------------------------------


def registry(quick: bool = False) -> dict:
    from .generative.retrieval import ContentANN, SidPrefix
    from .generative.sid_gr import SidGR
    from .models.heuristic import ItemKNN, PopRec
    from .models.mf import BPRMF
    from .models.seq import GRU4Rec, SASRec  # BERT4Rec 已移出对比矩阵，见下方 registry 注释
    from .models.two_tower import TwoTowerID, TwoTowerMM

    seq_cfg = dict(hidden=64, dropout=0.2, lr=1e-3, batch_size=512,
                   epochs=1 if quick else 25, patience=4, seed=42)
    # 双塔与序列模型共用超参，**只有检索范式不同**（历史池化 vs 时序建模），
    # 这样 twotower_id vs sasrec 的差距可以干净归因到"是否建模时序"。
    tower_cfg = dict(seq_cfg)
    sid_gr_cfg = dict(d_model=128, n_layers=2, n_heads=4,
                      dropout=0.1, lr=1e-3, batch_size=512,
                      epochs=1 if quick else 25, patience=4,
                      beam=20, seed=42)
    reg = {
        # ============ 对比矩阵（2026-09-14 定版，共 4 个模型）============
        # 保留：两个经典序列模型（gru4rec / sasrec）+ 双塔（twotower_id）
        #      + 一个内容检索探针（content_ann）。
        #
        # 【已移出对比矩阵的模型 —— 实现类一律保留，改一行即可加回】
        # · twotower_mm（2026-09-14 决定）：多模态双塔。
        #   - 实测 IandS 0.0322 / VG 0.0661，均低于同组 content_ann 之外的对照价值有限；
        #     且 coverage@10 偏低（0.0766 / 0.2106），打分分布过于集中。
        #   - 但它有一个**已验证且仍然成立**的结论保留在 `EVAL_PROTOCOL §4.1.1`：
        #     学出来的投影远强于直接用融合向量检索（content_ann→twotower_mm：
        #     IandS +11.8%、VG +178.9%）。需要时加回即可复现。
        #   - 产物归档到 baseline/_dropped/2026-09-14_cut_twotower_mm/。
        # · sid_prefix / sid_gr（2026-09-13 决定）：
        #   - sid_prefix 的 HR@10 比经典 baseline 高一倍。placebo 测试（打乱 SID 行）
        #     证明其分值来自语义结构而非 bug（0.0545→0.0113、0.0627→0.0047），
        #     但它吃到「LOO 下相邻购买同语义区」的协议红利（第 1 层共享率是对照的 7.9×~10.7×），
        #     且 maxlen=20 是次优配置（k=3 反而 +11~15%）⇒ 口径不干净，不宜当 SFT/RL 达标线。
        #   - sid_gr 候选集 = beam（≤20），与全库排序不可横比，作为对照意义有限。
        # · pop / itemknn / bprmf：经典非序列 baseline，本次精简后不进主榜
        #   （注意 pop 仍是重要的下限 sanity check，需要时可从归档取回对照）。
        # · bert4rec：更早决定移除，单轮 364~675s（sasrec 的 25 倍）而指标未超，买不回成本。
        # 全部产物已归档到 baseline/_dropped/2026-09-13_cut_models/（可回溯，不进结果表）。
        # =============================================================================
        "gru4rec":     (GRU4Rec,    "A_classic",    dict(seq_cfg, n_layers=1)),
        "sasrec":      (SASRec,     "A_classic",    dict(seq_cfg, n_layers=2, n_heads=2)),
        "twotower_id": (TwoTowerID, "A_classic",    dict(tower_cfg, feat="id")),
        "content_ann": (ContentANN, "B_retrieval",  dict(agg="mean")),
        # "twotower_mm": (TwoTowerMM, "B_retrieval",  dict(tower_cfg, feat="mm")),  # 2026-09-14 移出
    }
    return reg


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------


class Tee:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, msg: str = ""):
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# 单次运行
# ---------------------------------------------------------------------------


def run_one(name: str, domain: str, data, cls, group: str, cfg: dict,
            device: torch.device, args, log) -> dict:
    log(f"=== {domain} / {name} ({group}) ===")
    model = cls(n_items=data.n_items, n_users=data.n_users, maxlen=args.maxlen,
                 out_dir=os.path.join(RESULT_ROOT, domain, name), **cfg)
    t0 = time.time()

    # 数据子采样（仅冒烟用，正式跑不动它）
    if args.quick:
        keep = min(len(data.train_rows), 20000)
        data.train_rows = data.train_rows[:keep]
        log(f"  [quick] 训练样本截断到 {keep}")

    info = model.fit(data, device=device, log=log)
    fit_sec = time.time() - t0

    # --- 评估 ---
    evals = {}
    if isinstance(model, SidGR):
        for split in ("valid", "test"):
            ev = getattr(data, split)
            r = model.evaluate_custom(ev, device, batch_size=args.eval_batch, beam=args.beam)
            evals[split] = {"metrics": r["metrics"]}
            if split == "test":
                evals[split]["metrics"].update(N.summarize_ranks(r["ranks"]))
    else:
        for split in ("valid", "test"):
            ev = getattr(data, split)
            r = N.evaluate(model.score, ev, data.n_items, model.pad_id, device,
                           batch_size=args.eval_batch)
            evals[split] = {"metrics": r["metrics"]}
            if split == "test":
                evals[split]["metrics"].update(N.summarize_ranks(r["ranks"]))

    payload = {
        "domain": domain,
        "model": name,
        "group": group,
        "git_commit": git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "maxlen": args.maxlen,
        "quick": bool(args.quick),     # 冒烟结果（1 轮 + 子采样）不得进正式对比表
        "config": cfg,
        "model_info": model.info(),
        "data": {"n_items": data.n_items, "n_users": data.n_users, **data.meta},
        "valid": evals["valid"]["metrics"],
        "test": evals["test"]["metrics"],
        "fit_sec": round(fit_sec, 1),
        "total_sec": round(time.time() - t0, 1),
    }
    log(f"  → {domain}/{name} TEST: HR@10={payload['test']['HR@10']:.4f} "
        f"NDCG@10={payload['test']['NDCG@10']:.4f} MRR={payload['test']['MRR']:.4f} "
        f"coverage@10={payload['test']['coverage@10']:.4f}  用时 {payload['total_sec']}s")
    return payload


def main():
    ap = argparse.ArgumentParser(description="GenRetrieval baseline 统一入口")
    ap.add_argument("--model", default="all", help="逗号分隔的模型名，或 all")
    ap.add_argument("--domain", default="all", help="IandS / VG / all")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--maxlen", type=int, default=20)
    ap.add_argument("--eval_batch", type=int, default=512)
    ap.add_argument("--beam", type=int, default=20, help="仅 sid_gr：约束解码 beam 宽度")
    ap.add_argument("--quick", action="store_true", help="冒烟：1 轮 + 2 万样本")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = N.get_device() if args.device == "auto" else torch.device(args.device)

    reg = registry(quick=args.quick)
    names = list(reg) if args.model == "all" else [m.strip() for m in args.model.split(",") if m.strip()]
    for m in names:
        if m not in reg:
            raise SystemExit(f"未知模型 {m}，可选：{', '.join(reg)}")
    domains = ["IandS", "VG"] if args.domain == "all" else [args.domain]

    # 按域名分组处理，保证数据只加载一次
    by_domain: dict[str, list[str]] = {}
    for m in names:
        for d in domains:
            by_domain.setdefault(d, []).append(m)

    for domain, mlist in by_domain.items():
        # twotower_mm 已移出（2026-09-14）；加回时需同时取消 registry 里那行的注释。
        need_emb = any(m in ("content_ann", "twotower_mm") for m in mlist)
        data = load_domain(domain, root=ROOT, with_emb=need_emb)
        for m in mlist:
            cls, group, cfg = reg[m]
            if m == "sid_gr":
                cfg = dict(cfg, beam=args.beam)
            outdir = os.path.join(RESULT_ROOT, domain, m)
            log = Tee(os.path.join(outdir, "run.log"))
            log(f"domain={domain} model={m} device={device} commit={git_commit()} "
                f"n_items={data.n_items} n_users={data.n_users}")
            log(f"data.meta = {json.dumps(data.meta, ensure_ascii=False)}")
            try:
                payload = run_one(m, domain, data, cls, group, cfg, device, args, log)
                save_json(os.path.join(outdir, "metrics.json"), payload)
            except Exception as e:  # 单个模型失败不拖垮整批
                log(f"  ❌ {m} 失败：{type(e).__name__}: {e}")
                import traceback
                log(traceback.format_exc())
            finally:
                log.close()


if __name__ == "__main__":
    main()
