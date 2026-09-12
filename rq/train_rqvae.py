#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ-VAE 训练（SID tokenizer）—— Sinkhorn 解耦版
=============================================

设计原则
--------
1. **训练阶段不做 Sinkhorn。** `sk_epsilons` 默认全 0，量化一律走 `argmin(d)` 最近邻
   （`rq/models/vq.py:74`：`if not use_sk or self.sk_epsilon <= 0: indices = argmin(d)`）。
   Sinkhorn 只在导出阶段由 `build_sid_dual.py` 单独施加 —— 这样**同一个 ckpt 能导出
   「原始 SID」与「Sinkhorn 后 SID」两版**，消融干净，无需重训。

2. **超参全部对齐开源实现，不自己拍。**
   - 主体照抄上游 MiniOneRec `rq/rqvae.py` + `rq/rqvae.sh`：
     AdamW / lr 1e-3 / warmup 后 constant / grad_clip 1.0 / kmeans_init=True /
     beta 0.25 / num_emb_list [256,256,256] / e_dim 32 / layers [2048,1024,512,256,128,64]。
   - 损失函数与训练循环结构照抄上游 `rq/trainer.py`。
   - 参考 GRID（Snap, CIKM'25）`configs/experiment/rqvae_train_flat.yaml` 保留两个可选开关：
     `--learner adagrad` 与 `--normalize_latent`（encoder 输出 L2 归一化，GRID 默认开）。
     默认**关闭**，与上游基线一致；要做 A/B 时再手动打开。

3. **产物自包含。** 训练日志（stdout 由 shell tee 落盘）+ `train_metrics.json`
   （含完整超参、逐 epoch 曲线、best 指标）+ ckpt 全部写进同一个结果目录，
   便于 `results/sid/<域>/<模式>/` 直接归档对比。

为什么 epochs 要给到千级
------------------------
上游 `rq/rqvae.sh` 是 `--epochs 10000`，`rq/rqvae.py` 默认 5000。
`rq/trainer.py:28` 里 `max_steps = epochs * len(data_loader)`；
本数据集 25,847 items / bs 2048 = 13 batch/epoch → 5000 轮 ≈ **65,000 步**。
这是 RQ-VAE 码本收敛的量级要求，不是"能跑就行"的步数。

用法
----
    python rq/train_rqvae.py \
        --emb data/Amazon23/IandS/emb/long/emb_fused_gate_e60.npy \
        --out_dir results/sid/IandS/gate \
        --epochs 5000 --batch_size 2048 --eval_step 50
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from datasets import EmbDataset                      # noqa: E402
from models.rqvae import RQVAE                       # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="RQ-VAE (SID tokenizer) training")

    # ---- 数据 / 输出 ----
    p.add_argument("--emb", required=True, help="输入向量 .npy (N, d)")
    p.add_argument("--out_dir", required=True, help="结果目录，所有产物都写在这里")
    p.add_argument("--device", default="cuda:0")

    # ---- 优化（对齐上游 rq/rqvae.py 默认）----
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=0,
                   help="Windows 下必须 0，否则 DataLoader 多进程起不来")
    p.add_argument("--eval_step", type=int, default=50)
    p.add_argument("--learner", default="AdamW", choices=["AdamW", "Adam", "SGD", "Adagrad"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", default="constant", choices=["constant", "linear"])
    p.add_argument("--warmup_epochs", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # ---- 模型（对齐上游默认）----
    p.add_argument("--num_emb_list", type=int, nargs="+", default=[256, 256, 256])
    p.add_argument("--e_dim", type=int, default=32)
    p.add_argument("--layers", type=int, nargs="+", default=[2048, 1024, 512, 256, 128, 64])
    p.add_argument("--beta", type=float, default=0.25, help="commitment loss 权重")
    p.add_argument("--quant_loss_weight", type=float, default=1.0)
    p.add_argument("--dropout_prob", type=float, default=0.0)
    p.add_argument("--bn", action="store_true", help="MLP 里加 BatchNorm")
    p.add_argument("--loss_type", default="mse", choices=["mse", "l1"])
    p.add_argument("--kmeans_init", dest="kmeans_init", action="store_true", default=True)
    p.add_argument("--no_kmeans_init", dest="kmeans_init", action="store_false")
    p.add_argument("--kmeans_iters", type=int, default=100)

    # ---- Sinkhorn：训练期用法是一个超参，导出期做法另见 build_sid_dual.py ----
    # ⚠️ 关键：`sk_epsilons` 只有在 forward(use_sk=True) 时才起作用。
    # 上游 MiniOneRec / MQL4GRec 的 trainer 都是 `self.model(data)`（use_sk 默认 True），
    # 所以 MQL4GRec 传 [0,0,0,0.003] 时训练期最后一层**真的走 Sinkhorn**。
    # 而本脚本此前把 use_sk 硬编码为 False → sk_epsilons 完全失效（等同 MiniOneRec 口径）。
    p.add_argument("--sk_epsilons", type=float, nargs="+", default=None,
                   help="各层 Sinkhorn epsilon。**仅在 --train_use_sk 时于训练期生效**。\n"
                        "默认全 0（= 上游 MiniOneRec 默认）。")
    p.add_argument("--sk_iters", type=int, default=50)
    p.add_argument("--train_use_sk", action="store_true", default=False,
                   help="训练期前向是否走 Sinkhorn 分配（默认 False = 纯 argmin）。\n"
                        "  False —— MiniOneRec 口径：训练期不掺 Sinkhorn，解耦最干净（默认）\n"
                        "  True  —— MQL4GRec 口径：需配合 `--sk_epsilons 0 0 0.003`\n"
                        "注意：即便为 True，训练期监控的 collision 仍是 argmin 口径（见 eval 函数）。")

    # ---- ckpt 选择准则 ----
    p.add_argument("--select_ckpt", default="collision", choices=["loss", "collision"],
                   help="selected_model.pth 用哪个准则（默认 collision，与上游一致）：\n"
                        "  collision —— 唯一性最好。实测总 loss 在 ep120 触底后回升但重建 R² 与\n"
                        "               碰撞率都在持续变好，**loss 是假信号**，别按它选。\n"
                        "               必须配合 burn_in_frac 跳过 ep1（k-means 初始化会让\n"
                        "               未训练模型的碰撞率虚低到 0.044，而 R² 只有 0.01）\n"
                        "  loss      —— 只用于对照")
    p.add_argument("--burn_in_frac", type=float, default=0.1,
                   help="select_ckpt=collision 时，跳过前 X%% 轮次再选（避开 k-means 初始化假象）")

    # ---- GRID 可选开关（默认关，保持与上游基线一致）----
    p.add_argument("--normalize_latent", action="store_true",
                   help="GRID 做法：encoder 输出 L2 归一化后再量化")
    p.add_argument("--init_samples", type=int, default=0,
                   help=">0 时训练前用「随机抽样的 init_samples 条」先做完三层残差 k-means "
                        "(独立 init pass)，再开始训练；0 = 上游行为（第一个训练 batch 触发）。\n"
                        "实测 8192 即拿走大部分收益（基尼 0.344→0.176，量化起点 MSE -7.7%%），\n"
                        "全量 25847 耗时约 9s。不影响训练 batch_size 与训练动力学。")
    p.add_argument("--save_every", type=int, default=0, help=">0 时每 N 轮额外存一个周期 ckpt")
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def preinit_codebooks(model, embeddings, n_samples, seed, device):
    """训练前独立 k-means init pass：用 n_samples 条样本一次做完三层残差 k-means。

    与上游行为（训练第一步由第一个 batch=2048 触发 vq.init_emb）量化逻辑完全一致
    （逐层 kmeans → argmin 量化 → 取残差），只有两点不同：
      1. 样本量可远大于 2048 —— 质心对全量分布更有代表性
         （实测 I&S/text：基尼 0.344(2048) → 0.176(8192) → 0.128(25847)，
          inertia/N 0.000169 → 0.000156 → 0.000149，见 docs/KNOWLEDGE_BASE.md §5.11 Q6c）
      2. 样本抽样用 RandomState(seed) 固定 + layers.py 的 kmeans 已固定 random_state=0
         ⟹ 整个初始化端到端可复现（上游的「首 batch」受 DataLoader shuffle 影响不可复现）

    不改训练 batch_size / 训练动力学；完成后各层 initted=True，训练首步不再触发 init_emb。
    """
    rng = np.random.RandomState(seed)
    n = min(n_samples, len(embeddings))
    idx = rng.choice(len(embeddings), size=n, replace=False)
    x = torch.tensor(np.ascontiguousarray(embeddings[idx]), dtype=torch.float32,
                     device=device)
    t0 = time.time()
    latent = model.encoder(x)
    residual = latent
    layers_info = []
    for li, layer in enumerate(model.rq.vq_layers):
        layer.init_emb(residual)                       # kmeans(random_state=0) → copy_ → initted=True
        x_res, _, _ = layer(residual, use_sk=False)    # argmin 量化，与训练期口径一致
        residual = residual - x_res
        used = int((layer.embedding.weight.abs().sum(dim=1) > 1e-12).sum().item())
        layers_info.append({"layer": li, "used_codes": used, "n_codes": layer.n_e})
        print(f"[init ] L{li}: kmeans on {n:,} samples (seed={seed}) "
              f"-> 非零码 {used}/{layer.n_e}  residual‖·‖={float(residual.norm()):.4f}")
    print(f"[init ] 预初始化完成，耗时 {time.time()-t0:.1f}s", flush=True)
    return {"samples": int(n), "seed": int(seed), "layers": layers_info}


@torch.no_grad()
def eval_collision(model, loader, device):
    """上游 Trainer._valid_epoch 的同款指标：唯一 code 数 / 样本数。
    use_sk=False —— 训练期评估也不开 Sinkhorn，保证与导出版 sid_raw 口径一致。
    P-7：顺带返回各层用码数（get_indices 本来就返回逐层索引，零额外前向）——
    码本塌缩是单调正反馈，没有训练期监控就只能事后盲评。"""
    model.eval()
    inds = []
    n = 0
    for d in loader:
        d = d.to(device)
        ind = model.get_indices(d, use_sk=False)
        inds.append(ind.view(-1, ind.shape[-1]).cpu().numpy())
        n += ind.shape[0]
    model.train()
    ind = np.concatenate(inds, axis=0)
    seen = {",".join(str(int(x)) for x in row) for row in ind}
    per_layer = [int(np.unique(ind[:, l]).size) for l in range(ind.shape[1])]
    return 1.0 - len(seen) / max(n, 1), len(seen), n, per_layer


def main():
    args = parse_args()
    t_start = time.time()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 训练期 Sinkhorn 解耦：显式置 0，避免误传
    if args.sk_epsilons is None:
        args.sk_epsilons = [0.0] * len(args.num_emb_list)
    else:
        args.sk_epsilons = [0.0 if e <= 0 else e for e in args.sk_epsilons]
    if args.train_use_sk:
        if not any(e > 0 for e in args.sk_epsilons):
            raise SystemExit(
                "[error] --train_use_sk 生效需要至少一层 sk_epsilon > 0，"
                "请加 `--sk_epsilons 0 0 0.003`（MQL4GRec 配方）")
        print(f"[note] 训练期开启 Sinkhorn sk_epsilons={args.sk_epsilons} "
              f"（MQL4GRec 口径，用于对抗码本塌缩）")
    elif any(e > 0 for e in args.sk_epsilons):
        print(f"[note] sk_epsilons={args.sk_epsilons} 已传入，但 --train_use_sk 未开 → "
              f"训练期走纯 argmin，该参数**不影响训练**（仅 ckpt 里存着，导出期可再用）")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- 数据 ----
    data = EmbDataset(args.emb)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=False)
    n_batch = len(loader)

    # ---- 模型 ----
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    if args.normalize_latent:
        # GRID: encoder 输出先 L2 归一化再进量化器（此处包一层，不改变原类）
        orig_encoder = model.encoder
        import torch.nn.functional as F

        class _Enc(torch.nn.Module):
            def __init__(self, e):
                super().__init__()
                self.e = e

            def forward(self, x):
                return F.normalize(self.e(x), dim=-1, p=2)

        model.encoder = _Enc(orig_encoder)

    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())

    # ---- 优化器 / 调度（与上游 trainer.py 一致）----
    learner = args.learner.lower()
    if learner == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif learner == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif learner == "adagrad":
        opt = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    max_steps = args.epochs * n_batch
    warmup_steps = args.warmup_epochs * n_batch
    from transformers import (get_constant_schedule_with_warmup,
                              get_linear_schedule_with_warmup)
    if args.lr_scheduler_type == "linear":
        sched = get_linear_schedule_with_warmup(opt, warmup_steps, max_steps)
    else:
        sched = get_constant_schedule_with_warmup(opt, warmup_steps)

    print("=" * 78)
    print(f"[data ] {args.emb}  N={len(data):,}  in_dim={data.dim}")
    print(f"[model] in_dim={data.dim} -> layers={args.layers} -> e_dim={args.e_dim}")
    print(f"        num_emb_list={args.num_emb_list}  码本容量={int(np.prod(args.num_emb_list)):,}")
    print(f"        params={n_par:,}  device={device}")
    print(f"[train] epochs={args.epochs}  batch={args.batch_size}  batches/epoch={n_batch}  "
          f"-> max_steps={max_steps:,} (warmup {warmup_steps:,})")
    print(f"        learner={args.learner}  lr={args.lr}  beta={args.beta}  "
          f"kmeans_init={args.kmeans_init}  normalize_latent={args.normalize_latent}")
    print(f"[sk   ] sk_epsilons={args.sk_epsilons}  train_use_sk={args.train_use_sk}  "
          f"<-- {'训练期走 Sinkhorn（MQL4GRec 口径）' if args.train_use_sk else '训练期纯 argmin（= MiniOneRec 官方口径）'}")
    print(f"[sk   ] 导出期另由 build_sid_dual.py 出 raw / sk 两版，与上式独立")
    print("=" * 78, flush=True)

    curve = []
    best = {"loss": float("inf"), "collision": float("inf")}
    best_loss_ep = best_coll_ep = -1
    eval_loader = DataLoader(data, batch_size=4096, shuffle=False,
                             num_workers=0, pin_memory=True)

    # ---- 训练前独立 k-means init pass（可选，--init_samples > 0 时生效）----
    init_info = None
    if args.init_samples > 0:
        init_info = preinit_codebooks(model, data.embeddings, args.init_samples,
                                      seed=args.seed, device=device)
        cr0, uniq0, tot0, pl0 = eval_collision(model, eval_loader, device)
        print(f"[init ] 初始化完成后的起点碰撞率={cr0:.5f}  unique={uniq0:,}/{tot0:,}  "
              f"层用码={pl0}")

    # k-means 初始化会让 ep1 的碰撞率虚低（实测 0.044，但 R²≈0.01 等于没训），
    # 若按碰撞率选 ckpt 必须跳过 burn-in，否则会选到未训练模型。
    burn_in = max(1, int(args.epochs * args.burn_in_frac))

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, tot_recon = 0.0, 0.0
        for d in loader:
            d = d.to(device)
            opt.zero_grad(set_to_none=True)
            out, rq_loss, _ = model(d, use_sk=args.train_use_sk)
            loss, loss_recon = model.compute_loss(out, rq_loss, xs=d)
            if torch.isnan(loss):
                raise ValueError(f"Training loss is nan at epoch {ep}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            tot += float(loss)
            tot_recon += float(loss_recon)

        avg_loss = tot / n_batch
        avg_recon = tot_recon / n_batch
        rec = {"epoch": ep + 1, "loss": round(avg_loss, 6),
               "recon_loss": round(avg_recon, 6),
               "lr": round(float(sched.get_last_lr()[0]), 8),
               "sec": round(time.time() - t0, 2)}

        if (ep + 1) % args.eval_step == 0 or ep == 0:
            cr, n_uniq, n_tot, per_layer = eval_collision(model, eval_loader, device)
            rec["collision_rate"] = round(cr, 6)
            rec["n_unique"] = n_uniq
            rec["layer_usage"] = per_layer
            print(f"epoch {ep+1:>5}/{args.epochs}  loss={avg_loss:.5f}  "
                  f"recon={avg_recon:.5f}  collision={cr:.5f}  "
                  f"unique={n_uniq:,}/{n_tot:,}  usage={per_layer}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
            if cr < best["collision"] and (ep + 1) >= burn_in:
                best["collision"] = cr
                best_coll_ep = ep + 1
                torch.save({"args": args, "epoch": ep + 1, "state_dict": model.state_dict()},
                           os.path.join(ckpt_dir, "best_collision_model.pth"))
        else:
            print(f"epoch {ep+1:>5}/{args.epochs}  loss={avg_loss:.5f}  "
                  f"recon={avg_recon:.5f}  ({time.time()-t0:.1f}s)", flush=True)

        if avg_loss < best["loss"]:
            best["loss"] = avg_loss
            best_loss_ep = ep + 1
            torch.save({"args": args, "epoch": ep + 1, "state_dict": model.state_dict()},
                       os.path.join(ckpt_dir, "best_loss_model.pth"))

        if args.save_every and (ep + 1) % args.save_every == 0:
            torch.save({"args": args, "epoch": ep + 1, "state_dict": model.state_dict()},
                       os.path.join(ckpt_dir, f"epoch_{ep+1}_model.pth"))

        curve.append(rec)

    # 末轮模型也存一份（便于看"训满"的效果）
    torch.save({"args": args, "epoch": args.epochs, "state_dict": model.state_dict()},
               os.path.join(ckpt_dir, "last_model.pth"))

    # 下游统一引用这一份，避免各脚本各自硬编码 best_collision / best_loss
    chosen = "best_loss_model.pth" if args.select_ckpt == "loss" else "best_collision_model.pth"
    chosen_ep = best_loss_ep if args.select_ckpt == "loss" else best_coll_ep
    import shutil
    shutil.copyfile(os.path.join(ckpt_dir, chosen), os.path.join(ckpt_dir, "selected_model.pth"))

    metrics = {
        "script": "rq/train_rqvae.py",
        "emb": args.emb,
        "n_items": int(len(data)),
        "in_dim": int(data.dim),
        "n_params": int(n_par),
        "device": str(device),
        "args": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in vars(args).items()},
        "batches_per_epoch": n_batch,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "best": {"loss": round(best["loss"], 6), "loss_epoch": best_loss_ep,
                 "collision_rate": (round(best["collision"], 6)
                                    if best_coll_ep > 0 else None),
                 "collision_epoch": best_coll_ep},
        "select_ckpt": args.select_ckpt,
        "selected_model": os.path.join(ckpt_dir, "selected_model.pth"),
        "selected_epoch": chosen_ep,
        "burn_in_epoch": burn_in,
        "kmeans_preinit": init_info,
        "curve": curve,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    with open(os.path.join(args.out_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print(f"[done] {time.time()-t_start:.0f}s  best_loss={best['loss']:.5f}@ep{best_loss_ep}  "
          f"best_collision={best['collision']:.5f}@ep{best_coll_ep}")
    print(f"       ckpt -> {ckpt_dir}")
    print(f"       metrics -> {os.path.join(args.out_dir, 'train_metrics.json')}")


if __name__ == "__main__":
    main()
