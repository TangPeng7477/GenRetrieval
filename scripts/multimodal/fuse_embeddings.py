#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1-5: 多模态融合 + 融合质量评估
================================
输入: data/Amazon23/<short>/emb/emb_text_*.npy        (N, 1024) L2 归一化
      data/Amazon23/<short>/emb/emb_image_siglip.npy  (N, 768)  L2 归一化（缺图=零向量）
      data/Amazon23/<short>/emb/emb_image_mask.npy    (N,) bool
输出: data/Amazon23/<short>/emb/emb_fused_<mode>.npy
      data/Amazon23/<short>/emb/fusion_report.json

四种模式（--mode）
------------------
  text   : 纯文本向量（单模态基线，等价于"不做多模态"）
  concat : [e_text ; e_img] 拼接后 L2 归一化（可选 PCA/白化降到 --target_dim）
           无监督、线性：只找方差最大的方向，不知道"什么方向对推荐有用"
  mlp    : e = MLP([e_t ; e_i])，用共现对比目标(InfoNCE)训练
           有监督、非线性：网络自己学怎么混合两个模态，无显式门控结构
  gate   : 门控融合  e = g ⊙ W_t·e_t + (1-g) ⊙ W_i·e_i
           g = sigmoid(MLP([e_t ; e_i]))，同样用共现对比目标训练
           有监督 + 可解释：g 就是"每维该信文本还是信图像"的权重，可导出分析

  → 四者的关系是一条消融链：
     text(无图像) → concat(线性无监督) → mlp(非线性有监督) → gate(非线性有监督+门控)
     监督信号：正样本 = 同一用户行为序列中共现的物品对（从 train.inter 抽），
     让融合向量直接优化"推荐友好度"，而不是随便对齐两个模态

评估（--eval，两个独立代理指标，不依赖 SFT/RL，秒级出结果）
----------------------------------------------------------
  A) 模态对齐 recall@k：文本↔图像。**两边维度不同（1024 vs 768），直接点积无定义**，
     实现上先在拟合子集解岭回归线性映射，再到留出子集检索；衡量"线性可对齐程度"。
  B) 共现检索 recall@k：用融合向量检索 top-k，看命中物品是否与 query 共现
     （**这是真正决定 SID 质量的指标**；随机基线一并给出）

用法:
    # 建三种向量
    python scripts/multimodal/fuse_embeddings.py --short IandS --mode concat --target_dim 1024
    python scripts/multimodal/fuse_embeddings.py --short IandS --mode gate --epochs 3
    # 评估（对已存在的 mode 逐个跑）
    python scripts/multimodal/fuse_embeddings.py --short IandS --eval --modes text concat gate
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #

def l2n(x, eps=1e-8):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def load_pairs(inter_path, max_pairs, seed=42, max_hist=50):
    """从 train.inter 抽取共现正样本对 (i, j)：history 内两两 + (history_last, target)。"""
    rng = random.Random(seed)
    pairs = set()
    with open(inter_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 3:
                continue
            hist = [int(x) for x in p[1].split()] if p[1].strip() else []
            tgt = int(p[2])
            if not hist:
                continue
            hist = hist[-max_hist:]
            # target 与最近若干历史共现
            for a in hist[-5:]:
                pairs.add((tgt, a) if tgt < a else (a, tgt))
            # 历史内部滑窗共现（限制数量）
            if len(hist) >= 2:
                for _ in range(min(3, len(hist) - 1)):
                    a = rng.choice(hist)
                    b = rng.choice(hist)
                    if a != b:
                        pairs.add((a, b) if a < b else (b, a))
            if len(pairs) >= max_pairs * 1.5:
                break
    pairs = list(pairs)
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    return np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)


def split_pairs(inter_path, max_pairs=0, eval_frac=0.1, seed=42):
    """确定性切分 train / held-out，两者物品对不重叠。

    ⚠️ 为什么必须切：训练和评估若用同一份共现对（早期版本就是这么写的），
    召回率会随训练轮数单调上涨——那不是"学到更好的度量"，是**背下了这批对**。
    实测 mlp 在训练对上 R@10 能从 e3 的 0.171 涨到 e30 的 0.610，
    而留出对上只有 0.045（13 倍水分）。留出对才是可信的泛化指标。

    留出集只由 (inter_path, eval_frac, seed) 决定，**与 max_pairs 无关**——
    这样改训练数据量时评估口径不变，跨实验才可比（recall@k 会随正样本数浮动）。
    max_pairs=0 表示用全部剩余对作训练集。
    """
    pool = load_pairs(inter_path, 20_000_000, seed)      # 全量池
    rng = np.random.RandomState(seed + 1)                # 与 load_pairs 的 shuffle 解耦
    perm = rng.permutation(len(pool))
    n_eval = int(len(pool) * eval_frac)
    ev = pool[perm[:n_eval]]
    tr_all = pool[perm[n_eval:]]
    if max_pairs and max_pairs < len(tr_all):
        tr = tr_all[rng.permutation(len(tr_all))[:max_pairs]]
    else:
        tr = tr_all
    return tr, ev


class GateFusion(nn.Module):
    """e = g ⊙ W_t e_t + (1-g) ⊙ W_i e_i,  e 再 L2 归一化"""

    def __init__(self, dt, di, d_out=None, hidden=512):
        super().__init__()
        d_out = d_out or dt
        self.wt = nn.Linear(dt, d_out, bias=False)
        self.wi = nn.Linear(di, d_out, bias=False)
        self.gate = nn.Sequential(nn.Linear(dt + di, hidden), nn.GELU(), nn.Linear(hidden, d_out))

    def forward(self, et, ei):
        g = torch.sigmoid(self.gate(torch.cat([et, ei], dim=-1)))
        e = g * self.wt(et) + (1 - g) * self.wi(ei)
        return F.normalize(e, dim=-1)


class MLPFusion(nn.Module):
    """concat + MLP 非线性降维：e = MLP([e_t ; e_i])，再 L2 归一化。

    与 concat 模式的区别：concat 用 PCA——无监督、线性，只保留方差最大的方向；
    本模式用共现对比目标监督训练一个非线性投影，学到的是"对推荐有用的方向"。
    与 gate 模式的区别：不做乘性门控，纯粹让网络自己决定混合方式，
    参数量更集中在投影上（无 gate 分支），但缺少逐维模态权重的可解释性。
    """

    def __init__(self, dt, di, d_out=None, hidden=512, dropout=0.1):
        super().__init__()
        d_out = d_out or dt
        self.net = nn.Sequential(
            nn.Linear(dt + di, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, et, ei):
        return F.normalize(self.net(torch.cat([et, ei], dim=-1)), dim=-1)


def info_nce(anchor, pos, temperature=0.07):
    """in-batch negatives 对比损失"""
    logits = anchor @ pos.t() / temperature
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, labels)


def train_contrastive(model, t_emb, i_emb, pairs, epochs, bs, lr, dev, tag="train",
                      on_epoch_end=None):
    """共现对上的对称 InfoNCE 训练。gate / mlp 两个监督模式共用。

    on_epoch_end(epoch, model, et, ei) 每轮结束回调——用来在**留出对**上打点，
    一次长跑就能拿到完整的 epoch-性能曲线，不必为每个 epoch 数单独重训。
    """
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    et = torch.from_numpy(t_emb).to(dev)
    ei = torch.from_numpy(i_emb).to(dev)
    idx = torch.from_numpy(pairs)
    n = len(idx)
    curve = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for s in range(0, n, bs):
            batch = idx[perm[s: s + bs]].to(dev)
            a, b = batch[:, 0], batch[:, 1]
            ea = model(et[a], ei[a])
            eb = model(et[b], ei[b])
            loss = info_nce(ea, eb) + info_nce(eb, ea)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        print(f"  [{tag}] epoch {ep+1}/{epochs}  loss={tot/max(nb,1):.4f}", flush=True)
        if on_epoch_end is not None:
            rec = on_epoch_end(ep + 1, model, et, ei)
            if rec:
                curve.append({"epoch": ep + 1, "loss": round(tot / max(nb, 1), 4), **rec})
    return model, et, ei, curve


def make_holdout_eval(t_emb, i_emb, eval_pairs, n_query=2000, seed=7, k_list=(10, 50, 100)):
    """构造"每轮在留出共现对上打点"的回调。返回 (callback, meta)。"""
    pos_map = {}
    for a, b in eval_pairs:
        pos_map.setdefault(int(a), set()).add(int(b))
        pos_map.setdefault(int(b), set()).add(int(a))
    cand = np.array(sorted(pos_map), dtype=np.int64)
    rng = np.random.RandomState(seed)
    qs = rng.choice(cand, size=min(n_query, len(cand)), replace=False)
    gold = np.zeros((len(qs), t_emb.shape[0]), dtype=bool)
    for r, item in enumerate(qs):
        for p in pos_map[int(item)]:
            if p < t_emb.shape[0]:
                gold[r, p] = True
    meta = {"eval_pairs": int(len(eval_pairs)), "n_query": int(len(qs)),
            "avg_pos": float(gold.sum(axis=1).mean())}

    def cb(ep, model, et, ei):
        V = encode_all(model, et, ei)
        sim = V[qs] @ V.T
        sim[np.arange(len(qs)), qs] = -1e9          # 排除自身
        order = np.argsort(-sim, axis=1)[:, : max(k_list)]
        rec = {f"recall@{k}": round(float(np.take_along_axis(gold, order[:, :k], axis=1)
                                          .any(axis=1).mean()), 5) for k in k_list}
        print("    [holdout] " + "  ".join(f"{k}={v:.4f}" for k, v in rec.items()), flush=True)
        return rec

    return cb, meta


def encode_all(model, et, ei, chunk=4096):
    with torch.inference_mode():
        out = [model(et[s: s + chunk], ei[s: s + chunk]).float().cpu().numpy()
               for s in range(0, et.shape[0], chunk)]
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #

def cross_modal_align(t_emb, i_emb, mask, n_query=3000, seed=42, lam=1e-2):
    """跨模态对齐度：**不能**直接 t_emb @ i_emb.T。

    text 是 1024 维、image 是 768 维，分属两个独立的嵌入空间，点积没有定义
    （实测会直接抛 broadcast/matmul 维度错）。语义对齐是"能不能线性地互相还原"，
    所以先在拟合子集上解一个岭回归映射，再到留出子集上算 recall@k。
    报告的是**线性可对齐程度**，是 CLIP/SigLIP 式对齐质量的合理代理指标。
    """
    have = np.where(mask)[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(have))
    half = len(have) // 2
    fit, tst = have[perm[:half]], have[perm[half:]]

    def ridge_recall(src, dst, src_fit, src_tst, d_src, d_dst):
        Sf, Df = src[src_fit], dst[src_fit]
        St, Dt = src[src_tst], dst[src_tst]
        A = Sf.T @ Sf + lam * np.eye(d_src, dtype=np.float32)   # 岭正则，防奇异
        B = Sf.T @ Df
        if torch.cuda.is_available():
            W = torch.linalg.solve(torch.from_numpy(A).cuda(),
                                   torch.from_numpy(B).cuda()).cpu().numpy()
        else:
            W = np.linalg.solve(A.astype(np.float64), B.astype(np.float64)).astype(np.float32)
        P = l2n(St @ W)                       # 映射到目标空间再归一化
        Dn = l2n(Dt)
        sim = P @ Dn.T                        # 同空间，点积合法
        order = np.argsort(-sim, axis=1)[:, :10]
        gold = np.arange(len(St))[:, None]
        return {f"recall@{k}": float((order[:, :k] == gold).any(axis=1).mean())
                for k in (1, 5, 10)}

    q_t, q_i = tst[:n_query], tst[:n_query]
    return {
        "text_to_image": ridge_recall(t_emb, i_emb, fit, q_t, t_emb.shape[1], i_emb.shape[1]),
        "image_to_text": ridge_recall(i_emb, t_emb, fit, q_i, i_emb.shape[1], t_emb.shape[1]),
        "n_fit": int(len(fit)), "n_test": int(len(q_t)),
        "note": "先岭回归学线性映射再检索；跨空间直接点积无定义",
    }


def recall_at_k(sim, k_list, valid_q, valid_gold, exclude_self=True):
    """sim: (Q, N)；valid_gold: (Q, N) bool（True=命中）"""
    out = {}
    order = np.argsort(-sim, axis=1)[:, : max(k_list)]
    for k in k_list:
        topk = order[:, :k]
        hit = np.take_along_axis(valid_gold, topk, axis=1).any(axis=1)
        out[f"recall@{k}"] = float(hit.mean())
    return out


def _cooc_block(vecs, names, t_emb, pairs, n_query, rng):
    """在给定共现对上算 recall@k + 随机基线。"""
    pos_map = {}
    for a, b in pairs:
        pos_map.setdefault(int(a), set()).add(int(b))
        pos_map.setdefault(int(b), set()).add(int(a))
    cand = np.array(sorted(k for k in pos_map if k < t_emb.shape[0]))
    qs = rng.choice(cand, size=min(n_query, len(cand)), replace=False)
    gold = np.zeros((len(qs), t_emb.shape[0]), dtype=bool)
    for r, item in enumerate(qs):
        for p in pos_map[item]:
            if p < t_emb.shape[0]:
                gold[r, p] = True
    out = {}
    for name in names:
        v = vecs[name]
        sim = v[qs] @ v.T
        sim[np.arange(len(qs)), qs] = -1e9          # 排除自身
        out[name] = recall_at_k(sim, [10, 50, 100], len(qs), gold)
    avg_pos = float(gold.sum(axis=1).mean())
    for k in (10, 50, 100):
        out.setdefault("random_baseline", {})[f"recall@{k}"] = \
            round(min(1.0, avg_pos * k / t_emb.shape[0]), 6)
    out["_meta"] = {"n_query": int(len(qs)), "avg_pos": round(avg_pos, 4)}
    return out


def eval_report(vecs, names, t_emb, i_emb, mask, eval_pairs, n_query=3000, seed=42,
                train_pairs=None):
    """vecs: {name: (N,d)}；返回 JSON 可序列化的评估报告。

    eval_pairs 必须与训练对**不重叠**（见 split_pairs），否则召回率会随训练轮数
    单调虚高（背训练对）。若传入 train_pairs，额外报告训练对指标以量化记忆 gap。
    """
    rng = np.random.RandomState(seed)
    rep = {"n_items": int(t_emb.shape[0]), "n_eval_pairs": int(len(eval_pairs))}

    # --- A) 模态互检索（先学线性映射，跨空间点积无定义）---
    rep["modality_alignment"] = cross_modal_align(t_emb, i_emb, mask, n_query, seed)

    # --- B) 共现检索：留出对为主指标 ---
    if len(eval_pairs):
        sep = np.random.RandomState(seed)           # query 抽样用固定种子，两组可比
        rep["cooccurrence_retrieval_holdout"] = _cooc_block(
            vecs, names, t_emb, eval_pairs, n_query, sep)
    if train_pairs is not None and len(train_pairs):
        sep = np.random.RandomState(seed)
        rep["cooccurrence_retrieval_trainpairs"] = _cooc_block(
            vecs, names, t_emb, train_pairs, n_query, sep)
        rep["n_train_pairs"] = int(len(train_pairs))
    return rep


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="IandS")
    ap.add_argument("--data_dir", default="data/Amazon23")
    ap.add_argument("--mode", default="concat", choices=["text", "concat", "mlp", "gate"])
    ap.add_argument("--text_emb", default="", help="默认自动找 emb/ 下第一个 emb_text_*.npy")
    ap.add_argument("--target_dim", type=int, default=1024, help="降维目标（0=不降）")
    ap.add_argument("--hidden", type=int, default=512, help="mlp/gate 隐层宽度")
    ap.add_argument("--dropout", type=float, default=0.1, help="mlp 模式 dropout")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_pairs", type=int, default=0,
                    help="训练对上限（0=用全部剩余对；I&S 全池 80万、VG 186万）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["text", "concat", "mlp", "gate"])
    ap.add_argument("--n_query", type=int, default=3000)
    ap.add_argument("--out_name", default="", help="覆盖产物文件名后缀（默认=--mode），用于超参扫描不互相覆盖")
    ap.add_argument("--out_dir", default="", help="产物输出目录（默认 = 数据集的 emb/）")
    ap.add_argument("--eval_frac", type=float, default=0.1,
                    help="共现对池中划作**留出评估**的比例，与训练对不重叠（默认 0.1）")
    ap.add_argument("--eval_every", type=int, default=0,
                    help="每 N 轮在留出对上打点并记录曲线（0=不打点）。一次长跑即可拿到 scaling curve")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    root = os.path.join(args.data_dir, args.short)
    emb_dir = os.path.join(root, "emb")

    t_path = args.text_emb
    if not t_path:
        cands = sorted(f for f in os.listdir(emb_dir) if f.startswith("emb_text_") and f.endswith(".npy"))
        if not cands:
            raise SystemExit(f"未找到文本向量，请先跑 encode_text.py（{emb_dir}）")
        t_path = os.path.join(emb_dir, cands[0])
    t_emb = np.load(t_path).astype(np.float32)
    i_emb = np.load(os.path.join(emb_dir, "emb_image_siglip.npy")).astype(np.float32)
    mask = np.load(os.path.join(emb_dir, "emb_image_mask.npy"))
    print(f"[load] text={os.path.basename(t_path)} {t_emb.shape}  image={i_emb.shape}  "
          f"有图={mask.sum():,}/{len(mask):,}")

    def save(vec, name):
        name = args.out_name or name
        out_dir = args.out_dir or emb_dir
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"emb_fused_{name}.npy")
        np.save(p, vec.astype(np.float32))
        print(f"[save] {p}  shape={vec.shape}")
        return p

    if args.eval:
        names = [m for m in args.modes]
        vec_dir = args.out_dir or emb_dir
        vecs = {}
        for m in names:
            p = os.path.join(vec_dir, f"emb_fused_{m}.npy")
            if os.path.exists(p):
                vecs[m] = np.load(p).astype(np.float32)
            else:
                print(f"[skip] {m}: 缺 {p}")
        tr_pairs, ev_pairs = split_pairs(os.path.join(root, f"{args.short}.train.inter"),
                                         args.max_pairs, args.eval_frac, args.seed)
        print(f"[eval] 留出对={len(ev_pairs):,}  训练对={len(tr_pairs):,}  query={args.n_query}")
        rep = eval_report(vecs, list(vecs), t_emb, i_emb, mask, ev_pairs, args.n_query,
                          args.seed, train_pairs=tr_pairs)
        rep["vectors"] = {k: list(v.shape) for k, v in vecs.items()}
        rep["text_emb_file"] = os.path.basename(t_path)
        out = os.path.join(vec_dir, "fusion_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        print(f"[done] {out}")
        return

    if args.mode == "text":
        save(l2n(t_emb), "text")

    elif args.mode == "concat":
        cat = np.concatenate([t_emb, i_emb], axis=1)
        if args.target_dim and args.target_dim < cat.shape[1]:
            # PCA 白化（在拼接空间上做，等价于学到一组线性融合权重）
            t0 = time.time()
            mu = cat.mean(axis=0, keepdims=True)
            x = cat - mu
            # 25k×1792 的协方差 + 特征分解：CPU 单线程 BLAS 要 200+s，优先走 GPU
            if torch.cuda.is_available():
                try:
                    xt = torch.from_numpy(x).cuda()
                    cov = (xt.T @ xt) / (x.shape[0] - 1)
                    w, v = torch.linalg.eigh(cov.double())
                    v = torch.flip(v, dims=[1])[:, : args.target_dim]
                    cat = (xt @ v.float()).cpu().numpy()
                    dev_tag = "GPU"
                except RuntimeError as e:                       # 显存不足则回落 CPU
                    print(f"[concat] GPU 特征分解失败({e})，回落 CPU")
                    w, v = np.linalg.eigh(((x.T @ x) / (x.shape[0] - 1)).astype(np.float64))
                    cat = x @ v[:, ::-1][:, : args.target_dim]
                    dev_tag = "CPU"
            else:
                w, v = np.linalg.eigh(((x.T @ x) / (x.shape[0] - 1)).astype(np.float64))
                cat = x @ v[:, ::-1][:, : args.target_dim]
                dev_tag = "CPU"
            print(f"[concat] PCA 白化 {t_emb.shape[1]}+{i_emb.shape[1]} -> {cat.shape[1]} "
                  f"[{dev_tag}] ({time.time()-t0:.1f}s)")
        save(l2n(cat), "concat")

    elif args.mode in ("mlp", "gate"):
        pairs, ev_pairs = split_pairs(os.path.join(root, f"{args.short}.train.inter"),
                                     args.max_pairs, args.eval_frac, args.seed)
        print(f"[{args.mode}] 训练对={len(pairs):,}  留出对={len(ev_pairs):,}")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        d_out = args.target_dim or t_emb.shape[1]
        if args.mode == "mlp":
            model = MLPFusion(t_emb.shape[1], i_emb.shape[1], d_out,
                              hidden=args.hidden, dropout=args.dropout)
        else:
            model = GateFusion(t_emb.shape[1], i_emb.shape[1], d_out, hidden=args.hidden)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[{args.mode}] 参数量={n_par:,}  device={dev}  d_out={d_out}")

        cb, meta = (None, None)
        if args.eval_every > 0 and len(ev_pairs):
            cb, meta = make_holdout_eval(t_emb, i_emb, ev_pairs)
            print(f"[{args.mode}] 每 {args.eval_every} 轮在留出对上打点 "
                  f"(query={meta['n_query']}, 平均正样本={meta['avg_pos']:.2f})")

        def on_end(ep, m, e_t, e_i):
            if cb is None or ep % args.eval_every:
                return None
            return cb(ep, m, e_t, e_i)

        model, et, ei, curve = train_contrastive(model, t_emb, i_emb, pairs,
                                                 args.epochs, args.bs, args.lr, dev,
                                                 tag=args.mode, on_epoch_end=on_end)
        save(encode_all(model, et, ei), args.mode)
        if curve:
            out_dir = args.out_dir or emb_dir
            os.makedirs(out_dir, exist_ok=True)
            cp = os.path.join(out_dir, f"fusion_curve_{args.out_name or args.mode}.json")
            with open(cp, "w", encoding="utf-8") as f:
                json.dump({"mode": args.mode, "epochs": args.epochs, "bs": args.bs,
                           "lr": args.lr, "max_pairs": args.max_pairs,
                           "eval_frac": args.eval_frac, "holdout_meta": meta,
                           "curve": curve}, f, ensure_ascii=False, indent=2)
            print(f"[curve] {cp}")

    print("[done]")


if __name__ == "__main__":
    main()
