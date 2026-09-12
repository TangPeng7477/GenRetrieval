#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2-4.5: SID 质量评估三件套（uniqueness / fidelity / retrieval structure）
=========================================================================
迁移自 esci-ai-search 项目，适配本项目（Amazon 2023 双域 + 多模态融合向量）。

三层指标（对应 UPGRADE_PLAN §4.5）
----------------------------------
1. uniqueness   : SID 是否唯一 / 是否死码
   * ICR          唯一码率 = #distinct codes / N （越高越好，1.0 = 完全无碰撞）
   * collision    碰撞率 = 1 - ICR；并给出最大冲突组大小、冲突物品数
   * entropy      每层 codebook 使用熵（归一化到 [0,1]）、perplexity、死码率
2. fidelity     : 量化损失（需要 --ckpt 才能算；无 ckpt 则跳过）
   * recon MSE / R² / cosine（在**输入空间**度量，即 Dec(Σ_l C_l[code_l]) vs x）
   * 逐层累积 MSE：只用前 k 层重建的 MSE —— 直接回答"2/3/4 层选哪个"
3. structure    : SID 前缀是否真的承载语义（不依赖下游 SFT）
   * LCP           最近邻对的公共前缀长度 vs 随机对的公共前缀长度
                   lcp_gain = nn - random，lcp_ratio = nn / random
   * prefix hit    前缀长度 ≥ l 的命中率曲线（nn vs random）
   * cohesion[l]   按 l-前缀分组后组内平均余弦相似度 vs 随机分组（置换检验）
                   cohesion_ratio = 真实 / 随机 —— 与 esci 的 "prefix cohesion ×1.58~2.29" 同定义

为什么必须先过这三件套
----------------------
SID 是 SFT/RL 的输入。SID 一变，下游全部数字不可比。三件套是**秒级离线代理指标**，
用来在昂贵的 SFT 之前筛掉劣质 SID（V0 的 P6 教训：只靠端到端盲判，成本高且归因不清）。

用法
----
    # 只评 uniqueness + structure（不需要模型）
    python rq/eval_sid.py --short IandS --codes data/Amazon23/IandS/IandS.index.npy \
        --emb data/Amazon23/IandS/emb/emb_fused_gate.npy --out data/Amazon23/IandS/sid_eval_gate.json

    # 全量三件套（带 ckpt 才能算 fidelity）
    python rq/eval_sid.py --short IandS --codes data/Amazon23/IandS/IandS.index.npy \
        --emb data/Amazon23/IandS/emb/emb_fused_gate.npy \
        --ckpt rq/index/output/IandS/<ts>/best_collision_model.pth --out ...

    # 多方案对比（一次跑完，输出对比表）
    python rq/eval_sid.py --compare text=...npy,concat=...npy,gate=...npy ...
"""

import argparse
import json
import os
import re
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# 读取 SID codes：支持 .npy (N,L) 与 MiniOneRec 的 .index.json
# --------------------------------------------------------------------------- #

_PREFIX_RE = re.compile(r"^<([a-z])_(\d+)>$")


def load_codes(path):
    """返回 (codes: int ndarray (N,L), meta: dict)"""
    if path.endswith(".npy"):
        codes = np.load(path)
        if codes.ndim != 2:
            raise SystemExit(f"[codes] .npy 必须是 (N, L) 二维，实际 {codes.shape}")
        return codes.astype(np.int64), {"format": "npy", "n_items": int(codes.shape[0])}

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # MiniOneRec: {"0": ["<a_12>", "<b_3>", "<c_7>"], ...}  (也可能是 <a_12><b_3>... 的字符串列表)
    items = sorted(obj.items(), key=lambda kv: int(kv[0]))
    rows = []
    for _k, v in items:
        if isinstance(v, str):
            toks = re.findall(r"<[a-z]_\d+>", v)
        else:
            toks = [t if isinstance(t, str) else str(t) for t in v]
        row = []
        for t in toks:
            m = _PREFIX_RE.match(t.strip())
            if m:
                row.append(int(m.group(2)))
            else:                                    # 兼容纯数字 token
                row.append(int(re.sub(r"\D", "", t) or 0))
        rows.append(row)
    L = max(len(r) for r in rows)
    codes = np.zeros((len(rows), L), dtype=np.int64)
    for i, r in enumerate(rows):
        codes[i, : len(r)] = r
    return codes, {"format": "index.json", "n_items": len(rows), "n_layers": L}


# --------------------------------------------------------------------------- #
# 1) uniqueness
# --------------------------------------------------------------------------- #

def uniqueness_metrics(codes, n_emb_list=None):
    N, L = codes.shape
    keys = [tuple(r) for r in codes.tolist()]
    uniq = len(set(keys))
    icr = uniq / N

    # 冲突统计
    from collections import Counter
    cnt = Counter(keys)
    sizes = np.array(list(cnt.values()))
    collided_codes = int((sizes > 1).sum())
    collided_items = int(sizes[sizes > 1].sum())

    rep = {
        "n_items": int(N),
        "n_layers": int(L),
        "n_unique_codes": int(uniq),
        "icr": round(icr, 6),
        "collision_rate": round(1.0 - icr, 6),
        "n_collided_codes": collided_codes,
        "n_collided_items": collided_items,
        "max_code_frequency": int(sizes.max()) if len(sizes) else 0,
    }

    # per-layer 使用情况
    per_layer = []
    for l in range(L):
        col = codes[:, l]
        K = int(n_emb_list[l]) if n_emb_list and l < len(n_emb_list) else int(col.max() + 1)
        used, counts = np.unique(col, return_counts=True)
        p = counts / counts.sum()
        H = float(-(p * np.log(p)).sum())
        H_norm = H / np.log(K) if K > 1 else 0.0
        per_layer.append({
            "layer": l,
            "codebook_size": K,
            "n_used": int(len(used)),
            "dead_code_rate": round(1.0 - len(used) / K, 6),
            "entropy": round(H, 6),
            "entropy_norm": round(H_norm, 6),
            "perplexity": round(float(np.exp(H)), 3),
            "max_usage": int(counts.max()) if len(counts) else 0,
            "usage_p50": int(np.percentile(counts, 50)) if len(counts) else 0,
        })
    rep["per_layer"] = per_layer
    return rep


# --------------------------------------------------------------------------- #
# 2) fidelity（需要 ckpt）
# --------------------------------------------------------------------------- #

def load_rqvae(ckpt_path, in_dim, device="cpu"):
    """从 ckpt 重建 RQVAE（复用 rq/ 下的实现）"""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import torch
    from models.rqvae import RQVAE                    # noqa: E402

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = RQVAE(
        in_dim=in_dim,
        num_emb_list=a.num_emb_list, e_dim=a.e_dim, layers=a.layers,
        dropout_prob=getattr(a, "dropout_prob", 0.0), bn=getattr(a, "bn", False),
        loss_type=getattr(a, "loss_type", "mse"),
        quant_loss_weight=getattr(a, "quant_loss_weight", 1.0),
        beta=getattr(a, "beta", 0.25),
        kmeans_init=getattr(a, "kmeans_init", True),
        kmeans_iters=getattr(a, "kmeans_iters", 100),
        sk_epsilons=getattr(a, "sk_epsilons", [0.0] * len(a.num_emb_list)),
        sk_iters=getattr(a, "sk_iters", 50),
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    return model, a


def fidelity_metrics(codes, emb, ckpt_path, device="cpu", batch=2048):
    """重建保真度。

    报告两组数：
    A) 保存的 codes（即真正会进 SFT 的那份）的逐层累积重建 —— "实际交付的保真度"
    B) 纯 argmin codes（模型原生前向）的逐层累积重建 —— "量化器理论上限"
    A 与 B 在全层深度上的差 = **碰撞消解带来的保真度代价**（Sinkhorn 均衡会让物品
    偏离最近码，唯一性↑ 但保真度↓；码本容量不足时这个代价会非常大）。
    """
    import torch

    model, _args = load_rqvae(ckpt_path, emb.shape[1], device)
    codes = np.asarray(codes)
    L = codes.shape[1]                                   # 层数
    x = torch.from_numpy(emb.astype(np.float32)).to(device)
    N = x.shape[0]
    var = float(x.var().item())
    codes_t = torch.from_numpy(codes.astype(np.int64)).to(device)

    with torch.no_grad():                                # 不用 inference_mode：
        # inference tensor 不能作为 Parameter 的索引（会报 "cannot be saved for backward"）
        cb = model.rq.get_codebook().to(device)          # (L, K, e_dim)
        codes_arg = model.get_indices(x, use_sk=False).view(-1, L)

    def cum(recon_mse_list, cs):
        zq = torch.zeros((N, cb.shape[-1]), device=device)
        for k in range(L):
            zq = zq + cb[k][cs[:, k]]
            se = 0.0
            cos = 0.0
            for s in range(0, N, batch):
                r = model.decoder(zq[s: s + batch])
                se += float(((r - x[s: s + batch]) ** 2).sum().item())
                cos += float(torch.nn.functional.cosine_similarity(
                    r, x[s: s + batch], dim=1).sum().item())
            mse = se / (N * x.shape[1])
            recon_mse_list.append({
                "n_layers_used": k + 1,
                "recon_mse": round(mse, 8),
                "recon_rmse": round(mse ** 0.5, 6),
                "r2": round(1.0 - mse / var, 6) if var > 0 else None,
                "cosine": round(cos / N, 6),
            })

    saved, upper = [], []
    with torch.no_grad():
        cum(saved, codes_t)
        cum(upper, codes_arg)

        # codes 一致率（碰撞消解改动了多少码）
        changed = (codes_t != codes_arg)
        n_changed = float(changed.any(dim=1).float().mean())

    out = {
        "n_layers_total": int(L),
        "input_var": round(var, 8),
        "cumulative": saved,                 # A) 实际交付
        "cumulative_argmin": upper,          # B) 理论上限
        "codes_changed_by_collision_resolution": round(n_changed, 6),
    }
    if saved and upper:
        out["fidelity_cost_of_collision_resolution"] = round(
            saved[-1]["recon_mse"] - upper[-1]["recon_mse"], 8)
        out["final"] = saved[-1]
        out["final_argmin"] = upper[-1]
    return out


# --------------------------------------------------------------------------- #
# 3) retrieval structure
# --------------------------------------------------------------------------- #

def _nn_pairs(emb, n_query, seed=42):
    """返回 (q_idx, nn_idx)：每个 query 的最近邻（排除自身）"""
    rng = np.random.RandomState(seed)
    N = emb.shape[0]
    q = rng.choice(N, size=min(n_query, N), replace=False)
    try:
        import faiss
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb.astype(np.float32))
        _, I = idx.search(emb[q].astype(np.float32), 2)
        nn = I[:, 1]
    except Exception:                                  # noqa: BLE001
        nn = np.empty(len(q), dtype=np.int64)
        for s in range(0, len(q), 512):
            blk = q[s: s + 512]
            sim = emb[blk] @ emb.T
            sim[np.arange(len(blk)), blk] = -1e9
            nn[s: s + len(blk)] = sim.argmax(axis=1)
    return q.astype(np.int64), nn.astype(np.int64)


def _lcp_vec(codes, i_idx, j_idx):
    """向量化 LCP：eq 沿层累积相乘，和 = 公共前缀长度"""
    eq = codes[i_idx] == codes[j_idx]                  # (P, L)
    return np.cumprod(eq, axis=1).sum(axis=1)


def structure_metrics(codes, emb, n_query=3000, n_pairs=200000, n_perm=5,
                      seed=42, prefix_hit_k=5):
    N, L = codes.shape
    rng = np.random.RandomState(seed)
    rep = {}

    # ---- LCP: 最近邻对 vs 随机对 ----
    q, nn = _nn_pairs(emb, n_query, seed)
    lcp_nn = _lcp_vec(codes, q, nn)

    ri = rng.randint(0, N, size=n_pairs)
    rj = rng.randint(0, N, size=n_pairs)
    keep = ri != rj
    lcp_rand = _lcp_vec(codes, ri[keep], rj[keep])

    rep["lcp"] = {
        "L": int(L),
        "nn_mean": round(float(lcp_nn.mean()), 6),
        "random_mean": round(float(lcp_rand.mean()), 6),
        "nn_mean_norm": round(float(lcp_nn.mean() / L), 6),
        "lcp_gain": round(float(lcp_nn.mean() - lcp_rand.mean()), 6),
        "lcp_ratio": round(float(lcp_nn.mean() / max(lcp_rand.mean(), 1e-9)), 6),
        "n_query": int(len(q)),
        "n_random_pairs": int(keep.sum()),
    }

    # ---- prefix hit rate 曲线：P(prefix >= l) ----
    hit = []
    for l in range(1, L + 1):
        hn = float((lcp_nn >= l).mean())
        hr = float((lcp_rand >= l).mean())
        hit.append({"prefix_len": l, "nn_hit": round(hn, 6),
                    "random_hit": round(hr, 6),
                    "gain": round(hn - hr, 6),
                    "ratio": round(hn / hr, 6) if hr > 0 else None})
    rep["prefix_hit"] = hit

    # ---- cohesion：按 l-前缀分组的组内平均余弦相似度 vs 随机置换 ----
    def cohesion(labels, n_groups_hint=None):
        """labels: (N,) int；组内平均 cos sim（利用 ||Σv||² 恒等式，向量已 L2 归一化）"""
        uniq, inv = np.unique(labels, return_inverse=True)
        G = len(uniq)
        S = np.zeros((G, emb.shape[1]), dtype=np.float64)
        np.add.at(S, inv, emb)
        n = np.bincount(inv, minlength=G).astype(np.float64)
        num = (S ** 2).sum(axis=1) - n                 # 对角项减去（||v||=1）
        den = n * (n - 1.0)
        ok = den > 0
        if not ok.any():
            return float("nan")
        return float((num[ok] / den[ok] * n[ok]).sum() / n[ok].sum())

    coh = []
    # 各层前缀标签
    labels_per_level = []
    acc = np.zeros(N, dtype=np.int64)
    for l in range(L):
        acc = acc * 1000 + codes[:, l]                 # 唯一化前缀（K<=1000 足够）
        labels_per_level.append(acc.copy())

    # 注意：前缀层基数大时（如第 L 层接近唯一码），每组只有 1~2 个样本，
    # "组内平均相似度"退化成有限样本噪声 -> 必须报告组大小并标记可信度。
    # 判据：avg_group_size ∈ [3, N/10] 视为可信（太小=噪声，太大=退化成全局均值）。
    for l in range(L):
        lab = labels_per_level[l]
        true_c = cohesion(lab)
        nulls = []
        for p in range(n_perm):
            perm = np.random.RandomState(seed + 100 + p).permutation(N)
            nulls.append(cohesion(lab[perm]))
        # ICR=1.0 时第 3 层每组仅 1 个物品，cohesion 返回 nan（组内无样本对）。
        # 直接 nanmean 会触发 "Mean of empty slice" 警告并静默产出 nan，这里显式处理。
        nulls = [x for x in nulls if np.isfinite(x)]
        null_mean = float(np.mean(nulls)) if nulls else float("nan")
        g = np.bincount(np.unique(lab, return_inverse=True)[1])
        avg_gs = float(g.mean())
        pair_cov = float(g[g > 1].sum() / N)           # 落在"有多样本"组里的物品占比
        reliable = (3.0 <= avg_gs <= N / 10.0) and pair_cov > 0.5
        coh.append({
            "prefix_len": l + 1,
            "cohesion_true": round(true_c, 6),
            "cohesion_random": round(null_mean, 6),
            "cohesion_ratio": round(true_c / null_mean, 6) if null_mean > 1e-9 else None,
            "n_groups": int(len(g)),
            "avg_group_size": round(avg_gs, 2),
            "pair_coverage": round(pair_cov, 4),
            "reliable": bool(reliable),
        })
    rep["cohesion"] = coh

    # ---- 语义邻居在 top-k 里前缀命中（更接近"检索友好度"）----
    try:
        import faiss
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb.astype(np.float32))
        qk = rng.choice(N, size=min(n_query, N), replace=False)
        _, I = idx.search(emb[qk].astype(np.float32), prefix_hit_k + 1)
        topk = I[:, 1:]
        first_tok_hit = (codes[topk][:, :, 0:1] == codes[qk][:, None, 0:1])   # (Q,k)
        rep["semantic_topk_prefix_hit"] = {
            "k": int(prefix_hit_k),
            "first_token_match_rate": round(float(first_tok_hit.mean()), 6),
            "any_layer_match_rate": round(float(
                (codes[topk] == codes[qk][:, None, :]).all(axis=2).mean()), 6),
        }
    except Exception:                                  # noqa: BLE001
        pass

    return rep


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def pretty(rep, title):
    L = rep.get("uniqueness", {})
    print(f"\n{'='*74}\n {title}\n{'='*74}")
    if L:
        print(f"  [uniqueness]  N={L['n_items']:,}  L={L['n_layers']}  "
              f"ICR={L['icr']:.4f}  碰撞率={L['collision_rate']:.4f}  "
              f"最大冲突组={L['max_code_frequency']}")
        for p in L["per_layer"]:
            print(f"      layer{p['layer']}: used={p['n_used']:>5}/{p['codebook_size']:<5} "
                  f"dead={p['dead_code_rate']:.3f}  H_norm={p['entropy_norm']:.4f}  "
                  f"ppl={p['perplexity']:.1f}  max_use={p['max_usage']}")
    F = rep.get("fidelity")
    if F:
        print("  [fidelity]  A) 实际交付 codes 的逐层累积重建:")
        for c in F.get("cumulative", []):
            print(f"      {c['n_layers_used']} 层: MSE={c['recon_mse']:.6f}  "
                  f"R²={c['r2']:.4f}  cos={c['cosine']:.4f}")
        print("             B) 纯 argmin codes（量化器理论上限）:")
        for c in F.get("cumulative_argmin", []):
            print(f"      {c['n_layers_used']} 层: MSE={c['recon_mse']:.6f}  "
                  f"R²={c['r2']:.4f}  cos={c['cosine']:.4f}")
        print(f"             碰撞消解改动码比例={F.get('codes_changed_by_collision_resolution')}  "
              f"保真度代价 ΔMSE={F.get('fidelity_cost_of_collision_resolution')}")
    S = rep.get("structure")
    if S:
        lc = S.get("lcp", {})
        print(f"  [structure] LCP: nn={lc.get('nn_mean'):.4f} "
              f"random={lc.get('random_mean'):.4f} gain={lc.get('lcp_gain'):.4f} "
              f"ratio={lc.get('lcp_ratio'):.3f}  (norm {lc.get('nn_mean_norm'):.4f})")
        for c in S.get("cohesion", []):
            flag = "" if c.get("reliable") else "  [样本不足/退化，不可信]"
            print(f"      prefix {c['prefix_len']}: cohesion={c['cohesion_true']:.4f} "
                  f"random={c['cohesion_random']:.4f} ratio={c['cohesion_ratio']} "
                  f"(组数={c['n_groups']}, 均组大小={c['avg_group_size']}){flag}")
        st = S.get("semantic_topk_prefix_hit")
        if st:
            print(f"      top-{st['k']} 语义邻居: 首 token 命中={st['first_token_match_rate']:.4f} "
                  f"全匹配={st['any_layer_match_rate']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="IandS")
    ap.add_argument("--codes", default="", help="SID codes: .npy (N,L) 或 .index.json")
    ap.add_argument("--emb", default="", help="建 SID 用的输入向量 .npy（评估 structure/fidelity 必需）")
    ap.add_argument("--ckpt", default="", help="RQ-VAE ckpt，给了才评 fidelity")
    ap.add_argument("--out", default="", help="报告 json 输出路径")
    ap.add_argument("--num_emb_list", type=int, nargs="+", default=None)
    ap.add_argument("--n_query", type=int, default=3000)
    ap.add_argument("--n_pairs", type=int, default=200000)
    ap.add_argument("--n_perm", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", default="",
                    help="多方案对比，格式 name1=codes1=emb1,name2=..., 与 --ckpt 共用")
    # 便捷：默认按 short 推断路径
    ap.add_argument("--fused", default="", help="便捷：直接给融合向量名，自动找 codes/emb")
    args = ap.parse_args()

    root = os.path.join("data", "Amazon23", args.short)

    jobs = []
    if args.compare:
        for item in args.compare.split(","):
            parts = [p.strip() for p in item.split("=")]
            name = parts[0]
            codes = parts[1] if len(parts) > 1 and parts[1] else \
                os.path.join(root, f"{args.short}.index_{name}.npy")
            emb = parts[2] if len(parts) > 2 and parts[2] else \
                os.path.join(root, "emb", f"emb_fused_{name}.npy")
            jobs.append((name, codes, emb))
    else:
        codes = args.codes or os.path.join(root, f"{args.short}.index.npy")
        emb = args.emb or os.path.join(root, "emb", "emb_fused_gate.npy")
        jobs.append((args.short, codes, emb))

    all_rep = {}
    for name, codes_path, emb_path in jobs:
        print(f"\n>>> {name}: codes={codes_path}")
        codes, cmeta = load_codes(codes_path)
        emb = None
        if emb_path and os.path.exists(emb_path):
            emb = np.load(emb_path).astype(np.float32)
            n = np.linalg.norm(emb, axis=1, keepdims=True)
            n[n == 0] = 1.0
            emb = emb / n                              # structure 指标要求 L2 归一化
            if emb.shape[0] != codes.shape[0]:
                raise SystemExit(f"[dim] emb {emb.shape} 与 codes {codes.shape} 行数不一致")
        elif emb_path:
            print(f"    [warn] 缺 emb: {emb_path} -> 跳过 structure/fidelity")

        rep = {"name": name, "codes_file": codes_path, "codes_meta": cmeta,
               "emb_file": emb_path if emb is not None else None}
        rep["uniqueness"] = uniqueness_metrics(codes, args.num_emb_list)
        if emb is not None:
            rep["structure"] = structure_metrics(
                codes, emb, args.n_query, args.n_pairs, args.n_perm)
            if args.ckpt and os.path.exists(args.ckpt):
                try:
                    rep["fidelity"] = fidelity_metrics(codes, emb, args.ckpt, args.device)
                except Exception as e:                 # noqa: BLE001
                    print(f"    [warn] fidelity 失败: {type(e).__name__}: {e}")
        pretty(rep, f"{name}  ({args.short})")
        all_rep[name] = rep

    # 对比表
    if len(all_rep) > 1:
        print(f"\n{'='*74}\n 对比汇总\n{'='*74}")
        hdr = f"{'方案':<12}{'ICR':>8}{'dead1':>8}{'LCP_nn':>9}{'LCP_rnd':>9}{'ratio':>8}{'coh1':>8}"
        print(hdr)
        for name, r in all_rep.items():
            u, s = r["uniqueness"], r.get("structure", {})
            lc = s.get("lcp", {})
            coh = s.get("cohesion", [{}])
            print(f"{name:<12}{u['icr']:>8.4f}"
                  f"{u['per_layer'][0]['dead_code_rate']:>8.3f}"
                  f"{lc.get('nn_mean', float('nan')):>9.4f}"
                  f"{lc.get('random_mean', float('nan')):>9.4f}"
                  f"{lc.get('lcp_ratio', float('nan')):>8.3f}"
                  f"{coh[0].get('cohesion_ratio', float('nan')):>8.3f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_rep, f, ensure_ascii=False, indent=2)
        print(f"\n[done] 报告写入 {args.out}")


if __name__ == "__main__":
    main()
