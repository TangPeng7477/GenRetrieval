#!/usr/bin/env python
"""「图文向量映射到同一语义空间」的对齐方法横向对比 + 贡献正交拆解。

背景：现管线用 `fuse_embeddings.py::cross_modal_align` 的**岭回归**线性映射
（1024 维文本 -> 768 维图像），I&S 上 text->image R@10 = 0.6217。
本脚本在同一份 holdout 划分（seed=42, n_query=3000）上把「降维 / 白化 /
映射族 / 检索空间 / 正则强度」五个因素**正交拆开**，逐项量化增量。

⚠️ 关键设计（第一版踩过的坑）：检索是 `cos(l2n(query), l2n(target_pool))`，
所以「映射的输出空间」与「目标池所在空间」**必须是同一个**，否则量到的是
两个不同空间之间的余弦，数字没有语义。第一版把「白化后的输出」去对「未白化
的目标池」，得到了一个虚高的 0.8328。本版把「目标空间」作为显式因素枚举。

因素：
  A 文本侧预处理   raw / whiten(ZCA, fit 集统计)
  B 映射族         ridge(岭回归) / OP(正交 Procrustes)
  C 检索目标空间   img_raw / img_white
  D 维度           raw1024（现状基线, 仅一次）/ pca768
  E 正则强度       lambda 扫描；F  CCA 截断维度 k 扫描

方法编号：
  M0   raw1024 + ridge -> img_raw          现状基线（复现 0.6217）
  M1   pca768  + ridge -> img_raw          只加降维
  M2+  2(prep) x 2(map) x 2(tgt) 正交组合
  M9   CCA + 截断（正确用法）

用法：
  ./.venv/Scripts/python.exe scripts/multimodal/probe_alignment_methods.py IandS
  ./.venv/Scripts/python.exe scripts/multimodal/probe_alignment_methods.py VG
"""
import os
import json
import argparse

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


# ---------------------------------------------------------------- 基础算子
def l2n(x, eps=1e-8):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def fit_pca(X, k):
    """返回 (mu, P)，P 为 (d,k) 主方向；用协方差 eigh，比 full SVD 快。"""
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    order = np.argsort(-w)
    return mu.astype(np.float32), V[:, order[:k]].astype(np.float32)


def fit_partial_whiten(X, alpha=1.0):
    """partial whitening：alpha=0 -> 恒等（仅去均值）；alpha=1 -> 标准 ZCA 白化。"""
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    w = np.clip(w, 1e-8, None)
    W = (V * (w ** (-alpha / 2.0))) @ V.T
    return mu.astype(np.float32), W.astype(np.float32)


def fit_ridge(X, Y, lam):
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=np.float32)
    B = X.T @ Y
    return np.linalg.solve(A.astype(np.float64), B.astype(np.float64)).astype(np.float32)


def fit_op(X, Y):
    """正交 Procrustes，闭式解 Q = U V^T，SVD(X^T Y) = U S V^T。"""
    U, _, Vt = np.linalg.svd((X.T @ Y).astype(np.float64), full_matrices=False)
    return (U @ Vt).astype(np.float32)


def recall(P, D, ks=(1, 5, 10)):
    """P: 映射后的 query (m,d)；D: 目标池 (m,d)。gold = 对角线。"""
    sim = l2n(P) @ l2n(D).T
    order = np.argsort(-sim, axis=1)[:, : max(ks)]
    gold = np.arange(len(P))[:, None]
    return {f"R@{k}": float((order[:, :k] == gold).any(1).mean()) for k in ks}


def gap_stats(X, Y):
    """一阶（质心）+ 二阶（协方差）modality gap。先 L2 归一再比，量纲可比。"""
    Xn, Yn = l2n(X), l2n(Y)
    cg = float(np.linalg.norm(Xn.mean(0) - Yn.mean(0)))
    Cx, Cy = np.cov(Xn.T), np.cov(Yn.T)
    return cg, float(np.linalg.norm(Cx - Cy, "fro")), float(np.linalg.norm(Cx, "fro"))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("short", choices=["IandS", "VG"])
    ap.add_argument("--n_query", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--common_dim", type=int, default=768)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    emb = os.path.join(ROOT, "data", "Amazon23", args.short, "emb")
    tf = [f for f in sorted(os.listdir(emb))
          if f.startswith("emb_text_") and f.endswith(".npy")][0]
    T = np.load(os.path.join(emb, tf)).astype(np.float32)
    I = np.load(os.path.join(emb, "emb_image_siglip.npy")).astype(np.float32)
    mask = np.load(os.path.join(emb, "emb_image_mask.npy")).astype(bool)
    print(f"[load] {args.short}  text={tf} {T.shape}  image={I.shape}  "
          f"有图={mask.sum():,}/{len(mask):,}")

    # 与 cross_modal_align 完全相同的划分口径
    have = np.where(mask)[0]
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(have))
    half = len(have) // 2
    fit, tst = have[perm[:half]], have[perm[half:]]
    q = tst[: args.n_query]

    mu_p, P = fit_pca(T[fit], args.common_dim)
    Tp = (T - mu_p) @ P
    Tf_raw, If = T[fit], I[fit]
    Tq_raw = T[q]
    Tf_p, If_p = Tp[fit], I[fit]
    Tq_p, Iq = Tp[q], I[q]
    Xc_f = Tf_raw - Tf_raw.mean(0, keepdims=True)
    var_keep = float((((Xc_f @ P) ** 2).sum()) / ((Xc_f ** 2).sum()))
    print(f"[split] fit={len(fit):,}  test={len(tst):,}  query={len(q):,}")
    print(f"[dim]   text {T.shape[1]} -> {args.common_dim}（PCA 保留方差 {var_keep:.2%}）")

    # 白化统计量只用 fit 集拟合
    mu_wt, W_wt = fit_partial_whiten(Tf_p, 1.0)
    mu_wi, W_wi = fit_partial_whiten(If_p, 1.0)
    Tf_w, If_w = (Tf_p - mu_wt) @ W_wt, (If_p - mu_wi) @ W_wi
    Tq_w, Iq_w = (Tq_p - mu_wt) @ W_wt, (Iq - mu_wi) @ W_wi

    out = {"short": args.short, "n_fit": int(len(fit)), "n_query": int(len(q)),
           "text_file": tf, "text_dim_raw": int(T.shape[1]), "image_dim": int(I.shape[1]),
           "common_dim": args.common_dim, "text_pca_var_keep": var_keep, "lam": args.lam,
           "methods": {}, "gaps": {}, "sweeps": {}}

    rows = {}   # name -> avg R@10

    def record(name, fwd_t, bwd_t, tgt_t, tgt_i):
        """fwd_t/bwd_t 已含全部预处理与映射；tgt_* 是要比的目标池。"""
        r = {"text_to_image": recall(fwd_t, tgt_i),
             "image_to_text": recall(bwd_t, tgt_t)}
        avg = 0.5 * (r["text_to_image"]["R@10"] + r["image_to_text"]["R@10"])
        out["methods"][name] = r
        rows[name] = avg
        return avg

    # ---------------- M0 现状基线：raw1024 + ridge -> img_raw
    W0 = fit_ridge(Tf_raw, If, args.lam)
    W0b = fit_ridge(If, Tf_raw, args.lam)
    record("M0_raw1024_ridge__img_raw", Tq_raw @ W0, Iq @ W0b, Tq_raw, Iq)

    # ---------------- 因素正交组合：预处理 P(2) x 映射族(2) x 图像侧检索空间(2)
    #
    # 每个方向各自自洽 —— 「映射的输出空间」必须等于「目标池所在空间」：
    #   正向 T->I: 源=文本(经 P 预处理) -> 映射到 tname 图像空间; 目标池=该空间的图像 query
    #   反向 I->T: 源=图像(经 P 预处理) -> 映射到 P 文本空间;      目标池=该空间的文本 query
    def tr_txt(kind, X):
        return X if kind == "raw" else (X - mu_wt) @ W_wt

    def tr_img(kind, Y):
        return Y if kind == "raw" else (Y - mu_wi) @ W_wi

    tgt_pool = {"img_raw": (If_p, Iq), "img_white": (If_w, Iq_w)}

    for pname in ("raw", "whiten"):
        txt_f, txt_q = tr_txt(pname, Tf_p), tr_txt(pname, Tq_p)
        img_f, img_q = tr_img(pname, If_p), tr_img(pname, Iq)
        # 反向映射只依赖 pname（与 tname 无关），提到内层循环外
        Wb = fit_ridge(img_f, txt_f, args.lam)
        Qb = fit_op(img_f, txt_f)
        for tname, (Df, Dq) in tgt_pool.items():
            Wf = fit_ridge(txt_f, Df, args.lam)
            Qf = fit_op(txt_f, Df)
            record(f"P{pname}_ridge__{tname}", txt_q @ Wf, img_q @ Wb, txt_q, Dq)
            record(f"P{pname}_OP__{tname}", txt_q @ Qf, img_q @ Qb, txt_q, Dq)

    # ---------------- M5 两侧各自白化后直接点积（无监督，不学任何映射）
    # 这是对「白化闭合 modality gap => 可以直接跨模态检索」这一说法的压力测试：
    # 白化后两个模态的边缘分布都接近标准高斯，但**实例级对应关系并未建立**。
    record("M5_whiten_dot_unsupervised", Tq_w, Iq_w, Tq_w, Iq_w)

    # ---------------- M9 CCA + 截断（在白化空间，输出即白化空间）
    # 白化后 Xw^T Xw = (n-1) I，故典型相关系数 = s / (n-1)
    U_, S_, Vt_ = np.linalg.svd((Tf_w.T @ If_w).astype(np.float64), full_matrices=False)
    n_f = len(Tf_w)
    out["cca_canonical_corr_top20"] = [round(float(s / (n_f - 1)), 5) for s in S_[:20]]
    out["cca_singular_mean"] = float(S_.mean())
    print(f"\n[CCA] 典型相关系数（= s/(n-1)，前 20 个；越接近 1 说明该方向图文越一致）")
    print("  " + "  ".join(f"{v:.4f}" for v in out["cca_canonical_corr_top20"]))
    print(f"  全部 {len(S_)} 个方向的均值 = {S_.mean()/(n_f-1):.4f}"
          f"（谱越平坦 => 相关性越弥散，越需要大的 k）")
    for k in (16, 64, 128, 256, 768):
        Uu, Vv = U_[:, :k].astype(np.float32), Vt_[:k].T.astype(np.float32)
        record(f"M9_CCA_k{k}", Tq_w @ Uu, Iq_w @ Vv, Tq_w @ Uu, Iq_w @ Vv)

    # ---------------- gap 诊断（一阶 / 二阶）
    cg0, cvg0, cref0 = gap_stats(Tf_p, If_p)
    cg1, cvg1, cref1 = gap_stats(Tf_w, If_w)
    out["gaps"] = {"raw": {"centroid": cg0, "cov_gap": cvg0, "cov_ref": cref0,
                           "cov_ratio": cvg0 / cref0},
                   "whitened": {"centroid": cg1, "cov_gap": cvg1, "cov_ref": cref1,
                                "cov_ratio": cvg1 / cref1}}
    print(f"\n[gap] raw      centroid={cg0:.4f}  cov_gap={cvg0:.4f} "
          f"(ref {cref0:.4f}, ratio {cvg0/cref0:.3f})")
    print(f"[gap] whitened centroid={cg1:.4f}  cov_gap={cvg1:.4f} "
          f"(ref {cref1:.4f}, ratio {cvg1/cref1:.3f})")

    # ---------------- alpha 扫描：partial whitening，两侧同时、目标池同空间
    seq_a = []
    print(f"\n[alpha 扫描] partial whitening（0=仅去均值, 1=全白化），两侧同 alpha、目标池同空间")
    print(f"  {'alpha':>6} {'T->I R@10':>10} {'I->T R@10':>10} {'均值':>8} "
          f"{'centroid':>9} {'cov_ratio':>10}")
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        mt, Wt_ = fit_partial_whiten(Tf_p, a)
        mi, Wi_ = fit_partial_whiten(If_p, a)
        tf_a, if_a = (Tf_p - mt) @ Wt_, (If_p - mi) @ Wi_
        tq_a, iq_a = (Tq_p - mt) @ Wt_, (Iq - mi) @ Wi_
        Wf = fit_ridge(tf_a, if_a, args.lam)
        Wb = fit_ridge(if_a, tf_a, args.lam)
        r = {"text_to_image": recall(tq_a @ Wf, iq_a),
             "image_to_text": recall(iq_a @ Wb, tq_a)}
        avg = 0.5 * (r["text_to_image"]["R@10"] + r["image_to_text"]["R@10"])
        cg, cvg, cref = gap_stats(tf_a, if_a)
        print(f"  {a:>6.2f} {r['text_to_image']['R@10']:>10.4f} "
              f"{r['image_to_text']['R@10']:>10.4f} {avg:>8.4f} {cg:>9.4f} "
              f"{cvg/cref:>10.4f}")
        seq_a.append({"alpha": a, "avg_R@10": avg, "centroid": cg,
                      "cov_ratio": cvg / cref,
                      "t2i_R@10": r["text_to_image"]["R@10"],
                      "i2t_R@10": r["image_to_text"]["R@10"]})
    out["sweeps"]["alpha"] = seq_a

    # ---------------- lambda 扫描：判断白化收益是否只是「数值条件数」效应
    seq_l = []
    print(f"\n[lambda 扫描] ridge 正则强度 -> 双向 R@10 均值（目标池与映射输出同空间）")
    print(f"  {'lambda':>9} {'pca768':>10} {'whitened':>10}")
    for lam in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        a1, b1 = fit_ridge(Tf_p, If_p, lam), fit_ridge(If_p, Tf_p, lam)
        v1 = 0.5 * (recall(Tq_p @ a1, Iq)["R@10"] + recall(Iq @ b1, Tq_p)["R@10"])
        a2, b2 = fit_ridge(Tf_w, If_w, lam), fit_ridge(If_w, Tf_w, lam)
        v2 = 0.5 * (recall(Tq_w @ a2, Iq_w)["R@10"] + recall(Iq_w @ b2, Tq_w)["R@10"])
        print(f"  {lam:>9.0e} {v1:>10.4f} {v2:>10.4f}")
        seq_l.append({"lam": lam, "pca768": v1, "whitened": v2})
    out["sweeps"]["lambda"] = seq_l

    # ---------------- 汇总
    print(f"\n{'方法':<28} {'T->I R@1':>9} {'T->I R@10':>10} "
          f"{'I->T R@1':>9} {'I->T R@10':>10}  {'均值':>10}")
    print("-" * 88)
    for name, _ in sorted(rows.items(), key=lambda t: -t[1]):
        r = out["methods"][name]
        a, b = r["text_to_image"], r["image_to_text"]
        print(f"{name:<28} {a['R@1']:>9.4f} {a['R@10']:>10.4f} "
              f"{b['R@1']:>9.4f} {b['R@10']:>10.4f}  {rows[name]:>10.4f}")
    out["avg_R10"] = rows

    base = rows["M0_raw1024_ridge__img_raw"]
    print(f"\n[相对现状 M0（{base:.4f}）的增量]")
    for name, v in sorted(rows.items(), key=lambda t: -t[1]):
        print(f"  {name:<28} {v-base:+.4f}  ({(v/base-1)*100:+.1f}%)")

    out_path = args.out or os.path.join(ROOT, "results", "alignment_methods",
                                        f"{args.short}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    main()
