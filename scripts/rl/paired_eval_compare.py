"""逐行配对比较两个评估 dump —— 比边际 HR@K 灵敏得多的判别工具（不用 GPU）。

为什么需要它
------------
我们跑了 7 个 RL run，HR@10 全落在 0.0328~0.0342（±0.5σ）。但"边际率相同"**不等于"输出没变"**：
两个模型的 5,000 行预测可以逐行翻来翻去、只是净增益为零。这两件事的解释完全不同：

  · 若 b ≈ c 且都很小 ⟹ **模型几乎没动**（RL 没改变行为）⟹ 该查优化设置（lr / 步数 / KL）
  · 若 b ≈ c 但都很大 ⟹ **churn（来回翻）** ⟹ 该查奖励是否只在制造噪声
  · 若 b ≫ c ⟹ RL 真有净增益，只是被 c 抵消 ⟹ 该查哪些样本被"翻坏"（做 badcase）

方法：把两次评估在**同一批行**上配对（TEST_TAG 保证同一个 test CSV、同 random.seed），
对每行取 hit@K，做 **McNemar 精确检验**（只看不一致对 b/c，用二项分布），
比"比两个边际率的置信区间"灵敏得多（消掉了样本间方差）。

用法
----
    python scripts/rl/paired_eval_compare.py \
        --a results/sft/IandS-all/eval_IandS_beam50_u5k.json \
        --b results/sft/IandS-u5kr3/eval_IandS_beam50_u5k.json
    # 也可以只给 --a 一组目录名（会自动拼路径，并列出可选文件）

输出：逐行命中矩阵统计、McNemar p 值、集合 churn（top-K 重合度）、近失（前缀命中）分析。
"""

import argparse
import glob
import io
import json
import os
import re
import sys
from collections import Counter

# 允许 `python scripts/rl/paired_eval_compare.py` 直接跑（补仓库根到 sys.path）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SID = re.compile(r"<a_(\d+)><b_(\d+)><c_(\d+)>")


def load_dump(path):
    """返回 [(target_sid, [pred_sid, ...]), ...]。"""
    with io.open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for e in data:
        tgt = (e.get("output") or "").strip().strip('"')
        preds = [p.strip().strip('"') for p in (e.get("predict") or [])]
        rows.append((tgt, preds))
    return rows


def resolve(arg, domain="IandS"):
    """把 'IandS-u5kr3' 或文件路径解析成真实文件路径。"""
    if arg.endswith(".json") and os.path.exists(arg):
        return arg
    cands = sorted(glob.glob(f"results/sft/{arg}/eval_{domain}_beam*.json"))
    cands = [c for c in cands if not c.endswith((".meta.json", ".metrics.json"))]
    if not cands:
        raise SystemExit(f"[ERROR] 找不到 {arg} 的评估 dump：results/sft/{arg}/eval_{domain}_beam*.json")
    if len(cands) > 1:
        print(f"[warn] {arg} 有多个 dump，取最后一个：{cands[-1]}")
    return cands[-1]


def hit_at(preds, tgt, k):
    return tgt in preds[:k]


def prefix_len(sid):
    m = _SID.fullmatch(sid or "")
    return len(m.groups()) if m else 0


def prefix_match(pred, tgt, depth):
    """前 depth 层 SID 是否相同。"""
    mp, mt = _SID.fullmatch(pred or ""), _SID.fullmatch(tgt or "")
    if not (mp and mt):
        return False
    return mp.groups()[:depth] == mt.groups()[:depth]


def binom_two_sided_p(b, c):
    """McNemar 精确检验（n=b+c, p=0.5 双侧）。不依赖 scipy。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X<=k) * 2，截断到 1
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="基线 run（如 IandS-all 或 dump 路径）")
    ap.add_argument("--b", required=True, help="对照 run（如 IandS-u5kr3）")
    ap.add_argument("--domain", default="IandS")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--topk-churn", type=int, default=10, help="集合重合度按前多少条算")
    args = ap.parse_args()

    pa, pb = resolve(args.a, args.domain), resolve(args.b, args.domain)
    ra, rb = load_dump(pa), load_dump(pb)
    print(f"[A] {pa}  rows={len(ra)}")
    print(f"[B] {pb}  rows={len(rb)}")
    if len(ra) != len(rb):
        print(f"[warn] 行数不同（{len(ra)} vs {len(rb)}），取交集前 {min(len(ra), len(rb))} 行")
    n = min(len(ra), len(rb))

    K = args.k
    hit_a = [hit_at(ra[i][1], ra[i][0], K) for i in range(n)]
    hit_b = [hit_at(rb[i][1], rb[i][0], K) for i in range(n)]
    both = sum(1 for i in range(n) if hit_a[i] and hit_b[i])
    only_a = sum(1 for i in range(n) if hit_a[i] and not hit_b[i])   # A 命中、B 丢（=c）
    only_b = sum(1 for i in range(n) if hit_b[i] and not hit_a[i])   # B 新命中（=b）
    neither = n - both - only_a - only_b

    print(f"\n=== 逐行配对（hit@{K}，同一批 {n} 行）===")
    print(f"  都命中      {both:5d}")
    print(f"  A 命中 B 丢  {only_a:5d}   (锚点→RL 丢)")
    print(f"  B 命中 A 丢  {only_b:5d}   (RL 新命中)")
    print(f"  都不命中    {neither:5d}")
    print(f"  边际 HR@{K}：A = {sum(hit_a)/n:.4f}   B = {sum(hit_b)/n:.4f}   Δ = {(sum(hit_b)-sum(hit_a))/n:+.4f}")
    p = binom_two_sided_p(only_b, only_a)
    print(f"  McNemar 精确检验：b={only_b} c={only_a} ⟹ p = {p:.3f}"
          f"  {'（无显著差异）' if p > 0.05 else '（显著！）'}")

    if only_b + only_a:
        print(f"  ⟹ 翻转总数 {only_b + only_a} 行，其中净增益 {only_b - only_a:+d} 行；"
              f"判别：翻转少且净≈0 = 模型没动；翻转多而净≈0 = churn")

    # 集合层面的 churn：前 topk-churn 条预测的重合度 + 完全相同行数
    Tc = args.topk_churn
    same_top = 0
    jac_sum = 0.0
    for i in range(n):
        sa, sb = set(ra[i][1][:Tc]), set(rb[i][1][:Tc])
        if sa == sb:
            same_top += 1
        jac_sum += len(sa & sb) / max(1, len(sa | sb))
    print(f"\n=== 集合 churn（前 {Tc} 条预测）===")
    print(f"  完全相同行数 = {same_top}/{n} ({100*same_top/n:.1f}%)   平均 Jaccard = {jac_sum/n:.3f}")

    # 近失分析：target 的前 2 层是否出现在预测里（衡量"离命中多近"）
    print(f"\n=== 近失（target 的 SID 前缀命中）===")
    for name, rows in (("A", ra), ("B", rb)):
        d1 = sum(1 for i in range(n) if any(prefix_match(x, rows[i][0], 1) for x in rows[i][1]))
        d2 = sum(1 for i in range(n) if any(prefix_match(x, rows[i][0], 2) for x in rows[i][1]))
        print(f"  {name}: 前缀1 命中 {d1:5d} ({100*d1/n:.2f}%)   前缀2 命中 {d2:5d} ({100*d2/n:.2f}%)")

    # 若 B 新命中/丢命中，看看命中位次（区分"擦边进/出"）
    def rank_of_hits(rows):
        rs = []
        for i in range(n):
            t = rows[i][0]
            if t in rows[i][1][:K]:
                rs.append(rows[i][1].index(t) + 1)
        return Counter(rs)
    print(f"\n=== 命中位次分布（1 = top1）===")
    print(f"  A: {dict(sorted(rank_of_hits(ra).items()))}")
    print(f"  B: {dict(sorted(rank_of_hits(rb).items()))}")


if __name__ == "__main__":
    main()
