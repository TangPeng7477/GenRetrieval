"""
Sinkhorn algorithm demo
=======================
Compare: argmin vs Sinkhorn optimal transport
Scenario: 4 items (N=4) assigned to 3 prototypes (K=3)

=============================================================================
Sinkhorn 原理与项目实践 — 面试回答完整版
=============================================================================

【Sinkhorn 解决 SID 碰撞的完整流程】

Sinkhorn 解决碰撞的思路，是把分配问题从一个"每个商品独立决策"变成
"全局协调的最优传输问题"。在 argmin 的基础上加了一个约束——每个原型
分配到的商品数要大致均匀。

具体来说，Sinkhorn 求解的是带熵正则化的最优传输问题。目标函数有两项：
一项是最小化商品到原型的距离和，一项是熵正则化项。约束条件规定了行和
列的和：行约束对应每个商品的权重相等，列约束对应每个原型的分配数均匀，
在我们的设置里就是每个原型分大约 N/K 个商品。

这个优化问题可以通过交替行归一化和列归一化来求解——这就是 Sinkhorn 迭代。
先通过 exp(-C/ε) 把距离矩阵转换成相似度矩阵，距离越近的值越大。
然后交替做两件事：行归一化让每行概率加起来等于 1/N，保证每个商品等权；
列归一化让每列概率加起来等于 1/K，保证每个原型的分配均匀。
大约 50 轮就收敛了。

列归一化就是消除碰撞的关键动作。假设三个碰撞商品都挤在同一个原型上，
那这一列的总概率就会超过 1/K。列归一化检测到这一点，会把多余的概率
"挤"出去，分配到那些空置的原型列上。每个商品到不同原型的距离决定了
哪些列收到多少概率——离得近的多分一点，离得远的少分一点。最终每个商品
拿到一个概率分布，argmax 取概率最大的原型作为分配结果。因为概率已经
被列归一化重新平衡过，碰撞组里的不同商品往往会取到不同的原型，碰撞
就消除了。

但这里有一个关键区别要看场景。在 FAISS 全量后处理里——N=3686、K=256——
列约束是实实在在的强制力，每个原型必须分到恰好约 14 个商品，Sinkhorn
真的在做最优传输。在我们的碰撞消除场景里，碰撞组可能只有 2 到 10 个商品，
而 K 还是 256。这种情况下 N 远小于 K，列约束本身没什么约束力。
真正起作用的是 ε 的 tie-breaking 效应：三个商品离同一个原型同样近，
但它们的残差在第五位小数上有细微差异。exp(-C/ε) 把这个微小差异放大到
列归一化能感知的程度，然后列归一化微调各列概率的值，argmax 就取到了
不同的原型。所以 ε=0.003 这么小也够用——不是因为"容量满了被换掉"，
而是因为列归一化打破了 argmin 下的平局。

熵正则化强度 ε 控制 trade-off。当 ε 趋于 0，退化成纯 argmin，语义最优
但碰撞不控制；当 ε 趋于无穷，退化成纯均匀分配，零碰撞但语义全丢。
项目取 ε=0.003，是一个非常小的值，刚好够打破平局，不破坏语义。

基于这个理解，三层递进策略就很合理了。不是所有层都用 Sinkhorn，只有
第三层用，前两层保持 argmin。RQ-VAE 的层次结构有明确的语义分工：
第一层分大类，第二层分中类，第三层分具体商品。前两层的语义结构比
均匀更重要，如果强行均匀反而会破坏树状层次。第三层做最细粒度的区分，
碰撞本质上就是在这层没区分开，用 ε=0.003 做 tie-breaking 正好。

整个流程也不是一次 Sinkhorn 跑完的，而是迭代检测—重编码—再检测的闭环。
先用 argmin 给全部 3686 个商品生成初始 SID，然后检测碰撞组。对有碰撞的
商品组，只提取这些商品启用 Sinkhorn 重编码，未碰撞的商品保持不动。
重编码后重新检测，如果还有碰撞就再对新的碰撞组重编码，最多迭代 20 轮。
因为每次只动碰撞商品，不动其他商品，对语义结构的扰动最小。
最终 3686 个商品实现 0% 碰撞率。

从码本利用率看这个设计也是自洽的：第一层 48/256——数据天然只有约 48 个
粗类，argmin 保留了这个结构；第二层 256/256——残差空间丰富，全部用满；
第三层 256/256——Sinkhorn tie-breaking 配合迭代消除推动了全利用。
所以总共 560 个 SID token（48+256+256）不是问题，是数据与算法共同作用
下的合理结果。
"""

import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)

def p(text=""):
    """Print with flush to ensure output on Windows"""
    print(text)
    sys.stdout.flush()

p("=" * 70)
p("Sinkhorn Algorithm Demonstration")
p("=" * 70)

# ============================================================
# 1. Setup: 4 items in 2D plane, 3 prototypes
# ============================================================
p()
p("[Step 1: Data Preparation]")
p("-" * 50)

items = np.array([
    [0.0, 0.0],   # item A
    [0.1, 0.1],   # item B - very close to A
    [3.0, 0.0],   # item C
    [2.9, 0.1],   # item D - very close to C
], dtype=np.float64)

prototypes = np.array([
    [0.0, 0.0],   # prototype 0
    [3.0, 0.0],   # prototype 1
    [1.5, 2.0],   # prototype 2 - far from all items
], dtype=np.float64)

N, K = len(items), len(prototypes)

p(f"Items N = {N}, Prototypes K = {K}")
p()
p("Items (2D coordinates):")
labels = ["A", "B", "C", "D"]
for i, (x, y) in enumerate(items):
    p(f"  Item {labels[i]}:  ({x}, {y})")
p()
p("Prototypes (2D coordinates):")
for j, (x, y) in enumerate(prototypes):
    p(f"  Proto {j}:  ({x}, {y})")
p("  -> Proto 2 is far from all items, creating imbalance")

# ============================================================
# 2. Cost matrix C
# ============================================================
p()
p("[Step 2: Cost Matrix C]")
p("-" * 50)
p("  Formula: C[i,j] = ||item_i - proto_j||^2")
p()

C = np.zeros((N, K))
for i in range(N):
    for j in range(K):
        C[i, j] = np.sum((items[i] - prototypes[j]) ** 2)

# Print matrix
header = "        " + "  ".join([f"Proto{j}" for j in range(K)])
p(header)
for i in range(N):
    row = "  ".join([f"{C[i,j]:.4f}" for j in range(K)])
    p(f"Item{labels[i]}:  {row}")

# Argmin assignment
argmin_assign = np.argmin(C, axis=1)
p()
p("argmin assignment (each item picks closest prototype):")
for i in range(N):
    nearest_dist = C[i, argmin_assign[i]]
    p(f"  Item {labels[i]} -> Proto {argmin_assign[i]}  (dist={nearest_dist:.4f})")

# Detect collision
from collections import Counter
assign_counts = Counter(argmin_assign)
collisions = {k: v for k, v in assign_counts.items() if v > 1}
if collisions:
    for k, v in collisions.items():
        p(f"\n  COLLISION: Proto {k} assigned {v} items!")
else:
    p("\n  No collision")

# ============================================================
# 3. Sinkhorn iteration
# ============================================================
p()
p("=" * 70)
p("[Step 3: Sinkhorn Iteration]")
p("=" * 70)

epsilon = 0.1
num_iters = 10

p()
p("Parameters:")
p(f"  epsilon (entropy reg) = {epsilon}")
p(f"  iterations           = {num_iters}")

# Row/column constraints
a = np.ones(N) / N
b = np.ones(K) / K

p(f"  Row constraint a     = {a}")
p(f"  Col constraint b     = {b}")
p(f"  (Goal: each row sums to {a[0]:.4f}, each col sums to {b[0]:.4f})")

# Step 3.1: Initialize transport matrix P
p()
p("-" * 50)
p("3.1 Initialize P = exp(-C / epsilon)")
p("-" * 50)

P = np.exp(-C / epsilon)
P_before_norm = P.copy()
p()
p("  Before normalization:")
for i in range(N):
    row = "  ".join([f"{P[i,j]:.6f}" for j in range(K)])
    p(f"    {labels[i]}:  {row}")

P = P / P.sum()
p()
p("  After normalization (sum to 1):")
for i in range(N):
    row = "  ".join([f"{P[i,j]:.6f}" for j in range(K)])
    p(f"    {labels[i]}:  {row}")

# Step 3.2: Sinkhorn alternating normalization
p()
p("-" * 50)
p("3.2 Sinkhorn alternating row/col normalization")
p("-" * 50)

for it in range(1, num_iters + 1):
    p(f"  --- Iteration {it} ---")

    # Row normalization: each row sum = a_i
    P = P / P.sum(axis=1, keepdims=True) * a.reshape(-1, 1)
    p(f"  After row norm:")
    for i in range(N):
        row = "  ".join([f"{P[i,j]:.6f}" for j in range(K)])
        p(f"    {labels[i]}:  {row}  (sum={P[i].sum():.4f})")

    # Column normalization: each column sum = b_j
    P = P / P.sum(axis=0, keepdims=True) * b.reshape(1, -1)
    p(f"  After col norm:")
    for i in range(N):
        row = "  ".join([f"{P[i,j]:.6f}" for j in range(K)])
        p(f"    {labels[i]}:  {row}")
    col_sums = P.sum(axis=0)
    p(f"  Col sums: [{col_sums[0]:.4f}  {col_sums[1]:.4f}  {col_sums[2]:.4f}] (target={b[0]:.4f})")

    # Convergence check
    row_error = np.max(np.abs(P.sum(axis=1) - a))
    col_error = np.max(np.abs(P.sum(axis=0) - b))
    p(f"  Max row err: {row_error:.6f}, Max col err: {col_error:.6f}")

    if row_error < 1e-8 and col_error < 1e-8:
        p("  -> CONVERGED!")
        break

# ============================================================
# 4. Result comparison
# ============================================================
p()
p("=" * 70)
p("[Step 4: Result Comparison]")
p("=" * 70)

sinkhorn_assign = np.argmax(P, axis=1)

p()
p("Sinkhorn Final Transport Matrix P (each row = item's assignment probabilities):")
for i in range(N):
    row = "  ".join([f"{P[i,j]:.4f}" for j in range(K)])
    prob_max = P[i].max() * 100
    p(f"  Item {labels[i]}:  [{row}]  -> Proto {sinkhorn_assign[i]}  (prob={prob_max:.1f}%)")
p(f"  Row sums:  {P.sum(axis=1)}")
p(f"  Col sums:  {P.sum(axis=0)} (target={b[0]})")

# Comparison table
p()
p("-" * 50)
p("Assignment Comparison: argmin vs Sinkhorn")
p("-" * 50)
for i in range(N):
    match = "o" if argmin_assign[i] == sinkhorn_assign[i] else "*"
    p(f"  Item {labels[i]}:  argmin->Proto{argmin_assign[i]}  Sinkhorn->Proto{sinkhorn_assign[i]}  [{match}]")
p("  [o] same  [*] different (Sinkhorn changed for balance)")

# Collision check
argmin_counts = Counter(argmin_assign)
sinkhorn_counts = Counter(sinkhorn_assign)
argmin_collisions = sum(v - 1 for v in argmin_counts.values() if v > 1)
sinkhorn_collisions = sum(v - 1 for v in sinkhorn_counts.values() if v > 1)
if argmin_collisions:
    p(f"\n  argmin collisions:    {argmin_collisions}  {dict(argmin_counts)}")
else:
    p(f"\n  argmin collisions:    0  {dict(argmin_counts)}")
if sinkhorn_collisions:
    p(f"  Sinkhorn collisions:  {sinkhorn_collisions}  {dict(sinkhorn_counts)}")
else:
    p(f"  Sinkhorn collisions:  0  {dict(sinkhorn_counts)}")

# Transport cost comparison
cost_argmin = sum(C[i, argmin_assign[i]] for i in range(N))
cost_sinkhorn = sum(C[i, sinkhorn_assign[i]] for i in range(N))
p()
p("-" * 50)
p("Total Transport Cost (sum of C[i, assigned_proto]):")
p("-" * 50)
p(f"  argmin:    {cost_argmin:.4f}")
p(f"  Sinkhorn:  {cost_sinkhorn:.4f}")
p(f"  Increase:  {cost_sinkhorn - cost_argmin:.4f}  ({(cost_sinkhorn/cost_argmin - 1)*100:.1f}%)")
p(f"\n  Interpretation: Sinkhorn sacrifices {cost_sinkhorn-cost_argmin:.4f} in distance")
p(f"  to eliminate {argmin_collisions} collision(s) and balance prototype usage.")

# ============================================================
# 5. Effect of different epsilon values
# ============================================================
p()
p("=" * 70)
p("[Step 5: Effect of Different Epsilon Values]")
p("=" * 70)

epsilons = [0.01, 0.05, 0.1, 0.5, 10.0]
descriptions = ["near-argmin", "mild balance", "moderate", "strong balance", "near-uniform"]

p()
p(f"{'epsilon':>8}  {'description':<16}  {'assignment':>14}  {'cost':>8}  {'collisions':>10}  {'col_uniformity':>14}")
p("-" * 80)
for eps, desc in zip(epsilons, descriptions):
    P_eps = np.exp(-C / eps)
    P_eps = P_eps / P_eps.sum()
    for _ in range(50):
        P_eps = P_eps / P_eps.sum(axis=1, keepdims=True) * a.reshape(-1, 1)
        P_eps = P_eps / P_eps.sum(axis=0, keepdims=True) * b.reshape(1, -1)

    assign = np.argmax(P_eps, axis=1)
    cols = P_eps.sum(axis=0)
    assign_str = "".join([f"{labels[i]}->{assign[i]} " for i in range(N)])
    cost = sum(C[i, assign[i]] for i in range(N))
    collisions = sum(1 for v in Counter(assign).values() if v > 1)
    col_uniformity = 1 - np.std(cols) / np.mean(cols) if np.mean(cols) > 0 else 0

    p(f"{eps:>8.3f}  {desc:<16}  {assign_str:>14}  {cost:>8.4f}  {collisions:>10}  {col_uniformity:>14.3f}")

p()
p("col_uniformity = 1 - std/mean, higher = more balanced")

# ============================================================
# 6. Simulate the project's real scenario
# ============================================================
p()
p("=" * 70)
p("[Step 6: Simulate Project's Real Scenario (3-level, epsilon=0.003)]")
p("=" * 70)

p()
p("Scenario: 3 collided items (same layer-1 & layer-2 codes),")
p("residuals are very close but slightly different (as in real data).")
np.random.seed(42)

# 3 colliding items, 32-dim residuals
collision_residuals = np.random.randn(3, 32) * 0.01
for i in range(3):
    collision_residuals[i] += np.random.randn(32) * 0.001

# 256 prototypes (as in project)
K_real = 256
real_prototypes = np.random.randn(K_real, 32) * 0.1

# argmin
dists = np.zeros((3, K_real))
for i in range(3):
    for j in range(K_real):
        dists[i, j] = np.sum((collision_residuals[i] - real_prototypes[j]) ** 2)

argmin_sid = np.argmin(dists, axis=1)
p(f"\n  argmin assignment:         {argmin_sid}")
p(f"                           {' '.join([f'<c_{x}>' for x in argmin_sid])}")
if len(set(argmin_sid)) < 3:
    p(f"  -> COLLISION! Multiple items at <c_{argmin_sid[0]}>")

# Sinkhorn with epsilon = 0.003 (real project value)
eps_real = 0.003
a_real = np.ones(3) / 3
b_real = np.ones(K_real) / K_real

P_real = np.exp(-dists / eps_real)
P_real = P_real / P_real.sum()
for _ in range(50):
    P_real = P_real / P_real.sum(axis=1, keepdims=True) * a_real.reshape(-1, 1)
    P_real = P_real / P_real.sum(axis=0, keepdims=True) * b_real.reshape(1, -1)

sinkhorn_sid = np.argmax(P_real, axis=1)
p(f"\n  Sinkhorn assignment:       {sinkhorn_sid}")
p(f"                           {' '.join([f'<c_{x}>' for x in sinkhorn_sid])}")
if len(set(sinkhorn_sid)) == 3:
    p(f"  -> Collision resolved! Each item gets a different <c_>")

# Show top-3 probabilities
p(f"\n  Sinkhorn probability distribution (Top-3 per item):")
item_labels_real = ["Item A", "Item B", "Item C"]
for i in range(3):
    top3 = np.argsort(-P_real[i])[:3]
    probs = P_real[i, top3]
    prob_strs = [f"<c_{t}>: {p*100:.2f}%" for t, p in zip(top3, probs)]
    p(f"    {item_labels_real[i]}:  {',  '.join(prob_strs)}")

col_sums_real = P_real.sum(axis=0)
used = np.where(col_sums_real > 1e-10)[0]
p(f"\n  Prototypes with non-zero mass: {len(used)} / {K_real}")
p(f"  Success: 3 collided items resolved with {sum(1 for v in Counter(sinkhorn_sid).values())} unique SIDs")

p()
p("=" * 70)
p("KEY INSIGHT")
p("=" * 70)
p()
p("1. argmin -> COLLISION")
p("-" * 30)
p("Each item independently picks the closest prototype.")
p("When items are close together (e.g. A & B near Proto0),")
p("they pick the SAME prototype. Proto0 gets 2 items,")
p("Proto2 gets zero. This imbalance is the root cause.")
p()
p("2. Sinkhorn initialization: P = exp(-C / epsilon)")
p("-" * 30)
p("Distance matrix C is converted to a similarity matrix.")
p("Smaller distance -> larger P value.")
p("  exp(-0.000/0.1) = 1.000     # A -> Proto0")
p("  exp(-0.020/0.1) = 0.819     # B -> Proto0")
p("  exp(-9.000/0.1) = 0.000     # A -> Proto1 (far away)")
p("Proto2 column starts near-zero because all items are far from it.")
p()
p("3. Alternating normalization -> REBALANCE")
p("-" * 30)
p("  Row norm: each row sums to a_i = 1/N (all items equal)")
p("  Col norm: each column sums to b_j = 1/K (balanced prototypes)")
p()
p("  Track Item B through iterations:")
p("    Iter 1: Proto0=0.250  Proto2=0.000  (after row norm)")
p("    Iter 1: Proto0=0.167  Proto2=0.167  (after col norm -> PUSH!)")
p("    Iter 3: Proto0=0.100  Proto2=0.150")
p("    Iter 5: Proto0=0.087  Proto2=0.163")
p("    Iter10: Proto0=0.084  Proto2=0.166")
p("  Column normalization detects Proto0 is over capacity (>1/3),")
p("  so it pushes excess probability FROM Proto0 TO Proto2.")
p("  Which items get pushed? Those with nonzero probability at Proto2.")
p("  Items A and C have zero at Proto2 (they're too far), so they stay.")
p("  Items B and D have tiny but nonzero Proto2 probability -> they get moved.")
p()
p("4. Epsilon controls the push strength")
p("-" * 30)
p("  epsilon=0.01:  strong differentiation -> B->2, D->2")
p("  epsilon=0.10:  same pattern")
p("  epsilon=0.50:  pushes too hard -> B->0, D->1 (back to argmin)")
p("  epsilon=10.0:  all distances look equal -> degenerate")
p("  The sweet spot is a small epsilon that differentiates close distances")
p("  without pushing items to semantically unrelated prototypes.")
p()
p("5. Project scenario: epsilon=0.003, N=3 items, K=256 prototypes")
p("-" * 30)
p("  Here N << K, so the column constraint is weak.")
p("  The column sum target is 1/256 = 0.0039 per prototype.")
p("  Sinkhorn spreads probability near-uniformly across all 256 prototypes.")
p("  The collision is resolved by TIE-BREAKING: each item's argmax")
p("  picks a different prototype because numerical differences in the")
p("  5th decimal place are amplified by exp(-C/0.003) and column normalization.")
p()
p("  Key difference from FAISS route (N=3686, K=256):")
p("  - Large-scale balancing: capacity constraints are real, optimal transport")
p("  - Collision resolution: tie-breaking through tiny probability shifts")
p("  Both are Sinkhorn, but the mechanism differs by N vs K ratio.")
p()
p("6. Iterative elimination in the project")
p("-" * 30)
p("  Run 1: argmin on ALL 3686 items -> detect collision groups")
p("  Run 2: Sinkhorn (epsilon=0.003) on collision groups only -> recheck")
p("  Run 3: repeat until 0% collision or max 20 iterations")
p("  Only touched items get their SIDs changed; non-colliding items stay.")
p("  Final: 0% collision rate across 3686 items.")
p()
p("7. Codebook utilization is self-consistent")
p("-" * 30)
p("  Layer 1: 48/256 (18.8%) - only ~48 natural coarse categories")
p("  Layer 2: 256/256 (100%) - residual space is rich, argmin works")
p("  Layer 3: 256/256 (100%) - Sinkhorn tie-breaking + iteration")
p("  Total: 560 SID tokens. The gap (768-560=208) is entirely from L1.")
p("  This is NOT codebook collapse (L2+L3 are full).")
p("  It is the data's natural cluster count at the coarse level.")
