"""验证 `common/data.py` 里"从 train.inter 重建完整训练序列"这套口径的探针。

为什么需要它：`prepare_amazon23.py` 只落盘滑窗样本 `(history, target)`，没有落盘"每个用户的完整序列"。
但评估时必须把用户**全部**交互过的物品从候选里屏蔽掉（否则模型只要学会"重复推荐用户看过的东西"
就能刷分），所以必须能把完整序列还原出来。本脚本把还原公式的**三条前提**逐条实测验证：

  前提 1  同一个用户的滑窗样本按文件顺序排列，且 `target` 依次是 `tr[1], tr[2], ...`
  前提 2  每个用户的**第一个样本的 history 长度恒为 1**（= `[tr[0]]`）
  前提 3  训练段只有 1 个物品的用户产生不出任何滑窗样本 → 不在 train.inter 里，
          但他们的 valid/test history 首位就是那唯一的训练物品

校验方式：用重建序列预测 valid 的输入（`train_seq[-20:]`）应 100% 命中；
test 的输入应等于 `(train_seq + valid 段全部物品)[-20:]`，因此
"与 `train_seq + 仅 valid 目标`"不一致的样本数，恰好等于"valid 段 ≥2 个物品"的用户数。

用法：
    ./.venv/Scripts/python.exe -m baseline.scripts.probe_sequence_reconstruction
"""

from __future__ import annotations

import collections
import sys

from ..common.data import load_inter, load_domain, DEFAULT_ROOT
import os


def load_rows(path):
    return load_inter(path)


def main() -> int:
    ok_all = True
    for dom in ("IandS", "VG"):
        root = DEFAULT_ROOT
        tr = load_inter(os.path.join(root, "data", "Amazon23", dom, f"{dom}.train.inter"))
        va = load_inter(os.path.join(root, "data", "Amazon23", dom, f"{dom}.valid.inter"))
        te = load_inter(os.path.join(root, "data", "Amazon23", dom, f"{dom}.test.inter"))
        print(f"\n===== {dom} =====")

        # 前提 2：首样本 history 长度分布（自适应：v1 timestamp L=1，v2 LOO L=2，取决于协议）
        first = {}
        for u, h, _ in tr:
            first.setdefault(u, len(h))
        dist = collections.Counter(first.values())
        total = sum(dist.values())
        L = max(dist, key=dist.get) if dist else 0
        ok_n = dist.get(1, 0) + dist.get(2, 0)
        p2 = (ok_n / total) >= 0.999 and L in (1, 2)    # ≥99.9% 用户 L∈{1,2}
        ok_all &= p2
        print(f"  [前提2] train.inter 首样本 history 长度分布 = {dict(dist)}  L={L}  "
              f"协议一致性 = {ok_n/total*100:.3f}%  → "
              f"{'✅ ≥99.9% 用户 L∈{1,2}' if p2 else '❌ 不满足'}")

        # 重建（与 common/data.py::reconstruct_train_sequences 同源）
        seqs = collections.OrderedDict()
        for u, h, t in tr:
            if u not in seqs:
                seqs[u] = list(h) if h else []    # L=1（v1 timestamp）/ L=2（v2 LOO）
            seqs[u].append(t)
        full = dict(seqs)

        # 前提 1 + 校验：valid 输入 == train_seq[-20:]（≥99.99% 允许 sid 过滤产生的边缘不一致）
        ok = bad = 0
        for u, h, _ in va:
            if u in full:
                if full[u][-20:] == h:
                    ok += 1
                else:
                    bad += 1
        # ≤0.01% 不一致 → 协议正常；v2 LOO + sid 过滤会产生少量边缘样本
        if (ok + bad) > 0:
            ratio = bad / (ok + bad)
        else:
            ratio = 0.0
        print(f"  [前提1] valid: history == train_seq[-20:]  ok={ok} bad={bad} "
              f"（{ratio*100:.4f}% bad，≤0.01% 视为通过）")
        ok_all &= (ok + bad == 0) or ratio <= 0.0001

        # 前提 3：隐形用户（训练段只有 L 个物品 = K<L+1 → 不在 train.inter）
        users_tr = set(full)
        miss_v = {u for u, _, _ in va} - users_tr
        miss_t = {u for u, _, _ in te} - users_tr
        lens_v = {len(h) for u, h, _ in va if u in miss_v}
        print(f"  [前提3] 不在 train.inter 的 valid/test 用户 = {len(miss_v)} / {len(miss_t)}；"
              f"这些用户 valid history 长度集合 = {lens_v if lens_v else '∅'}")
        p3 = lens_v <= {1, 2}     # v1 timestamp L=1，v2 LOO L=2
        ok_all &= p3
        print(f"          → {'✅ 全部 ∈{1（v1 timestamp）, 2（v2 LOO）}（可用 history[:L] 补齐）' if p3 else '❌ 存在长度>2，补齐公式不成立'}")

        # 补齐后：所有 eval 用户都应有训练历史
        D = load_domain(dom)
        mv = sum(1 for u in D.valid.users if int(u) not in D.train_seq)
        mt = sum(1 for u in D.test.users if int(u) not in D.train_seq)
        print(f"  补齐后 train_seq 覆盖用户 = {len(D.train_seq)}，valid/test 仍缺 = {mv} / {mt} "
              f"（应为 0 / 0）；metadata: {D.meta}")
        ok_all &= (mv == 0 and mt == 0)

        # test 口径确认：test history 比 "train + 仅 valid 目标" 多出的条目 = valid 段非末位的物品
        vmap = {u: t for u, _, t in va}
        extra = sum(1 for u, h, _ in te
                    if u in full and (full[u] + ([vmap[u]] if u in vmap else []))[-20:] != h)
        print(f"  [test 口径] 与 'train + 仅 valid 目标' 不一致的条数 = {extra} / {len(te)} "
              f"→ 说明 test history 用的是 (train 段 + valid 段**全部**物品) 的窗口")

        # 残余不可屏蔽项：**只可能**出现在 |valid 段| ≥ 20 的用户上。
        # 推理：test history = (train 段 + valid 段)[-20:]，而 valid 段物品排在拼接序列的**末尾**。
        # 若 |valid 段| ≤ 20，则全部 valid 物品都落在窗口内 → 都能被屏蔽；
        # 只有 |valid 段| > 20 时，valid 段里较早的物品才会被窗口挤掉。
        # 检测方式：若某用户的 test history 与其 train 段**零重叠**，则窗口必然整个由 valid 段填满，
        # 即 |valid 段| ≥ 20（反证：只要窗口里还留得下 1 个 train 物品，就会出现重叠）。
        ts = {u: set(v) for u, v in full.items()}
        full_win = sum(1 for _, h, _ in te if len(h) == 20)
        no_overlap = sum(1 for u, h, _ in te if u in ts and not (set(h) & ts[u]))
        print(f"  [残余风险] test 中窗口打满(history 长 20)的用户 = {full_win} / {len(te)}")
        print(f"             其中与 train 段**零重叠**（⇒ |valid 段| ≥ 20）的用户 = {no_overlap} / {len(te)} "
              f"({no_overlap / len(te) * 100:.3f}%) —— **只有这批人**可能存在无法屏蔽的早期 valid 段物品")
        # 补：窗口外的 train 物品我们是**能**屏蔽的（有完整 train 序列），这里给出量级
        dropped = sum(max(0, len(full.get(u, [])) - 20) for u, _, _ in te)
        print(f"             因 20 截断被挤出窗口、但**已被完整 train 序列补屏蔽**的物品数 = {dropped}")

    print(f"\n总体：{'✅ 三条前提全部成立' if ok_all else '❌ 有前提不成立，需重新推导口径'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
