"""训练一个给 GRPO 当 Reward Model 的 SASRec（reward_type="sasrec" / cf_reward）。

为什么单独写这个脚本
--------------------
`rl.py` 的 `cf_reward` 需要一个 SASRec 来给「每条 rollout 解码出的 item」打分：

    logits = model.logits(seq)                 # [B, item_num]（baseline net）
    reward = logits.gather(1, pred_id)         # 取 rollout 那个 item 的分数

`baseline/results/IandS/sasrec/` 里训过一个（valid HR@10 0.0494 / NDCG@10 0.030），
但**只存了 metrics.json / run.log，没存权重** ⟹ 要当 RM 必须重训一份权重出来。

🔴 版本史（这段是踩坑记录，别删）
--------------------------------
v1（2026-09-27 上午）用**根 `sasrec.py` 的 SASRec 类** + 1 负例 BCE：
    valid HR@10 = 0.0134（best ep9，末 ep 已过拟合到 0.0066）
v2 把 BCE 换成**全库 softmax CE**：0.0192（best ep7）—— 涨了 1.4×，但仍只有 baseline 的 40%。
v3（本版）改回 **baseline 自己的 `SASRecNet`**：与 baseline 逐项对齐。

**v2 → baseline 差 2.5× 的原因（逐项核对源码，非猜测）**：

| 项 | 根 `sasrec.py` 的 SASRec（v1/v2） | baseline 的 SASRecNet |
|---|---|---|
| **层数** | **只有 1 层**！`self.mh_attn` 是单个模块、不是 ModuleList，<br>传 `num_heads=2` 只改**头数**，层数恒为 1 | `n_layers=2`（ModuleList + FFN(GELU) + 双 LN） |
| **输出层** | `self.s_fc = nn.Linear(hidden, item_num)` 独立线性层<br>（1.65M 参数，占我 3.36M 的一半） | **共享 item embedding 做 dot-product**<br>`encode(hist)[:, -1] @ item_emb.weight.T` |
| 填充 | **右**填充 + `gather(len-1)` ⟹ 末位位置随长度变 | **左**填充 + `[:, -1]` ⟹ 末位恒在 maxlen-1，位置语义一致 |
| 输入缩放 | 无（×√hidden 那行被注释掉了） | `item_emb * sqrt(hidden)`，init std 0.02 |
| 优化器 | `torch.optim.Adam(lr=1e-3)`，无 wd / 无梯度裁剪 | `AdamW(lr=1e-3, wd=0)` + `clip_grad_norm_(5.0)` |
| batch / 早停 | 128 / patience 5 / monitor HR@10 | **512** / patience 4 / monitor **NDCG@10** |

⚠️ 我 v2 时以为「把 heads 从 1 加到 2 = 架构对齐」，其实**层数一直是 1** ——
   所以那次「升级架构」几乎没效果（0.0134 → 0.0120，BCE 版甚至略降）。

`[实测]` **评估口径不是主因**：baseline 评估默认 `mask_seen=True`（屏蔽该用户
history ∪ 完整训练序列，平均只屏蔽 6.1 个 item）；我拿 v2 的权重实测
不屏蔽 0.0192 → 屏蔽 history 0.0196 → 屏蔽 history∪train_seq **0.0196**
⟹ 只值 **+0.0004**，2.5× 的差距**不在评估协议，在模型与训练配置**。

用法
----
    # 默认 = baseline 架构（推荐，与 baseline/results 那份同配置）
    ./.venv/Scripts/python.exe scripts/rl/train_sasrec_cf.py --domain IandS

    # 退回根 sasrec.py 的实现（对照用）
    ./.venv/Scripts/python.exe scripts/rl/train_sasrec_cf.py --domain IandS --net root

权重写 `models/sasrec_<DOMAIN>_cf.pt`，云端用 `rl_run0.sh` 的 `CF_PATH` 指过去。
⚠️ 换 `--net` / `--hidden` / `--layers` / `--state-size` 后，**必须同步改 `rl.py` 的
   `CF_*` 常量**，否则 `load_state_dict` 会形状不符而报错（故意让它响亮失败）。
"""

import argparse
import ast
import collections
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 本脚本在 scripts/rl/ 下，Python 只把脚本所在目录加入 sys.path，
# 而根 sasrec.py / baseline 包都在仓库根 —— 必须显式补上。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sasrec import SASRec                       # 根实现（--net root）
from baseline.models.seq import SASRecNet       # baseline 实现（默认）

# ⚠️ 默认值必须与 rl.py 的 CF_* 常量逐项一致
NET = "baseline"
HIDDEN = 64
LAYERS = 2
STATE_SIZE = 20          # maxlen
DROPOUT = 0.2
NUM_HEADS = 2
LR = 1e-3
WD = 0.0
CLIP = 5.0


def parse_item_list(cell):
    """history_item_id 列形如 "[4526, 18956]"（字符串）或已是 list。"""
    if isinstance(cell, (list, tuple)):
        return list(cell)
    try:
        return list(ast.literal_eval(cell))
    except (ValueError, SyntaxError):
        return []


def build_sequences(csv_path, item_num, state_size, max_rows=0, left_pad=True):
    """返回 (seqs, lens, targets, hists, users, max_hist)。

    left_pad=True  -> 历史靠右、pad 在左（baseline / SASRecNet 口径，末位恒为最新 item）
    left_pad=False -> 历史靠左、pad 在右（根 sasrec.py 口径，需 len_states 定位末位）
    """
    df = pd.read_csv(csv_path)
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)

    seqs, lens, targets, hists, users = [], [], [], [], []
    max_hist = 0
    for hist, tgt, u in zip(df["history_item_id"], df["item_id"], df["user_id"]):
        hist = [int(x) for x in parse_item_list(hist)]
        max_hist = max(max_hist, len(hist))
        hist = hist[-state_size:]              # 超长截断到最近 state_size 个
        n = len(hist)
        if n == 0:                             # 无历史：跳过（SASRec 需要至少 1 个）
            continue
        if left_pad:
            seq = [item_num] * (state_size - n) + hist
        else:
            seq = hist + [item_num] * (state_size - n)
        seqs.append(seq)
        lens.append(n)
        targets.append(int(tgt))
        hists.append(hist)
        # ⚠️ user_id 是字符串（形如 "A20679"），**不能 int()**（`[实测]` 直接抛
        #    ValueError: invalid literal for int() with base 10: 'A20679'）
        users.append(u)
    return (
        torch.LongTensor(seqs),
        torch.LongTensor(lens),
        torch.LongTensor(targets),
        hists,
        users,
        max_hist,
    )


def build_train_seq(csv_path):
    """每用户的完整训练序列（含 history 与 target）—— 用于 mask_seen 对照。"""
    df = pd.read_csv(csv_path)
    seq = collections.defaultdict(set)
    for u, h, t in zip(df["user_id"], df["history_item_id"], df["item_id"]):
        s = seq[u]                      # user_id 是字符串，直接用原值做 key
        s.update(int(x) for x in parse_item_list(h))
        s.add(int(t))
    return seq


def count_items(info_path):
    with open(info_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def score_batch(model, net, s, l):
    """统一打分入口：[B, state] -> [B, item_num]。"""
    return model.logits(s) if net == "baseline" else model.forward_eval(s, l)


@torch.no_grad()
def evaluate(model, net, seqs, lens, targets, item_num, device,
             hists=None, users=None, train_seq=None, batch=512, k=10):
    """全库排序的 HR@K / NDCG@K。

    默认**不 mask**（与 v1/v2 同口径，可纵向比）；传了 hists/users/train_seq 会额外
    报一版 mask_seen（baseline 口径），用于与 baseline/results 横向比。
    """
    model.eval()
    hits, ndcgs, n = 0.0, 0.0, 0
    hits_m, ndcgs_m = 0.0, 0.0
    do_mask = hists is not None and train_seq is not None
    for i in range(0, len(seqs), batch):
        s = seqs[i : i + batch].to(device)
        l = lens[i : i + batch].to(device)
        t = targets[i : i + batch].to(device)
        logits = score_batch(model, net, s, l)                     # [B, item_num]

        def _rank_of(x):
            r = x.argsort(dim=-1, descending=True).argsort(dim=-1)
            return r.gather(1, t.view(-1, 1)).view(-1).float()

        rt = _rank_of(logits)
        hits += (rt < k).sum().item()
        ndcgs += (1.0 / torch.log2(rt + 2) * (rt < k)).sum().item()
        n += t.numel()

        if do_mask:
            x = logits.clone()
            for j in range(x.shape[0]):
                m = set(hists[i + j]) | train_seq[users[i + j]]
                idx = torch.tensor([v for v in m if v < item_num], device=device, dtype=torch.long)
                if idx.numel():
                    x[j, idx] = -1e9
            rt_m = _rank_of(x)
            hits_m += (rt_m < k).sum().item()
            ndcgs_m += (1.0 / torch.log2(rt_m + 2) * (rt_m < k)).sum().item()
    model.train()
    out = (hits / n, ndcgs / n)
    if do_mask:
        out = out + (hits_m / n, ndcgs_m / n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="IandS")
    ap.add_argument("--sft-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--net", choices=["baseline", "root"], default=NET)
    ap.add_argument("--epochs", type=int, default=25)          # baseline: 25
    ap.add_argument("--batch", type=int, default=512)          # baseline: 512
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--wd", type=float, default=WD)
    ap.add_argument("--clip", type=float, default=CLIP)
    ap.add_argument("--max-train-rows", type=int, default=0, help="0 = 全部；调试用小样本")
    ap.add_argument("--max-valid-rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=4)         # baseline: 4
    ap.add_argument("--monitor", choices=["hr", "ndcg"], default="ndcg")  # baseline: NDCG@10
    # 🔴 monitor 必须用 **mask_seen 那一版**：baseline 的 valid_fn 走的是 evaluate 默认
    #    mask_seen=True。用不屏蔽的 NDCG 选 epoch 会**选早**（`[实测]` 全量 valid 下不屏蔽
    #    NDCG 的峰在 ep3=0.0228，而 mask_seen 的峰在 ep6=0.0300）⟹ 存下来的权重差 9%：
    #    ep3 mask HR@10=0.0472 vs ep6 **0.0513**（baseline 报的是 0.0517）。
    ap.add_argument("--monitor-seen", type=int, default=1, help="1 = 用 mask_seen 版指标选 epoch")
    # 架构参数：必须与 rl.py 的 CF_* 一致，改了这里就要同步改 rl.py
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--state-size", type=int, default=STATE_SIZE)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--heads", type=int, default=NUM_HEADS)
    args = ap.parse_args()

    d = args.domain
    sft = args.sft_dir or os.path.join("data", "Amazon23", d, "sft")
    out = args.out or os.path.join("models", f"sasrec_{d}_cf.pt")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info_path = os.path.join(sft, "info", f"{d}.item_info.txt")
    item_num = count_items(info_path)
    left_pad = args.net == "baseline"
    print(f"[data] domain={d} item_num={item_num} device={device} net={args.net} "
          f"pad={'left' if left_pad else 'right'}")

    tr_path = os.path.join(sft, "train", f"{d}_5_train.csv")
    va_path = os.path.join(sft, "valid", f"{d}_5_valid.csv")
    st = args.state_size
    tr_s, tr_l, tr_t, _, _, max_tr = build_sequences(tr_path, item_num, st, args.max_train_rows, left_pad)
    va_s, va_l, va_t, va_h, va_u, max_va = build_sequences(va_path, item_num, st, args.max_valid_rows, left_pad)
    train_seq = build_train_seq(tr_path)
    print(f"[data] train={len(tr_t)} valid={len(va_t)} max_history_len={max(max_tr, max_va)}")
    if max(max_tr, max_va) > st:
        print(f"[warn] 历史最长 {max(max_tr, max_va)} > state_size {st}，已截断到最近 {st} 个")

    if args.net == "baseline":
        model = SASRecNet(item_num, pad_id=item_num, maxlen=st, hidden=args.hidden,
                          n_layers=args.layers, n_heads=args.heads, dropout=args.dropout).to(device)
    else:
        model = SASRec(args.hidden, item_num, st, args.dropout, device,
                       num_heads=args.heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] net={args.net} hidden={args.hidden} layers={args.layers} state={st} "
          f"heads={args.heads} dropout={args.dropout} 参数量 {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()

    steps = int(np.ceil(len(tr_t) / args.batch))
    print(f"[train] epochs={args.epochs} batch={args.batch} steps/epoch={steps} "
          f"lr={args.lr} wd={args.wd} clip={args.clip} monitor={'NDCG@10' if args.monitor=='ndcg' else 'HR@10'}")
    # 🔴 必须按 valid 指标保存**最佳**权重，不能存最后一个 epoch：
    #    `[实测]` v1（32/10 + BCE）训 25 ep 时 best 在 ep9（0.0134），末 epoch 已过拟合到 0.0066。
    best, best_ep, bad_ep = -1.0, -1, 0
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr_t))
        total = 0.0
        for i in range(steps):
            idx = perm[i * args.batch : (i + 1) * args.batch]
            s = tr_s[idx].to(device)
            l = tr_l[idx].to(device)
            pos = tr_t[idx].to(device)
            if args.net == "baseline":
                loss = model.loss(s, pos)                      # 内部就是全库 softmax CE
            else:
                loss = ce(model.forward(s, l), pos)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += loss.item()

        res = evaluate(model, args.net, va_s, va_l, va_t, item_num, device,
                       hists=va_h, users=va_u, train_seq=train_seq)
        hr, nd, hr_m, nd_m = res
        m_nd, m_hr = (nd_m, hr_m) if args.monitor_seen else (nd, hr)
        cur = m_nd if args.monitor == "ndcg" else m_hr
        flag = ""
        if cur > best:
            best, best_ep, bad_ep = cur, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " <- best"
        else:
            bad_ep += 1
        print(f"[train] epoch {ep}/{args.epochs} loss={total/steps:.4f} "
              f"valid HR@10={hr:.4f} NDCG@10={nd:.4f} | mask_seen HR@10={hr_m:.4f} "
              f"NDCG@10={nd_m:.4f}{flag}")
        if bad_ep >= args.patience:
            print(f"[early-stop] {args.patience} 轮未提升，停在 epoch {ep}（best = epoch {best_ep}）")
            break

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), out)
    print(f"[save] {out}  (best epoch {best_ep}, monitor={best:.4f})")
    print(f"[next] rl_run0.sh 用 CF_PATH 指向它；RL 跑 REWARD_TYPE=sasrec")


if __name__ == "__main__":
    main()
