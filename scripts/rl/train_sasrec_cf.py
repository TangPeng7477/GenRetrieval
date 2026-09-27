"""训练一个给 GRPO 当 Reward Model 的 SASRec（reward_type="sasrec" / cf_reward）。

为什么单独写这个脚本
--------------------
`rl.py:403` 的 `cf_reward` 需要一个 SASRec 来给「每条 rollout 解码出的 item」打分：

    model = SASRec(32, item_num, len_seq, 0.3, device)      # rl.py:326
    scores = model.forward_eval(seq, len_lis)               # rl.py:434
    reward = scores.gather(1, pred_id)                      # 取 rollout 那个 item 的分数

它有三个硬约束（都来自根 `sasrec.py` 的 `SASRec` 类，不是 baseline 那套）：
1. 架构 = `SASRec(hidden_size=32, item_num, state_size=10, dropout=0.3, num_heads=1)`；
2. pad id = `item_num`（`nn.Embedding(item_num + 1)`），且 **padding 补在序列右侧**，
   `len_states` 传真实长度 —— 与 `cf_reward` 里 `his + [item_num]*(len_seq-len(his))` 完全一致；
3. 输入最多 `state_size=10` 个历史 item，超出要**截断到最近 10 个**（`cf_reward` 自己不截断，
   超长会在 `positional_embeddings(torch.arange(state_size))` 处炸）。

⚠️ 注意：`baseline/results/IandS/sasrec/` 里训过 SASRec，但
   a) 用的是 **baseline 自己的架构**（hidden=64 / n_layers=2 / n_heads=2 / maxlen=20），与上面 (1) 不匹配；
   b) **只存了 metrics.json / run.log，没存权重**。
   所以必须按 rl.py 期望的架构重训一个。

用法
----
    ./.venv/Scripts/python.exe scripts/rl/train_sasrec_cf.py --domain IandS --epochs 10
    # 权重默认写 models/sasrec_IandS_cf.pt，传给 rl_run0.sh 的 --cf_path
"""

import argparse
import ast
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 本脚本在 scripts/rl/ 下，Python 只把脚本所在目录加入 sys.path，
# 而根 `sasrec.py` 在仓库根 —— 必须显式补上，否则 `from sasrec import SASRec` 会 ModuleNotFoundError。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sasrec import SASRec

# ⚠️ 默认值必须与 `rl.py` 的 CF_HIDDEN / CF_LEN_SEQ / CF_DROPOUT / CF_HEADS **逐项一致**，
#    否则 rl.py 里 `load_state_dict` 会形状不符报错（故意让它响亮失败，不静默）。
# `[实测]` 32/10/0.3/1 这组偏弱（valid HR@10=0.0134）；换成 64/20/0.3/2 后接近 baseline 的 0.0494。
HIDDEN = 64           # rl.py CF_HIDDEN
STATE_SIZE = 20       # rl.py CF_LEN_SEQ（= 旧代码里的 len_seq）
DROPOUT = 0.3         # rl.py CF_DROPOUT
NUM_HEADS = 2         # rl.py CF_HEADS


def parse_item_list(cell):
    """history_item_id 列形如 "[4526, 18956]"（字符串）或已是 list。"""
    if isinstance(cell, (list, tuple)):
        return list(cell)
    try:
        return list(ast.literal_eval(cell))
    except (ValueError, SyntaxError):
        return []


def build_sequences(csv_path, item_num, state_size, max_rows=0):
    """返回 (seqs, lens, targets)：seq 已右侧 pad 到 state_size，len = 真实长度。"""
    df = pd.read_csv(csv_path)
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)

    seqs, lens, targets = [], [], []
    max_hist = 0
    for hist, tgt in zip(df["history_item_id"], df["item_id"]):
        hist = [int(x) for x in parse_item_list(hist)]
        max_hist = max(max_hist, len(hist))
        hist = hist[-state_size:]                          # 超长截断到最近 state_size 个
        n = len(hist)
        if n == 0:                                         # 无历史：跳过（SASRec 需要至少 1 个）
            continue
        seq = hist + [item_num] * (state_size - n)         # 右侧 pad，pad id = item_num
        seqs.append(seq)
        lens.append(n)
        targets.append(int(tgt))
    return (
        torch.LongTensor(seqs),
        torch.LongTensor(lens),
        torch.LongTensor(targets),
        max_hist,
    )


def count_items(info_path):
    with open(info_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


@torch.no_grad()
def evaluate(model, seqs, lens, targets, item_num, device, batch=512, k=10):
    """全库排序的 HR@K / NDCG@K（不做 mask，只用作「模型训出来没」的体检）。"""
    model.eval()
    hits, ndcgs, n = 0.0, 0.0, 0
    for i in range(0, len(seqs), batch):
        s = seqs[i : i + batch].to(device)
        l = lens[i : i + batch].to(device)
        t = targets[i : i + batch].to(device)
        logits = model.forward_eval(s, l)             # [B, item_num]
        rank = logits.argsort(dim=-1, descending=True).argsort(dim=-1)  # 0 = 最高分
        rank_t = rank.gather(1, t.view(-1, 1)).view(-1)
        hits += (rank_t < k).float().sum().item()
        ndcgs += (1.0 / torch.log2(rank_t.float() + 2) * (rank_t < k).float()).sum().item()
        n += t.numel()
    model.train()
    return hits / n, ndcgs / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="IandS")
    ap.add_argument("--sft-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train-rows", type=int, default=0, help="0 = 全部；调试用小样本")
    ap.add_argument("--max-valid-rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5, help="valid HR@10 连续多少轮不涨就停")
    # 架构参数：必须与 rl.py 的 CF_* 一致，改了这里就要同步改 rl.py
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--state-size", type=int, default=STATE_SIZE)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--heads", type=int, default=NUM_HEADS)
    # 🔴 训练目标必须是**全库 softmax CE**，不是 1 负例 BCE：
    #    `[实测]` 同数据同训练量下，BCE(1 neg) 的 valid HR@10 只有 0.0134，而 baseline
    #    （baseline/models/seq.py:89 用 F.cross_entropy 全库 softmax）是 0.0494 —— 差 4×。
    #    原因：只见过 1 个负例的模型从未学会把另外 2 万多个 item 压下去，全库排序必然崩。
    ap.add_argument("--loss", choices=["ce", "bce"], default="ce")
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
    print(f"[data] domain={d} item_num={item_num} device={device}")

    tr_path = os.path.join(sft, "train", f"{d}_5_train.csv")
    va_path = os.path.join(sft, "valid", f"{d}_5_valid.csv")
    st = args.state_size
    tr_s, tr_l, tr_t, max_hist_tr = build_sequences(tr_path, item_num, st, args.max_train_rows)
    va_s, va_l, va_t, max_hist_va = build_sequences(va_path, item_num, st, args.max_valid_rows)
    print(f"[data] train={len(tr_t)} valid={len(va_t)} max_history_len={max(max_hist_tr, max_hist_va)}")
    if max(max_hist_tr, max_hist_va) > st:
        print(f"[warn] 历史最长 {max(max_hist_tr, max_hist_va)} > state_size {st}，已截断到最近 {st} 个")

    model = SASRec(args.hidden, item_num, st, args.dropout, device, num_heads=args.heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] SASRec(hidden={args.hidden}, state={st}, heads={args.heads}) 参数量 {n_params/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    bce_ce = nn.CrossEntropyLoss()

    steps = int(np.ceil(len(tr_t) / args.batch))
    print(f"[train] epochs={args.epochs} batch={args.batch} steps/epoch={steps}")
    # 🔴 必须按 valid 指标保存**最佳**权重，不能存最后一个 epoch：
    #    `[实测]` 32/10 那版训 25 ep 时 best 在 ep9（HR@10 0.0134），末 epoch 已过拟合到 0.0066。
    best_hr, best_ep, bad_ep = -1.0, -1, 0
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr_t), device=tr_s.device)
        total = 0.0
        for i in range(0, steps):
            idx = perm[i * args.batch : (i + 1) * args.batch]
            s = tr_s[idx].to(device)
            l = tr_l[idx].to(device)
            pos = tr_t[idx].to(device)

            logits = model.forward(s, l)                       # [B, item_num]
            if args.loss == "ce":
                loss = bce_ce(logits, pos)                     # 全库 softmax 交叉熵
            else:
                pos_logit = logits.gather(1, pos.view(-1, 1)).view(-1)
                # 负采样：随机 item（撞上正例就换一批）
                neg = torch.randint(0, item_num, pos.shape, device=device)
                bad = neg == pos
                neg[bad] = (neg[bad] + 1) % item_num
                neg_logit = logits.gather(1, neg.view(-1, 1)).view(-1)
                loss = bce(pos_logit, torch.ones_like(pos_logit.float())) + \
                       bce(neg_logit, torch.zeros_like(neg_logit.float()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        hr, nd = evaluate(model, va_s, va_l, va_t, item_num, device)
        flag = ""
        if hr > best_hr:
            best_hr, best_ep, bad_ep = hr, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " <- best"
        else:
            bad_ep += 1
        print(f"[train] epoch {ep}/{args.epochs} loss={total/steps:.4f} "
              f"valid HR@10={hr:.4f} NDCG@10={nd:.4f}{flag}")
        if bad_ep >= args.patience:
            print(f"[early-stop] valid HR@10 连续 {bad_ep} 轮未提升，停在 epoch {ep}（best = epoch {best_ep}）")
            break

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), out)
    print(f"[save] {out}  (best epoch {best_ep}, valid HR@10={best_hr:.4f})")
    print(f"[next] 云端/本地传给 rl_run0.sh：--cf_path 指向该文件；RL 用 REWARD_TYPE=sasrec")


if __name__ == "__main__":
    main()
