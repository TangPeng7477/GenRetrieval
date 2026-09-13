"""sid_prefix 的证伪诊断 —— 回答「0.0545 是不是虚高」。

设计原则：**所有变体复用同一条 `N.evaluate` 链路**（同一份屏蔽集、同一个 rank 定义），
只有被怀疑的那个因素被替换掉。这样任何差异都只能归因于该因素。

变体：
  1. baseline      —— 原样 sid_prefix（应与 baseline/results 里的数字一致，作为自检）
  2. shuffle_item  —— SID 行随机打乱：破坏「物品 ↔ SID」的对应关系，
                      但保留 SID 的边际分布（每层各码的出现频次完全不变）。
                      → placebo 测试：若 HR@10 崩到 pop 水平，说明高分确实来自语义结构。
  3. random_sid    —— 完全随机 3 层 SID（每层均匀 0~255）。
                      → 连分布结构都破坏，双重保险。
  4. hist_last{k}  —— history 只保留最后 k 条（k=1/3/5/10），mask 屏蔽集仍为完整历史。
                      → 拆「20 条历史累积投票」贡献了多少。

另有一项纯统计诊断：target 与 history 各位置共享各层前缀的比例，对照「target vs 随机物品」。

用法（项目根目录）：
    ./.venv/Scripts/python.exe baseline/scripts/diag_sidprefix.py --domain IandS
产物：只打印，**不写 baseline/results/**（诊断结果不得污染正式主榜）。
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ..common import nn as N
from ..common.data import load_domain

ROOT_DEFAULT = None  # 由 --root 指定


def _silent(*_a, **_kw):
    """吞掉 model.fit 的日志。"""
    return None


def eval_one(model, data, split: str, device) -> dict:
    ev = getattr(data, split)
    r = N.evaluate(model.score, ev, data.n_items, model.pad_id, device,
                   batch_size=512, mask_seen=True)
    return r["metrics"]


def truncated_copy(data, k: int):
    """返回一个 data 的浅拷贝副本，其 test/valid 的 hist 被截断到最后 k 条。

    注意：masks **保持原样**（仍屏蔽完整历史），这样截短 history 只会让模型看到的信号变少，
    不会意外把历史物品放回候选 —— 否则会造成虚假提升。
    """
    import copy
    d = copy.copy(data)
    for name in ("valid", "test"):
        ev = getattr(data, name)
        nev = copy.copy(ev)
        nev.hist = [h[-k:] for h in ev.hist]
        nev._flat = None
        setattr(d, name, nev)
    return d


def prefix_sharing_stats(data, sid, split: str = "test", seed: int = 42):
    """统计 target 与 history 各位置共享前缀的比例，并给出随机对照。"""
    ev = getattr(data, split)
    rng = np.random.default_rng(seed)
    n_levels = sid.shape[1]
    # history 倒数第 j 个位置（j=1 表示最后一条）
    res = {}
    for j in (1, 2, 3, 5, 10):
        hits = {l: [] for l in range(n_levels)}
        rand_hits = {l: [] for l in range(n_levels)}
        n_ok = 0
        for h, t in zip(ev.hist, ev.targets):
            if len(h) < j:
                continue
            src = h[-j]
            # 随机对照：随机抽一个物品
            rnd = int(rng.integers(0, data.n_items))
            n_ok += 1
            for l in range(n_levels):
                hits[l].append(bool(sid[t, l] == sid[src, l]))
                rand_hits[l].append(bool(sid[t, l] == sid[rnd, l]))
        res[j] = {
            "n": n_ok,
            "same_with_hist_minus_j": {l: float(np.mean(hits[l])) for l in range(n_levels)},
            "same_with_random_item": {l: float(np.mean(rand_hits[l])) for l in range(n_levels)},
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="IandS")
    ap.add_argument("--root", default=r"D:\Codings\GenRetrieval")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"domain={args.domain}  device={device}")
    print("=" * 78)

    data = load_domain(args.domain, root=args.root)
    sid_orig = np.asarray(data.sid, dtype=np.int64)
    print(f"SID shape={sid_orig.shape}  n_items={data.n_items}  test={len(data.test)}")
    print()

    from ..generative.retrieval import SidPrefix

    def run(tag: str, sid=None, k=None):
        d = truncated_copy(data, k) if k else data
        if sid is None:
            sid = sid_orig
        d.sid = sid
        m = SidPrefix(n_items=d.n_items, n_users=d.n_users, maxlen=20)
        N.set_seed(42)
        m.fit(d, device=torch.device("cpu"), log=_silent)
        tr = eval_one(m, d, "test", device)
        print(f"  {tag:22} HR@10={tr['HR@10']:.4f}  NDCG@10={tr['NDCG@10']:.4f}  "
              f"MRR={tr['MRR']:.4f}  cov@10={tr['coverage@10']:.4f}")
        return tr

    print("【A 组】placebo：破坏 SID 语义")
    print("-" * 78)
    base = run("baseline (原样)")
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(sid_orig))
    run("shuffle_item", sid=sid_orig[perm])
    sid_rand = rng.integers(0, 256, size=sid_orig.shape, dtype=np.int64)
    run("random_sid", sid=sid_rand)

    print()
    print("【B 组】history 长度消融（模型只看最后 k 条）")
    print("-" * 78)
    for k in (1, 3, 5, 10, 20):
        run(f"hist_last{k}", k=k)

    print()
    print("【C 组】纯统计：target 与 history 倒数第 j 条共享第 l 层前缀的比例")
    print("-" * 78)
    st = prefix_sharing_stats(data, sid_orig, "test")
    print(f"  {'位置':>8} | {'第1层':>18} | {'第2层':>18} | {'第3层':>18}")
    print(f"  {'(vs 随机物品)':>8} | {'(vs 随机物品)':>18} | {'(vs 随机物品)':>18} |")
    for j, v in st.items():
        row = f"  h[-{j}]"
        for l in range(sid_orig.shape[1]):
            hi = v["same_with_hist_minus_j"][l]
            rd = v["same_with_random_item"][l]
            row += f" | {hi:7.4f} / {rd:7.4f}"
        print(row + "   (n={})".format(v["n"]))
    print()
    print("  读法：左侧是「target 与历史第 j 条」共享该层码的比例，右侧是「target 与随机物品」的对照。")
    print("        若左侧远高于右侧 ⇒ 相邻购买确实落在同一语义区，sid_prefix 在吃这个红利。")


if __name__ == "__main__":
    main()
