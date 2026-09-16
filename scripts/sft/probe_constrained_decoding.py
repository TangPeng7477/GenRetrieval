# -*- coding: utf-8 -*-
"""约束解码链路实测探针（Run-0 前）

回答三个问题（全部用真实数据 + 真实 tokenizer，不靠推断）：

  A. 训练端 prompt 与评估端 prompt 是否逐 token 一致？
     —— 这是 "训练/评估要不要同一套约束" 的真正落点。
        约束解码本身只在推理时需要；但 prompt 模板 + 终止约定必须两边一致。

  B. evaluate.py 的 Trie（hash_dict）构建出来是几步、每步允许什么？
     —— 验证 prefix_index=3 的假设，以及与训练目标 [a,b,c,\n,EOS] 的对应。

  C. evaluate.py 用的裸 tokenizer（add_special_tokens 默认 True）与 data.py 的
     Tokenizer.encode（同样默认）在 "### Response:\n" 上切分是否一致？
     —— prefix_index 是硬编码常量，两边切分不同就会静默错位。

用法：
  ./.venv/Scripts/python.exe scripts/sft/probe_constrained_decoding.py --domain IandS --n-rows 5
"""
import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PASS, FAIL = "PASS", "FAIL"
_fail = 0


def check(tag, msg, ok):
    global _fail
    if not ok:
        _fail += 1
    print(f"  [{PASS if ok else FAIL}] {tag:28s} {msg}")
    return ok


def get_hash(x):
    """与 evaluate.py:24 完全一致"""
    x = [str(_) for _ in x]
    return "-".join(x)


# ---------------------------------------------------------------- prompt 构造
CAT_NAME = {"IandS": "Industrial_and_Scientific", "VG": "Video_Games"}


def build_real_datasets(csv, tok, n_rows, domain):
    """直接实例化 data.py 的真实 Dataset 类取 input_ids。

    [为什么不复刻] 我第一版在这里手抄了 get_history 的句子，结果 data.py 修好之后
    探针仍然 FAIL —— 手抄的副本不会跟着真实代码走。改为直接调用真实类，
    这个探针就从"复刻说明"升级成了"回归测试"。
    """
    import data as D
    kw = dict(train_file=csv, tokenizer=tok, max_len=99999, sample=n_rows,
              test=True, seed=0, category=CAT_NAME.get(domain, ""))
    return D.SidSFTDataset(**kw), D.EvalSidDataset(**kw)


def enc(tok, s, bos=False, eos=False):
    """复刻 data.py:13 Tokenizer.encode"""
    t = tok.encode(s)
    bos_id, eos_id = tok.bos_token_id, tok.eos_token_id
    while t and bos_id is not None and t[0] == bos_id:
        t = t[1:]
    while t and eos_id is not None and t[-1] == eos_id:
        t = t[:-1]
    if bos and bos_id is not None:
        t = [bos_id] + t
    if eos and eos_id is not None:
        t = t + [eos_id]
    return t


# ---------------------------------------------------------------------- Trie
def build_hash_dict(tokenizer, info_path, prefix_index=3):
    """复刻 evaluate.py:61-119（semantic 侧）"""
    with open(info_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    semantic_ids = [ln.split("\t")[0].strip() + "\n" for ln in lines]
    info_semantic = [f"### Response:\n{_}" for _ in semantic_ids]
    prefixID = [tokenizer(_).input_ids for _ in info_semantic]

    hash_dict = {}
    for ID in prefixID:
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            h = get_hash(ID[:i]) if i == prefix_index else get_hash(ID[prefix_index:i])
            hash_dict.setdefault(h, set()).add(ID[i])
    for k in hash_dict:
        hash_dict[k] = sorted(hash_dict[k])
    return hash_dict, prefixID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="IandS")
    ap.add_argument("--model-dir", default="models/Qwen3-0.6B")
    ap.add_argument("--n-rows", type=int, default=5)
    args = ap.parse_args()

    os.chdir(ROOT)
    sft = f"data/Amazon23/{args.domain}/sft"
    csv = f"{sft}/train/{args.domain}_5_train.csv"
    info = f"{sft}/info/{args.domain}.item_info.txt"

    print("=" * 78)
    print(f"约束解码链路探针  domain={args.domain}  model={args.model_dir}")
    print("=" * 78)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    # [必须] 先注册 SID 词表 —— 与 sft.py:241-306 同一口径（sid_vocab.json，码序）。
    # 不注册的话 <a_115> 会被切成 17 个 token，整条链路都测不出真东西。
    with open(f"{sft}/info/sid_vocab.json", encoding="utf-8") as f:
        sid_vocab = json.load(f)
    n_added = tok.add_tokens(sid_vocab)
    print(f"\n[env] vocab={len(tok)}  eos={tok.eos_token_id}({tok.eos_token!r})  bos={tok.bos_token_id}")
    print(f"[env] 注册 SID: {n_added}/{len(sid_vocab)} 个新 token，id 从 {len(tok)-n_added} 起")

    df = pd.read_csv(csv, nrows=args.n_rows)
    print(f"[env] csv rows={len(df)}  info rows={sum(1 for _ in open(info, encoding='utf-8'))}")

    # ---------------------------------------------------------------- A
    print("\n" + "-" * 78)
    print("A. 训练端 vs 评估端 prompt 一致性（直接调用 data.py 真实类）")
    print("-" * 78)
    ds_tr, ds_ev = build_real_datasets(csv, tok, args.n_rows, args.domain)
    r = ds_tr.data.iloc[1].copy()
    r_ev = ds_ev.data.iloc[1].copy()

    tr_ids = ds_tr[1]["input_ids"]
    ev_ids = ds_ev[1]["input_ids"]
    tr_str = tok.decode(tr_ids)
    ev_str = tok.decode(ev_ids)
    # get_history 会原地改写 row（row['history_item_sid'] = eval(...)），必须传副本
    tr_tgt = ds_tr.get_history(r)["output"]
    ev_tgt = ds_ev.get_history(r_ev)["output"]
    print(f"  训练端类 = {type(ds_tr).__name__}  评估端类 = {type(ds_ev).__name__}")
    check("两侧取样同一行商品", f"{r['item_sid']}", r["item_sid"] == r_ev["item_sid"])

    print(f"  训练端 prompt 尾部 80 字符: ...{tr_str[-80:]!r}")
    print(f"  评估端 prompt 尾部 80 字符: ...{ev_str[-80:]!r}")
    print()
    print(f"  训练端 len={len(tr_ids)}   评估端 len={len(ev_ids)}   diff={len(ev_ids)-len(tr_ids)}")

    same = tr_ids == ev_ids
    check("prompt 逐 token 一致", f"{'是' if same else '否 —— train/eval 不一致'}", same)

    # 定位第一个分歧位置
    if not same:
        for i, (a, b) in enumerate(zip(tr_ids, ev_ids)):
            if a != b:
                print(f"       首个分歧 @ token #{i}")
                print(f"         训练端: {tr_ids[max(0,i-4):i+8]}")
                print(f"                {tok.decode(tr_ids[max(0,i-4):i+8])!r}")
                print(f"         评估端: {ev_ids[max(0,i-4):i+8]}")
                print(f"                {tok.decode(ev_ids[max(0,i-4):i+8])!r}")
                lo = min(len(tr_ids), len(ev_ids))
                print(f"       共同前缀长度 = {i} / {lo}")
                break

    # 目标一致性（target 决定 Trie 与 labels 的对齐）
    tr_lab = enc(tok, tr_tgt, bos=False, eos=True)
    ev_lab = enc(tok, ev_tgt, bos=False, eos=True)
    print()
    print(f"  训练 target ids: {tr_lab}  -> {tok.decode(tr_lab)!r}")
    check("训练/评估 target 一致", f"len={len(tr_lab)}",
          tr_lab == ev_lab and len(tr_lab) == 5)
    sid_ids = enc(tok, str(r["item_sid"]), bos=False, eos=False)
    nl_ids = enc(tok, chr(10), bos=False, eos=False)
    print(f"  target 结构: {len(sid_ids)} SID {sid_ids} + '\\n'{nl_ids} + EOS({tok.eos_token_id})")
    check("target = 3 SID + \\n + EOS", f"{tr_lab}",
          tr_lab[:3] == sid_ids and tr_lab[-1] == tok.eos_token_id)

    # prompt 末尾 3 token（Trie 第 0 步的 key）
    print()
    for tag, ids in (("训练端", tr_ids), ("评估端", ev_ids)):
        tail = ids[-3:]
        print(f"  {tag} prompt 末 3 token: {tail} -> {tok.decode(tail)!r}")
    tail_ok = ev_ids[-3:] == enc(tok, "### Response:\n", False, False)
    check("评估端末尾 == '### Response:\\n'", f"{ev_ids[-3:]}", tail_ok)

    # ---------------------------------------------------------------- B
    print("\n" + "-" * 78)
    print("B. evaluate.py 的 Trie（hash_dict）形状")
    print("-" * 78)
    hash_dict, prefixID = build_hash_dict(tok, info)
    L0 = len(prefixID[0])
    print(f"  info 条数={len(prefixID)}  单条 SID token 长度={L0}（含 eos）")
    print(f"  hash_dict 层数 = {L0 - 3}  （= len(ID)-prefix_index = 8-3）")
    print()

    probe = prefixID[0]
    for step in range(L0 - 3):
        i = 3 + step
        h = get_hash(probe[:i]) if i == 3 else get_hash(probe[3:i])
        allowed = hash_dict.get(h, [])
        sample = tok.decode(allowed[:3]) if allowed else "(none)"
        dec = tok.decode([probe[i]])
        print(f"    step{step}  i={i}  key={h[:42]:42s} "
              f"n_allowed={len(allowed):5d}  e.g.{sample!r}  <- 真值 {probe[i]}({dec!r})")
    check("Trie 步数 == 5", f"{L0 - 3}", (L0 - 3) == 5)

    # 每步的真值必须在允许集合内（否则第一步就崩）
    ok_all = True
    for step in range(L0 - 3):
        i = 3 + step
        h = get_hash(probe[:i]) if i == 3 else get_hash(probe[3:i])
        if probe[i] not in hash_dict.get(h, []):
            ok_all = False
            print(f"       [!!] step{step} 真值 {probe[i]} 不在允许集合")
    check("真值全在允许集合", "是" if ok_all else "否", ok_all)

    # 每步的候选规模（信息量：a/b/c 层应各 256，\\n 与 EOS 应各 1）
    print()
    sizes = []
    for step in range(L0 - 3):
        i = 3 + step
        h = get_hash(probe[:i]) if i == 3 else get_hash(probe[3:i])
        sizes.append(len(hash_dict.get(h, [])))
    print(f"  某条 SID 逐层候选数: {sizes}")
    check("第4步(\\n)候选==1", f"{sizes[3] if len(sizes)>3 else '?'}", len(sizes) > 3 and sizes[3] == 1)
    check("第5步(EOS)候选==1", f"{sizes[4] if len(sizes)>4 else '?'}", len(sizes) > 4 and sizes[4] == 1)

    # ---------------------------------------------------------------- C
    print("\n" + "-" * 78)
    print("C. prefix_index=3 的前提：'### Response:\\n' 的切分")
    print("-" * 78)
    s = "### Response:\n"
    a = tok(s).input_ids                      # evaluate.py 用法（默认 add_special_tokens=True）
    b = tok.encode(s, add_special_tokens=False)
    c = enc(tok, s, bos=False, eos=False)     # data.py Tokenizer.encode
    print(f"  tokenizer(s)                 -> {a}  len={len(a)}")
    print(f"  encode(..., add_special=False)-> {b}  len={len(b)}")
    print(f"  data.py Tokenizer.encode     -> {c}  len={len(c)}  (去掉首尾bos/eos后)")
    check("三种路径长度均为 3", f"{len(a)}/{len(b)}/{len(c)}", len(a) == len(b) == len(c) == 3)
    check("三种路径完全相同", f"{a} == {b} == {c}", a == b == c)

    print("\n" + "=" * 78)
    print(f"结果: {'全部通过' if _fail == 0 else f'{_fail} 项未通过'}")
    print("=" * 78)
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
