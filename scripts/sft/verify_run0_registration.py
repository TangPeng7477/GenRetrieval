"""Run-0 前置自检：SID 词表注册 + 数据集端到端 token 化。

为什么需要它
------------
SFT 一旦上 3090 开跑，数据侧的问题要等到 loss 不降才暴露，试错成本很高。
本脚本用**真实 tokenizer + 真实构造产物**把"注册 SID"和"读 CSV"这两步
先验穿，把问题拦在本地。

检查项
------
A. 词表    : sid_vocab.json 的码序、index.json 覆盖性、注册后 id 落点
B. T1      : SidSFTDataset 的 input_ids/labels 构造，目标恰好 3 个 SID + EOS
C. T2a/T2b : SidItemFeatDataset，SID 在 input 侧 3 token / title 在 output 侧
D. 长度    : 与 cutoff_len 的关系（P50 / P95 / max）

用法
----
    ./.venv/Scripts/python.exe scripts/sft/verify_run0_registration.py --domain IandS
    ./.venv/Scripts/python.exe scripts/sft/verify_run0_registration.py --domain IandS --n-sample -1
    ./.venv/Scripts/python.exe scripts/sft/verify_run0_registration.py --domain IandS --with-model
"""

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CATEGORY_MAP = {
    "IandS": "industrial and scientific items",
    "Video_Games": "video games",
    "VG": "video games",
}


def hr(title):
    print()
    print("=" * 66)
    print(f"  {title}")
    print("=" * 66)


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="IandS", help="IandS | VG")
    p.add_argument("--base-model", default=os.path.join(ROOT, "models", "Qwen3-0.6B"))
    p.add_argument("--data-root", default=os.path.join(ROOT, "data", "Amazon23"))
    p.add_argument("--n-sample", type=int, default=512, help="-1 为全量")
    p.add_argument("--cutoff-len", type=int, default=400)
    p.add_argument("--with-model", action="store_true",
                   help="额外加载权重，验证 resize_token_embeddings 的真实形状")
    return p.parse_args()


def main():
    args = parse_args()
    dom = args.domain
    sft_dir = os.path.join(args.data_root, dom, "sft")
    vocab_path = os.path.join(sft_dir, "info", "sid_vocab.json")
    index_path = os.path.join(sft_dir, "index", f"{dom}.index.json")
    item_path = os.path.join(sft_dir, "index", f"{dom}.item.json")
    train_csv = os.path.join(sft_dir, "train", f"{dom}_5_train.csv")
    valid_csv = os.path.join(sft_dir, "valid", f"{dom}_5_valid.csv")
    category = CATEGORY_MAP.get(dom, "industrial and scientific items")

    for p in (vocab_path, index_path, item_path, train_csv, valid_csv):
        if not os.path.exists(p):
            print(f"[FATAL] 缺文件: {p}")
            return 2

    # ---------------------------------------------------------------- A
    hr(f"A. 词表注册  [{dom}]")
    from sft import SidVocabLoader  # 复用真实训练路径上的代码

    loader = SidVocabLoader(vocab_path)
    vocab = loader.tokens
    print(f"  vocab file        : {os.path.relpath(vocab_path, ROOT)}")
    print(f"  n_tokens          : {len(vocab)}")
    print(f"  head              : {vocab[:4]}")
    print(f"  tail              : {vocab[-4:]}")

    # 分层结构（应为 <a_*>,<b_*>,<c_*> 各 256）
    layers = collections.Counter(t[1:t.index(">")].split("_")[0] for t in vocab)
    print(f"  layer histogram   : {dict(layers)}")

    used, missing = loader.check_coverage(index_path)
    print(f"  index coverage    : used={len(used)} / vocab={len(vocab)} / missing={len(missing)}")
    if missing:
        print(f"  [FAIL] missing tokens: {missing[:10]}")
        return 1
    if len(used) < len(vocab):
        print(f"  [WARN] {len(vocab)-len(used)} 个码在 index 中未被使用（死码，非致命）")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    before = len(tok)
    added = tok.add_tokens(vocab)
    after = len(tok)
    print(f"  len(tokenizer)    : {before} -> {after}  (+{added} requested {len(vocab)})")

    ids = [tok.convert_tokens_to_ids(t) for t in vocab]
    print(f"  id range          : [{min(ids)}, {max(ids)}]")
    monotonic = ids == list(range(ids[0], ids[0] + len(ids)))
    print(f"  码序 == id 连续   : {monotonic}   <- 必须 True，否则 M4 码本初始化会错位")
    if not monotonic:
        return 1

    probe = "<a_115><b_51><c_233>"
    enc = tok.encode(probe, add_special_tokens=False)
    print(f"  probe '{probe}' -> {enc} (len={len(enc)})")
    if len(enc) != 3:
        print("  [FAIL] SID 未按 3 token 编码")
        return 1

    # --- A2. Trie 前置：评估端硬编码 prefix_index=3，取决于模板结尾的切分 ---
    pre = tok.encode("### Response:\n", add_special_tokens=False)
    print(f"  '### Response:\\n' -> {pre} len={len(pre)} "
          f"{[tok.decode([i]) for i in pre]}")
    print(f"  prefix_index 匹配  : {len(pre) == 3}  "
          f"(evaluate.py:84 / LogitProcessor.py:41 硬编码 3)")
    if len(pre) != 3:
        print("  [FAIL] prefix_index 与实际切分不符，Trie 会整体错位")
        return 1

    if args.with_model:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
        rows_before = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tok))
        rows_after = model.get_input_embeddings().weight.shape[0]
        print(f"  embedding rows    : {rows_before} -> {rows_after}")
        if rows_after != after:
            print("  [FAIL] embedding 行数与 tokenizer 不一致")
            return 1
        del model

    # ---------------------------------------------------------------- B
    hr("B. T1 seq2sid  (SidSFTDataset)")
    from data import SidSFTDataset

    ds = SidSFTDataset(train_file=train_csv, tokenizer=tok, max_len=args.cutoff_len,
                       sample=args.n_sample, seed=42, category=category)
    print(f"  样本数            : {len(ds)}")
    rec = ds[0]
    prompt_len = sum(1 for x in rec["labels"] if x == -100)
    label_ids = [x for x in rec["labels"] if x != -100]
    print(f"  一条样本          : len(input_ids)={len(rec['input_ids'])} "
          f"prompt_len={prompt_len} label_len={len(label_ids)}")
    print(f"  label 解码        : {tok.decode(label_ids, skip_special_tokens=False)!r}")
    print(f"  eos_token_id      : {tok.eos_token_id}")

    # 目标结构 = [3 个 SID] + [\n, EOS]
    # \n 不是脏数据：LogitProcessor 第 4 步只放 \n、第 5 步才放 EOS，
    # 与 data.py:417 的 output = target_item + "\n" 完全自洽。
    sid_ids = set(ids)
    nl = tok.encode("\n", add_special_tokens=False)
    nl_id = nl[0] if len(nl) == 1 else None
    print(f"  newline_id        : {nl_id}")

    bad = layer_bad = 0
    for i in range(len(ds)):
        lab = [x for x in ds[i]["labels"] if x != -100]
        core = [x for x in lab if x in sid_ids]
        tail = [x for x in lab if x not in sid_ids]
        if len(core) != 3 or tail != [nl_id, tok.eos_token_id]:
            bad += 1
            if bad <= 3:
                print(f"  [FAIL] idx={i} core={len(core)} tail={tail} lab={lab}")
            continue
        if [(x - ids[0]) // 256 for x in core] != [0, 1, 2]:
            layer_bad += 1
            if layer_bad <= 3:
                print(f"  [FAIL] idx={i} 层序非 a->b->c: {core}")
    print(f"  目标结构校验      : {len(ds)-bad}/{len(ds)} 通过   (期望 [3 SID]+[\\n,EOS])")
    print(f"  层序 a->b->c      : {len(ds)-layer_bad}/{len(ds)} 通过")
    if bad or layer_bad:
        print("  [FAIL] 目标结构不满足")
        return 1

    lens = [len(ds[i]["input_ids"]) for i in range(min(len(ds), 2000))]
    print(f"  长度(P50/P95/max) : {pct(lens,.5)} / {pct(lens,.95)} / {max(lens)}"
          f"   cutoff={args.cutoff_len}  truncated={sum(1 for L in lens if L >= args.cutoff_len)}")

    # ---------------------------------------------------------------- C
    hr("C. T2a/T2b  item feature  (SidItemFeatDataset)")
    from data import SidItemFeatDataset

    feats = SidItemFeatDataset(item_file=item_path, index_file=index_path, tokenizer=tok,
                               max_len=args.cutoff_len, sample=args.n_sample, seed=42,
                               category=category)
    tasks = collections.Counter(d["task"] for d in feats.data)
    print(f"  样本数            : {len(feats)}  {dict(tasks)}")
    seen = set()
    for i in range(len(feats)):
        t = feats.data[i]["task"]
        if t in seen:
            continue
        seen.add(t)
        r = feats[i]
        lab = [x for x in r["labels"] if x != -100]
        sid_in_lab = [x for x in lab if x in sid_ids]
        print(f"  [{t:9s}] in={len(r['input_ids'])} lab={len(lab)} "
              f"sid_in_target={len(sid_in_lab)} target={tok.decode(lab, skip_special_tokens=False)!r}")
        if t == "sid2title" and sid_in_lab:
            print("  [FAIL] sid2title 的目标不该是 SID")
            return 1
        if t == "title2sid" and len(sid_in_lab) != 3:
            print(f"  [FAIL] title2sid 的目标应为 3 个 SID，实得 {len(sid_in_lab)}")
            return 1

    # ---------------------------------------------------------------- D
    hr("结论")
    print(f"  A 词表注册        : PASS  ({len(vocab)} tokens, 码序, id {min(ids)}..{max(ids)})")
    print(f"  B T1 目标         : PASS  ({len(ds)} 条结构 = [3 SID] + [\\n, EOS])")
    print(f"  C T2a/T2b         : PASS")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
