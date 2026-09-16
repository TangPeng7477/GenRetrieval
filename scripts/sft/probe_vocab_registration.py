# -*- coding: utf-8 -*-
"""SID 词表注册前的体检探针。

结论都记在 `docs/SFT_PIPELINE.md §6.4(3)`。改训练脚本之前先跑它，确认四件事：

  1. `len(tokenizer)` 与 `config.vocab_size` 的 gap
     —— Qwen3 上 len()=151669 而 vocab_size=151936（=1187x128 对齐），差 267
  2. 码序（`info/sid_vocab.json`）vs 字典序（MiniOneRec `TokenExtender` 的 sorted()）
     的 id 差异 —— 实测 765/768 个不同，直接影响 M4 码本语义初始化能否对齐
  3. `index/` 里的 token 集合 vs `sid_vocab.json`
     —— VG 的 index 只有 759 个（缺 9 个 a 层死码），按 index 收集会少注册
  4. resize 后新增行的**真实初始化分布**
     —— transformers 4.57.1 的 resize_token_embeddings 默认 mean_resizing=True
        （旧行 mean + 协方差多元正态），不是 initializer_range=0.02

用法：
    ./.venv/Scripts/python.exe scripts/sft/probe_vocab_registration.py
    ./.venv/Scripts/python.exe scripts/sft/probe_vocab_registration.py --domain VG
    ./.venv/Scripts/python.exe scripts/sft/probe_vocab_registration.py --skip-model
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "Amazon23")


def check_token_files(dom: str) -> dict:
    """index.json 的 token 集合 vs sid_vocab.json。"""
    idx_p = os.path.join(DATA, dom, "sft", "index", f"{dom}.index.json")
    voc_p = os.path.join(DATA, dom, "sft", "info", "sid_vocab.json")
    idx = json.load(open(idx_p, encoding="utf-8"))
    voc = json.load(open(voc_p, encoding="utf-8"))

    used: set[str] = set()
    for v in idx.values():
        used.update(v)
    vocab_set = set(voc)
    dead = [t for t in voc if t not in used]          # vocab 有、index 没用过
    extra = sorted(used - vocab_set)                  # index 有、vocab 没有
    layers: dict[str, int] = {}
    for t in dead:
        layers[t[1]] = layers.get(t[1], 0) + 1

    print(f"[3] token 集合  ({dom})")
    print(f"  index.json 出现过          : {len(used)}")
    print(f"  sid_vocab.json             : {len(vocab_set)}")
    print(f"  死码（vocab 有、index 没用过）: {len(dead)}  分层 {layers or '{}'}")
    if dead:
        print(f"    e.g. {dead[:8]}")
    print(f"  index 有但 vocab 没有        : {len(extra)}" + (f"  {extra[:5]}" if extra else ""))
    if len(used) != len(vocab_set):
        print("  🔴 二者不等 → 按 index.json 收集 token 会少注册（MiniOneRec 的 TokenExtender 就是这个行为）")
    return {"n_index": len(used), "n_vocab": len(vocab_set), "n_dead": len(dead), "dead_layers": layers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="IandS", choices=["IandS", "VG"])
    ap.add_argument("--base-model", default="models/Qwen3-0.6B")
    ap.add_argument("--skip-model", action="store_true", help="跳过需要加载权重的第 2/4 节")
    args = ap.parse_args()

    os.chdir(ROOT)
    base = args.base_model
    vocab = json.load(open(os.path.join(DATA, args.domain, "sft", "info", "sid_vocab.json"),
                           encoding="utf-8"))

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(base)
    n_tok = len(tk)

    print("=" * 74)
    print(f"[1] tokenizer 现状  ({base})")
    print(f"  len(tokenizer)         = {n_tok}")
    print(f"  eos = {tk.eos_token!r}  id = {tk.eos_token_id}")
    print(f"  sid_vocab.json         = {len(vocab)} 个 token")
    for s in ["<a_0>", "<b_123>", "<c_255>"]:
        e = tk.encode(s, add_special_tokens=False)
        print(f"  未注册 {s!r:10s} -> {len(e)} tok  ids={e}")

    check_token_files(args.domain)

    if not args.skip_model:
        import torch
        from transformers import AutoModelForCausalLM

        t_code = AutoTokenizer.from_pretrained(base)
        t_code.add_tokens(vocab)                          # 码序
        t_lex = AutoTokenizer.from_pretrained(base)
        t_lex.add_tokens(sorted(vocab))                   # 字典序（MiniOneRec 行为）

        print(f"\n[2] 两种顺序的 id 映射  ({args.domain})")
        diff = [s for s in vocab if t_code.convert_tokens_to_ids(s) != t_lex.convert_tokens_to_ids(s)]
        for label, t in (("码序  ", t_code), ("字典序", t_lex)):
            print(f"  {label}: <a_0>={t.convert_tokens_to_ids('<a_0>')}  "
                  f"<a_1>={t.convert_tokens_to_ids('<a_1>')}  "
                  f"<c_255>={t.convert_tokens_to_ids('<c_255>')}")
        print(f"  字典序前 12: {sorted(vocab)[:12]}")
        print(f"  两序 id 不同的 token: {len(diff)} / {len(vocab)}")
        n_new_tok = len(t_code)

        print(f"\n[4] resize_token_embeddings 的真实初始化")
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        w_old = model.get_input_embeddings().weight.data.clone()
        n_cfg = w_old.shape[0]
        print(f"  config.vocab_size = {n_cfg}  (= {n_cfg // 128} x 128)   "
              f"gap = vocab_size - len(tokenizer) = {n_cfg - n_tok}")
        print(f"  注册后 len(tokenizer) = {n_new_tok}")
        print(f"  新 token id 区间 = [{n_tok}, {n_new_tok}) 共 {n_new_tok - n_tok}")
        print(f"    ├─ [{n_tok}, {n_cfg})  长度 {n_cfg - n_tok}  ← 落在旧 padding 行（未被 resize 触及）")
        print(f"    └─ [{n_cfg}, {n_new_tok})  长度 {n_new_tok - n_cfg}  ← 全新分配的行")

        model.resize_token_embeddings(n_new_tok)          # 默认 mean_resizing=True
        w_new = model.get_input_embeddings().weight.data
        print(f"  embedding 行数 {n_cfg} -> {w_new.shape[0]}  (增 {w_new.shape[0] - n_cfg})")
        print(f"  旧行逐位保留: {torch.equal(w_old, w_new[:n_cfg])}")
        for label, zone in (("旧行 [0,        n_cfg)", w_old),
                            ("旧 padding 区       ", w_new[n_tok:n_cfg]),
                            ("真新行              ", w_new[n_cfg:])):
            print(f"  {label:22s} n={zone.shape[0]:6d}  mean={zone.mean().item():+.5f}  "
                  f"std={zone.std().item():.5f}")

        m2 = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        m2.resize_token_embeddings(n_new_tok, mean_resizing=False)
        w2 = m2.get_input_embeddings().weight.data
        print(f"  [对照] mean_resizing=False 真新行 std = {w2[n_cfg:].std().item():.5f}")

        print(f"\n  config.vocab_size 现在 = {model.config.vocab_size}  "
              f"是 128 的倍数: {model.config.vocab_size % 128 == 0}  "
              f"(pad_to_multiple_of=128 -> {((n_new_tok + 127) // 128) * 128})")
        print(f"  旧 151,936 行是否有预训练语义: 是（std {w_old.std().item():.5f}）")
    print("=" * 74)


if __name__ == "__main__":
    sys.exit(main())
