#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GRPO(LoRA) 显存账本探针 —— 回答"本地 4GB 能不能跑 GRPO + LoRA"。

为什么需要它：GRPO 的显存不是"权重 × N"这种简单倍数，而是四块拼起来的：
    ① 基座权重（bf16）
    ② 可训参数 + 梯度 + 优化器状态  <- LoRA 的 modules_to_save 在这里放大 30 倍
    ③ 生成阶段的 KV cache（展开成 B × num_generations 条序列）
    ④ 前向的 logits 张量（B*G × (C+1) × vocab，**全词表**）—— 最容易被忽略的一块
    ⑤ 参考模型：PEFT 下 **不存在**（ref_model=None，靠 disable_adapter 复用同一份权重）

本探针不估算，逐块实测（真实权重 + 真实 LoRA + 真实 generate + 真实 backward）。

用法：
    python scripts/sft/probe_rl_memory.py                       # 默认：无 modules_to_save
    python scripts/sft/probe_rl_memory.py --modules-to-save embed_tokens,lm_head
    python scripts/sft/probe_rl_memory.py --optim adamw_bnb_8bit
    python scripts/sft/probe_rl_memory.py --batch 1 --num-generations 2   # 最小可跑配置
"""
import argparse
import gc
import json
import os
import sys
import time

import torch


def gib(b):
    return b / 1073741824


def fmt_mib(b):
    return f"{b / 1048576:8.1f} MiB"


_GLOBAL = {"peak": 0}


def reset_peak():
    # 注意：reset 会清零计数器，所以每次清零前先把当前峰值汇入全局，
    # 否则会漏掉优化器 step 那类"一次性瞬时峰值"（曾因此把带 modules_to_save 的
    # 4300.7 MiB 漏报成 3756 MiB）。
    _GLOBAL["peak"] = max(_GLOBAL["peak"], torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


def snap(tag, t0=None):
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    dt = f"  {time.time() - t0:5.1f}s" if t0 else ""
    print(f"  [{tag:26s}] allocated {fmt_mib(alloc)}   peak {fmt_mib(peak)}{dt}")
    return alloc, peak


# --------------------------------------------------------------------- 参数分组
def param_breakdown(model):
    """把参数按「组」拆开 —— 只关心显存，不关心名字细节。"""
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            g = "LoRA (attention)"
        elif "modules_to_save" in n:
            g = "modules_to_save (embed/lm_head)"
        else:
            g = "其他可训"
        d = groups.setdefault(g, {"n": 0, "elems": 0, "bytes": 0, "dtype": set()})
        d["n"] += 1
        d["elems"] += p.numel()
        d["bytes"] += p.numel() * p.element_size()
        d["dtype"].add(str(p.dtype).replace("torch.", ""))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3-0.6B")
    ap.add_argument("--domain", default="IandS")
    ap.add_argument("--sft-dir", default="")
    ap.add_argument("--batch", type=int, default=4, help="= per_device_train_batch_size（唯一 prompt 数）")
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=0, help="0 = 从真实数据量出来")
    ap.add_argument("--completion-len", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--modules-to-save", default="",
                    help="逗号分隔；SFT 必须 'embed_tokens,lm_head'，RL 可以留空省 2.6 GiB")
    ap.add_argument("--optim", default="adamw_torch",
                    choices=["adamw_torch", "adamw_bnb_8bit", "paged_adamw_32bit", "sgd"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-backward", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="开梯度检查点（用时间换激活内存）；PEFT 下需同时 enable_input_require_grads")
    ap.add_argument("--no-input-grads", action="store_true",
                    help="[诊断用] 故意不调 enable_input_require_grads，验证梯度是否会静默变 None")
    a = ap.parse_args()

    sft_dir = a.sft_dir or f"data/Amazon23/{a.domain}/sft"
    vocab_path = os.path.join(sft_dir, "info", "sid_vocab.json")
    csv_path = os.path.join(sft_dir, "train", f"{a.domain}_5_train.csv")
    info_path = os.path.join(sft_dir, "info", f"{a.domain}.item_info.txt")

    dt_map = {"bf16": torch.bfloat16, "fp16": torch.float32, "fp32": torch.float32}
    load_dt = dt_map[a.dtype]
    amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[a.dtype]

    print("=" * 78)
    print("GRPO(LoRA) 显存账本探针")
    print("=" * 78)
    print(f"  device      : {torch.cuda.get_device_name(0)}")
    print(f"  总显存      : {gib(torch.cuda.get_device_properties(0).total_memory):.2f} GiB")
    print(f"  precision   : {a.dtype}  (权重 dtype={load_dt})")
    print(f"  LoRA        : r={a.lora_r} alpha={a.lora_alpha} targets={a.lora_targets}")
    print(f"  modules_to_save : {a.modules_to_save or '(空)'}")
    print(f"  optim       : {a.optim}")
    print(f"  batch x gen : {a.batch} x {a.num_generations} = {a.batch * a.num_generations} 条序列")
    print()

    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
    from peft import LoraConfig, get_peft_model

    # ------------------------------------------------------------------ ① 基座
    reset_peak()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=load_dt, attn_implementation="sdpa")
    model.to("cuda")
    m_base, _ = snap("① 基座权重", t0)
    base_params = sum(p.numel() for p in model.parameters())

    # tokenizer：注册 SID（真实流程里这一步发生在 SFT 输出目录）
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    voc = json.load(open(vocab_path, encoding="utf-8"))
    added = tok.add_tokens(voc)
    # 🔴 真实流程里 sft.py 会 resize_token_embeddings(len(tokenizer)) —— 少了这一步
    #    embed 还是 151936 行，账本会偏小。这里如实复现。
    model.resize_token_embeddings(len(tok))
    print(f"  tokenizer   : {len(tok)}  (新增 {added} 个 SID)  -> resize 到 {model.get_input_embeddings().weight.shape[0]} 行")
    tied_before = model.get_output_embeddings().weight is model.get_input_embeddings().weight

    # ------------------------------------------------------------------ ② LoRA
    msave = [x for x in a.modules_to_save.split(",") if x]
    lcfg = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=[x for x in a.lora_targets.split(",") if x],
        modules_to_save=msave or None,
    )
    model = get_peft_model(model, lcfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    tied_after = None
    same_tensor = None
    try:
        m = model.base_model.model                      # Qwen3ForCausalLM
        w_emb = m.model.embed_tokens.weight
        w_head = m.lm_head.weight
        tied_after = w_emb is w_head
        same_tensor = (w_emb.data_ptr() == w_head.data_ptr())
    except Exception as e:
        tied_after = f"(探测不到: {type(e).__name__})"
    print(f"  tie_word_embeddings: resize 后 {tied_before}  -> 套 LoRA 后 {tied_after}")
    if same_tensor is False:
        print(f"  ⚠️ embed_tokens 与 lm_head 已是**两份独立张量** -> modules_to_save 会让词表参数翻倍")
    elif same_tensor is True:
        print(f"  两份 module 仍共享同一张量（tie 未破）")

    if a.grad_ckpt:
        # PEFT + gradient checkpointing：必须让输入参与计算图，否则 LoRA 梯度为 None
        model.gradient_checkpointing_enable()
        if a.no_input_grads:
            print("  enable_input_require_grads: **故意跳过**（诊断）")
        else:
            model.enable_input_require_grads()
        # 🔴 检查点只在 self.training=True 时生效；而 from_pretrained 默认给的是 eval 模式，
        #    不显式 train() 的话上面那句是**空操作**（曾因此误判"梯度检查点无效"）。
        model.train()
        _gc_on = [n for n, m in model.named_modules()
                  if getattr(m, "gradient_checkpointing", False)]
        print(f"  gradient_checkpointing: ON  (training={model.training}, "
              f"生效模块 {len(_gc_on)} 个, 例: {_gc_on[:2]})")

    print()
    print("--- 可训参数分组 ---")
    groups = param_breakdown(model)
    for g, d in sorted(groups.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"  {g:34s} {d['n']:6d} 张量  {d['elems']:>13,} 元素  {d['bytes']/1048576:9.1f} MiB  {sorted(d['dtype'])}")
    print(f"  {'合计':34s} {'':6s}        {n_train:>13,} 元素  {sum(p.numel()*p.element_size() for p in trainable)/1048576:9.1f} MiB")
    print(f"  占基座比例: {100*n_train/base_params:.2f}%   (基座 {base_params:,} 元素)")

    # ------------------------------------------------------------------ ③ 优化器
    print()
    print("--- 优化器状态（真实创建 + 灌假梯度 + step）---")
    if a.optim == "adamw_torch":
        opt = torch.optim.AdamW(trainable, lr=1e-5)
    elif a.optim == "adamw_bnb_8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=1e-5)
    elif a.optim == "paged_adamw_32bit":
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW32bit(trainable, lr=1e-5)
    else:
        opt = torch.optim.SGD(trainable, lr=1e-5)

    before = torch.cuda.memory_allocated()
    for p in trainable:
        p.grad = torch.zeros_like(p)
    after_grad = torch.cuda.memory_allocated()
    opt.step()
    after = torch.cuda.memory_allocated()
    m_grad = after_grad - before          # 梯度：dtype 跟随参数
    m_opt = after - after_grad            # 优化器状态
    # 关键细节：torch.optim.AdamW 用 zeros_like(p) 建状态 ⟹ **状态 dtype 跟随参数 dtype**
    #   参数 bf16 -> 状态 bf16（4 B/param），不是普遍以为的 fp32 8 B/param
    #   而 bitsandbytes 的 *32bit 变体强制 fp32 状态（8 B/param），所以它反而更大
    n_f32 = sum(p.numel() for p in trainable if p.dtype == torch.float32)
    n_bf16 = sum(p.numel() for p in trainable if p.dtype != torch.float32)
    print(f"  梯度实测    : {fmt_mib(m_grad)}")
    print(f"  状态实测    : {fmt_mib(m_opt)}   "
          f"（fp32 参数 {n_f32:,} 元素 × 8B + 低精参数 {n_bf16:,} 元素 × 4B "
          f"= {(8*n_f32+4*n_bf16)/1048576:.1f} MiB）")
    snap("② + 梯度 + 优化器 累计")
    del opt
    for p in trainable:
        p.grad = None
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------ ④ 生成
    m_gen_peak = None
    if not a.skip_generate:
        print()
        print("--- ④ 生成阶段（真实 generate，B x G 条序列）---")
        prompts = None
        plen = a.prompt_len
        if not plen:
            import pandas as pd
            df = pd.read_csv(csv_path, nrows=a.batch)
            rows = []
            for _, r in df.iterrows():
                h = eval(r["history_item_sid"])
                history = ", ".join(h)
                inp = (f"The user has interacted with items {history} in chronological order. "
                       f"Can you predict the next possible item that the user may expect?")
                instruction = ("Below is an instruction that describes a task, paired with an input "
                               "that provides further context. Write a response that appropriately "
                               "completes the request. \n\n### Instruction:\n"
                               "Can you predict the next possible item that the user may expect?\n\n")
                rows.append(instruction + f"### User Input: \n{inp}\n\n### Response:\n")
            prompts = rows
            plen = max(len(tok(x, add_special_tokens=False).input_ids) for x in prompts)
            print(f"  [数据] 真实 prompt 长度 max = {plen}")

        x = torch.randint(100, 1000, (a.batch, plen), device="cuda")
        attn = torch.ones_like(x)
        gen_kw = dict(max_new_tokens=a.completion_len, do_sample=True, temperature=1.0,
                      num_return_sequences=a.num_generations, pad_token_id=tok.eos_token_id)
        # 注意：不要把 dtype 塞进 generate 的 kwargs —— transformers 的
        # _validate_model_kwargs 会拒绝未知 kwarg（'The following model_kwargs are not used: [dtype]'）。
        # 精度由外层 autocast 负责。
        ctx = torch.autocast("cuda", dtype=amp_dt) if amp_dt is not None else torch.autocast("cuda", enabled=False)
        reset_peak()
        t0 = time.time()
        with torch.inference_mode(), ctx:
            out = model.generate(input_ids=x, attention_mask=attn, **gen_kw)
        m_gen_peak = torch.cuda.max_memory_allocated()
        print(f"  生成输出 shape   : {tuple(out.shape)}   (= B*G, P+C)")
        snap("④ 生成阶段 peak", t0)
        kv_calc = (a.batch * a.num_generations) * (plen + a.completion_len) * \
                  model.config.num_hidden_layers * 2 * model.config.num_key_value_heads * \
                  model.config.head_dim * 2
        print(f"  KV cache 推算    : {kv_calc/1048576:.1f} MiB  "
              f"(B*G x (P+C) x layers x 2 x kv_heads x head_dim x 2B)")
        del out, x, attn
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------ ⑤ 训练步
    m_step_peak = None
    if not a.skip_backward:
        print()
        print("--- ⑤ GRPO 训练步（前向 + logits + ref 前向 + backward）---")
        from trl.trainer.utils import selective_log_softmax
        B = a.batch * a.num_generations
        P = a.prompt_len or 177
        C = a.completion_len
        ids = torch.randint(100, 1000, (B, P + C), device="cuda")
        attn = torch.ones_like(ids)
        reset_peak()
        t0 = time.time()
        ctx = torch.autocast("cuda", dtype=amp_dt) if amp_dt is not None else torch.autocast("cuda", enabled=False)
        model.train()                              # 训练步本该在 train 模式（检查点/dropout 都依赖它）
        if a.grad_ckpt:
            model.config.use_cache = False         # 检查点与 KV cache 不兼容
        with ctx:
            logits = model(input_ids=ids, attention_mask=attn, logits_to_keep=C + 1).logits
            logits = logits[:, :-1, :]
            per_tok = selective_log_softmax(logits, ids[:, -C:])
            loss = -per_tok.mean()
        m_logits = torch.cuda.max_memory_allocated()
        print(f"  logits shape      : {tuple(logits.shape)}  "
              f"= {logits.numel()*logits.element_size()/1048576:.1f} MiB  <- 全词表，最容易忽略")
        print(f"  策略前向后 peak   : {fmt_mib(m_logits)}")
        # ref 前向（PEFT 下走 disable_adapter，无额外权重）
        with torch.inference_mode(), ctx:
            with model.disable_adapter():
                rl = model(input_ids=ids, attention_mask=attn, logits_to_keep=C + 1).logits
                _ = selective_log_softmax(rl[:, :-1, :], ids[:, -C:])
        del rl
        print(f"  ref 前向后 peak   : {fmt_mib(torch.cuda.max_memory_allocated())}  "
              f"(无额外权重，只多一次前向)")
        del logits, per_tok
        torch.cuda.empty_cache()
        loss.backward()
        # 🔴 正确性校验：PEFT + gradient checkpointing 下，若检查点层的输入不 requires_grad，
        #    PyTorch 会只打印一条 warning 然后把梯度置 None —— 显存看着很美，其实没在训练。
        n_ok = sum(1 for p in trainable if p.grad is not None)
        n_nz = sum(1 for p in trainable if p.grad is not None and torch.count_nonzero(p.grad) > 0)
        print(f"  梯度覆盖: {n_ok}/{len(trainable)} 个可训张量拿到梯度，其中 {n_nz} 个非全零"
              + ("   [OK]" if n_ok == len(trainable) and n_nz > 0 else "   *** 异常 ***"))
        m_step_peak = torch.cuda.max_memory_allocated()
        snap("⑤ 训练步 total peak", t0)
        del loss, ids, attn
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------ 汇总
    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    total = torch.cuda.get_device_properties(0).total_memory
    gp = max(_GLOBAL["peak"], torch.cuda.max_memory_allocated())
    rows = [
        ("① 基座权重", m_base),
        ("② 可训参数(含 modules_to_save)", sum(p.numel() * p.element_size() for p in trainable)),
        ("③ 优化器状态", m_opt),
        ("④ 生成阶段 peak 增量", (m_gen_peak - m_base) if m_gen_peak else None),
        ("⑤ 训练步 peak 增量", (m_step_peak - m_base) if m_step_peak else None),
    ]
    for name, v in rows:
        if v is None:
            print(f"  {name:34s} (跳过)")
        else:
            print(f"  {name:34s} {fmt_mib(v)}   {100*v/total:5.1f}% of 总显存")
    if m_step_peak:
        print()
        print(f"  训练步峰值（仅第⑤段） = {fmt_mib(m_step_peak)}")
        print(f"  >>> 全流程实测峰值     = {fmt_mib(gp)}  ({gib(gp):.2f} GiB) / 物理 {gib(total):.2f} GiB")
        if gp > total:
            print(f"  >>> 超出物理显存 -> Windows WDDM 会落共享内存换页（不 OOM 但很慢）")
        else:
            print(f"  >>> 可装入物理显存（余量 {fmt_mib(total - gp)}），不换页")


if __name__ == "__main__":
    main()
