"""本地 LoRA 训练可行性实测：真实模型 + 真实 LoRA 配置 + 真实数据集，测单步耗时与峰值显存。

用途：回答「本地 4GB 卡能不能把 LoRA 训练跑完」。不做任何估算——
      直接加载真实权重、套上与 `sft.py` 同一套 LoRA 配置、用真实 `SidSFTDataset` 出 batch。

用法：
    ./.venv/Scripts/python.exe scripts/sft/bench_local_training.py
    ./.venv/Scripts/python.exe scripts/sft/bench_local_training.py --cutoff-len 320 --steps 6
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

CATEGORY = {"IandS": "Industrial_and_Scientific", "VG": "Video_Games"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="IandS")
    ap.add_argument("--sft-dir", default=None)
    ap.add_argument("--model-dir", default="models/Qwen3-0.6B")
    ap.add_argument("--cutoff-len", type=int, default=320)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6, help="训练计时步数")
    ap.add_argument("--eval-steps", type=int, default=8, help="eval 计时步数")
    ap.add_argument("--n-rows", type=int, default=64)
    ap.add_argument("--optim", default="adamw_torch",
                    help="adamw_torch | adamw_bnb_8bit | sgd")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="bf16（Ampere+）| fp16（V100/Volta 必须用这个）| fp32")
    ap.add_argument("--no-model", action="store_true", help="只统计序列长度，不加载模型")
    a = ap.parse_args()

    sft_dir = a.sft_dir or f"data/Amazon23/{a.domain}/sft"
    category = CATEGORY[a.domain]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

    # ---- 1) tokenizer + SID 词表：复用 sft.py 的真实加载器 ----
    import sft as sftmod
    tokenizer = AutoTokenizer.from_pretrained(a.model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"          # 与 sft.py:239 一致（训练用 right）
    loader = sftmod.SidVocabLoader(f"{sft_dir}/info/sid_vocab.json")
    new_tokens = loader.tokens
    tokenizer.add_tokens(new_tokens)
    print(f"[vocab] 注册 {len(new_tokens)} 个 SID token  len(tokenizer)={len(tokenizer)}")

    # ---- 2) 真实数据集（sample 限制行数，避免 tokenize 全量）----
    from data import SidSFTDataset
    csv = f"{sft_dir}/train/{a.domain}_5_train.csv"
    ds = SidSFTDataset(train_file=csv, tokenizer=tokenizer, max_len=a.cutoff_len,
                       sample=a.n_rows, seed=0, category=category)
    lens = sorted(len(ds[i]["input_ids"]) for i in range(len(ds)))
    p = lambda q: lens[min(int(len(lens) * q), len(lens) - 1)]
    print(f"[data ] {len(ds)} 条  cutoff={a.cutoff_len}  "
          f"实际长度 min={lens[0]} p50={p(.5)} p90={p(.9)} max={lens[-1]}  "
          f"均值={sum(lens)/len(lens):.0f}")
    if a.no_model:
        return

    # ---- 3) 真实模型 + 与 sft.py 完全一致的 LoRA 配置 ----
    from peft import LoraConfig, get_peft_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[gpu  ] {torch.cuda.get_device_name(0)}  "
              f"total={total/2**30:.2f} GiB  free={free/2**30:.2f} GiB")
    t0 = time.time()
    # 加载 dtype 与 autocast dtype 分离 —— 严格对齐 Trainer：
    #   bf16: 权重 bf16，无 scaler
    #   fp16: 权重 **fp32** + autocast(fp16) + GradScaler（GradScaler 不能 unscale fp16 梯度）
    _LOAD_DT = {"bf16": torch.bfloat16, "fp16": torch.float32, "fp32": torch.float32}[a.dtype]
    _AMP_DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=_LOAD_DT)
    model.resize_token_embeddings(len(tokenizer))
    model.to(dev)
    print(f"[model] 加载完成 {time.time()-t0:.1f}s")

    cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     modules_to_save=["embed_tokens", "lm_head"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    tr = sum(x.numel() for x in model.parameters() if x.requires_grad)
    al = sum(x.numel() for x in model.parameters())
    print(f"[lora ] trainable={tr:,} || all={al:,} || trainable%={100*tr/al:.2f}"
          f"   (与 sft.py 同配置: r=32 alpha=64 dropout=0.05 q/k/v/o + modules_to_save)")

    coll = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8,
                                 return_tensors="pt", padding=True)
    opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=5e-4)
    if a.optim == "sgd":
        opt = torch.optim.SGD([x for x in model.parameters() if x.requires_grad], lr=5e-4)
        print("[optim] SGD（无优化器状态 —— 用于隔离「显存溢出」对速度的影响）")
    elif a.optim == "adamw_bnb_8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit([x for x in model.parameters() if x.requires_grad], lr=5e-4)
        print(f"[optim] bitsandbytes AdamW8bit（优化器状态减半以上）bnb={bnb.__version__}")
    else:
        print("[optim] AdamW (fp32 状态，与 sft.py 的 optim=\"adamw_torch\" 一致)")
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def batch_of(i):
        b = coll([ds[i % len(ds)]])
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in b.items()}

    # ---- AMP：复现 Trainer 的 bf16 / fp16 行为 ----
    # 🔴 fp16 必须配 GradScaler：梯度常落在 1e-8..1e-4，低于 fp16 最小正规数 6.1e-5 会下溢成 0。
    #    Trainer 在 fp16=True 时自动挂 scaler；这里手写循环，所以要自己加，否则测出来的不是真行为。
    _amp_dtype = _AMP_DT
    scaler = torch.amp.GradScaler("cuda", enabled=(a.dtype == "fp16" and dev == "cuda"))
    print(f"[amp  ] dtype={a.dtype}  autocast={'off' if _amp_dtype is None else str(_amp_dtype).split('.')[-1]}"
          f"  GradScaler={'on' if scaler.is_enabled() else 'off'}")

    def fwd(batch):
        if _amp_dtype is None:
            return model(**batch).loss
        with torch.amp.autocast(device_type=("cuda" if dev == "cuda" else "cpu"), dtype=_amp_dtype):
            return model(**batch).loss

    def train_step(batch):
        loss = fwd(batch)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        return loss.item()

    # ---- 4) 训练步：fwd + bwd + opt.step ----
    model.train()
    for i in range(2):                       # 预热（不计时）
        train_step(batch_of(i))
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    losses = []
    for i in range(a.steps):
        losses.append(train_step(batch_of(i + 100)))
    if dev == "cuda":
        torch.cuda.synchronize()
    t_train = (time.time() - t0) / a.steps
    peak_train = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
    print(f"[train] {a.micro_batch} 条/步  {t_train:.3f} s/微批  峰值显存 {peak_train:.2f} GiB  "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f}"
          + (f"  GradScaler scale={scaler.get_scale():.0f}" if scaler.is_enabled() else ""))

    # ---- 5) eval 步：纯前向（Trainer 的 per_device_eval_batch_size = micro_batch）----
    model.eval()
    with torch.no_grad():
        fwd(batch_of(0))                     # 预热
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for i in range(a.eval_steps):
            fwd(batch_of(i + 200))
    if dev == "cuda":
        torch.cuda.synchronize()
    t_eval = (time.time() - t0) / a.eval_steps
    peak_eval = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
    print(f"[eval ] {a.micro_batch} 条/步  {t_eval:.3f} s/微批  峰值显存 {peak_eval:.2f} GiB")

    # ---- 6) 外推 ----
    MB = a.micro_batch
    train_micro_batches = 208999 * 3 / MB          # T1 全量 3 epoch
    eval_passes = 50984 / MB
    print()
    print("== 外推（T1 全量 208,999 条 × 3 epoch，micro=%d）==" % MB)
    print(f"  训练微批总数        : {train_micro_batches:,.0f}")
    print(f"  纯训练耗时          : {train_micro_batches*t_train/3600:,.1f} 小时")
    print(f"  单次 eval 前向       : {eval_passes:,.0f} 次  -> {eval_passes*t_eval/60:,.1f} 分钟")
    print(f"  20 次 eval 总耗时    : {20*eval_passes*t_eval/3600:,.1f} 小时")
    tot = (train_micro_batches*t_train + 20*eval_passes*t_eval) / 3600
    print(f"  >>> 合计约          : {tot:,.1f} 小时（{tot/24:,.1f} 天）")


if __name__ == "__main__":
    main()
