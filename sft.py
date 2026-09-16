import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset


class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens


class SidVocabLoader:
    """加载 SID 码表（**码序**），用于 tokenizer 注册。

    与 MiniOneRec `TokenExtender` 的差异是有意为之，两个硬理由：

    1) **顺序**：`TokenExtender` 从 `.index.json` 现收 token 后 `sorted()`，
       得到的是字典序（`<a_0>, <a_100>, ..., <a_109>, <a_10>...`）。
       而 `sid_vocab.json` 是**码序**（`<a_0>, <a_1>, ..., <a_255>`）。
       [实测] 二者在 765/768 个 token 上给出的 id 不同。
       后果：M4 码本语义初始化按 `codebook.npy` 的行下标对齐（`(3,256,32)`），
       `<a_k>` 必须落在 `cb[0][k]` —— 用字典序会**静默错位**。

    2) **集合**：`index.json` 只含"被用过"的码。[实测] VG 只有 759 个
       （缺 9 个 a 层死码，3.52%），IandS 恰好 768。
       后果：两域词表大小不一致，且无法覆盖码本全部行。

    因此注册以 `sid_vocab.json` 为准，并用 `check_coverage()` 断言
    `index.json` 用到的 token 一个不漏 —— 漏了就是静默掉码。
    """

    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.tokens = self._load()

    def _load(self):
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        if not isinstance(vocab, list) or not vocab:
            raise ValueError(f'sid_vocab.json 应为非空 list: {self.vocab_path}')
        if len(set(vocab)) != len(vocab):
            raise ValueError(f'sid_vocab.json 含重复 token: {self.vocab_path}')
        return vocab

    def check_coverage(self, index_path):
        """断言 index.json 用到的 token 全部在码表内。

        Returns (used:set, missing:list) —— missing 非空即为致命错。
        """
        with open(index_path, 'r', encoding='utf-8') as f:
            indices = json.load(f)
        used = set()
        for sids in indices.values():
            used.update(sids)
        return used, sorted(used - set(self.tokens))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



# ---------------- SFT 任务注册表 ----------------
# [口径] T1 = 主任务（唯一进 EVAL_PROTOCOL 召回指标）；T2a/T2b/T3 = 辅助任务。
#        默认四个全开 == MiniOneRec 的 ConcatDataset 行为（Run-0 锚点）。
TASK_REGISTRY = {
    "T1":  ("SidSFTDataset",       "seq2sid  ", "历史 SID -> 目标 SID"),
    "T2a": ("SidItemFeatDataset",  "sid2title", "SID      -> title"),
    "T2b": ("SidItemFeatDataset",  "title2sid", "title    -> SID"),
    "T3":  ("FusionSeqRecDataset", "seq2title", "历史 SID -> 目标 title"),
}


def resolve_tasks(tasks):
    """解析 --tasks 字符串 -> (selected, warns)。抽在 train() 外，便于单测与探针复用。"""
    selected = [t.strip() for t in str(tasks).split(",") if t.strip()]
    unknown = [t for t in selected if t not in TASK_REGISTRY]
    if unknown:
        raise ValueError(f"未知任务 {unknown}；可选 {list(TASK_REGISTRY)}")
    if not selected:
        raise ValueError("--tasks 不能为空；可选 " + ",".join(TASK_REGISTRY))
    warns = []
    # T2a/T2b 由同一个类同时产出（data.py:711-725 两个循环，各 25,847 条），类外拆不开
    if ("T2a" in selected) != ("T2b" in selected):
        warns.append("T2a/T2b 由同一个 SidItemFeatDataset 同时产出，无法只取其一 —— "
                     "两路都会进训练集。要精确拆分需改 data.py:711-725。")
    return selected, warns


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str="",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    sid_vocab_path: str = "",      # 留空则从 sid_index_path 推导 <sft>/info/sid_vocab.json
    tasks: str = "T1,T2a,T2b,T3",  # 逗号分隔，选择哪些任务进训练集。默认全开 == MiniOneRec 的
                                   # ConcatDataset 行为（Run-0 锚点）。传 "T1" 即单任务消融。
    torch_compile: bool = False,   # [红线] 默认关。Qwen3 + 动态 padding 下 torch.compile 会
                                   # 反复重编译（每次 batch 长度不同）而拖慢甚至 OOM。
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.environ['WANDB_PROJECT'] = wandb_project
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "sdpa"
        print("flash-attn not found, using PyTorch SDPA (memory-efficient) instead")

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    original_vocab_size = len(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # 训练用 right padding（标准做法）：真实 token 从位置 0 起算，与预训练分布一致。
    # 生成端（evaluate.py:140 + 手工左填充 evaluate.py:166）单独设 left，不受这里影响。
    # 注：Qwen3 纯 RoPE 下 left/right 数学等价（attention 只依赖相对距离），
    #     但 left 是生成端设置的复制粘贴，换 sliding window / rope_scaling 就会错。
    tokenizer.padding_side = "right"
    new_tokens = []
    register_info = {}

    if not sid_vocab_path and sid_index_path:
        cand = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(sid_index_path))),
            "info", "sid_vocab.json",
        )
        if os.path.exists(cand):
            sid_vocab_path = cand

    if sid_vocab_path and os.path.exists(sid_vocab_path):
        loader = SidVocabLoader(sid_vocab_path)
        new_tokens = loader.tokens
        print(f"[SID] vocab={len(new_tokens)} (code order) <- {sid_vocab_path}")

        # 覆盖性断言：index.json 用到的 token 必须全在码表内，否则是静默掉码
        if sid_index_path and os.path.exists(sid_index_path):
            used, missing = loader.check_coverage(sid_index_path)
            if missing:
                raise ValueError(
                    f"index 用到 {len(used)} 个 token，其中 {len(missing)} 个不在 "
                    f"sid_vocab.json 内: {missing[:10]}"
                )
            print(f"[SID] coverage OK: index uses {len(used)}/{len(new_tokens)} tokens")
        else:
            print("[SID] WARN: sid_index_path 缺失，跳过覆盖性断言")

        # 真实落点（[实测] Qwen3-0.6B）：注册前 len(tokenizer)=151669（= config.vocab_size
        # 151936 - 旧 padding 区 267）。故 SID id 从 151669 起，旧 267 行是"僵尸区"
        # （从未被 resize 触及、也无预训练语义），可接受。
        added = tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print(f"[SID] add_tokens +{added} (requested {len(new_tokens)}), "
              f"len(tokenizer)={len(tokenizer)}")
        register_info = {
            "source": sid_vocab_path,
            "order": "code",
            "n_tokens": len(new_tokens),
            "vocab_size_before": original_vocab_size,
            "vocab_size_after": len(tokenizer),
            "token_to_id": {t: tokenizer.convert_tokens_to_ids(t) for t in new_tokens},
        }
    elif sid_index_path and os.path.exists(sid_index_path):
        # 兼容 MiniOneRec 原路径 —— 不推荐：字典序会让 M4 码本语义初始化静默错位
        print("[SID] WARN: 未找到 sid_vocab.json，回退 MiniOneRec TokenExtender(sorted)")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))
            register_info = {
                "source": sid_index_path,
                "order": "sorted",
                "n_tokens": len(new_tokens),
                "vocab_size_before": original_vocab_size,
                "vocab_size_after": len(tokenizer),
            }

    if register_info:
        map_path = os.path.join(output_dir, "sid_token_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(register_info, f, ensure_ascii=False, indent=1)
        print(f"[SID] token map saved -> {map_path}")

    # Freeze LLM parameters if required
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    # grad shape: [vocab_size, hidden_dim]
                    grad[:original_vocab_size].zero_()
                    return grad
                
                embedding_layer.weight.register_hook(mask_grad)

                print(f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})")

        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        # Print the number of trainable parameters (it will still report the size of the entire embedding matrix, but only the newly added rows will have non-zero gradients).
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    # ---------------- 训练集按任务开关拼装（--tasks） ----------------
    # 单任务消融传 --tasks T1；辅助任务消融传 --tasks T1,T2a 等。
    selected, _warns = resolve_tasks(tasks)
    for _w in _warns:
        print(f"[WARN] {_w}")
    print("[TASKS] 任务 -> 数据类映射：")
    for _t in selected:
        _cls, _name, _io = TASK_REGISTRY[_t]
        print(f"        {_t:4s} {_cls:22s} {_name}  {_io}")

    train_datasets = []
    if "T1" in selected:
        train_datasets.append(SidSFTDataset(train_file=train_file, tokenizer=tokenizer,
                                            max_len=cutoff_len, sample=sample, seed=seed, category=category))
    if "T2a" in selected or "T2b" in selected:
        train_datasets.append(SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path,
                                                 tokenizer=tokenizer, max_len=cutoff_len, sample=sample,
                                                 seed=seed, category=category))
    if "T3" in selected:
        train_datasets.append(FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path,
                                                  index_file=sid_index_path, tokenizer=tokenizer,
                                                  max_len=cutoff_len, sample=sample, seed=seed, category=category))
    # 未接线（保留 MiniOneRec 原样，需要时再开）：
    #   SFTData                   -> 与 T1 同构（注释里的 train_data4）
    #   TitleHistory2SidSFTDataset-> 历史用 title 而非 SID（注释里的 train_data5）
    #   T4 text2sid（tasks/text2sid.jsonl 25,847 条）本仓 data.py 无对应类，需新写

    for _i, _d in enumerate(train_datasets, start=1):
        print(f"        {_i}. {type(_d).__name__:22s} {len(_d):>9,} 条")
    train_data = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    print(f"[TASKS] 训练集合计 {len(train_data):,} 条")

    # 验证集恒为 T1：评测口径（EvalSidDataset）就是 T1 seq2sid，跟着 --tasks 变会让 val_loss 不可比
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # val_data = SFTData(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=20000, seed=seed, category=category)
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    sample_frac = 1
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)

    print(hf_train_dataset)
    print(hf_val_dataset)
    eval_step = 0.05
    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step, 
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if wandb_project or wandb_run_name else "none",
            torch_compile=torch_compile,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    fire.Fire(train)
