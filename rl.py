from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import random
import re
import numpy as np
import torch
from data import D3Dataset, SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset, RLSid2TitleDataset, RLSidhis2TitleDataset
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from minionerec_trainer import ReReTrainer
from sasrec import SASRec
# 🔴 复用 sft.py 的逗号参数解析器（单一实现，别在本文件重写一份）：
#    fire 会把命令行 `a,b,c` 解析成 **tuple**，直接 str(v).split(",") 会得到脏元素。
#    这个 bug 在 SFT 侧实测踩过（见 sft.py:147 的注释），RL 侧新增的 LoRA 参数同源。
from sft import parse_csv_list
# 🔴 提示词模板真源（与 `data.py:17` 同款导入）。本文件的**前置护栏**要用
#    `pt.response_prefix()` / `pt.source()`；此前漏了这一行 ⟹ `train()` 一进来就
#    `NameError: name 'pt' is not defined`（[实测] 2026-09-25 云端，命令行参数全解析完才炸）。
#    注意：`data.py` 里导入过 `pt` 是**另一个模块的命名空间**，本文件不会继承。
import prompt_templates as pt
from fire import Fire
import pickle
import math
import json
from sklearn.metrics import ndcg_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---- [本项目新增 2026-09-25] 部分信用奖励（阶段 0 判决：rule 的 0/1 精确匹配信号过稀 ⟹ 多数步零梯度）----
# completion 与 target 都是 "<a_x><b_y><c_z>"（Trie 约束下必为合法 SID 路径）。
# 档位：全对 1.0 / 前两位对 0.6 / 仅首位对 0.3 / 不匹配或格式非法 0.0。
# 判据（RL_PIPELINE §6.9 阶段 0）：reward 非零步占比应从 ~13% 升到 >50%（起跑 2 min 即可看）。
_SID_RE = re.compile(r'<a_(\d+)><b_(\d+)><c_(\d+)>')


def sid_partial_credit(completion: str, target: str) -> float:
    """0/1 精确匹配 → 逐位部分信用。格式非法一律 0（宁缺毋滥，不给噪声梯度）。"""
    c = _SID_RE.fullmatch(completion.strip("\n\" "))
    t = _SID_RE.fullmatch(target.strip("\n\" "))
    if c is None or t is None:
        return 0.0
    if c.groups() == t.groups():
        return 1.0
    if c.group(1) == t.group(1) and c.group(2) == t.group(2):
        return 0.6
    if c.group(1) == t.group(1):
        return 0.3
    return 0.0


def train(    # model/data params
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",
    
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    
    # training hyperparams
    output_dir: str = "",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    num_generations: int = 16,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    max_completion_length: int = 16,
    reward_type: str = "rule",
    sample_train: bool = False,

    # ---- [本项目新增] RL 数据集构成：任务子集 / 按规模缩放 ----
    # 用途：做"训练与评估同一批用户"的小规模迭代时，把**无法按用户切分**的 T2 去掉（只留 T1,T3）。
    #   `T1` = SidDataset            （历史 SID → 目标 SID）
    #   `T2` = RLTitle2SidDataset    （title → SID ＋ description → SID；**item-level，无用户维度**）
    #   `T3` = RLSeqTitle2SidDataset （历史**标题** → 目标 SID）
    # 🔴 与 SFT 的 `--tasks` **不是一套**：RL 无 T2a/T2b 之分，且三路 target **全是 SID**（`rule` 奖励要求）。
    # 默认 "T1,T2,T3" + 下面两个上限，保证与原版行为**逐位相同**。
    rl_tasks: str = "T1,T2,T3",
    # rl_t2_sample / rl_t3_sample：给 T2 / T3 单独设行数上限（<=0 视为不限）。
    # 为什么需要：T1 会被"按用户过滤后的 train CSV"缩掉，而 **T2 是 item-level 缩不掉**、
    # T3 原本的 10000 又是**硬编码绝对值** ⟹ 不设上限则任务配比严重走样。
    # [实测] 5,000 用户：T1 20,496 行；若不 cap，T2 51,433 + T3 10,000 会让 T2 占比
    # 从全量的 19% 跳到 63%、单次迭代 38 min → 3.5 h。→ RL_PIPELINE §6.9 阶段 0
    rl_t2_sample: int = -1,
    rl_t3_sample: int = 10000,
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    dapo: bool = False,
    gspo: bool = False,
    resume_from_checkpoint: str = None,
    # ---- 本项目新增（与 sft.py 同口径）----
    # torch_compile：[红线] 默认关。GRPO 每步生成长度不定 + 动态 padding，
    #   torch.compile 会反复重编译（本仓已在 SFT 阶段实测过这个坑）。
    torch_compile: bool = False,
    # save 频率与上限。Trainer/GRPOConfig 语义：**< 1 = 占训练总步数的比例；>= 1 = 绝对步数**。
    # ⚠️ 全参 GRPO 的单个 checkpoint 很大（bf16 权重 + paged_adamw_32bit 的 fp32 状态
    #    ≈ 6 GB），原版 save_total_limit=20 会占 ~120 GB 磁盘，本仓下调。
    save_steps: float = 0.1,
    save_total_limit: int = 3,
    # 优化器：paged_adamw_32bit 需 bitsandbytes（本机实测 0.48.1 可用）
    optim: str = "paged_adamw_32bit",

    # 计算精度：bf16（Ampere+ 默认）| fp16（V100 等 Volta 必须用这个）| fp32
    precision: str = "bf16",

    # ---- LoRA（[本项目新增] 本文件原本完全不支持 LoRA）----
    # 为什么 RL 侧**不需要** modules_to_save（与 SFT 相反）：SFT 时 SID 是刚 add_tokens
    # 出来的新 token，embed_tokens 不训就永远随机初始化；而 RL 是在 SFT 产物之上接着训，
    # SID 的 embedding 已经训好了 ⟹ 冻结它们、只训 attention 是 GRPO+LoRA 的常规做法。
    # [实测] 3050Ti 4GB / 16 条序列（RL_PIPELINE §6.1 / §6.3）：
    #   带 embed_tokens,lm_head -> 可训 321.4M，优化器 step 瞬时峰值 4.20 GiB（超物理）
    #   不带                    -> 可训   9.2M，峰值            2.59 GiB
    # 另一个坑：带 modules_to_save 会**破坏 tie_word_embeddings** —— [实测] resize 后
    #   tie=True，套上 LoRA 后变成两份独立张量，词表参数直接翻倍。
    use_lora: bool = False,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_targets: str = "q_proj,k_proj,v_proj,o_proj",
    lora_modules_to_save: str = "",   # 留空 = 不训 embedding / lm_head（RL 推荐）

    # 梯度检查点：用时间换激活内存。**本地小卡必须开**。
    # [实测] 3050Ti 4GB / 16 条序列（prompt 177 + completion 16）：
    #   关 = 9.33 GiB（必然换页）| 开 = 2.59 GiB。64 条序列时 3.74 GiB 仍可装。
    grad_ckpt: bool = False,

    # 训练步数上限（-1 = 按 num_train_epochs）。本地冒烟用它把训练截到 1~2 步。
    max_steps: int = -1,
):

    # ---- 计算精度（[本项目新增] 原本三处硬编码 bf16）----
    # 🔴 bf16 只在 Ampere+（sm_80）有原生支持；V100 是 Volta（sm_70），必须走 fp16。
    # 🔴 **fp16 不能把模型加载成 fp16**：Trainer 在 fp16=True 时挂 GradScaler，
    #    而 GradScaler 拒绝 unscale fp16 梯度 —— [实测] 直接报
    #    "ValueError: Attempting to unscale FP16 gradients"。
    #    正确做法 = **fp32 主权重 + autocast 把计算降到 fp16**（AMP 的标准形态）。
    #    代价是权重显存翻倍（0.6B: 1.2G -> 2.4G），换来数值稳定。
    if precision == "bf16":
        _dt, _bf16, _fp16 = torch.bfloat16, True, False
    elif precision == "fp16":
        _dt, _bf16, _fp16 = torch.float32, False, True
    elif precision == "fp32":
        _dt, _bf16, _fp16 = torch.float32, False, False
    else:
        raise ValueError(f"precision 只支持 ['bf16', 'fp16', 'fp32']，收到 {precision!r}")
    _attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        _attn_impl = "sdpa"
        print("flash-attn not found, using PyTorch SDPA (memory-efficient) instead")
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ---- 前置护栏：[实测] rl.py 与 ReReTrainer 都**不做** add_tokens / resize_token_embeddings，
    #      完全依赖 --model_path 那个目录自带的扩展 tokenizer + 已 resize 的 embedding。
    #      指回原始基座（models/Qwen3-0.6B）会让 SID 碎成子 token、约束映射全废，而且**不报错**。
    #      所以这里主动拦下来，别等训完才发现。
    _tok_probe = AutoTokenizer.from_pretrained(model_path)
    _sid_probe = _tok_probe.encode("<a_0>", add_special_tokens=False)
    if len(_sid_probe) != 1:
        raise ValueError(
            f"--model_path 必须是 **SFT 训练产物**（自带扩展 tokenizer），"
            f"但 '{model_path}' 把 '<a_0>' 切成了 {len(_sid_probe)} 个 token：{_sid_probe}。\n"
            f"  正确用法: --model_path outputs/<SFT_EXP_ID>/final_checkpoint\n"
            f"  指回原始基座会让 SID 碎裂、约束映射全废，且不会报错。"
        )
    # 响应前缀由模板真源给出（不再手抄 '### Response:\n'）—— 它必须与 data.py 渲染的
    # prompt 末尾、以及 LogitProcessor/ReReTrainer 硬编码的 prefix_index 三者一致。
    _prefix = pt.response_prefix()
    _pfx = _tok_probe.encode(_prefix, add_special_tokens=False)
    if len(_pfx) != 3:
        raise ValueError(
            f"响应前缀 {_prefix!r} 在 '{model_path}' 的 tokenizer 下切成 {len(_pfx)} 个 token"
            f"（{_pfx}），而 LogitProcessor/ReReTrainer 硬编码 prefix_index=3。\n"
            f"  模板真源 = {pt.source(anchor=model_path)}\n"
            f"  换基座 / 换 tokenizer / 换格式时必须重测这条，并同步 prefix_index。"
        )
    print(f"[guard] SID tokenizer OK: '<a_0>'=1 token, 响应前缀 {_prefix!r}={len(_pfx)} tokens "
          f"(prefix_index=3 成立)  vocab={len(_tok_probe)}")
    print(f"[guard] precision={precision} -> dtype={_dt}  "
          f"(bf16 需 Ampere+；V100/Volta 请用 fp16)")
    print(f"[guard] use_lora={use_lora}  grad_ckpt={grad_ckpt}  max_steps={max_steps}")
    if not grad_ckpt:
        print("[guard] \u26a0\ufe0f gradient_checkpointing=OFF —— [实测] 3050Ti 4GB 上 16 条序列要 "
              "9.33 GiB，本地跑请加 --grad_ckpt True；3090 24G 无所谓")


    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    
    
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Extract semantic_id (first column) from the format: semantic_id \t item_title \t item_id
        item_name = [_.split('\t')[0].strip() for _ in info]
        item2id = {name: i for i, name in enumerate(item_name)}

    sample = -1

    # ---- [本项目新增] 任务子集（默认 T1,T2,T3 = 原版行为）----
    # 🔴 必须用 `parse_csv_list`，不能 `str(rl_tasks).split(",")`：
    #    fire 会把命令行里的 `T1,T3` 解析成 **tuple** ⟹ `str(('T1','T3'))` = `"('T1', 'T3')"`
    #    ⟹ split 出 `["('T1'", " 'T3')"]` ⟹ 全部"未知任务"直接 raise。
    #    [实测] 2026-09-25 云端 `RL_TASKS=T1,T3` 就是这么崩的（默认值是 Python 字符串时不会暴露）。
    #    这与 `sft.py:147` 记的是**同一个坑**（`rl.py:15` 早就 import 了这个助手，我却又手写了一遍错的）。
    _rl_tasks = parse_csv_list(rl_tasks)
    _valid = ("T1", "T2", "T3")
    if not _rl_tasks:
        raise ValueError("rl_tasks 不能为空（默认 'T1,T2,T3'）")
    _bad = [t for t in _rl_tasks if t not in _valid]
    if _bad:
        raise ValueError(f"rl_tasks 只接受 {_valid} 的子集，收到 {_bad!r}（完整值={rl_tasks!r}）")
    _t2_sample = rl_t2_sample if rl_t2_sample and rl_t2_sample > 0 else -1
    _t3_sample = rl_t3_sample if rl_t3_sample and rl_t3_sample > 0 else -1
    print(f"[RL_TASKS] 启用 {','.join(_rl_tasks)}"
          f"   (T1=SidDataset seq2sid / T2=RLTitle2SidDataset title2sid+desc2sid(item-level)"
          f" / T3=RLSeqTitle2SidDataset seqtitle2sid)")
    print(f"[RL_TASKS] 行数上限：T2={_t2_sample}  T3={_t3_sample}  (<=0 = 不限)")

    train_datasets = []
    # train_data = D3Dataset(train_file, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data)
    if "T1" in _rl_tasks:
        train_datasets.append(SidDataset(train_file, category=category_dict[category], sample=sample))
    if "T2" in _rl_tasks:
        train_datasets.append(RLTitle2SidDataset(item_file=item_meta_path, index_file=sid_index_path,
                                                category=category_dict[category], sample=_t2_sample))
    if "T3" in _rl_tasks:
        train_datasets.append(RLSeqTitle2SidDataset(train_file, category=category_dict[category], sample=_t3_sample))
    # train_data4 = RLSid2TitleDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data4)
    # train_data5 = RLSidhis2TitleDataset(train_file, item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data5)
    # train_data6 = RLTitle2Sid_1LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data6)
    # train_data7 = RLTitle2Sid_2LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data7)
    for _i, _d in enumerate(train_datasets, 1):
        print(f"  {_i}. {type(_d).__name__:<24s} {len(_d):>9,} 条")
    train_data = ConcatDataset(train_datasets)
    print(f"  [RL_TASKS] 训练集合计 {len(train_data):,} 条")
    # eval_data = D3Dataset(eval_file, category=category_dict[category], sample=sample)
    # ⚠️ 验证集恒为 SidDataset（= T1 格式），与 rl_tasks 无关 —— 与 sft.py:471 同款设计。
    eval_data = SidDataset(eval_file, category=category_dict[category], sample=sample)

    train_dataset = Dataset.from_dict({k : [elm[k] for elm in train_data] for k in train_data[0].keys()})
    train_dataset = train_dataset.shuffle(seed=seed) 
    if sample_train and "sft" in model_path:
        train_dataset = train_dataset.select(range(int(0.2 * len(train_dataset)), len(train_dataset)))
    eval_dataset = Dataset.from_dict({k : [elm[k] for elm in eval_data] for k in eval_data[0].keys()})
    eval_dataset = eval_dataset.shuffle(seed=seed)
    

    # prompt2history = {**train_data.prompt2history, **eval_data.prompt2history}
    # history2target = {**train_data.history2target, **eval_data.history2target}

    prompt2history = {}
    history2target = {}
    
    # Collect prompt2history and history2target from all train datasets
    for dataset in train_datasets:
        if hasattr(dataset, 'prompt2history'):
            prompt2history.update(dataset.prompt2history)
        if hasattr(dataset, 'history2target'):
            history2target.update(dataset.history2target)
    
    # Add eval_data mappings
    if hasattr(eval_data, 'prompt2history'):
        prompt2history.update(eval_data.prompt2history)
    if hasattr(eval_data, 'history2target'):
        history2target.update(eval_data.history2target)

    print("train_dataset: ", train_dataset)
    print("eval_dataset: ", eval_dataset)

    # 🔴 [本项目修正] 原版这里**无条件**加载一份 llm_model，但它只被 reward_type=="semantic"
    #    用到（下面 item_ada_embd.to(llm_model.device)）。rule/ranking 奖励下它是**纯重复**：
    #    训练用的模型由 ReReTrainer 按 model_init_kwargs 自己再加载一份 ⟹ 白白多占一份权重
    #    （0.6B bf16 = 1.14 GiB）。而这 1.14 GiB 往往正是"本地能不能跑"的分界线。
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if reward_type == "semantic":
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=_dt, device_map="auto", attn_implementation=_attn_impl)
        device = llm_model.device
    else:
        llm_model = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[mem] reward_type={reward_type} 不需要额外模型副本，跳过 llm_model 加载"
              f"（省约 1.14 GiB）  device={device}")
    
    len_seq = 10
    item_num = len(item_name)
    print(f"item_num: {item_num}")

    if reward_type == "sasrec":
        if not cf_path or not os.path.exists(cf_path):
            raise ValueError(f"reward_type='sasrec' 需要 --cf_path（SASRec 的 state_dict），"
                             f"当前 cf_path={cf_path!r}。本仓尚无该权重，需先用根目录 sasrec.py 训练。")
        model = SASRec(32, item_num, len_seq, 0.3, device)
        model.to(device)
        model.load_state_dict(torch.load(cf_path))
        model.eval()
    if reward_type == "semantic":
        if not ada_path or not os.path.exists(ada_path):
            raise ValueError(f"reward_type='semantic' 需要 --ada_path（item embedding 的 pickle），"
                             f"当前 ada_path={ada_path!r}。本仓尚无该文件。")
        with open(ada_path, "rb") as f:
            item_ada_embd = pickle.load(f)
        item_ada_embd = torch.tensor(item_ada_embd).to(llm_model.device)
        print(f"Load item_ada_embd successfully. shape={tuple(item_ada_embd.shape)}")

    ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
    ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]


    def ndcg_rule_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        repeat = num_generations
        rewards = []
        flag = False
        lis = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\"") == targets[i].strip("\n\""):
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i%num_generations])
            
            if (i+1)%num_generations == 0:
                if flag:
                    rewards.extend(lis)
                else:
                    rewards.extend([0.0] * repeat)
                flag = False
                lis = []
        
        return rewards

    def rule_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        rewards = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\" ") == targets[i].strip("\n\" "):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    def partial_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        return [sid_partial_credit(completions[i], targets[i]) for i in range(len(completions))]

    def semantic_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        target_ids = [item2id[elm.strip("\"\n")] for elm in targets]
        completions = [elm.strip("\"\n") for elm in completions]
        for i, completion in enumerate(completions):
            if completion not in item2id:
                print("==============================")
                print(prompts[i])
                print(f"Invalid item: {completion}")
                print("==============================")
        completion_ids = [item2id[elm] for elm in completions]
        rewards =  torch.cosine_similarity(item_ada_embd[target_ids], item_ada_embd[completion_ids], dim=-1)
        print(rewards)
        return rewards

    def cf_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        history_list = [elm.split("::") for elm in history]
        pred_ids = []
        for i, elm in enumerate(completions):
            elm = elm.strip("\n\"")
            if elm not in item_name:
                # print("========Invalid Item========")
                # print(f"Invalid item: {elm}")
                # print(f"Prompt: {prompts[i]}")
                # print("============================")
                pred_ids.append(random.randint(0, item_num-1))
            else:
                pred_ids.append(item2id[elm])
        
        len_lis = []
        history_ids = []
        for his in history_list:
            his = [item2id[elm] for elm in his]
            len_lis.append(len(his))
            if len(his) < len_seq: 
                his = his + [item_num] * (len_seq - len(his))
            history_ids.append(his)
        
        seq = torch.LongTensor(history_ids).to(device)
        pred = torch.LongTensor(pred_ids).to(device)    
        
        with torch.no_grad():
            predictions = model.forward_eval(seq, torch.tensor(np.array(len_lis)).to(device))
            scores = torch.gather(predictions, 1,  pred.view(-1, 1)).view(-1)
        return scores
    


    if reward_type == "rule":
        reward_fun = rule_reward
    elif reward_type == "ranking":
        reward_fun = [rule_reward, ndcg_rule_reward]
    elif reward_type == "ranking_only":
        reward_fun = ndcg_rule_reward
    elif reward_type == "semantic":
        reward_fun = semantic_reward
    elif reward_type == "sasrec":
        reward_fun = cf_reward
    elif reward_type == "partial":
        reward_fun = partial_reward
    else:
        raise ValueError(
            f"reward_type 只接受 rule/ranking/ranking_only/semantic/sasrec/partial，收到 {reward_type!r}"
        )
    
    os.environ['WANDB_PROJECT'] = wandb_project
    report_to = "wandb" if wandb_project or wandb_run_name else "none"

    training_args = GRPOConfig(output_dir=output_dir,
                                # 训练用的模型由 ReReTrainer 自己加载；dtype 必须在这里给，
                                # 否则 fp16 精度下它会按 config.json 的默认 dtype 加载。
                                model_init_kwargs={"attn_implementation": _attn_impl,
                                                   "dtype": _dt},
                                save_steps=save_steps,
                                save_total_limit=save_total_limit,
                                eval_strategy="steps",
                                max_completion_length=max_completion_length,
                                num_generations=num_generations,
                                temperature=temperature,
                                sync_ref_model=sync_ref_model,
                                per_device_eval_batch_size=eval_batch_size,
                                per_device_train_batch_size=train_batch_size,
                                gradient_accumulation_steps=gradient_accumulation_steps,  
                                eval_steps=eval_step, 
                                logging_steps=1, 
                                learning_rate=learning_rate,
                                beta=beta,
                                warmup_ratio=0.03,
                                max_grad_norm= 0.3,
                                num_train_epochs=num_train_epochs,
                                bf16=_bf16,
                                fp16=_fp16,
                                optim=optim,
                                lr_scheduler_type="cosine", 
                                save_strategy="steps",
                                report_to=report_to,
                                run_name=wandb_run_name,
                                torch_compile=torch_compile,
                                gradient_checkpointing=grad_ckpt,
                                **({"max_steps": max_steps} if max_steps and max_steps > 0 else {}),
                            )
    # ---- LoRA：构造 PeftConfig 交给 ReReTrainer（而不是在这里预 wrap 模型）----
    # 依据 minionerec_trainer.py:280-292 —— 它拿到 peft_config 会 get_peft_model()，
    # 且 is_peft_model(model) 为真时把 **self.ref_model 置 None**：参考模型不额外占权重，
    # 靠 disable_adapter() 复用同一份（:901-904）。这就是"GRPO 显存要翻倍"这个说法
    # 在 PEFT 下不成立的机制。
    peft_config = None
    if use_lora:
        from peft import LoraConfig
        _targets = parse_csv_list(lora_targets)
        _msave = parse_csv_list(lora_modules_to_save)
        if not _targets:
            raise ValueError(f"--lora_targets 解析为空：{lora_targets!r}")
        peft_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
            task_type="CAUSAL_LM", target_modules=_targets, modules_to_save=_msave or None,
        )
        print(f"[LoRA] r={lora_r} alpha={lora_alpha} dropout={lora_dropout} targets={_targets}")
        _risky = [m for m in _msave if m in ("embed_tokens", "lm_head")]
        print(f"[LoRA] modules_to_save={_msave or '（空）'}"
              + (f"   ⚠️ 含 {_risky} -> 词表参数翻倍、破坏 tie；[实测] 优化器 step 峰值 "
                 f"2.59 -> 4.20 GiB，本地 4GB 会超" if _risky else
                 "   （推荐：SID embedding 在 SFT 已训好，RL 不必再训）"))

    trainer = ReReTrainer(
        model=model_path,
        peft_config=peft_config,
        base_model=model_path,
        dapo=dapo,
        gspo=gspo,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        info_file=info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # 🔴 LoRA + gradient_checkpointing 必须补这一句，否则**梯度静默变成 None**：
    #    PyTorch 的 checkpoint 要求该段计算的输入 requires_grad，而 LoRA 冻结了 base、
    #    embed_tokens 也不可训 ⟹ 第一层检查点的输入不 require grad，只会打印一条 warning
    #    然后把梯度置 None —— 显存看着很美，其实模型没在学。
    #    [实测] transformers/modeling_utils.py:2849 有这个 API，但 Trainer / TRL / ReReTrainer
    #    **没有任何一方自动调用**（已 grep 三处确认）。
    if use_lora and grad_ckpt:
        trainer.model.enable_input_require_grads()
        print("[LoRA] enable_input_require_grads() 已调用（gradient checkpointing 下保梯度）")
    if use_lora:
        _n = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        _tot = sum(p.numel() for p in trainer.model.parameters())
        print(f"[LoRA] 可训参数 {_n:,} / {_tot:,} = {100*_n/_tot:.2f}%")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    # 🔴 LoRA 下 trainer.model 是 PeftModel，直接 save_pretrained 只落 adapter
    #    （此时 OUTPUT_DIR 根目录那份就是 adapter），而 evaluate.py 用
    #    AutoModelForCausalLM.from_pretrained 原样加载 ⟹ 会失败。必须 merge 后存。
    final_model = trainer.model
    if use_lora:
        try:
            final_model = final_model.merge_and_unload()
            print("[LoRA] adapter merged -> final_checkpoint/ 落完整权重")
        except Exception as e:
            print(f"[LoRA] merge_and_unload 失败（{type(e).__name__}: {e}），final_checkpoint/ 是 adapter")
    final_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
if __name__ == "__main__":
    Fire(train)
