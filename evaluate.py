
import pandas as pd
import fire
import torch
import json
import os
from transformers import GenerationConfig,  AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper
from data import  EvalD3Dataset, EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor
import prompt_templates as pt   # 提示词/响应前缀的唯一真源
from accelerate import Accelerator
import random
import bitsandbytes as bnb



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
P = 998244353
MOD = int(1e9 + 9)
import numpy as np

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 4,
    K: int = 0,
    seed: int = 42,
    length_penalty: float=0.0,
    # 🔴 [本项目 2026-09-19 提效] 默认 256 -> 8。
    #    依据：目标 SID 恒为 3 个 token + 1 个 EOS（Trie 在 sid_vocab 里把 EOS 追加为第 4 步），
    #    实测 2000 条生成结果长度分布 16~21 字符（= 3 个 SID token），**无一例外**。
    #    原值 256 意味着每条序列白跑 252 步 ⟹ 解码耗时虚高约 60 倍。
    #    ⚠️ 若将来改 SID 层数或放开"最多 N 个候选 SID"的多目标生成，必须同步调大这个值。
    max_new_tokens: int = 8,
    num_beams: int = 50,
    sid_vocab_path: str = "",   # [本项目新增] 非空则现场注册 SID 词表（dry-run / 未训练基座用）
    max_samples: int = 0,       # [本项目新增] 0=全部；>0 只随机取 N 条（dry-run 用，显著提速）

    # ---- 解码采样（[本项目新增] 原本完全靠继承基座，不可控也不可见）----
    # 🔴 背景：本函数构造 GenerationConfig 时若**不传** do_sample，generate() 内部的
    #    `_prepare_generation_config` 会用**基座 generation_config.json 的非默认值**填充它。
    #    Qwen3-0.6B / Qwen2.5-0.5B 两个基座都写着 `do_sample: true` ⟹ 原来**一直在采样**
    #    （beam sampling，不是纯 beam search），且本仓代码里看不到这个事实。
    #    实测溯源见 docs/DECODING_STRATEGIES.md §2.2/§2.3 与 docs/SFT_PIPELINE.md §3.5.4。
    # ⟹ 现在**显式**传参：默认关闭采样（确定性 beam search，可复现），
    #    需要采样消融时显式 DO_SAMPLE=True + TEMPERATURE / TOP_P。
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,         # 1.0 = 不截断（HF 语义：>=1 视为不启用核采样）

    # 计算精度：bf16（Ampere+ 默认）| fp16（V100 等 Volta 必须用这个）| fp32
    precision: str = "bf16",

    # attention 后端：[本项目 2026-09-19 提效] sdpa（默认，零依赖）| eager | flash_attention_2
    attn_impl: str = "sdpa",
):
    random.seed(seed)

    # ---- 计算精度（[本项目新增] 原本硬编码 bf16）----
    # 纯推理，没有 Trainer、没有 GradScaler ⟹ 直接用 fp16 权重即可（V100 上正确且省显存）。
    # ⚠️ 训练侧不同：那边 fp16 必须用 fp32 主权重，见 sft.py / rl.py 同名段落。
    _DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if precision not in _DTYPE:
        raise ValueError(f"precision 只支持 {sorted(_DTYPE)}，收到 {precision!r}")
    _dt = _DTYPE[precision]
    set_seed(seed)
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    category = category_dict[category]
    print(category)

    # ---- 解码口径显式回显（[本项目新增]）----
    # 🔴 这条日志是"口径可追溯"的一部分：采样与否会改变 HR（束采样引入随机性），
    #    以前它由基座静默决定、日志里查不到，跨版本比数时无从对账。
    _mode = "BEAM_SAMPLE（束采样，含随机性）" if (do_sample and num_beams > 1) else \
            "SAMPLE（纯采样）" if do_sample else \
            "BEAM_SEARCH（纯束搜索，确定性）" if num_beams > 1 else "GREEDY（贪心）"
    print(f"[解码] mode={_mode}")
    print(f"[解码] num_beams={num_beams}  do_sample={do_sample}  "
          f"temperature={temperature}  top_p={top_p if top_p < 1.0 else 'None(不截断)'}")
    if do_sample and num_beams > 1:
        print("[解码] ⚠️ 束采样下结果依赖随机种子，同模型重跑会有抖动；"
              "与 do_sample=False 的结果不可直接横比。")

    # [本项目 2026-09-19 提效] 显式指定 attention 后端。
    #   sdpa = PyTorch 原生 scaled_dot_product_attention，在 Ada（4090D, sm_89）上会自动走
    #   FlashAttention-2 内核，**零额外依赖**（不需要 flash_attn 包）。
    #   ⚠️ 若环境装了 flash_attn 且想用，传 ATTN_IMPL=flash_attention_2。
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=_dt, device_map="auto",
        attn_implementation=attn_impl,
    )
    print(f"[模型] dtype={precision}  attn={attn_impl}  device={model.device}")
    model.eval()
    model_device = next(model.parameters()).device
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Parse new format: semantic_id \t item_title \t item_id
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]
        
        # Format for tokenization
        # 🔴 约束解码的 key 前缀必须与 data.py 渲染的 prompt 末尾**同一个真源**
        #    （chatml = '<|im_start|>assistant\n'，实测 3 token ⟹ prefix_index=3 成立）。
        info_semantic = [pt.response_prefix(anchor=info_file) + _ for _ in semantic_ids]
        info_titles = [pt.response_prefix(anchor=info_file) + _ for _ in item_titles]


    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # [本项目新增] 可选：现场注册 SID 词表。口径与 sft.py:241-306 完全一致（读 sid_vocab.json，码序）。
    # 用途一：未训练基座也能跑通整条评估链路（evaluator 冒烟测试，不用先烧 GPU）。
    # 用途二：把「评估端词表必须与训练端一致」从"靠人记得指对目录"变成代码保证。
    # ⚠️ 训练产物自带扩展后的 tokenizer 时，add_tokens 幂等（新增 0 个），resize 也会被跳过。
    if sid_vocab_path:
        with open(sid_vocab_path, encoding="utf-8") as _f:
            _sid_vocab = json.load(_f)
        _added = tokenizer.add_tokens(_sid_vocab)
        _emb_rows = model.get_input_embeddings().weight.shape[0]
        print(f"[SID] 注册 {len(_sid_vocab)} 个 SID token（新增 {_added}）；"
              f"tokenizer={len(tokenizer)}  模型 embedding 行数={_emb_rows}")
        if _emb_rows != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
            print(f"[SID] resize -> {len(tokenizer)}"
                  f"（⚠️ 新增行是随机初始化 —— 只有未训练的基座才会走到这里）")
        else:
            print("[SID] 模型 embedding 已对齐，跳过 resize")

    # Create prefixID for semantic IDs (existing functionality)
    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids[1:] for _ in info_titles]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]
    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3
    
    # Build hash_dict for semantic IDs (existing functionality)
    hash_dict = dict()
    # print(f"eos token: {tokenizer.eos_token_id}")
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    # Build hash_dict_title for item titles (new functionality)
    hash_dict_title = dict()
    for index, ID in enumerate(prefixTitleID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict_title:
                hash_dict_title[hash_number] = set()
            hash_dict_title[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    # Convert sets to lists for both dictionaries
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    for key in hash_dict_title.keys():
        hash_dict_title[key] = list(hash_dict_title[key])

    # Define prefix constraint functions
    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []
        
    def prefix_allowed_tokens_fn_title(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict_title:
            return hash_dict_title[hash_number]
        return []

    # Default to semantic constraints (backward compatibility)
    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_semantic
    # prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_title
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # val_dataset = EvalD3Dataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
    val_dataset = EvalSidDataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed,
                                 sample=(max_samples if max_samples > 0 else -1))
    if max_samples > 0:
        print(f"[DRY-RUN] 只取 {len(val_dataset)} 条（seed={seed} 随机采样，非前 N 条）")
        
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    # encodings = [val_dataset[i] for i in indexes]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate(
            encodings,
            num_beams=10,
            max_new_tokens=8,
            length_penalty=1.0,
            **kwargs,
    ):
        # [本项目 2026-09-19 提效] padding 长度按**本批**最大值算，不再用全局 maxLen。
        #   原实现把所有批 pad 到全数据集最长样本的长度 ⟹ 每个 batch 都在算大量 pad token
        #   （decode 阶段 pad 位置也要过 attention）。按批 pad 后总 FLOPs 显著下降，
        #   且**不改变任何一条样本的生成结果**（左 padding + attention_mask 语义等价）。
        maxLen = max([len(_["input_ids"]) for _ in encodings])

        padding_encodings = {"input_ids": []}
        attention_mask = []

        for  _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L) 
        
        # print(f"num_beams: {num_beams}")
        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id = model.config.pad_token_id,
            eos_token_id = model.config.eos_token_id,
            max_new_tokens = max_new_tokens,
            top_k=None,
            # [本项目新增] 显式固定"是否采样"这条口径，不再让它静默继承基座。
            #   只传这两个值即可决定模式：do_sample=False -> BEAM_SEARCH；
            #   do_sample=True -> BEAM_SAMPLE（HF 的 get_generation_mode 判定）。
            #   ⚠️ temperature / top_p 只在 do_sample=True 时才有意义，否则 HF 会忽略。
            do_sample=do_sample,
            temperature=temperature,
            top_p=(top_p if top_p < 1.0 else None),
            **kwargs
        )
        
        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=base_model,
                eos_token_id=model.config.eos_token_id
            )
            logits_processor = LogitsProcessorList([clp])

            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(model_device),
                attention_mask=torch.tensor(attention_mask).to(model_device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
                # 🔴 [本项目新增] 必须显式关掉"用模型默认值回填"，否则上面的
                #    do_sample=False 会被基座 generation_config.json 的 true 覆盖掉 ——
                #    因为 HF 的合并规则是「传入值 == 全局默认值 且 模型值 != 全局默认值 ⟹ 取模型值」，
                #    而 False 恰好 == 全局默认值 ⟹ **传 False 等于没传**（实测已验证）。
                #    use_model_defaults=False 会禁用整个回填逻辑，让显式参数真正生效。
                #    行为细节见 transformers/generation/utils.py:_prepare_generation_config。
                use_model_defaults=False,
            )
       
        batched_completions = generation_output.sequences[:, maxLen:]
       
        
        if base_model.lower().find("llama") > -1:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
            
        output = [_.split("Response:\n")[-1].strip() for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs
    
    model = model.to(model_device)

    from tqdm import tqdm
    outputs = []
    new_encodings = []
    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size])

    
    # [本项目 2026-09-19 提效] 打印解码规模，便于估时间/显存。
    _nseq = batch_size * num_beams
    print(f"[解码] 样本={len(encodings)}  批={batch_size}  beam={num_beams}  "
          f"=> 每批 {_nseq} 条序列  max_new_tokens={max_new_tokens}  批数={BLOCK}")
    if _nseq > 160:
        print(f"  ⚠️ batch×beam={_nseq} 偏大，若 OOM 请降 BATCH_SIZE（别降 NUM_BEAMS —— 它是 HR@K 硬上限）")

    import time as _time
    _t0 = _time.time()
    for idx, encodings in enumerate(tqdm(new_encodings)):
        # Use standard evaluation
        output = evaluate(encodings, max_new_tokens=max_new_tokens, num_beams=num_beams, length_penalty=length_penalty)
        
        outputs = outputs + output
    _el = _time.time() - _t0
    print(f"[解码] 完成 {BLOCK} 批，耗时 {_el:.1f}s（{_el/max(BLOCK,1)*1000:.0f} ms/批，"
          f"{_el*1000/max(len(encodings),1):.0f} ms/样本）")
    if torch.cuda.is_available():
        print(f"[解码] 峰值显存 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
       
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
  

    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')  
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(main)




