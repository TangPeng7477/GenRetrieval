# RL Pipeline（GRPO）— 唯一入口

> 2026-09-17 ｜ 本文件是 **RL 阶段的唯一入口**：上游依赖 / 参数映射 / 实测 / 改动清单都在这里。
> SFT 侧见 [`docs/SFT_PIPELINE.md`](SFT_PIPELINE.md)；指标口径见 [`docs/EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md)。
> 升级计划与消融矩阵（R0~R3）见 [`docs/UPGRADE_PLAN.md`](UPGRADE_PLAN.md) §6。

## 0. TL;DR

| 项 | 结论 |
|---|---|
| **入口** | **`bash rl_run0.sh`**（根目录）。⚠️ `rl.sh` / `rl_3090.sh` 是 MiniOneRec **原版**，路径写死 `./data/Amazon/...`（本仓不存在），**不要直接用** |
| 算法 | **GRPO** = `trl.GRPOConfig` + 本仓自定义 `minionerec_trainer.ReReTrainer`（`[实测]` trl 0.24.0 + transformers 4.57.1 下 `import` 通过） |
| 🔴 **数据集** | **零新增**。三个在用的 Dataset 类**全部直接读 SFT 阶段已落盘的同名产物**（§1） |
| 🔴 **模型起点** | **必须是 SFT 训练产物目录**（`outputs/<SFT_EXP_ID>/final_checkpoint`）。指回原始基座会让 SID 碎裂、约束映射全废，**且不报错**（§2 已有主动护栏） |
| 奖励 | 立即可用：`rule` / `ranking` / `ranking_only`（R0 锚点 = `rule`）。`semantic` / `sasrec` **缺 artifact**（§4） |
| 当前状态 | ⏸ **被 SFT Run-0 阻塞**——还没有 SFT 模型可接着训 |

---

## 1. 数据集：**零新增**，全部复用 SFT 阶段产物

`rl.py` 用三个 Dataset 类（`rl.py:130/132/134`），它们不落任何新文件，
prompt 全在 `pre()` 里现场拼：

| RL 类（`data.py`） | 构造参数 | 读的上游文件 | 与 SFT 阶段的关系 |
|---|---|---|---|
| `SidDataset`（`:354`） | `train_file` | `train/<域>_5_train.csv` | **同一个 CSV**（SFT 的 T1 用的就是它） |
| `RLTitle2SidDataset`（`:794`） | `item_file` + `index_file` | `index/<域>.item.json` + `index/<域>.index.json` | **同两个文件**（SFT 的 T2b 用的就是它） |
| `RLSeqTitle2SidDataset`（`:895`） | `train_file` | `train/<域>_5_train.csv`（用 `history_item_title` 列） | **同一个 CSV** |

加上 `info_file`（`info/<域>.item_info.txt`）与 `eval_file`（`valid/<域>_5_valid.csv`）——
**这五类文件 SFT 阶段已经全部产出**，RL 侧不需要跑任何 `prepare_*` 脚本。

> `[实测]` 三个类的字段需求与 CSV 表头完全咬合：
> `history_item_sid` / `history_item_title` / `item_sid` / `item_id` / `history_item_id` 均在表内。

### 1.1 任务构成（与 SFT 的对应关系）

| RL 数据集 | 任务形态 | 对应 SFT 任务 |
|---|---|---|
| `SidDataset` | 历史 SID → 目标 SID（主任务） | **T1** `seq2sid` |
| `RLTitle2SidDataset` | title/description → SID | **T2b** `title2sid` |
| `RLSeqTitle2SidDataset` | 历史**标题**序列 → 目标 SID | 无直接对应（SFT 的 T3 是 seq→title，方向相反） |

> ⚠️ `rl.py:134` 里 `RLSeqTitle2SidDataset` 的 `sample=10000` 是**硬编码**的上限，
> 想要全量要改代码或加参数（当前未改，作为已知差异记录）。

---

## 2. 入口：`rl_run0.sh`

```bash
# 默认：IandS 单域，从 IandS-run0 的 SFT 产物接着训
bash rl_run0.sh

# 换 SFT 来源
SFT_EXP_ID=IandS-S0 bash rl_run0.sh

# 换奖励 / 消融标签
REWARD_TYPE=ranking RUN_TAG=R1 bash rl_run0.sh
```

### 2.1 上游产物 → 参数映射

| 上游（SFT 阶段已产出） | 传给 `rl.py` 的参数 |
|---|---|
| `data/Amazon23/<域>/sft/train/<域>_5_train.csv` | `--train_file`（`SidDataset` + `RLSeqTitle2SidDataset` 共用） |
| `data/Amazon23/<域>/sft/valid/<域>_5_valid.csv` | `--eval_file` |
| `data/Amazon23/<域>/sft/index/<域>.item.json` | `--item_meta_path` |
| `data/Amazon23/<域>/sft/index/<域>.index.json` | `--sid_index_path` |
| `data/Amazon23/<域>/sft/info/<域>.item_info.txt` | `--info_file`（**约束映射的唯一来源**，见 §3） |
| `outputs/<SFT_EXP_ID>/final_checkpoint/` | `--model_path`（**必须**含扩展 tokenizer + resize 过的 embedding） |

### 2.2 前置检查（`rl_run0.sh` 内，缺文件立刻停）

- 5 个上游文件存在性
- `--model_path/model.safetensors` 存在
- 🔴 `grep -q '<a_0>' <model_path>/tokenizer.json` —— 比"目录存在"强得多，
  能拦住"指回原始基座"这种**静默失败**

---

## 3. 约束解码映射 `[实测]`

RL 的约束生成**与 `evaluate.py` 同源**：两者都从 `info_file` 第 1 列（SID 串）建前缀映射。

**建表**（`minionerec_trainer.py:530-572`，ReReTrainer `__init__`）：

```
info_file 每行 -> split('\t')[0] + "\n" -> 前置**响应前缀**（pt.response_prefix()，默认 chatml = <|im_start|>assistant\n）
              -> tokenizer(...) -> prefixID
对每个 prefixID：append(eos)，然后 i 从 prefix_index=3 到末尾：
    key = hash(ID[:3])            (i==3)      -> value = ID[3]
    key = hash(ID[3:i])           (i>3)       -> value = ID[i]
```

**查表**（`LogitProcessor.py:51-54`）：

```python
if self.count == 0:  hash_key = sent[-self.prefix_index:]   # 只取 prompt 末 3 个 token
else:                hash_key = sent[-self.count:]          # 倒数 count 个
```

⟹ 长 prompt 不影响：**只要 prompt 末尾恰好是响应前缀**。默认 chatml 的
`<|im_start|>assistant\n` = `[151644, 77091, 198]`，alpaca 的 `### Response:\n` = `[14374, 5949, 510]`
—— **两者都恰好 3 个 token**，所以 `prefix_index=3` 两种格式都成立。
前缀一律从 `pt.response_prefix()` 取（`minionerec_trainer.py:538-539`），**不准手抄**。
而 `minionerec_trainer.py:675-678` 用的是
`maybe_apply_chat_template`（我们的输入是纯字符串、非 conversational ⟹ **不套 chat template**）
+ `add_special_tokens=False`，所以不会被追加尾 token。两条合起来前提成立。

### 3.1 探针实测（原 `scripts/sft/probe_rl_constraint_map.py`，**已于 2026-09-18 移除**）

⚠️ 下列数字是 **alpaca 口径**时的输出。换成 chatml 后前缀 token 变成 `[151644, 77091, 198]`，
但**结构与候选数完全不变** —— 已由 `probe_constrained_decoding.py` 在 chatml 下复验（`[256,98,1,1,1]`）。

```
1. prefix_index = 3 的前提
   add_special_tokens=True  -> [14374, 5949, 510]  len=3  ['###', ' Response', ':\n']
   add_special_tokens=False -> [14374, 5949, 510]  len=3
   [PASS] 两种调用都 = 3 token 且相同

2. 复刻建表（25,847 条）
   tokenize 后长度分布 = {7: 25847}      # 3 前缀 + 3 SID + 1 换行
   SID 段长度分布      = {3: 25847}
   首段 != '### Response:\n' 的条目数 = 0

3. 映射形状
   深度 1 [1 个 SID (a)]      键数=   256    (a 层每个码 -> 其后可能的 b)
   深度 2 [2 个 SID (a,b)]    键数= 18653
   深度 3 [3 SID 或 prompt 末3] 键数= 24767   (其中 1 个是 prompt 前缀，候选 256)
   深度 4 [3 SID + 换行]      键数= 24766

4. 模拟解码（按查表规则走 5 步）
   step 0: 候选 256 -> <a_0>     step 3: 候选 1 -> '\n'(198)
   step 1: 候选  67 -> <b_14>    step 4: 候选 1 -> EOS(151645)
   step 2: 候选   1 -> <c_199>
```

**结构断言（4 条全 PASS）**：

| 断言 | 结果 |
|---|---|
| prompt 末 3 token → 候选数 == a 层码数（256） | ✅ |
| a 层键数 == a 层码数（256） | ✅ |
| 3-SID 键（24,766 个）候选恒为 `[换行 198]` | ✅ |
| 3SID+换行 键（24,766 个）候选恒为 `[EOS]` | ✅ |

⟹ 与 `SFT_PIPELINE §3.4` 里 `evaluate.py` 的 Trie（5 步 `[256, 98, 1, 1, 1]`）**结构同构**。
⚠️ `prefix_index=3` 是 `minionerec_trainer.py` 与 `LogitProcessor.py` **两处硬编码**，
换基座 / 换 tokenizer 必须重测「响应前缀是否恰好 3 token」——
跑 `scripts/sft/probe_constrained_decoding.py` 的 C 段（原 `probe_rl_constraint_map.py` 已删）。

---

## 4. 奖励函数现状（`rl.py:207-305`）

| `--reward_type` | 组成 | 额外 artifact | 本仓可用？ |
|---|---|---|---|
| `rule` | `rule_reward`（二值：完全匹配=1.0） | 无 | ✅ **R0 锚点默认** |
| `ranking` | `[rule_reward, ndcg_rule_reward]` | 无 | ✅（原版 shell 脚本用这个） |
| `ranking_only` | `ndcg_rule_reward` | 无 | ✅ |
| `semantic` | 目标/预测 item 的 embedding 余弦相似度 | `--ada_path`（item embedding 的 **pickle**） | ❌ **缺文件**，`rl.py` 已加护栏报错 |
| `sasrec` | SASRec 对该 item 的打分 | `--cf_path`（**根 `sasrec.py`** 的 `SASRec` state_dict） | ❌ **缺文件**（仓里只有 RQ-VAE 量化器的 `.pth`，不是这个） |

> ⚠️ `rule_reward` 是**二值奖励**——UPGRADE_PLAN §P3 记录的"奖励稀疏、大量 group 全 0 无梯度"
> 就是指它。`ranking` 把连续 ndcg 信号加进来，是原版对稀疏性的缓解。
> 分层奖励设计（R1）见 UPGRADE_PLAN §6.1（**尚未实施**）。
>
> ✅ 本轮修掉一处会误导的日志：`rl.py` 原先**无条件**打印
> `"Load item_ada_embd successfully."`（非 semantic 奖励时也打印，且变量未定义）。
> 现改为只在 semantic 分支内打印，并带上 shape。

---

## 5. 训练内指标 & 日志

RL **不产出 `results/` 文件**，指标走 `ReReTrainer` 的 metrics 通道
（`minionerec_trainer.py:998-1001` 写 `HR@k` / `NDCG@k`，`:1081-1094` 的 `log()` 求均值后合并进
stdout）：

```
HR@3 / HR@5 / HR@10 / HR@20
NDCG@3 / NDCG@5 / NDCG@10 / NDCG@20
reward / reward_std / categorical_diversity / token_diversity
```

查看方式：

```bash
grep -E 'HR@|NDCG@|reward' logs/rl/IandS-rl0/rl.log | tail -30
```

> 评测口径 = **训练内 beam 搜索**（`test_beam`，默认 10），与 `EVAL_PROTOCOL` 的
> 全库排序口径**不是一回事**，不可直接和 `baseline/RESULTS.md` 数字横比。
> 正式对齐仍需训完后用 `evaluate_run0.sh` 跑一遍（`--base_model` 指 `final_checkpoint`）。

`[实测]` `:1090` 有 transformers 版本分支：`>= 4.47.0.dev0` 走 `super().log(logs, start_time)`
——本仓 4.57.1 走这条 ✅。

---

## 6. 成本

### 6.1 显存：**参考模型不占额外权重**（PEFT 下）—— 实测

⚠️ 先纠正一个常见误解：**"GRPO 要额外一份参考模型、所以显存翻倍"——在 LoRA/PEFT 下不成立。**

```
minionerec_trainer.py:287-293            参考模型怎么来的
    if   is_deepspeed_zero3_enabled():  ref_model = from_pretrained(...)          # 只有 ZeRO-3 另载
    elif not is_peft_model(model):      ref_model = create_reference_model(model) # 全参：真多一份
    else:                               ref_model = None                          # PEFT：**不加载**

minionerec_trainer.py:898-906            参考 logps 怎么算
    if self.ref_model is not None:  ref = self._get_per_token_logps(self.ref_model, ...)
    else:  with unwrap_model(self.model).disable_adapter():                       # 复用同一份权重
               ref = self._get_per_token_logps(self.model, ...)
```

⟹ PEFT 下参考 logps 靠 **`disable_adapter()` + 同一份权重**算出来，代价只是**多一次前向**
（[实测] 16 条序列时 +约 385 MiB，且在 `torch.inference_mode()` 下、不留计算图）。

真正的显存构成是四块（[实测] 3050Ti **4.00 GiB**，prompt 177 + completion 16）：

| 块 | 量 | 说明 |
|---|---|---|
| ① 基座权重 bf16 | **1136.9 MiB** | 0.6B × 2 B |
| ② 可训参数 | **9.2 MiB**（无 `modules_to_save`）／ **595.5 MiB**（带） | LoRA 只挂 attention = 9,175,040 个（fp32） |
| ③ 梯度 + 优化器状态 | **105 MiB** ／ **1893 MiB** | ⚠️ **AdamW 的状态 dtype 跟随参数 dtype**：bf16 参数只吃 4 B/param，不是常见说法的 8 B/param |
| ④ 生成 KV + logits + 激活 | 与 `B × num_generations` 成正比 | 见 §6.2 |

> `[实测]` 词表参数翻倍这件事值得单记：带 `modules_to_save` 会**破坏 `tie_word_embeddings`** ——
> resize 后 `lm_head.weight is embed_tokens.weight` 为 **True**，套上 LoRA 后变成**两份独立张量**，
> 词表参数直接翻倍（312,190,976 元素 = 2 × 152,437 × 1024）。

### 6.2 `gradient_checkpointing` 才是本地那把钥匙（含我自己的两次测量翻车）

`[实测]`（3050Ti 4.00 GiB / prompt 177 + completion 16 / **全流程峰值**，含优化器 step 的瞬时峰值）：

| 序列数 `B×G` | `grad_ckpt=off` | `grad_ckpt=on` |
|---:|---:|---:|
| 4 | 3.25 GiB | **2.59 GiB** |
| 8 | 5.28 GiB | — |
| **16（`rl_run0.sh` 默认 4×4）** | **9.33 GiB** ✗ | **2.59 GiB** ✓ |
| 32 | — | 2.59 GiB ✓ |
| 64 | — | 3.74 GiB ✓ |
| 128 | — | 6.34 GiB ✗ |
| 512 | —（光生成阶段就 8.55 GiB） | ✗ |

🔴 **踩坑 1（我自己的测量错误）**：第一轮我报"梯度检查点几乎无效（9.33 → 9.03）"——**那是空操作**。
`from_pretrained` 默认把模型置于 **eval 模式**，而 HF 的检查点只在 `self.training=True` 时生效。
显式 `model.train()` 之后，效果是 **16 条序列 9.33 → 2.59 GiB（5.2×）**。

🔴 **踩坑 2（会静默失败，后果更严重）**：LoRA + 检查点**必须**调 `enable_input_require_grads()`。
PyTorch 的 checkpoint 要求该段计算的输入 `requires_grad=True`；LoRA 冻结了 base、`embed_tokens`
也不可训 ⟹ 第一层检查点的输入不 require grad，PyTorch 只打一条 warning 然后**把梯度置 None**——
显存看着很美，其实模型没在学。该 API 在 `transformers/modeling_utils.py:2849`，
而 **Trainer / TRL / ReReTrainer 三方都没有自动调用**（已 grep 确认）⟹ 必须由调用方补。
本仓已在 `rl.py` 补上，并加了"可训张量拿到梯度 / 非全零"的覆盖率打印。

### 6.3 `modules_to_save` 在 RL 侧应当**留空**（与 SFT 相反）

| | SFT | RL |
|---|---|---|
| 要不要训 `embed_tokens`/`lm_head` | **必须**（SID 是刚 `add_tokens` 的新 token，不训就永远随机初始化） | **不必**（SFT 产物里 SID 的 embedding 已经训好） |
| 可训参数 | 321.4M | **9.2M** |
| `[实测]` 峰值（16 序列 + grad_ckpt） | **4.20 GiB** ✗ 超物理 | **2.59 GiB** ✓ |

### 6.4 顺带修掉一处隐性浪费：`llm_model` 重复加载

原版 `rl.py` **无条件**加载一份 `llm_model`，而它只被 `reward_type == "semantic"` 用到
（`item_ada_embd.to(llm_model.device)`）—— rule/ranking 奖励下它是**纯重复**：训练用的模型
由 `ReReTrainer` 按 `model_init_kwargs` **再加载一份**。0.6B bf16 = **1.14 GiB** 白占，
而这 1.14 GiB 往往正是"本地能不能跑"的分界线。已改为按需加载。

### 6.5 本地配方（3050Ti 4 GB，实测可跑）

```bash
MODEL_PATH=outputs/IandS-run0/final_checkpoint \
TRAIN_BATCH_SIZE=4 NUM_GENERATIONS=4 \
USE_LORA=True GRAD_CKPT=True \
LORA_MODULES_TO_SAVE="" \
MAX_STEPS=2 TEST_DURING_TRAINING=False BEAM_SEARCH=False \
SAVE_STEPS=999 EVAL_STEP=999 \
bash rl_run0.sh
```

⚠️ **能跑 ≠ 能得结论**：`B×G = 4` 只有**1 个 prompt 的组内比较**，GRPO 的 advantage 方差估计
在这个规模上极不稳定。本地定位 = **冒烟 + 调试**；正式训练仍上云（`rl_run0.sh` 默认 4 prompt ×
4 gen × grad_acc 8）。

`[实测] 端到端（伪 SFT 产物 = 基座 + resize 过的 embedding，`exit=0`）`：

```
[mem] reward_type=rule 不需要额外模型副本，跳过 llm_model 加载（省约 1.14 GiB）
[LoRA] 可训参数 9,175,040 / 605,737,984 = 1.51%
[LoRA] enable_input_require_grads() 已调用（gradient checkpointing 下保梯度）
2/2 [00:16, 8.45s/it]  completion_length 5.0  categorical_diversity 1.0
      <- 约束解码在真实 trainer 里精确产出 [3 SID + \n + EOS]
[LoRA] adapter merged -> final_checkpoint/ 落完整权重
```

- `nvidia-smi` 1s 采样（含桌面基线 183 MiB）：峰值 **3048 MiB ⟹ 训练进程约 2.80 GiB**，
  与探针的 2655 MiB 一致（差额是 CUDA context / 碎片）。
- 单步 **8.45 s**（16 条序列，3050Ti）。

⚠️ **`TRAIN_BATCH_SIZE` 必须能被 `NUM_GENERATIONS` 整除**：本地想省显存设成 `1 × 4` 会直接报
`The global train batch size (1 x 1) must be evenly divisible by the number of generations per prompt (4)`。
这不是显存问题，是 GRPO 的结构约束（每个 prompt 要复制成 G 条做组内比较）。
合法的小配置是 `4 × 4`（16 条，实测 2.59 GiB）或 `4 × 2`（8 条）。`rl_run0.sh` 已加前置护栏。

### 6.6 `reward=0` 是**预期**的，不能当"跑通了"的证据（`add_gt` 诊断）

在**未训练的基座**上跑 rule 奖励，`reward` 几乎必然恒为 0：rule 奖励 = "生成的 SID 与目标 SID
**完全相等**"，而随机初始化的模型在 25,847 个候选里命中那一个的概率约 **4e-5**。
后果是 **advantage 全 0 ⟹ `loss`≈0、`grad_norm`=0** —— 此时**看不出 LoRA 是否真的在学**
（哪怕梯度全是 None，输出也是这个样子）。

所以本地要验证梯度链，必须开 `ADD_GT=True`（`minionerec_trainer.py:856-873`：把每组里的一条
候选**替换成 ground truth**）。`[实测]` 对照：

| 配置 | reward | reward_std | grad_norm | 能说明什么 |
|---|---:|---:|---:|---|
| `ADD_GT=False`（正式口径，与 `rl.sh` / `rl_3090.sh` 相同） | 0.0 | 0.0 | **0.0** | 只说明链路没崩，**不能**证明在学 |
| `ADD_GT=True`（**仅诊断**，会扰动训练语义） | 0.25 | 0.5 | **22.69 / 18.58** | ✅ LoRA 梯度确实回流到了参数 |

还有一条很漂亮的自洽证据：两步的 **`kl` 都是 0.0** —— 因为 LoRA 的 `lora_B` 初始化为 0，
step 0 时 adapter 输出恒为 0 ⟹ 策略与参考模型**逐位相同**，KL 必然为 0。
（这条同时说明 `disable_adapter()` 那条参考路径是通的。）

### 6.7 云端（3090）推算

| 项 | 量级 | 依据 |
|---|---|---|
| 显存（全参 GRPO） | 权重 1.2G + grads 1.2G + `paged_adamw_32bit` 状态 ~4.8G + 激活/KV ≈ **8–12 GB** | ⚠️ **未实测**；24G 单卡应可容纳 |
| ⚠️ 若换 V100 | 必须 `PRECISION=fp16`，且**显存约翻倍**（AMP 要 fp32 主权重）；16 GB 版跑不动，32 GB 版可以 | 见 [`PRECISION_GUIDE.md`](PRECISION_GUIDE.md) §5 |
| 单个 checkpoint | ≈ **6 GB**（权重 + fp32 优化器状态） | `[设计推算]` |
| checkpoint 总量 | `save_total_limit=3` → ~18 GB，加 root + `final_checkpoint` 两份 ≈ **21 GB** | 原版 `save_total_limit=20` 会到 ~120 GB，**本仓已下调** |

---

## 7. 本项目对 MiniOneRec 原版的改动清单

| 文件 | 位置 | 改了什么 | 为什么 |
|---|---|---|---|
| `rl.py` | `:73` | 新增 `--torch_compile`（默认 **False**） | 原版硬编码 `True`；GRPO 每步生成长度不定 + 动态 padding 会反复重编译（SFT 侧已实测这个坑） |
| `rl.py` | `:77-78` | 新增 `--save_steps` / `--save_total_limit`（默认 0.1 / **3**） | 原版 20 会占 ~120 GB 磁盘 |
| `rl.py` | `:80` | 新增 `--optim`（默认 `paged_adamw_32bit`） | 参数化，便于本地/云端切换 |
| `rl.py` | `:91-113` | **新增前置护栏**：`<a_0>` 必须 1 token、**响应前缀必须 3 token**（取自 `pt.response_prefix()`，默认 chatml） | 指回原始基座会静默崩（SID 碎裂 + 约束映射全废）；换 tokenizer 会让 `prefix_index=3` 失效 |
| `rl.py` | `:185-201` | 奖励 artifact 缺失时**报错**；修掉无条件误导日志 | 原版会抛 `KeyError`/`NameError`，看不出根因 |
| `rl_run0.sh` | 新增 | RL 入口：EXP_ID 命名 / 前置检查 / venv 探测 / 写 `run.meta.json` | 对齐 `sft_run0.sh`；原两个 `.sh` 路径不可用 |
| `rl.py` | `--use_lora` 等 6 个参数 | **新增 LoRA**（本文件原本完全不支持）：`--use_lora/--lora_r/--lora_alpha/--lora_dropout/--lora_targets/--lora_modules_to_save`。传 `peft_config` 给 `ReReTrainer`（不预 wrap），由它 `get_peft_model` 并把 `ref_model` 置 None | 本地 4GB 唯一可行路径；UPGRADE_PLAN §5.2 的双轨 |
| `rl.py` | `--grad_ckpt` / `--max_steps` | 梯度检查点（**默认 False**，`rl_run0.sh` 里默认 True）；训练步上限（本地冒烟用） | 16 序列 9.33 → 2.59 GiB |
| `rl.py` | `trainer.model.enable_input_require_grads()` | LoRA + 检查点时**必须补**，否则梯度静默为 None | Trainer/TRL/ReReTrainer 三方都不自动调（见 §6.2 踩坑 2） |
| `rl.py` | `llm_model` 改为按需加载 | 只 `reward_type=="semantic"` 才加载 | 原版无条件加载，rule 奖励下白占 1.14 GiB（§6.4） |
| `rl.py` | 保存段 | LoRA 下先 `merge_and_unload()` 再落 `final_checkpoint/` | 否则 `evaluate.py` 的 `from_pretrained` 加载不了 |
| `rl_run0.sh` | 前置检查 | 新增 `TRAIN_BATCH_SIZE % NUM_GENERATIONS == 0` 护栏 | [实测] 本地 `1×4` 直接报错，报错文本不提示是结构约束 |
| `rl_run0.sh` | 新增 `ADD_GT` / `GRAD_CKPT` / `MAX_STEPS` / `USE_LORA` 等环境变量 | 正式默认值与原版一致（`ADD_GT=False`） | 本地冒烟与梯度链诊断需要它们（§6.5/§6.6） |

**刻意没改的**：
- `category_dict`（`rl.py:120`）仍是 5 个硬编码键、**无 `Video_Games`** —— 与 `sft.py` 保持同口径，
  上 VG 时两处要一起加（`SFT_PIPELINE §3.2` 已记同样的一条）。
- `RLSeqTitle2SidDataset` 的 `sample=10000` 硬编码。
- 奖励函数本身（分层设计属 R1，见 UPGRADE_PLAN §6.1）。

---

## 8. 待办 / 阻塞

| # | 项 | 状态 |
|---|---|---|
| 1 | **SFT Run-0** 产出 `outputs/IandS-run0/final_checkpoint` | ⏸ **硬阻塞**（RL 无从接着训） |
| 2 | 3090 上实测**全参** RL 显存与单 checkpoint 体积（§6.6 仍是推算） | ⏸ |
| 2b | 本地 LoRA 冒烟 | ✅ `[实测]` 见 §6.5：`exit=0`，峰值 2.80 GiB、单步 8.45 s；梯度链由 §6.6 的 `ADD_GT` 诊断验穿 |
| 2c | 用**真实 SFT 产物**（而非伪产物）跑一次本地冒烟 | ⏸ 等 `outputs/IandS-run0/final_checkpoint` |
| 3 | `semantic` 奖励的 `ada_path`（item embedding pickle） | ⏸ 缺 |
| 4 | `sasrec` 奖励的 `cf_path`（根 `sasrec.py` 的权重） | ⏸ 缺 |
| 5 | 分层奖励（R1）/ OPD 蒸馏（R2/R3） | ⏸ 见 UPGRADE_PLAN §6 |
| 6 | RL 侧的 EXP_ID 与评估打通（训完用 `evaluate_run0.sh` 对齐全库口径） | ⏸ |
