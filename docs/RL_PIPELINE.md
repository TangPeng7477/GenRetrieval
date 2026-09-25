# RL Pipeline（GRPO）— 唯一入口

> 2026-09-17 ｜ 本文件是 **RL 阶段的唯一入口**：上游依赖 / 参数映射 / 实测 / 改动清单都在这里。
> SFT 侧见 [`docs/SFT_PIPELINE.md`](SFT_PIPELINE.md)；指标口径见 [`docs/EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md)。
> 升级计划与消融矩阵（R0~R3）见 [`docs/UPGRADE_PLAN.md`](UPGRADE_PLAN.md) §6。

## 0. TL;DR

| 项 | 结论 |
|---|---|
| **入口** | **`bash rl_run0.sh`**（根目录）。⚠️ `rl.sh` 是 MiniOneRec **原版**，路径写死 `./data/Amazon/...`（本仓不存在），**不要直接用**（原 `rl_3090.sh` 已删） |
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
>
> ⚠️ `RLTitle2SidDataset` 一个类实际产出**两路**（`data.py:836-850`，`'task': 'title2sid'`
> 与 `'task': 'description2sid'` 两个循环）。所以"三个 Dataset 类"≠"三个任务"，实际是 **4 路**。

### 1.2 为什么 RL 的三路和 SFT 的四路**不是同一套**（结构原因，不是随意选的）

🔴 **RL 三路的 target 全部是 SID** —— 这不是巧合，是 `rule` 奖励的硬约束。

`rule` 奖励的定义是「生成出的 SID 与目标 SID **完全相等**」（`rl.py:207-305`）。
所以凡是**目标为 title** 的任务（SFT 的 T2a `sid2title`、T3 `seq2title`）在 RL 里**拿不到规则奖励**
（要判定得用 `semantic` 奖励，而它缺 `--ada_path`，见 §8 待办 #3）。
于是 MiniOneRec 把它们**换成反方向的"→SID"**：

| SFT 任务 | 目标 | RL 里为什么不用 / 换成什么 |
|---|---|---|
| T1 `seq2sid` | SID | ✅ 原样保留（`SidDataset`，主任务） |
| T2b `title2sid` | SID | ✅ 原样保留（`RLTitle2SidDataset` 的 title 分支） |
| T2a `sid2title` | **title** | ❌ 规则奖励判不了 ⟹ RL 侧对应类是 `RLSid2TitleDataset`，**已被注释掉**（`rl.py:192`） |
| T3 `seq2title` | **title** | ❌ 同上 ⟹ 换成**反方向** `seqtitle2sid`（历史**标题**序列 → SID） |

**组织方式与 SFT 完全一致**：`ConcatDataset` 拼起来 + `shuffle(seed)`，单阶段混洗、
无任务权重、无 task tag（`rl.py:200-205`）。这一点和 SFT §6.7 的定版配方是同构的。

**验证集只用一个 `SidDataset`**（`rl.py:202`）⟹ 与 **SFT 的验证集完全同源、恒为 T1**
（对应 `SFT_PIPELINE` 里「验证集恒为 T1」那条，`sft.py:471`）。
⟹ 训练内 `HR@k` 是**纯 T1 口径**的轨迹，只看相对变化，别拿去和别的口径横比。

⚠️ **两路 prompt 格式是 SFT 从没训过的**（RL 会自己适应，但早期步会很噪）：

| 路 | SFT 有没有训过 | 说明 |
|---|---|---|
| `seqtitle2sid` | ❌ **没有** | SFT 有 `seq2title`（历史 SID → title），方向相反、格式不同 |
| `description2sid` | ❌ **没有** | 走 `text2sid` 模板但只填 description；SFT 的 T4 是 title+brand+categories+features，且**没有 Dataset 类**（`SFT_PIPELINE §7`） |

这两路合计约占总步数的 **~22%**（50k + 10k / 269k）。打通流程阶段建议**保留默认**
（SFT 的教训正是"辅助任务的语义锚不能砍"），但要知道**早期 RL 的一部分"学习"其实是在适应新格式**，
不是 T1 变好。若想拿更干净的信号，可注释掉 `train_data3`（`rl.py:190`）或 `description2sid` 循环。

---

## 2. 入口：`rl_run0.sh`

```bash
# ✅ 当前唯一有合格起点的用法：SFT 定版产物 IandS-all（2026-09-25）
SFT_EXP_ID=IandS-all bash rl_run0.sh
#   = MODEL_PATH=outputs/IandS-all/final_checkpoint

# 默认值（IandS-run0）现在**没有对应产物**，直接用会撞"上游产物不齐"
bash rl_run0.sh          # ⚠️ 会 exit 1

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
MODEL_PATH=outputs/IandS-all/final_checkpoint \
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
| `ADD_GT=False`（正式口径，与 V0 的 `rl.sh` 相同） | 0.0 | 0.0 | **0.0** | 只说明链路没崩，**不能**证明在学 |
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

### 6.8 `[实测+推算]` 起点换成 `IandS-all` 之后：RL 第一次能真学到东西

> 2026-09-25：SFT 定版产物 `outputs/IandS-all/final_checkpoint` 就位（`HR@10 = 0.0356`）。
> 这让 §6.6 那条「reward 恒 0 是预期的」**不再无条件成立**。

**① `ADD_GT` 不再是必需品**：rule 奖励 =「生成 SID 与目标 SID 完全相等」，
未训练基座命中率 ~4e-5 ⟹ 恒 0；而 `IandS-all` 的 `HR@1 = 0.0067` ⟹ **正式配置 `ADD_GT=False`
也应该能看到 `reward > 0` / `reward_std > 0` / `grad_norm > 0`**。
🔴 这是本轮最强的判据：**不需要**用 `ADD_GT=True` 扰动训练语义去验梯度链了。

**② 但稀疏度是真瓶颈**（别指望一步到位）。⚠️ 先纠正一个我一度写错的机制：

🔴 **`BEAM_SEARCH=True` 不等于"确定性 beam search"**。`minionerec_trainer.py:480-495` 里它是：

```python
if self.beam_search:
    GenerationConfig(num_beams=self.num_generations,        # 4
                     num_return_sequences=self.num_generations,
                     top_k=None, top_p=None, temperature=self.temperature,  # 1.0
                     do_sample=True)                        # 🔴
```

`num_beams>1` **且** `do_sample=True` ⟹ HF 判定的模式是 **`BEAM_SAMPLE`（束采样，随机）**，
**不是 `BEAM_SEARCH`**（`configuration_utils.py:382`；同 §`SFT_PIPELINE` 里那条 HF 模式判定红线）。
⟹ `BEAM_SEARCH=True/False` 两者的实际差别**很小**（都是采样，只是一个在 4 条子里挑）。

| 配置 | 实际 HF 模式 | 单条候选命中率 | 组（G=4）至少一个命中 |
|---|---|---:|---:|
| `BEAM_SEARCH=True`（默认） | **BEAM_SAMPLE**（随机） | **0.52%** `[实测]` | **2.1%** `[实测]` |
| `BEAM_SEARCH=False` | SAMPLE（纯采样） | ≈ `HR@1` = 0.67% | ≈ 2.7% |

实测口径：`IandS-rlfix` 30 步 × 128 条候选 = **3,840 条命中 20 条**（0.52%/条）；
30 步 × 32 组 = **960 组命中 20 组**（2.1%/组）；**有信号的优化步 = 5/30 ≈ 17%**。
⟹ 与 `HR@1`/`HR@4` 的预测同量级 ✓（**奖励稀疏是真的，但比"97~98% 全 0"稍好**）。

✅ **命中率有实测了，且与"窗口分母"无关**（`E[reward]` 直接给出，推导见 §6.10 ⑻）：
**每候选 0.90%**（`rl300`，150 步）/ **0.52%**（`rlfix`，30 步）；**每组（= 每 prompt）3.58% / 2.08%**
—— 与 `HR@1` = 0.67% 及 `1−(1−HR@1)⁴` = 2.65% 吻合 ✓。**有信号的优化步 = 24% / 17%。**

**②b `reward_type` 的四个取值 —— 以及为什么换 `ranking` 救不了稀疏**

`rl.py:368-377`：`rule` → `rule_reward`（单项）；**`ranking` → `[rule_reward, ndcg_rule_reward]`（两项）**；
`ranking_only` → 只 ndcg；`semantic` / `sasrec` 因缺文件不可用。入口开关 = `REWARD_TYPE`（`rl_run0.sh:74`，默认 `rule`）。

🔴 **本项目各 run 用的都是默认的 `rule`（单项）**，由日志直接证明：只有一个 `rewards/*` 键
（`rewards/rule_reward`），且 **`reward` 与该键逐行完全相同**；若两项生效，`reward = Σ wᵢ·rᵢ` 会与 rule 单独不等，
并会多出 `rewards/ndcg_rule_reward` 键。（顺带：`reward` 的取值落在干净的 `k/32` 网格上，也只有在单项 rule 下才可能。）

🔴 **换成 `ranking` 不会提高"有信号步"的比例**：`ndcg_rule_reward`（`rl.py:279-303`）被 `flag` 门控 ——
**组内一个都没命中时整组强制为 0**；只有已命中的组才拿到 `lis`，其中**命中的那条给 `0.0`、
其余按位置给负值**（`ndcg_rewards = [-1/log2(i+2)]` 归一化到和 = −1，`rl.py:275-276`，全为负）。
⟹ 它只改变**命中组内部的相对奖励**，对那 76% 零 advantage 的窗口毫无作用。

**它具体改了什么**（`ndcg_rewards`（G=4）= `[-0.3904, -0.2463, -0.1952, -0.1681]`；按 `reward_weights` 各 1 计；
优势 = GRPO 的组内标准化且 `torch.std` 默认无偏）：

| 命中位置 | `rule` only（现状） | `rule` + `ndcg` | 优势对比 |
|---|---|---|---|
| 位置 0（最高分） | `adv = [1.50, -0.50, -0.50, -0.50]` | `adv = [**1.495**, -0.604, -0.479, -0.412]` | 命中者几乎不变（−0.005） |
| 位置 2 | 同上（与位置无关） | `adv = [-0.717, -0.452, **1.478**, -0.309]` | 同上 |

🔑 **净效果 = 命中者优势不变（≈+1.5），但"错误候选"的负优势被按位置分档 —— 位置越靠前罚得越重。**
而束采样的返回顺序 ≈ 按累计分数排 ⟹ 语义正是「**模型最自信的那些错误答案，被罚得最狠**」，
方向与排序目标一致。**代价只是多算一个标量函数，所以值得作为消融项试一次**（一个环境变量）：

```bash
REWARD_TYPE=ranking ... bash rl_run0.sh     # 与 rule 同一 RUN_TAG 体系外的 tag，避免覆盖
```

⚠️ 但它**不改变 76% 零 advantage 窗口** ⟹ 这是**二阶改进**，救不了稀疏；且它让 reward 不再落在
干净的 `k/32` 网格上（可作为"是否真的启用了两项"的旁证）。

✅ **好消息**：束采样是**随机**的 ⟹ **探索是有的** —— 我早先担心的
"确定性 beam 会让目标不在 top-4 时永远没有探索机会" **不成立**（那是我按 `BEAM_SEARCH` 的字面意思推的，错）。
真实增益仍然要靠堆量：`MAX_STEPS`、`NUM_GENERATIONS`↑、或调 `TEMPERATURE`（现为 1.0，分布很散）。

**③ 成本（务必先限步，别直接全量）** `[实测，2026-09-25]`：

```
训练数据 = 270,432（208,999 SidDataset + 51,433 RLTitle2Sid + 10,000 RLSeqTitle2Sid）
步数/epoch = 270,432 / (TRAIN_BATCH_SIZE 4 × GRAD_ACC_STEPS 8 = 32) = 8,451
步速       = 4.9 s/step（4090D，全参 GRPO + grad_ckpt，两次 run 互校）
⟹ 1 epoch ≈ 11.5 h，2 epoch ≈ 23.6 h
```

🔴 **`TEST_DURING_TRAINING` 的内部 eval 是最大隐藏开销**：`EVAL_STEP=0.0999` ⟹ 约 10 次全量验证集
（50,984 条）beam=10 评估。`[实测]` **单次 ≈ 6.0 h**（进度条 `683/50984 [04:40<5:56:55]`，
`EVAL_BATCH_SIZE=4` + `max_new_tokens=16` + 每步 LogitProcessor，比 `evaluate.py` 贵得多）
—— 我原先估的 30 min **差了 12 倍**。⟹ **短跑/长跑都建议 `TEST_DURING_TRAINING=False` + `EVAL_STEP=999`**。

### 6.9 推荐跑法：四阶段（先小规模迭代，再通链路，最后要数字）

> `EXP_ID = <域>-<RUN_TAG>`（`rl_run0.sh:37`），所以换 `RUN_TAG` 就能隔离产物。

**阶段 0 · 小规模快迭代（~15–40 min，判"RL 到底有没有在推动 HR"）**

🔑 **不需要改代码 —— `MAX_STEPS` 本身就是"子集"**：`rl.py:210` 的 `shuffle(seed=seed)`
（`seed=42`，`set_seed(42)`）是**固定排列**，且 `--seed` 也透传给了 HF（决定 `RandomSampler` 的每轮顺序）
⟹ `MAX_STEPS=N` 恰好等于"只用这个固定排列的前 `N × 32` 个 prompt"，LR 的 cosine 也在这 N 步内退火完。
**统计上等价于"从全量里随机抽 32N 条、跑 1 个 epoch"**（固定种子，可跨 run 复现）。

```bash
MODEL_PATH=outputs/IandS-all/final_checkpoint USE_LORA=False RUN_TAG=rlfast \
MAX_STEPS=470 NUM_TRAIN_EPOCHS=1 TEST_DURING_TRAINING=False EVAL_STEP=999 SAVE_STEPS=999 \
bash rl_run0.sh
```

| 相当于多少条 | `MAX_STEPS` | 训练时长（4.9 s/步） | 其中有梯度的步（24%） |
|---:|---:|---|---:|
| 5,000 | 157 | ~13 min | ~38 |
| 15,000 | 470 | ~38 min | ~113 |
| 30,000 | 940 | ~1.3 h | ~226 |

⚠️ **`sample` 是硬编码的，没有 CLI 开关**：`rl.py:187` 的 `sample = -1`（全量）、
`:195` 的 `sample=10000`（T3 上限）。`sample_train`（`:211`）**不是子采样** —— 它只在
`model_path` 含 `"sft"` 时丢掉前 20%。**只有要"N 条 × E epochs"那种语义才需要改代码**；本阶段不需要。

评估侧本来就是现成的：`evaluate_run0.sh` 的 `MAX_SAMPLES` 走 `EvalSidDataset(sample=N)` +
`random.seed(42)` ⟹ **每次抽同一批，跨 run 可比**。

| `MAX_SAMPLES` | 时长（`BATCH_SIZE=12`） | `HR@10 ≈ 0.036` 时的相对误差 |
|---:|---|---:|
| 1,000 | ~3 min | ±16% |
| 2,000 | ~6 min | ±12% |
| 5,000 | ~16 min | **±7%** |
| 50,982 | ~2.6 h | ±0.8% |

🔴 **锚点必须与迭代 run 同 `MAX_SAMPLES`（否则数字不可比）—— 这是"训练/评估要不要同一批子集"的真正落点。**

先分清两个"子集"，它们**按设计就不该重合**：

| | 来源 | 内容 |
|---|---|---|
| **训练子集** | `IandS_5_train.csv`（LOO 的 **train** split） | 每用户**早于倒数第二条**的历史窗口（≈4.1 行/用户） |
| **评估子集** | `IandS_5_test.csv`（LOO 的 **test** split） | 每用户**最后一条**作为目标（1 行/用户） |

⟹ 在 LOO 下"训练与评估用同一批行"**不可能**，也不需要。真正必须恒定的是**跨 run 的评估集** ——
而这一点现在**有坑**：

| `IandS-all` 的现有锚点 | `HR@10` | 怎么来的 |
|---|---:|---|
| 全量 `n=50,982` | **0.0356** | 默认 `TEST_FILE` |
| **同一批 5,000 用户的 test** | **0.0342** | ✅ **快迭代统一用这个**（`TEST_FILE=…u5k.csv` + `MAX_SAMPLES=0`） |
| 随机 `n=100`（`MAX_SAMPLES=100`，在**全量** test 上抽） | **0.0200** | 比全量低 44% |

⟹ `n=100` 与全量差 44%，是"不同 N 不可比"的实证；而**同一批用户的 5,000 条与全量只差 3.9%
（0.5σ，统计上不可区分）** ⟹ 它是全量锚点的**忠实代理**（详细对照见下方「已实现」）。

🔴 **两个"5,000"不要混用**：
- `MAX_SAMPLES=5000`（在**全量** test 上**随机抽** 5,000 行）—— 另一种口径，**不等于**上面的锚点；
- 快迭代统一走 **"同一批 5,000 用户的子集 `TEST_FILE` + `MAX_SAMPLES=0`"**（见下方「已实现」），
  好处是**训练用户 ≡ 评估用户**。

此后每个 fast run 都用**同一个** `TEST_FILE`（+ `MAX_SAMPLES=0`）评 ⟹ 落在完全相同的 5,000 行上
（`random.seed(42)` 固定）⟹ 是**行级配对比较**，实际判别力优于表里的边际 ±7%。

⚠️ **"按用户对齐"现在已由子集方案覆盖**（T1/T3 用同一批用户的 `train`、评估用同一批用户的 `test`）
⟹ 原先"训练前缀覆盖 ≈3,700 用户 vs 评估 5,000 用户、交集仅 ~7%"这个 confound 不再存在。
代价：任务构成变成 **T1+T3（无 T2）**，外推需假设 T2 非关键变量。

### ✅ 已实现：「同一批用户」的小规模迭代（`T1+T3` 版，2026-09-25）

**为什么值得做**：全量 run 里训练用户 ⊇ 评估用户 = **100%**；而 `MAX_STEPS` 前缀方案只有
**约 7% 交集**（470 步 ≈3,700 用户 vs 评估 5,000 用户）⟹ 它不是全量 run 的忠实缩放版。

**做法**：按**同一批用户**切 train/test，并把按用户切不动的 `T2` 去掉。

| 事实（本机实测） | 值 |
|---|---|
| `train` / `test` CSV 是否带 `user_id` | **都带** ✓ |
| `train` 行/用户 | **4.10**（208,999 / 50,985；p50=2、p95=13、max=200） |
| `test` 行/用户 | **1.00**（50,982 / 50,982）⟹ 取 N 个用户 = N 条评估行 |
| `train ∩ test` 用户 | **50,982**（每个 test 用户都有 train 行）⟹ 抽样池充足 |

🔴 **为什么必须去掉 T2**：`T2 = RLTitle2SidDataset` 读 `item_info` / `sid_index`（**商品目录**），
**item-level、没有用户维度** ⟹ 按用户缩不动。若留着它（51,433 行，与用户无关），配比会被顶到
T2 占 **63%**、单次迭代 **3.5 h**。而 `T1`（`SidDataset`）与 `T3`（`RLSeqTitle2SidDataset`）
**都是从 train CSV 派生的** ⟹ 都能按用户切 ⟹ `RL_TASKS=T1,T3`。

**新增的三件工具**（默认路径全部**逐位不变**）：

| 文件 / 开关 | 作用 |
|---|---|
| `scripts/rl/make_user_subset.py` | 抽 N 个用户（种子固定）⟹ 产出 `train/*.<tag>.csv`、`test/*.<tag>.csv`、`info/*.user_subset.<tag>.json`（清单含 seed / 用户列表 / 行数 / 建议参数）。**流式 `csv` 读写、不整表进内存**（train CSV 222 MB），并保留字段原始文本（下游有 `eval(row['history_item_id'])`） |
| `RL_TASKS` / `RL_T2_SAMPLE` / `RL_T3_SAMPLE` | 任务子集 + 行数上限。`RL_TASKS` 带 **shell 级真值陷阱防护**（放错位置会被前置检查抢掉——已修正到配置区最前）。`rl.py` 侧同名参数，非法值直接 `raise` |
| `TRAIN_FILE` / `EVAL_FILE` / `SID_INDEX` / `ITEM_META` / `INFO_FILE` / `TEST_FILE` | **五条 RL 路径 + 评估的 `TEST_FILE` 改为 `${VAR:-default}`**（原来都是写死的普通赋值）⟹ 可指向子集 CSV |

`[实测]` 本机实跑 `--n-users 5000 --seed 42`：

```
train ∩ test 用户 = 50,982 ；seed=42 抽 5,000 个
  train/IandS_5_train.u5k.csv   208,999 -> 20,418 行 (21.2 MB)
  test /IandS_5_test.u5k.csv     50,982 ->  5,000 行 ( 5.5 MB)
  建议 RL_T3_SAMPLE = 977（= round(20,418 × 10,000 / 208,999)，保住全量的 T1:T3 配比）
  训练集 = 20,418(T1) + 977(T3) = 21,395 行 ⟹ 668 步/epoch ≈ 55 min @4.9 s/step
```

**完整性已判**：CRLF=**0**（全 LF）、列名与源一致、两文件 `user_id` 全部 ∈ S、
**`train 用户集合 == S == test 用户集合`**（这才是目标）、`history_item_id` / `history_item_sid` 仍可 `eval()`、
SID 串全部匹配 `<a_x><b_y><c_z>`、无 NaN。

**✅ 锚点已测（2026-09-25，`EXP_ID=IandS-all`，同一批 5,000 用户的 test，beam=50）**

| 口径 | HR@1 | HR@10 | HR@20 | HR@50 |
|---|---:|---:|---:|---:|
| 全量 `n=50,982` | 0.0067 | 0.0356 | 0.0541 | 0.0856 |
| **子集 `n=5,000`（新锚点）** | **0.0046** | **0.0342** | 0.0552 | 0.0862 |
| 相对差 | −31% | **−3.9%** | +2.0% | +0.7% |

`HR@10` 在 `n=5,000` 上的 1σ 抽样误差 ≈ ±0.0026 ⟹ 实测差 −0.0014 = **0.5σ** ⟹ **统计上不可区分**。
只有 `HR@1` 偏低（−0.0021 ≈ 2σ，对子集构成更敏感），且它不影响 `HR@10/20/50`。
⟹ **5,000 用户子集是全量锚点的忠实代理**，快迭代可放心用它比较。
耗时 19m32s（全量 158m47s，比值 8.1 ≈ 数据量比 10.2×）；峰值显存 15.56 GB（`BATCH_SIZE=12`）。

🔴 **踩过的坑：换了评估集但文件名没变 ⟹ 全量锚点被静默顶掉。**
`SAMPLE_TAG` 只在 `MAX_SAMPLES != 0` 时加后缀，而子集评估用的是 `MAX_SAMPLES=0`（"全部" = 子集的 5,000 行）
⟹ 与全量评估写入**同一个** `eval_IandS_beam50.json` ⟹ 汇总表里 `IandS-all n=50,982` 那行被 `n=5,000` 顶掉
（`0.0356` 现只剩在 `SFT_PIPELINE §6.7`、README 与 git 历史里；云端那份全量预测 json 已被覆盖）。
**已修**：`evaluate_run0.sh` 新增 `TEST_TAG` —— 非默认 `TEST_FILE` 自动加后缀
（`IandS_5_test.u5k.csv` -> `_u5k`，即 `eval_IandS_beam50_u5k.json`）；默认路径不加后缀。
六条判据已验证（不传/子集/子集+n100/显式默认/VG 默认/怪名字），**旧路径与旧行为逐位不变**。
（想把 `n=50,982` 那行也放回表里，需要重跑一次全量评估 ≈2.6 h；不急可先不做。）

**跑法**（①②一次性，③可反复）：

```bash
# ① 生成子集（云端跑；本机已生成同一份 —— seed 相同 ⟹ 内容一致）
python scripts/rl/make_user_subset.py --domain IandS --n-users 5000 --seed 42
# ② 锚点：IandS-all 在**同一批用户**的 test 上  ← ✅ 已跑（HR@10=0.0342）；需重测时用这条
TEST_FILE=data/Amazon23/IandS/sft/test/IandS_5_test.u5k.csv \
MODEL_PATH=outputs/IandS-all/final_checkpoint MAX_SAMPLES=0 BATCH_SIZE=12 bash evaluate_run0.sh
# ③ 快迭代训练（训练用户 ≡ 评估用户）
TRAIN_FILE=data/Amazon23/IandS/sft/train/IandS_5_train.u5k.csv \
RL_TASKS=T1,T3 RL_T3_SAMPLE=977 \
MODEL_PATH=outputs/IandS-all/final_checkpoint USE_LORA=False RUN_TAG=u5k \
NUM_TRAIN_EPOCHS=1 TEST_DURING_TRAINING=False EVAL_STEP=999 SAVE_STEPS=999 bash rl_run0.sh
```

⚠️ **两处诚实边界**：① "同用户"只覆盖 **T1/T3**（占全量 81%，也正是承载用户信号的两路），
T2 无用户维度、被整体去掉；② 于是任务构成变成 `T1:T3`（无 T2）⟹ 与全量 run 的 `T1:T2:T3` 不同，
外推结论需假设 **T2 不是关键变量**（它只做"哪个商品有这个标题"，不含用户信号）。

🔴 **短跑的两个陷阱**：① 只有约 24% 的步给梯度 ⟹ 470 步里真正更新参数的只有约 110 步，
**别用 reward 的绝对值判断"学没学到"**；② 命中率在各个 prompt 间不均（探针复现的那段开头 160 个 prompt
在 `rlfix` 里同样是 0 命中）⟹ **短跑的 reward 水平受"开头运气"影响**，只看"有没有从恒 0 变成出现非 0"。

**阶段 A · 冒烟（约 5 min，唯一目标是验链路）**

```bash
cd ~/GenRetrieval && git pull && source .venv/bin/activate

MODEL_PATH=outputs/IandS-all/final_checkpoint \
RUN_TAG=rlsmoke MAX_STEPS=5 \
TEST_DURING_TRAINING=False SAVE_STEPS=999 EVAL_STEP=999 \
bash rl_run0.sh
#  -> outputs/IandS-rlsmoke/
```

**判据（4 条全过才算通）**：

| # | 看什么 | 期望 | 不过说明什么 |
|---|---|---|---|
| 1 | 前置检查 | 无 `[MISSING]` / `[BAD]` | `final_checkpoint` 缺文件或 tokenizer 里没有 `<a_0>` |
| 2 | `completion_length` | **5.0**，且 `categorical_diversity 1.0` | 约束解码没生效（生成长度失控） |
| 3 | `reward` / `reward_std` | **> 0**（量级 0.01~0.07，见 §6.8②） | 起点模型或约束映射有问题 |
| 4 | `grad_norm` | **> 0** | 梯度没回流（检查 `GRAD_CKPT` + `enable_input_require_grads`） |

⚠️ 若 3/4 仍恒 0 ——先用 `ADD_GT=True` 跑同样的 5 步做**对照**（§6.6）：
`ADD_GT` 下有梯度而正常配置没有 ⟹ 说明是**奖励太稀疏**而非链路坏；两边都 0 ⟹ 链路坏。

**阶段 B · 短跑（~25 min，要轨迹）**

```bash
MODEL_PATH=outputs/IandS-all/final_checkpoint \
RUN_TAG=rl0 MAX_STEPS=300 \
TEST_DURING_TRAINING=False EVAL_STEP=999 SAVE_STEPS=999 \
bash rl_run0.sh
#  -> outputs/IandS-rl0/
```

- `MAX_STEPS=300` 把训练钉在 ~25 min（`[实测]` 4.9 s/step × 300 ≈ 25 min）。
- 🔴 **必须 `EVAL_STEP=999`** —— 若留着 `EVAL_STEP=0.5`，HF 层 `evaluate()` 会在 step 150/300 各跑一次
  **全量验证集（50,984 条）beam=10**，单次 ≈ **6 h** ⟹ 这个"短跑"会变成 ~12 h（§6.10 ⑺）。
  我早先在这里写过 `EVAL_STEP=0.5`，是**没验证过内部 eval 成本**就下的配方，已改。
- 🔴 **别指望 `TEST_DURING_TRAINING` 给 HR 轨迹**（我原先这么写，错了）：它**每步都跑**、样本只有
  当前 micro-batch 的 **1~4 条** prompt、而且评的是**训练 batch 不是验证集** ⟹ `HR@k` 恒为 0 是必然。
  `EVAL_STEP` 对该路径**无效**。详见 **§6.10 ⑸**。
  ⟹ 短跑只需看 `reward` / `reward_std` / `grad_norm` / `kl` 的走势；**要 HR 就走阶段 C 的
  `evaluate_run0.sh`**（beam=50，与 `IandS-all` 同口径）。
- ⚠️ 训练内 `HR@k`（beam=10）与 `evaluate_run0.sh`（beam=50）、`baseline/`（全库排序）**三者两两不可横比**。

**阶段 C · 口径对齐评估（这才是判决点）**

```bash
# 先 5000 条冒烟（~15 min），确认后再上全量（~2.6 h）
MODEL_PATH=outputs/IandS-rl0/final_checkpoint MAX_SAMPLES=5000 BATCH_SIZE=12 bash evaluate_run0.sh
MODEL_PATH=outputs/IandS-rl0/final_checkpoint BATCH_SIZE=12 bash evaluate_run0.sh
```

判据：**与 `IandS-all` 的 `HR@10 = 0.0356` 比**（同 beam=50、同 n、同宽松口径 ⟹ 唯一合法对照）。

**显存兜底**：默认 `USE_LORA=False`（全参）。⚠️ 全参下 `ref_model` 会**真多一份权重**
（§6.1：`not is_peft_model` ⟹ `create_reference_model`）⟹ 24 GB 应够（§6.7 估 8~12 GB），
若 OOM 就 `USE_LORA=True`（参考模型靠 `disable_adapter()` 复用，不额外占权重）。

### 6.10 `[实测]` 首次云端冒烟（2026-09-25，`IandS-rlsmoke`，`MAX_STEPS=5`）

起点 = `IandS-all/final_checkpoint`（SFT 定版）+ `USE_LORA=False`（全参）。

**通过项（链路全通）**：

| 项 | 实测 | 判读 |
|---|---|---|
| 前置护栏 | `<a_0>`=1 token，响应前缀 `<\|im_start\|>assistant\n`=3 token，`prefix_index=3` 成立，`vocab=152437` | ✅ 词表注册与模板真源一致 |
| 训练集 | `num_rows = 270,432` | 208,999 + **51,433** + 10,000 |
| 验证集 | `num_rows = 50,984`（= `valid/*.csv`） | ✅ |
| `completion_length` | **5.0**（每步都是） | ✅ 约束解码在真实 trainer 里精确产出 `[3 SID + \n + EOS]` |
| `categorical_diversity` | **1.0** | ✅ 4 条 beam 候选互不相同 |
| `reward` / `reward_std` | 前 4 步 **0.0**，第 5 步 **0.03125 / 0.0625** | ✅ **起点模型第一次给出非零信号** |
| `grad_norm` | 前 4 步 0，第 5 步 **1.6484** | ✅ 梯度确实回流（**不需要 `ADD_GT` 了**） |

🔴 **修正 §6.8 的成本推算**：我按 SFT 的 `SidItemFeatDataset`（50,440）估 T2，但 RL 用的是
`RLTitle2SidDataset` —— 它产出 `title2sid` **＋`description2sid`**（§1.2），实测 **51,433**。
⟹ 合计 **270,432**（非 269,439）。

**⑵ 步速与全量成本 `[实测]`**：

```
train_runtime = 25.08 s / 5 步 = 5.02 s/步（train_steps_per_second = 0.199）
步数 = 270,432 / (TRAIN_BATCH_SIZE 4 × GRAD_ACC 8) = 8,451 步/epoch × 2 = 16,902 步
⟹ 全量 2 epoch ≈ 23.6 h，另加训练内 eval
```

⚠️ 我早先按 3050Ti 实测外推出的"4090D 约 9~14 h"**偏快一倍**，已由 §6.8③ 的实测取代 ——
根因是全参 GRPO 的 ref 前向 + beam 生成都按 128 条序列/步算，比 SFT 的 teacher-forcing 贵得多。

🔴 **缩短时长：抬 `TRAIN_BATCH_SIZE` 没用**（我初版这么写过，错的）。总成本 ∝ **总生成条数**：

```
总条数 = epoch × 样本数 × NUM_GENERATIONS = 2 × 270,432 × 4 ≈ 2.16 M 条
5 s / 128 条 ⟹ 39 ms/条 ⟹ 2.16 M × 0.039 ≈ 84,000 s ≈ 23.3 h   ✓ 与上面吻合
```

抬 batch 只是把同一批工作换个切法（步数少了，但每步按比例更重）⟹ **壁钟时间不变**。
真正的杠杆只有三个：

| 杠杆 | 效果 | 代价 |
|---|---|---|
| **`NUM_TRAIN_EPOCHS=1`** | 23.6 h → **≈11.8 h** | 少一轮 |
| 减样本（`rl.py` 里 `sample` 硬编码 `-1`，**要改代码**） | 线性 | 改代码 |
| `MAX_STEPS=N` 截断 | 线性 | 只跑一部分数据 |

⚠️ 往下压 `NUM_GENERATIONS`（4→2）虽然线性省钱，但会**同时降低组内命中概率**（`HR@2` < `HR@4`）
⟹ 信号更稀疏，不划算（§6.8②）。

**⑶ ✅ `kl ≡ 0` 已定论：不是空转，但**幅度极小

现象：`kl` 在全部 5 步都是 `0.0`。而 `use_lora=False` ⟹ `ref_model = create_reference_model(model)`
（**真副本**，非 `disable_adapter` 路径），`kl` 用 k3 估计量 `exp(Δ)−Δ−1 ≥ 0`
（`minionerec_trainer.py:1049/1070`）⟹ 恰好 0.0 意味着 ref 与 policy 的逐 token logprob **逐位相同**。

**判决式检验（已跑）** —— 逐张量比对起点与产物的 `model.safetensors`：

| 观测 | 值 |
|---|---|
| 张量数 | 310 / 310（key 集合一致） |
| **有差异的张量** | **208 / 310** |
| `max\|d\|`（最大差异） | **9.537e-07** |

⟹ **权重确实被更新了，不是空转。** 同时这**自我解释了 `kl` 为什么恰好为 0**：
更新幅度只有 ~1e-6 量级，**在 bf16 logits 的分辨率下逐位不可观测** ⟹ k3 估计量为 0。
（第 1 步 `learning_rate = 0.0`（warmup）⟹ 该步本就无更新，kl=0 正常。）

🔴 **判据修正**：**`kl` 在本项目（bf16 前向）不是"策略有没有动"的可靠指标**。
要验证策略是否更新，**比权重**，不要看 `kl`。

⚠️ **遗留未解释**（诚实标注，别当结论）：**102/310 张量逐位未变**；且各张量的 `max|d|`
几乎都等于同一个常数 `9.537e-07`。前者像"更新被存储精度舍入"，后者更像某种归一化步长的
一致性 —— 目前**没有可证实的机制**。若后续要深挖，切口是查 `_get_per_token_logps` 的
dtype 与 `paged_adamw_32bit` 下参数的存储精度。

⚠️ 另注：`max|d| ≈ 1e-6/5 步` ⟹ **5 步不足以判断长期是否有效**（若更新准相干，8451 步可累积到 ~1e-2；
若近随机游走，`√8451 × 1e-6 ≈ 9e-5` 可忽略）。**必须靠更长的短跑（≥300 步）看 `reward`/`HR` 轨迹**。

**⑷ 🔴 首跑暴露的 trainer bug：测试路径复用了训练侧的 `ConstrainedLogitsProcessor`（已修）**

`TEST_DURING_TRAINING=True` 一开就崩（此前所有冒烟都带 `=False` ⟹ 测试路径**从未被执行过**）：

```
LogitProcessor.py:49  input_ids.view(-1, self._num_beams, input_ids.shape[-1])
RuntimeError: shape '[-1, 4, 93]' is invalid for input of size 930
```

**根因**：`minionerec_trainer.py:689-700` 只造了**一个** `ccc` 实例，两处共用：

| 路径 | beam 宽度 | 用的实例 |
|---|---|---|
| 训练 | `num_generations`（4） | `self.logits_processor = [TemperatureLogitsWarper, ccc]` |
| 测试 | `test_beam`（10，见 `test_generation_config`） | `self.test_lp_list = [ccc]` ← 🔴 **同一个对象** |

`ccc._num_beams` 是按训练侧算的（4），测试时却有 10 条 beam ⟹ `view(-1, 4, L)` 维度对不上。

🔴 **只在 `(prompt 数 × test_beam) % num_generations ≠ 0` 时才崩** ⟹ 是否复现**取决于 batch 大小**：

| prompt 数 | 行数 | `view(-1,4,93)` 可整除 | 结果 |
|---:|---:|:--:|---|
| 1 | 10 | ❌ | **崩**（= 云端这次） |
| 2 | 20 | ✅ | 不崩（所以它"挑 batch"，更像偶发） |

**修复**：测试侧另建一个 `num_beams=self.test_beam` 的独立实例（`ccc_test`）。
⚠️ 顺带说明 `count`（解码步计数）**不是**问题：`ccc` 在 `_prepare_inputs` 里每步重建 ⟹ 自然归零；
但独立实例能彻底避免两条路径互相污染。

**验证**（本机 `torch` 隔离测试，与云端报错逐字一致）：
```
A. 复用训练侧实例（nb=4）：1 prompt -> RuntimeError: shape '[-1, 4, 93]' ... size 930  ✅ 复现
                          2 prompts -> OK
B. 测试侧独立实例（nb=10）：1 / 2 prompts -> OK  ✅
C. 连续两次 __call__ 后 count=2 ⟹ 每步重建即归零
```
⚠️ **这是"此前从未执行过的代码路径"里藏的第一个 bug，可能不止一个** ⟹ 短跑要盯完整日志。

**⑸ 🔴 `TEST_DURING_TRAINING` 的真实语义：每步都跑，且评的是**训练 batch**（不是验证集）**

读代码（`minionerec_trainer.py:669/753`）后确认三件事，全部与直觉相反：

| 项 | 实际 |
|---|---|
| 触发频率 | **每个训练步都跑** —— `rl.py:397` 的 `eval_steps=eval_step` 只写进 `GRPOConfig`，而这个自定义 trainer **从不读 `eval_steps`**（`grep eval_step minionerec_trainer.py` = 空）⟹ **`EVAL_STEP` 对本路径完全无效** |
| 评测样本 | **当前 micro-batch 的 prompt**（`dedup_prompt` 取自 `prompt_ids`、`dedup_target` 取自该 batch 的 `targets`），**不是** `eval_dataset` |
| 样本量 | `dedup` 取 `i % num_generations == 0` ⟹ 每步约 **1~4 条** prompt |

🔴 后果：**`HR@3/5/10/20` 几乎必然恒为 0**（`HR@10≈0.035 × 4 prompt` ⟹ 期望命中 0.14 次），
**它不是验证集指标、也没有轨迹可言**。
⟹ ⚠️ **本文件 §6.9 里"用 `EVAL_STEP=0.5` 拿到 2 次 HR 轨迹"的说法是错的**（我当时的假设未经验证）——
`TEST_DURING_TRAINING` 目前**不能**用来判断 RL 是否在学。

**RL 增益的唯一合法判据仍然是 §6.9 阶段 C**：跑完用 `evaluate_run0.sh`（beam=50、全量/`MAX_SAMPLES`）评
`final_checkpoint`，与 `IandS-all` 的 `HR@10 = 0.0356` 比。**长跑建议直接 `TEST_DURING_TRAINING=False`**
（既省掉每步的 beam-10 生成，又不损失任何有效信息）。

**⑹ ✅ 已结案：约束解码吐违规 token（`No valid tokens found for hash_key [17] at step 1`）**

**先排除映射侧**（本机离线实测，用真实 tokenizer + `item_info.txt` 复现 trainer 的构建逻辑）：

| 检查 | 结果 |
|---|---|
| `hash_dict` 键数 | 68,442 |
| 前缀键 `'151644-77091-198'` | **存在** ✓ |
| 该键允许的 token | 恰好 **256** 个，id 范围 **[151669, 151924]** = `<a_*>` 全集 ✓ |
| token id **17** 是否在其中 | **否** ✗ |
| 四路 RL prompt 末 3 token | **全部** `[151644, 77091, 198]` ✓ |
| `item_info.txt` 25,847 行 | 全部 7 token、前缀齐、第 4 token 全在 a 码区间 ✓ |

**再排除"prompt 结尾不对"**：完整日志里 **`at step 0` 的告警 = 0 条**
（step1×65 / step2×14 / step3×4 / step4×4，共 87 条）⟹ 第 0 步的 key 全部合法。
且违规 key 是 `[17][16][11][10][15][14][0][1][9][7][13][12][8]` —— **13 个值全在 [0,18)**，
**每次以完全相同的顺序重复** ⟹ 这是**位置型索引**（tie-break）的特征，不是采样出来的 token。

**🔴 真正的机制（HF beam-sampler × 逐步约束的结构性冲突）：**

`transformers/generation/utils.py::_get_top_k_continuations` 每步会要
`beams_to_keep = max(2, 1+n_eos) * num_beams` 张**互不重复**的票：

```python
# do_sample=True（BEAM_SAMPLE）
topk_indices = torch.multinomial(softmax(accumulated_log_probs),
                                 num_samples=beams_to_keep, replacement=False)
# do_sample=False（BEAM_SEARCH）同理，torch.topk(..., k=beams_to_keep)
```

而 SID 的 Trie 约束在 **`c` / `\n` / `eos` 这几位只放行 1 个 token** ⟹
**支持集 < 票数** ⟹ `torch.multinomial` 只能拿**零概率位置凑数**。实测这些凑数位置的
**扁平索引就是 `0..k-1`**（= beam0 的 token `0..k-1`），落地后正是日志里那批 ASCII/数字 id：

| 配置 | 实测逃逸 id |
|---|---|
| `num_beams=10`（k=20，测试路径） | `{1,3,7,8,9,15,16,17,18,19}` |
| `num_beams=4`（k=8，训练路径） | `{1,3,4,7}`（⊂[0,8)） |

正常情况下这些凑数条目分数是 `-inf`、排名垫底、**进不了 beam**（实测三组配置"垃圾=0"）。
🔴 **但只要某些 beam 的合法项分数本身不是有限值**（`#finite < num_beams`），
`_get_running_beams_for_next_iteration` 的 `topk(k=num_beams)` 就**必须**保留它们
⟹ 位置型垃圾 token 被写进序列 ⟹ 下一步 `hash_key` 失效 ⟹ 告警刷屏 + 生成被污染。

**判决式验证**（真实 tokenizer + `hash_dict` + 微型模型，注入"合法码分数非有限"复刻云端）：

| 判据 | 旧版 | 新版 |
|---|---|---|
| 正常情形生成序列**逐位一致** | — | **True** ✅ |
| 合法码分数非有限 ⟹ 违规候选 | **30/30** | **0/30** ✅ |
| 同场景 `do_sample=True` | 直接抛 `probability tensor contains inf/nan` | 正常 ✅ |

（旧版在"整行非有限"时崩溃，反证云端必然是**部分**行非有限 —— 与"没有崩"一致。）

**修复（两处，零回归）：**

1. **`LogitProcessor.py`** —— 合法项分数若为非有限（`-inf`/NaN），兜底成 `row_max - 1000`
   （**有限值**），保证**每个 beam 至少贡献 1 张有限票** ⟹ `#finite ≥ num_beams` 恒成立
   ⟹ 凑数条目永不进 beam。不允许位仍是 `-inf`（排名绝对垫底，不引入新风险）。
   ⚠️ 分数正常时**逐位不改**（上表判据 1 已证）⟹ SFT/评估结果不变。
   顺带：改为**直接返回 mask**（与旧 `scores+mask` 等价）、fallback 的 eos 位补 `isfinite` 兜底、
   新增「`cur_len` 回退即重置 `count`」的实例复用自愈。
2. **`minionerec_trainer.py`** —— 4 处 `generate()` 全加 **`use_model_defaults=False`**，
   `test_generation_config` 显式补 `temperature=1.0 / top_k=None / top_p=None`。
   原因：HF 的默认值回填规则（「传入值 == `GenerationConfig` 全局默认 且 基座值 != 全局默认 ⟹ 取基座值」）
   使 `do_sample=False`（== 全局默认）被 Qwen3 基座的 `true` 覆盖（日志原样打出覆盖清单），
   且 `Temperature/TopK/TopP` 三个 warper 会被**追加在约束处理器之后**（`_get_logits_processor`
   里 warper 是在 merge 之后 append）。这与 **`evaluate.py` 早已修过的坑同源**
   —— 修复后测试路径恢复为**确定性 `BEAM_SEARCH`**，训练路径保持 `do_sample=True`（探索所需）但不再带 warper。

🔴 **通用红线**：任何"逐步约束"都必须保证「每步允许集 ≥ HF 需要的候选数（`2×num_beams`）」，
或保证"合法项分数恒有限"；否则 `multinomial`/`topk` 的凑数行为会**静默绕过约束**。
排查一行命令：`grep -o "at step [0-9]*" <log> | sort | uniq -c`（**`at step 0` 为 0 ⟹ 掩码在第 0 步是对的**）。

### ✅ 云端验证（2026-09-25，`IandS-rlfix`）

配置刻意让**两条生成路径同时被走到**：`MAX_STEPS=30` + `TEST_DURING_TRAINING=True`
（告警只在这条路径产生）+ `EVAL_STEP=999`（避开 ⑺ 那次 6 h 的内部 eval）+ `SAVE_STEPS=999`。

| 判据 | 结果 |
|---|---|
| `grep -c "No valid tokens" <log>` | **0**（修复前：头 15 步就有 **87** 条） |
| 覆盖的生成路径 | 测试路径 beam=10 **与** 训练路径 beam=4，**两者都跑过、都零告警** |
| `completion_length` | 5.0（每步）—— 仍是精确的 `[3 SID + \n + EOS]` |
| `categorical_diversity` | 1.0 |

⟹ **判决式通过**：约束在两条路径上都真正生效了。

**⑺ 🔴 内部 `evaluate()` 单次 ≈ 6 小时 —— 它才是长跑的主导成本（我原估 30 min，差 12×）**

`rl.py:389/397` 设了 `eval_strategy="steps"` + `eval_steps=eval_step` ⟹ **HF 层的 `evaluate()` 会对
整个 `eval_dataset`（50,984 条）跑 beam=10 生成**。`[实测]` 进度条：

```
683/50984 [04:40<5:56:55, 2.35it/s]      ⟹ 单次 eval ≈ 6.0 h
```

**`eval_steps` 的精确语义**（`trainer_callback.py:161-166 compute_steps`）：

```python
if num_steps < 1:  num_steps = math.ceil(max_steps * num_steps)   # <1 = 比例
# 否则保持原值 = 绝对步数
```

| `EVAL_STEP` | `MAX_STEPS=300` 时的行为 | 内部 eval 次数 | 额外耗时 |
|---|---|---|---|
| `0.5` | `ceil(300×0.5)=150` ⟹ step 150 / 300 | 2 | **~12 h** |
| `0.99` | `ceil(297)=297` ⟹ 仅 step 297 | 1 | ~6 h |
| **`999`（> MAX_STEPS）** | 绝对步数 999 > 300 ⟹ **永不触发** | 0 | 0 |
| `1.0` | ⚠️ 绝对 1 步 ⟹ **每步都 eval**（灾难） | 300 | — |

🔴 **要快速拿 checkpoint：`EVAL_STEP=999`（关内部 eval）+ `TEST_DURING_TRAINING=False`**，
300 步 ≈ **23 min**；HR 随后用 `evaluate_run0.sh` 拿（beam=50、`MAX_SAMPLES=5000` ≈ 15 min），
比内部 eval 又便宜一个量级且**口径与 `IandS-all` 可比**。

**⑻ `[实测]` 梯度链与奖励稀疏度**

`rule_reward` 是**二值 0/1**（`rl.py:305-316`，精确字符串匹配）。`log()` 每次平均后 `clear()`
（`minionerec_trainer.py:1138`），而 `_metrics["reward"]` 每 micro-batch append 一次（`:1037`）。
⚠️ **分母（一个日志窗口含多少条候选）见本节末 —— 它尚未核定，所以下表的"反解"只在给定分母时成立：**

| 日志值 | 若分母 = 128（`4 prompt × 4 gen × grad_accum 8`） | 若分母 = 32 |
|---|---|---|
| `reward = 0.03125` = 1/32 | 命中 4 条 | 命中 1 条 |
| `reward_std = 0.0625`（`std_grouped_rewards.mean()`，无偏；二值组 `[1,0,0,0]` 的 std = 0.5） | Σ 组内 std/32 组 = 2.0 ⟹ 4 个组各中 1 条 | Σ 组内 std/8 组 = 0.5 ⟹ 1 个组中 1 条 |

两种口径给出**完全相同**的 (reward, reward_std) 对，所以只能靠**取值网格**区分（见本节末）。
**与分母无关、可直接引用的结论**：`grad_norm` 由"窗口内是否存在命中"门控。

| 观测（`IandS-rlfix` 30 步 / `rl300` 150 步） | 值 |
|---|---|
| 有信号的优化步 | **`rlfix` 5/30 ≈ 17%；`rl300` 36/150 = 24%**（两者同量级） |
| 有信号步的 `grad_norm` | 1.23 ~ 2.30 |
| 无信号步的 `grad_norm` | **恰好 0.0**（前 8 步）；策略被推动过之后为 ~5e-4 |
| `kl` | 0.0（前 8 步）→ 5e-4 起（与 `grad_norm` 同步出现） |
| 步速 | **4.9 s/step** |

🔴 **比早先那张表更准的一点**：`advantage = 0` **且** `kl = 0`（策略尚未被推动）时
`loss ≡ 0` ⟹ `grad_norm` **恰好 0.0**，而不是"约 5e-4"。只有当策略已被推过一次（`kl > 0`）后，
零 advantage 的步才会剩下约 5e-4 的 KL 正则梯度。这解释了前 8 步的**精确 0.0**。
⟹ 结论不变但更硬：**没有命中的步完全不贡献梯度**，有效更新 ≈ 17% × 总步数。

⚠️ **反常结论已撤回（2026-09-25，用最长 run 实测）**：我据 30 步断言「`reward` 非零时恒为
`0.03125` ⟹ 每步恰好 4 命中」。统计 150 步后**该说法是错的**：

```bash
grep -o "'reward': [0-9.]*" logs/rl/IandS-rl300/rl.log | sort | uniq -c
#  114 'reward': 0.0   30 'reward': 0.03125   5 'reward': 0.0625   1 'reward': 0.09375
```

| 统计量 | 值 |
|---|---|
| 有信号的窗口 | **36 / 150 = 24%**（`rlfix` 的 5/30 = 17% 属同量级） |
| 计数分布 | 命中 1 次 ×30、2 次 ×5、3 次 ×1 |
| 与 `Poisson(λ = 43/150 = 0.287)` 的拟合 | 预期 112.6 / 32.3 / 4.6 / 0.4 vs 实测 **114 / 30 / 5 / 1** —— **拟合极好** |

⟹ **命中在窗口之间是独立的，计数没有被量化**（值域是 `k/32`，k 只取到 1/2/3）。
原判断的错因：只看了 30 步的 `rlfix`，而那 30 步恰好只出现 k=1。

✅ **分母这件事其实不影响任何"率" —— 上一条的顾虑撤回。**

`reward = h/W`（W = 窗口内候选数，h = 命中数）⟹ `E[reward] = E[h]/W`；而
「每候选命中率」≡ `E[reward]`、「每组（= 每 prompt）命中率」≡ `4·E[reward]` —— **两者都与 W 无关**。
所以 W 是 32 还是 128 只改变"计数怎么读"，不改变任何可引用的比例：

| 量 | 公式 | `rl300`（150 步） | `rlfix`（30 步） | 与 SFT 口径对照 |
|---|---|---:|---:|---|
| 每候选命中率 | `E[reward]` | **0.90%** | 0.52% | `HR@1` = 0.67% ✓ 同量级 |
| 每组 / 每 prompt 命中率 | `4·E[reward]` | **3.58%** | 2.08% | `1−(1−HR@1)⁴` = 2.65% ✓ |
| 有信号的优化步 | — | **36/150 = 24%** | 5/30 = 17% | — |

（`E[reward]` 直接算：`rl300` = `(30×0.03125 + 5×0.0625 + 1×0.09375)/150` = **0.00896**。）

⚠️ **判决实验已跑，结论是"无法判定"**：`GRAD_ACC_STEPS=1 MAX_STEPS=40`（1 micro-batch/步）
→ **0 个非零 reward**（`grep` 只回 `40 'reward': 0.0`）⟹ **没有非零值就没有网格可读**。

但这个"0"本身**不反常，且已被解释**：`shuffle(seed=42)` 是确定性的，两次 run 消费的 prompt 顺序相同 ——
探针跑掉的正是**前 160 个 prompt**，而 `rlfix` 的第一次命中在第 8 步（≈第 224 个 prompt 之后），
也就是说**同一段开头在 `rlfix` 里同样是 0 命中**。两次完全一致 ✓。
（注：`rlfix` 的 2.08% 率下，160 个 prompt 的期望命中约 3.4 个，单看探针像是 1/30 的事件；
对齐 prompt 顺序后才知道它只是复现了同一段"硬开头"。）

**辅助结论**：步速 探针 `26.51 s / 40 步 = 0.66 s/step`，`rlfix` `147.03 / 30 = 4.90 s/step`，
比值 **7.40 ≈ `GRAD_ACC_STEPS` 8** ⟹ 成本确实 ∝ grad_accum（每优化步含 8 个工作单元）。

**⟹ 此事不再投入 GPU**：既然所有率都与 W 无关，就不再追 W；本记录到此收口。

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
| 1 | **SFT 定版产物** `outputs/IandS-all/final_checkpoint` | ✅ **2026-09-25 已就位**（`HR@10=0.0356`，§6.8）⟹ RL 硬阻塞解除 |
| 2 | 3090 上实测**全参** RL 显存与单 checkpoint 体积（§6.6 仍是推算） | ⏸ |
| 2b | 本地 LoRA 冒烟 | ✅ `[实测]` 见 §6.5：`exit=0`，峰值 2.80 GiB、单步 8.45 s；梯度链由 §6.6 的 `ADD_GT` 诊断验穿 |
| 2c | 用**真实 SFT 产物**跑一次冒烟（伪产物阶段结束） | ⏸ **下一步**：§6.8 ③ 的三阶段跑法 |
| 3 | `semantic` 奖励的 `ada_path`（item embedding pickle） | ⏸ 缺 |
| 4 | `sasrec` 奖励的 `cf_path`（根 `sasrec.py` 的权重） | ⏸ 缺 |
| 5 | 分层奖励（R1）/ OPD 蒸馏（R2/R3） | ⏸ 见 UPGRADE_PLAN §6 |
| 6 | RL 侧的 EXP_ID 与评估打通（训完用 `evaluate_run0.sh` 对齐全库口径） | ⏸ |
