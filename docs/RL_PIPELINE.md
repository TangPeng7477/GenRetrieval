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

**建表**（`minionerec_trainer.py:529-572`，ReReTrainer `__init__`）：

```
info_file 每行 -> split('\t')[0] + "\n" -> 前置 "### Response:\n"
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

⟹ 长 prompt 不影响：**只要 prompt 末尾恰好是 `### Response:\n`**。
而 `minionerec_trainer.py:675-678` 用的是
`maybe_apply_chat_template`（我们的输入是纯字符串、非 conversational ⟹ **不套 chat template**）
+ `add_special_tokens=False`，所以不会被追加尾 token。两条合起来前提成立。

### 3.1 探针实测（`scripts/sft/probe_rl_constraint_map.py`）

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
换基座 / 换 tokenizer 必须重跑该探针。

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

## 6. 成本（**推算，待 3090 实测**）

| 项 | 量级 | 依据 |
|---|---|---|
| 显存 | policy bf16 1.2G + ref bf16 1.2G + grads 1.2G + `paged_adamw_32bit` 状态 ~4.8G + 激活/KV ≈ **9–12 GB** | 24G 单卡应可容纳；⚠️ **未实测** |
| ⚠️ 若换 V100 | 必须 `PRECISION=fp16`，且**显存约翻倍**（fp32 主权重）；16 GB 版跑不动，32 GB 版可以 | 见 [`PRECISION_GUIDE.md`](PRECISION_GUIDE.md) §5 |
| 单个 checkpoint | ≈ **6 GB**（权重 + fp32 优化器状态） | `[设计推算]` |
| checkpoint 总量 | `save_total_limit=3` → ~18 GB，加 root + `final_checkpoint` 两份 ≈ **21 GB** | 原版 `save_total_limit=20` 会到 ~120 GB，**本仓已下调** |

---

## 7. 本项目对 MiniOneRec 原版的改动清单

| 文件 | 位置 | 改了什么 | 为什么 |
|---|---|---|---|
| `rl.py` | `:73` | 新增 `--torch_compile`（默认 **False**） | 原版硬编码 `True`；GRPO 每步生成长度不定 + 动态 padding 会反复重编译（SFT 侧已实测这个坑） |
| `rl.py` | `:77-78` | 新增 `--save_steps` / `--save_total_limit`（默认 0.1 / **3**） | 原版 20 会占 ~120 GB 磁盘 |
| `rl.py` | `:80` | 新增 `--optim`（默认 `paged_adamw_32bit`） | 参数化，便于本地/云端切换 |
| `rl.py` | `:91-113` | **新增前置护栏**：`<a_0>` 必须 1 token、`### Response:\n` 必须 3 token | 指回原始基座会静默崩（SID 碎裂 + 约束映射全废）；换 tokenizer 会让 `prefix_index=3` 失效 |
| `rl.py` | `:185-201` | 奖励 artifact 缺失时**报错**；修掉无条件误导日志 | 原版会抛 `KeyError`/`NameError`，看不出根因 |
| `rl_run0.sh` | 新增 | RL 入口：EXP_ID 命名 / 前置检查 / venv 探测 / 写 `run.meta.json` | 对齐 `sft_run0.sh`；原两个 `.sh` 路径不可用 |

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
| 2 | 3090 上实测 RL 显存与单 checkpoint 体积（§6 是推算） | ⏸ |
| 3 | `semantic` 奖励的 `ada_path`（item embedding pickle） | ⏸ 缺 |
| 4 | `sasrec` 奖励的 `cf_path`（根 `sasrec.py` 的权重） | ⏸ 缺 |
| 5 | 分层奖励（R1）/ OPD 蒸馏（R2/R3） | ⏸ 见 UPGRADE_PLAN §6 |
| 6 | RL 侧的 EXP_ID 与评估打通（训完用 `evaluate_run0.sh` 对齐全库口径） | ⏸ |
