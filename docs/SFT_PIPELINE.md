# SFT 阶段流水线（唯一入口）

> **状态**：2026-09-14 建立 ｜ **M3/M4 阶段**（`docs/UPGRADE_PLAN.md §5`）
> **本文只管「SFT 数据怎么构造、提示词为什么这么写、训练/解码怎么接」**。
> 指标定义一律以 **[`docs/EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md)** 为准（召回效果）与
> [`docs/SID_PIPELINE.md`](SID_PIPELINE.md)（SID 质量），本文只给读法 + 回指。
>
> **三条来源标注**（沿用全仓铁律）：`[实测]` 本机跑出来的数字 ｜ `[代码]` 本仓/开源代码可读证 ｜ `[文献]` 论文结论（标核验等级）。

---

## 0. TL;DR

| 项 | 定版 | 来源 |
|---|---|---|
| 输入 | LOO 的 `*.inter` + 定版 `sid_raw`（`gate + init8192 + 5000 轮`，3 层 × 256 码） | `[实测]` |
| SID token 形式 | `<a_12><b_34><c_56>`，每码 **1 个 token**，共 768 个新 token | `[实测]` 见 §4 ① |
| 主任务 T1 | 历史 SID 序列 → 目标 SID，**提示词与 MiniOneRec 逐字一致** | `[代码]` `data.py::SidSFTDataset` |
| 辅助任务 | T2 `sid↔title` ／ T3 `seq2title` ／ T4 `text2sid`（本项目新增） | §2 |
| 样本数 | I&S 209k/51k/51k ／ VG 436k/95k/95k（train/valid/test） | `[实测]` |
| `cutoff_len` | **400**（只训 T1/T2/T3 时 320 就够；**带 T4 必须 400**，全量 max 392） | `[实测]` §3.1 表 |
| 上游落点 | `data/Amazon23/<域>/sft/`（域代号 `IandS` / `VG`） | `[实测]` §3.2 |
| 训练端 | MiniOneRec `sft.py` + `data.py` 为骨架，**本项目已改 5 处**（SID 注册 / `--tasks` 开关 / padding / `torch_compile` / `--sid_vocab_path`） | `[实测]` §3.2 · §4.6 |
| 碰撞（语义桶） | 主榜**严格口径**（每 SID 桶取 1 个 representative），另报宽松上界 | §5.1 |

---

## 1. 提示词为什么这样设计（文献对照）

生成式推荐的「提示词」其实不是一回事，分成**三个流派**，选错了会直接改变训练信号的落点：

| 流派 | 代表 | 样本形态 | 训练信号落在哪 |
|---|---|---|---|
| **A. 无提示词，平铺 NTP** | **TIGER**（NeurIPS'23） | `user_5 hist: <a_1><b_2><c_3> … → <a_x><b_y><c_z>`，整条序列做 next-token prediction | **整条序列**（含历史段） |
| **B. 指令问答对（prompt→target）** | **P5**（RecSys'22，最早把推荐 verbalize 成 instruction）、**MiniOneRec**、**LC-Rec** | `### Instruction: … ### User Input: … ### Response: …` | **只落在 target 段**（prompt 段 label = −100） |
| **C. 文本化 ID** | **IDGenRec** | 用 LLM 生成**可读文本 ID**（如属性关键词），再在 prompt 里拼历史文本 ID + Trie 约束解码 | 只落在 target 段 |

**本项目走 B**（与 MiniOneRec 同构），三条理由：

1. **样本效率**：B 只对 target 段计 loss，历史段是纯条件；A 对历史段也算 loss，
   在 0.6B 小模型 + 20 万样本上，A 会把容量浪费在"复述历史"上。
2. **辅助任务能挂上去**：只有 B 形态能自然地把 T2/T4 这类"SID ↔ 文本"对齐任务
   塞进同一个 batch（同一套 instruction 外壳）。LC-Rec 的 alignment 主张（§1.1）就靠这个落地。
3. **与 C 的取舍**：C（IDGenRec 的文本化 ID）对**冷启动/零样本**更友好（新商品有文本就能出 ID），
   但本项目 SID 已定版为 RQ-VAE 离散码（`SID_PIPELINE.md`），换路线等于推倒重来。
   **C 的思想用 T4 任务部分吸收**（让模型学会"从富文本反查 SID"，见 §2）。

> ⚠️ **曾经写在第 1 条的理由已作废**（2026-09-14 自我纠正）：
> "走 B 是为了保住 V0 锚点（HR@10 = 0.093）的可比性" —— **这个理由不成立**。
> V0 是 Amazon18 + 全局时间 8:1:1 + Qwen2.5-0.5B，本项目是 Amazon23 + LOO + Qwen3-0.6B，
> 数据集都不同，本来就比不了（详见 §5.3）。删掉这条后，B 仍是首选，但理由是上面的 1/2/3。

### 1.1 LC-Rec 的 alignment tuning（本项目的辅助任务直接来源）

**LC-Rec**（Zheng et al., **ICDE 2024**，arXiv:2311.09049，`[文献]` 核验：GitHub README 的 `run.sh`
任务清单为 `seqrec, item2index, index2item, fusionseqrec, itemsearch, preferenceobtain`）
的核心主张是：**光有 seq2sid 不够，必须额外做「SID ↔ 自然语言」的双向对齐任务**，
否则 LLM 学到的 SID 是"无意义的符号"。它的三类任务：

| LC-Rec 任务 | 含义 | 本项目对应 |
|---|---|---|
| `seqrec` | 历史 SID → 下一个 SID | **T1** |
| `item2index` / `index2item` | 物品文本 ↔ SID 双向 | **T2**（title 版）+ **T4**（富文本版） |
| `fusionseqrec` | 历史 SID → 下一个物品**标题** | **T3** |
| `itemsearch` / `preferenceobtain` | 按自然语言意图检索 / 生成用户偏好描述 | ⏸ 待办（需额外标注，见 §6） |

> ⚠️ **不要写成 SIGIR**：LC-Rec 是 **ICDE 2024**（arXiv 2311.09049，GitHub README 的 bibtex 明确
> `booktitle = ICDE`）。网上不少二手资料误标 SIGIR'24。

**MiniOneRec 的任务组合是 LC-Rec 的子集** `[代码]`（本仓 `sft.py:350-357` 实际 concat 的三个）：
`SidSFTDataset`(T1) + `SidItemFeatDataset`(T2) + `FusionSeqRecDataset`(T3)，**1:1:1 均匀混合**。

### 1.2 本项目相对 MiniOneRec 的两处改动

| 改动 | 内容 | 理由 |
|---|---|---|
| **新增 T4 `text2sid`** | `title + brand + categories + features` → SID | Amazon23 有 `features` 字段（要点列表），信息密度高于 description；MiniOneRec 只有 title。这是 LC-Rec `item2index` 的富文本变体 |
| **任务配比可配置** | 四任务**分别落盘**，训练端按配比组合 | M4 要消融 `1:1:1` / `2:1:1` / `1:1:2` / `+T4`（`UPGRADE_PLAN §5.3`）。写死就无法消融 |

**不改的**：T1/T2/T3 的提示词**逐字沿用 MiniOneRec**（含它那个拼写错误 `palyed` 也不改——
改了就不是同一条 prompt，V0 锚点失效）。

---

## 2. 六个任务的定义（模板单一真源 `config/prompt_templates.json`）

外层骨架有 **三种**，全在 `config/prompt_templates.json` 里定义（模板细节见 §3.1）：

| 格式 | 形态 | 用途 |
|---|---|---|
| **`chatml`（默认，2026-09-18 起）** | `<\|im_start\|>system … <\|im_end\|>\n<\|im_start\|>user … <\|im_end\|>\n<\|im_start\|>assistant\n` | Qwen3 原生，与预训练一致 |
| `alpaca` | 下框（MiniOneRec 原样） | **V0 用的是它**；保留作格式消融对照 |
| `verbatim` | 同 alpaca 但带引号（MiniOneRec 逐字复刻） | 只用于交叉校验，不进训练 |

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### User Input: 
{input}

### Response:
{output}
```

⚠️ 默认格式已是 `chatml` ⟹ **Run-0 相对 V0 同时改了两个变量**（基座 `Qwen2.5-0.5B`→`Qwen3-0.6B`
**和** 骨架 `alpaca`→`chatml`）。这是"适配 Qwen3"的主动决定，但要写进归因（§6.4(5)）。

| 任务 | instruction | input（示例） | output | 数据来源 |
|---|---|---|---|---|
| **T1 `seq2sid`** | `Can you predict the next possible item that the user may expect?` | `The user has interacted with items <a_147><b_81><c_146>, <a_148><b_184><c_140> in chronological order. Can you predict the next possible item that the user may expect?` | `<a_5><b_23><c_66>` | `{train,valid,test}/*.csv` |
| **T2a `sid2title`** | `Answer the question about item identification.` | `What is the title of item <a_5><b_23><c_66>?` | 商品标题 | `index/*.item.json` |
| **T2b `title2sid`** | 同上 | `Which item has the title: 3Doodler "What Will You Create? Project Book?` | `<a_5><b_23><c_66>` | 同上 |
| **T3 `seq2title`** | `Can you recommend the next item for the user based on their interaction history?` | `The user has sequentially interacted with items <a_147>…, <a_148>…. Can you recommend the next item for him? Tell me the title of the item` | 商品标题 | `train/*.csv` |
| **T4 `text2sid`** | `Answer the question about item identification.` | `An item can be described as follows: 3Doodler … \| Brand: WobbleWorks \| Categories: … \| Features: …. Which item is it describing?` | `<a_5><b_23><c_66>` | ⏸ 训练端**无对应 Dataset 类**，未接线 |
| **T5 `seqtitle2sid`**（RL 侧） | `Answer the question about item identification.` | `Given the title sequence of user historical interactive items: <title>, <title> … Which item will be next?` | `<a_5><b_23><c_66>` | `RLSeqTitle2SidDataset`（`data.py`） |

（T4/T5 也写在真源里，但**不在 Run-0 的 `--tasks` 默认值**内。）

**历史段的分隔**：物品内三层 token **直接拼接**（`<a_5><b_23><c_66>`），物品之间用 `", "` 分隔
——与 MiniOneRec `data.py::get_history` 一致，这样"一个物品 = 连续 3 个 token"，Trie 约束解码的层级才对得上。

### 2.1 引号口径（2026-09-14 定版）：**全部裸写，一律不加引号**

MiniOneRec 原文是 T2a 的 SID 带双引号、T2b 的 title 不带（不对称）。本项目统一为
**SID / title / text 全部裸写**，理由：

1. `<a_5><b_23><c_66>` 的尖括号本身就是天然定界符，不需要引号；
2. **completion 里的 SID 必须裸写** —— 否则 Trie 约束解码的首个 token 会变成 `"` 而不是 `<a_*>`，
   要连 `LogitProcessor.py` 的起点一起改；
3. T4 的 text 内部本身含未闭合双引号（如 `3Doodler "What Will You Create? Project Book`），
   外部再包一层引号会让边界更乱。

---

## 3. 产物规格

```
data/Amazon23/<域>/sft/
├── index/<域>.index.json      {item_id(str): ["<a_12>","<b_34>","<c_56>"]}   ← sft.py 的 --sid_index_path
├── index/<域>.item.json       {item_id(str): {"title","description","text"}}  ← sft.py 的 --item_meta_path
├── train|valid|test/<域>_5_<split>.csv    T1 主任务，列与 MiniOneRec 完全一致（7 列）
├── tasks/itemfeat.jsonl        T2（每物品正反 2 条 = 2N）
├── tasks/seq2title_<split>.jsonl  T3
├── tasks/text2sid.jsonl        T4（N 条）
├── info/<域>.item_info.txt     sid \t title \t item_id（convert_dataset.py 同格式）
├── info/sid2items.json         sid → {items[], representative, size}（碰撞桶）
├── info/sid_vocab.json         有序 768 token（tokenizer.add_tokens 用）
├── info/codebook.npy           (3,256,32)，RQ-VAE 码本，供 M4 语义初始化
├── info/prompt_templates.json  模板**快照**（真源在 config/，由 prepare 拷入；读取入口 prompt_templates.py）
└── stats.json                  全部计数
```

**CSV 7 列**（`user_id, history_item_title, item_title, history_item_id, item_id, history_item_sid, item_sid`）
—— 后四个是 list 的 Python repr 字符串，因为 MiniOneRec 用 `eval(row[...])` 读 `[代码] data.py:404`。

### 复现命令

```bash
./.venv/Scripts/python.exe scripts/data/prepare_sft_data.py --domain IandS
./.venv/Scripts/python.exe scripts/data/prepare_sft_data.py --domain VG
# （原 verify_sft_data.py 体检脚本已于 2026-09-18 删除，见 §7「已移除的开发期脚本」）
```

### 3.1 提示词模板：单一真源（2026-09-18 收敛）

**此前同一套模板有 3 份实现**（① `data.py` 的 f-string ② `prepare_sft_data.py` 的
`PROMPT_TEMPLATES` ③ `build_sft_prompts.py` 自带的两套），改一处忘一处就是静默漂移。
现已收敛为：

| 角色 | 文件 | 说明 |
|---|---|---|
| **真源（入 git）** | `config/prompt_templates.json` | 3 种格式骨架 + 6 个任务的 instruction/input + `response_prefix` + `completion` 约定 |
| **唯一读取入口** | `prompt_templates.py` | 纯 stdlib 渲染器；`data.py` / `evaluate.py` / `minionerec_trainer.py` 都经它取模板 |
| 建数据时的**快照** | `data/Amazon23/<域>/sft/info/prompt_templates.json` | 由 `prepare_sft_data.py` 从 config 拷入；锚定 `data_path` 时**优先用快照** ⟹ 旧数据永远不会被后来的模板改动污染 |

三个要知道的点：

- 🔴 **响应前缀必须从 `pt.response_prefix()` 取，不准手抄字符串**。它同时是约束解码的 key
  （`evaluate.py` 的 Trie、`minionerec_trainer.py` 的查表、`scripts/sft/probe_constrained_decoding.py`），
  手抄一处就是"约束静默失效"。
- 🔴 **`completion` 尾哨兵 = `\n` + EOS** ⟹ T1 目标恒为 **5 token** `[a,b,c,\n,<|im_end|>]`。
  Trie 的 step3 是"只允许 `\n`"、step4"只允许 EOS"，所以这两层是**设计**。
  （旧文档写"completion 末尾不加 `\n`"——**那句是错的**，与实际训练 target 不符，已作废。）
- ⚠️ **换基座后必须重测响应前缀的 token 数**：`prefix_index=3` 依赖它恰好切成 3 个 token。
  chatml 的 `<|im_start|>assistant\n` = `[151644, 77091, 198]` ✓（alpaca 的 `### Response:\n`
  也是 3 ✓）。重测办法：`probe_constrained_decoding.py` 的 C 段。

**格式消融不再需要预渲染数据集**：`PROMPT_FORMAT=alpaca bash sft_run0.sh` 即可切换
（训练端与评估端都走同一模块，自动一致；默认 `chatml`）。

> 🗑️ 被删掉的是**预渲染明文**：`build_sft_prompts.py` 及其产出 `prompts/`（2.6 GB）
> 与 `tasks/`（0.36 GB）。零消费方，而它能提供的唯一价值（两套明文供消融）已被
> `PROMPT_FORMAT` 这个运行时开关取代。清单见 §7「已移除的开发期脚本」。

**全量长度实测**（格式无关，2026-09-14 全量跑出，仍然有效）：

| 任务 | 样本数 I&S / VG | prompt 均值 | total_max（含 completion+EOS） | >320 |
|---|---:|---:|---:|---:|
| T1 `seq2sid` | 208,999 / 435,534 | 107.8 / 111.9 | **179 / 179** | 0 / 0 |
| T2a `sid2title` | 25,847 / 25,611 | 57.0 / 57.0 | **161 / 143** | 0 / 0 |
| T2b `title2sid` | 25,847 / 25,611 | 86.6 / 69.7 | **160 / 141** | 0 / 0 |
| T3 `seq2title` | 208,999 / 435,534 | 111.8 / 115.9 | **277 / 251** | 0 / 0 |
| T4 `text2sid` | 25,847 / 25,611 | 192.0 / 165.7 | **391 / 362** | **18 / 3** |

→ **`cutoff_len`：不带 T4 用 320；带 T4 用 400**。T4 只占 4.3% 样本，但把 max 从 277 顶到 392。
⚠️ MiniOneRec 是 `tokens[-max_len:]` 左侧截断，超长会砍掉 instruction 头部。

---

### 3.2 上游产物 → 训练/评估参数映射（跑训练前照这张表填）

域代号是 **`IandS` / `VG`**（**不是** `Industrial_and_Scientific` —— 那个全名只用在 `--category`）。
下表以 `IandS` 为例，VG 只需把路径里的 `IandS` 换成 `VG`。

| `sft.py` 参数 | 指向上游哪个文件 | IandS `[实测]` 规模 |
|---|---|---|
| `--train_file` | `data/Amazon23/IandS/sft/train/IandS_5_train.csv` | 208,999 行 |
| `--eval_file` | `data/Amazon23/IandS/sft/valid/IandS_5_valid.csv` | 50,984 行 |
| `--sid_index_path` | `data/Amazon23/IandS/sft/index/IandS.index.json` | 25,847 item × 3 层 |
| `--item_meta_path` | `data/Amazon23/IandS/sft/index/IandS.item.json` | 含 `title`/`description`/`text` |
| `--sid_vocab_path` | `data/Amazon23/IandS/sft/info/sid_vocab.json` | **本项目新增参数**；留空自动推导（§6.4(4)） |
| `--category` | 类目**全名**（域代号无效） | `Industrial_and_Scientific` |
| `--cutoff_len` | — | 400（含 T4）/ 320（不含 T4），见 §3.1 |

| `evaluate.py` 参数 | 指向 | 备注 |
|---|---|---|
| `--test_data_path` | `.../sft/test/IandS_5_test.csv` | 50,982 行 |
| `--info_file` | `.../sft/info/IandS.item_info.txt` | 制表符分隔 `sid \t title \t item_id` |
| `--base_model` | **训练输出目录**（`outputs/sft_IandS/`），**不是** `models/Qwen3-0.6B` | 🔴 见下 |

🔴 **`--base_model` 必须指向训练输出目录**：`evaluate.py` 里**没有任何 `add_tokens`**
（`evaluate.py:74` 只做 `AutoTokenizer.from_pretrained(base_model)`），它依赖
`sft.py:435` 的 `tokenizer.save_pretrained(output_dir)` —— 即**训练产物自带扩展后的 tokenizer**。
指回原始基座的话，SID 会被切成碎片（`<a_1>` → 6 个 token），Trie 全挂。这是 MiniOneRec 的既有设计。
> ✅ 2026-09-16 起 `evaluate.py --sid_vocab_path` 可在**评估端现场注册**，但**只用于 dry-run**
> （§3.5）；正式评估仍必须指向训练产物 —— `evaluate_run0.sh` 前置检查会用
> `grep '<a_0>' tokenizer.json` 拦住指错目录的情况（§3.6）。

> ⚠️ **诚实边界（未接线项，不影响 Run-0 可跑）**：`tasks/*.jsonl` 与
> `prompts/{alpaca,chatml}/*.jsonl` **目前没有消费方**。训练端 T1/T2/T3 的数据由 `data.py`
> 三个 Dataset 类**从 `index/` + CSV 现场重建**（`sft.py:353` 的 `SidItemFeatDataset` 用
> `item.json` + `index.json` 现拼 sid2title/title2sid 对；`sft.py:357` 的 `FusionSeqRecDataset` 同）。
> 明文 prompt 渲染产物是给后续「格式消融 / 训练端直读明文」预留的，暂未接线。
>
> 📦 **上云传什么**（训练端只读 `train/valid/test/*.csv` + `index/*.json` + `info/`）
> → **README §4.1**；本文件不重复那些数字。

### 3.3 分任务训练开关 `--tasks`（2026-09-16 落地）

训练端原本**无条件**把 3 路 Dataset 拼成一份（MiniOneRec 原样）。现改为 `--tasks` 可选，
**默认 `T1,T2a,T2b,T3` 全开，行为与 MiniOneRec 的 `ConcatDataset` 完全一致** —— Run-0 锚点不变。

`[实测]` 任务 ↔ 数据类 ↔ 规模（IandS；探针 `probe_task_switch.py` **已于 2026-09-18 移除**，数字为当时实测）：

| 键 | 数据类 | 任务 | 输入 → 目标 | 条数 | 占比 |
|---|---|---|---|---:|---:|
| `T1` | `SidSFTDataset` | seq2sid | 历史 SID 序列 → 目标 **SID** | 208,999 | 44.5% |
| `T2a` | `SidItemFeatDataset` | sid2title | SID → title | 25,847 | 5.5% |
| `T2b` | `SidItemFeatDataset` | title2sid | title → SID | 25,847 | 5.5% |
| `T3` | `FusionSeqRecDataset` | seq2title | 历史 SID 序列 → 目标 **title 文本** | 208,999 | 44.5% |
| | | | **合计（默认全开）** | **469,692** | 100% |

> 注意 **T1 与 T3 共用同一份 `train CSV`**（同一条用户序列），只是目标空间不同
> （SID vs title）—— 这正是 concat 它们的理由：让模型同时学会"SID 序列 ↔ title"两个方向。

用法（输出目录**按任务组合自动加后缀**，消融跑不会覆盖 Run-0 产物）：

```bash
bash sft_run0.sh                  # 默认四路      -> outputs/sft_IandS_run0
TASKS=T1 bash sft_run0.sh         # 单任务消融    -> outputs/sft_IandS_T1
TASKS=T1,T3 bash sft_run0.sh      # 双任务        -> outputs/sft_IandS_T1-T3
```

🔴 **三条必须知道的限制**：

1. **T2a/T2b 拆不开**：二者由同一个 `SidItemFeatDataset` 在一次构造里同时产出
   （`data.py:711-725` 两个循环，各 25,847 条）。`selected` 只给一半时 `resolve_tasks`
   （`sft.py:147`）会打 WARN，**实际两路都会进训练集**。要精确拆分需改 `data.py`。
2. **验证集恒为 T1**：`val_data` 固定 `SidSFTDataset(valid CSV)`，不随 `--tasks` 变 ——
   否则 val_loss 跨 run 不可比（评测口径 `EvalSidDataset` 就是 T1）。
3. 🔴 **只有 T1 进主指标**：`evaluate.py:163` 的 `EvalSidDataset` 只测 seq2sid
   （Trie 约束生成 3 SID → HR@K / NDCG@K）。所以"分别跑"的价值 =
   **消融定位每个辅助任务对 T1 主指标的贡献**，而不是分别得到各自的指标；
   单跑 T2a/T3 训出的模型，也只能拿它去跑 T1 评测看跨任务迁移。
   → 与 §6.5 执行队列「每项只改一个变量」配套使用。

> **未接线（本仓 `data.py` 无对应类，要用需新写）**：T4 text2sid
> （`tasks/text2sid.jsonl`，25,847 条）；注释里的 `SFTData`（与 T1 同构）、
> `TitleHistory2SidSFTDataset`（历史用 title 而非 SID，`data.py:1337`）。

### 3.4 约束解码：训练端 vs 评估端各需要什么（2026-09-16 实测）

**结论**：**约束束搜索只在评估端需要；训练端不需要，也不该加。**

| 环节 | 用约束解码？ | 实现位置 |
|---|---|---|
| 训练（`sft.py`） | ❌ 不用 | 无 —— 只有 `Trainer`（`sft.py:393`），全程没有 `generate` |
| 评估（`evaluate.py`） | ✅ 用 | `LogitProcessor.py:24 ConstrainedLogitsProcessor`，挂载在 `evaluate.py:206-212`，默认 `num_beams=50` |
| 零训练基线（`baseline/generative/sid_gr.py`） | ✅ 用，**另一套独立实现** | numpy 逐层 mask `prefix_allow` + 手写 beam（`baseline/generative/sid_gr.py:131-166`） |

**训练端为什么不需要**：SFT 是 teacher forcing —— label 由数据给定，模型不"选" token，
不存在生成出非法 SID 的机会。约束解码解决的是"**自由生成时如何保证输出合法**"，
这个问题在训练时根本不存在；硬加只会污染 loss。

> 🔴 **真正必须两边一致的是「prompt 模板 + 目标格式」，不是束搜索本身。**
> 训练端靠**数据**教约束（target 本身就是合法序列 `[a,b,c,\n,EOS]`），
> 评估端靠 **Trie** 强制约束。两者必须描述同一件事，否则训练学到的东西在推理时用不上。

`[实测]` 探针 `scripts/sft/probe_constrained_decoding.py`（直接实例化 `data.py` 的真实类，**不复刻模板**）：

```
A prompt 一致性 : SidSFTDataset vs EvalSidDataset   逐 token 一致（len=85）
B Trie 形状     : 5 步  [256, 98, 1, 1, 1]
                    step0  key="### Response:\n"      -> 允许全部 256 个 <a_k>
                    step1  key="<a_37>"               -> 允许 98 个 <b_k>
                    step2  key="<a_37><b_32>"         -> 允许 1 个 <c_k>
                    step3  key="<a_37><b_32><c_60>"   -> 只允许 \n (198)
                    step4  key="..<c_60>\n"          -> 只允许 EOS (151645)
C prefix_index  : "### Response:\n" 三种 encode 路径均 = 3 token [14374, 5949, 510]
```

B 段与训练 target `[<a>,<b>,<c>,\n,EOS]` **逐位对应** —— 这就是 §6.4(2)
「SID 定长 3 层 → 终止靠 Trie 不靠 EOS」在代码层面的落地证据。

⚠️ `prefix_index=3`（`LogitProcessor.py:41` 硬编码）的前提是 prompt 末尾恰为
`"### Response:\n"` 且切成 3 个 token —— C 段实测成立。**换基座 / 换 tokenizer 必须重测这条。**

🔴 **本轮抓到的上游遗留 bug（已修）**：`data.py:623-631` 的 `EvalSidDataset.get_history`
原文把输入句式注释掉、换成了另一句，而三个训练类
（`:375 SidDataset` / `:416 SidSFTDataset` / `:507 SidSFTDataset_GPR`）用的是原句 ——
**train/eval prompt 不一致**：

```
训练端: "The user has interacted with items {history} in chronological order.
         Can you predict the next possible item that the user may expect?"
评估端: "Can you predict the next possible item the user may expect,
         given the following chronological interaction history: {history}"   <- 原版
```

共同前缀仅 **49** token、总长差 **4**。指令部分（`:425 / :516 / :638`）三处相同，
所以模型能**部分泛化、不会崩到 0** —— 但会静默掉点，且从指标上很难察觉。已统一回训练端口径。

> 防复发：`evaluate_run0.sh` 前置检查已接入该探针（跳过用 `SKIP_PROBE=1`）。

### 3.5 不训练也能评测吗？—— 能跑通，但数字没意义（2026-09-16 实测）

**结论分三层**：

| 问题 | 答案 |
|---|---|
| 词表 + `evaluate.py` 写对了，不训练能不能跑评测/推理？ | **能** —— 纯工程问题，与训练无关 |
| 跑出来的指标有意义吗？ | **没有** —— Trie 只保证"**合法**"，不保证"**正确**" |
| 那这个能力有什么用？ | **evaluator 冒烟测试** + **随机下界锚点** |

`[实测]` 用**完全未训练**的 `models/Qwen3-0.6B` 现场注册词表，跑真实 `evaluate.py`：

```bash
# 用通用入口（EXP_ID 显式给，因为这里用的是原始基座而不是训练产物）
EXP_ID=dryrun-untrained MODEL_PATH=models/Qwen3-0.6B \
  SID_VOCAB_PATH=data/Amazon23/IandS/sft/info/sid_vocab.json \
  MAX_SAMPLES=300 NUM_BEAMS=20 bash evaluate_run0.sh
```

```
[SID] 注册 768 个 SID token（新增 768）；tokenizer=152437  模型 embedding 行数=151936
[SID] resize -> 152437（⚠️ 新增行是随机初始化 —— 只有未训练的基座才会走到这里）
[DRY-RUN] 只取 300 条（seed=42 随机采样）
75/75 [10:14]              <- 300 条 / batch=4 / beam=20，在 3050Ti 上约 10 分钟

calc.py:
  NDCG: [0. 0. 0. 0. 0.]
  HR  : [0. 0. 0. 0. 0.]   <- K = 1/3/5/10/20 全 0
  CC  : 0                  <- 生成但不属于 item_dict 的 SID 数 = 0
```

**两个读法**：

1. **`CC = 0` 是正面结论**：768 个新增 token 注册正确、Trie 每步掩码正确、
   `batch_decode(skip_special_tokens=True)` 没把 SID 抹掉、SID → 物品可映射 ——
   **整条评估链路正确**。这一步不依赖训练，所以可以（也应该）在烧 GPU 前做。
2. **`HR = 0` 是预期结果**：随机命中率 = `20/25847 = 0.077%`/条，300 条期望命中 **0.23** 次，
   实测 0 次完全落在随机区间内。

生成样例（未训练模型；样本目标 `<a_19><b_139><c_135>`）：

```
beam[0]  = <a_2><b_33><c_179>     未命中
beam 前3 = [<a_2><b_33><c_179>, <a_7><b_27><c_95>, <a_7><b_27><c_148>]
```

可见 beam 内部有明显前缀偏好（`<a_7><b_27>` 连续出现）—— 那是随机初始化 embedding 的偶然偏置，不含语义。

**随机下界锚点**（同一 test 口径，IandS）：

| 模型 | HR@10 | 来源 |
|---|---:|---|
| **未训练 Qwen3-0.6B**（beam=20） | **0.0000** | 本节实测，300 条 |
| `sid_gr`（零 LLM，但训练了自己的解码器） | 0.0168 | `EVAL_PROTOCOL §3.4` |
| `sasrec`（全库排序口径） | 0.0395 | `baseline/RESULTS.md` |

⟹ **0.0000 与 0.0168 之间的差，就是"训练"这件事的量化贡献起点。**

#### 3.5.1 `[实测]` 换 chatml 后复跑（2026-09-18，模板收敛的验收）

模板从"3 份实现"收敛为 `config/prompt_templates.json` + `prompt_templates.py` 后，用**同一套未训练基座**
路径复跑，专门验「响应前缀必须从真源取」这个改动（漏改 = 约束静默失效）：

```bash
SID_VOCAB_PATH=data/Amazon23/IandS/sft/info/sid_vocab.json \
MODEL_PATH=models/Qwen3-0.6B EXP_ID=IandS-untrained-tplcheck \
MAX_SAMPLES=32 BATCH_SIZE=2 NUM_BEAMS=20 bash evaluate_run0.sh      # 实测 1m41s
```

| 检查项 | 结果 |
|---|---|
| `evaluate_run0.sh` 前置闸门（`probe_constrained_decoding.py`） | ✅ 通过（prompt 逐 token 一致 / Trie 5 步） |
| 现场注册词表 | ✅ 768 个；`tokenizer` 152437 / `embedding` 151936 → 触发 resize |
| **`predict` 候选全为合法 `<a_x><b_y><c_z>`** | ✅ **640/640 = 100%** |
| **首个候选是空串的样本数** | ✅ **0 / 32** |
| `calc.py` 的 `CC`（= metrics 的 `n_generated_not_in_item_dict`） | ✅ **0** |
| `No valid tokens found for hash_key` 告警 | ✅ **0 次** |
| HR@1..20 / NDCG@1..20 | 0.0（未训练，预期见上文） |

🔴 **为什么"100% 合法"能证伪"前缀写错"**：`LogitProcessor.py:58-66` 查不到 key 时是**强制 EOS 后
`continue`** —— 生成会变成**空串**，而不是畸形 SID。所以只要有一条合法生成，就说明**第 0 步的
`hash_key = prompt[-3:]` 在 Trie 里查到了**。探针同时给出闭环：Trie 的 step0 key =
`151644-77091-198`（正是 chatml 前缀）、候选 256；评估端 prompt 末 3 token 也是 `[151644, 77091, 198]`。

> 反例（本该失败的样子）：不给 `SID_VOCAB_PATH`、把 `MODEL_PATH` 指回 `models/Qwen3-0.6B`，
> 前置检查会直接拦下并打印正确命令（`tokenizer.json` 里没有 `<a_0>`）。已实测。


> 📁 **该锚点已作为一个正式版本落盘**（`RUN_TAG=untrained`），产物在
> `results/sft/IandS-untrained/`，与训练后的结果同一套命名、可直接并列比较：
> ```bash
> EXP_ID=IandS-untrained MODEL_PATH=models/Qwen3-0.6B \
>   SID_VOCAB_PATH=data/Amazon23/IandS/sft/info/sid_vocab.json \
>   MAX_SAMPLES=300 NUM_BEAMS=20 bash evaluate_run0.sh
> ```
> 文件名带 `_n300` ⟹ 与将来**全量**（无 `_n` 后缀）的结果天然区分、不会覆盖。
> `meta.json` 里 `registered_at_eval=true` / `base_model_has_sid_token_map=false`
> 会如实标记"这是未训练 + 现场注册"，不会与训练产物混淆。
>
> ⚠️ **本地跑的显存约束（实测踩坑）**：显存由 **`BATCH_SIZE × NUM_BEAMS`**
> （beam 展开后的序列总数）决定，**不是 batch 单独决定**。4GB 卡（3050Ti）：
> `batch4 × beam20 = 80` 条序列 ✓ 跑得动；`batch8 × beam50 = 400` 条 → **CUDA OOM**。
> 本地想跑就压 **batch**，**别压 beam**（beam 宽度 = HR@K 的硬上限，压它等于自降天花板）。
> 锚点用 `beam20` 就够 —— 未训练模型的 HR 在所有 K 上恒为 0，与 beam 宽度无关。
>
> ⚠️ **别在脚本运行期间改它**：bash 是流式读文件的，编辑到一半被读到会报
> `syntax error near unexpected token`（本轮实测踩到；事后 `bash -n` 却是通过的，
> 因为文件最终是完整的 —— 这种"检查不出来"的失败最迷惑人）。要改就等任务结束。

⚠️ **beam 宽度是硬天花板**：生成式的 HR@10 上界 = `beam_ceiling@10`。
`sid_gr` 在 beam=20 时 `beam_ceiling@20` 才 0.0312（**< sasrec 0.0395**）——
所以 sasrec 这条达标线**光靠"beam 内排序"打不过**，必须真的提高"能不能生成出来"。
`evaluate_run0.sh` 默认 `--num_beams 50` 比 sid_gr 宽，是留了 headroom。

⚠️ **MRR 对生成式禁止横向比较**（`EVAL_PROTOCOL §3.4`：MRR ≈ 1/beam 是结构常数）。

### 3.6 实验命名规范：一个 `EXP_ID` 串起训练与评估（2026-09-16）

**动机**：跑消融时最怕"这个 `final_result.json` 到底是谁的结果"。原先
`evaluate_run0.sh` 用 `basename(MODEL_PATH)` 当结果目录名，而 `MODEL_PATH` 末级恒为
`final_checkpoint` ⟹ **所有版本的评估结果写进同一个文件、互相覆盖**（本轮已修）。

**规范**：

```
EXP_ID = <域>-<RUN_TAG>[-<任务集>]      例 IandS-run0 / IandS-run0-T1T3 / IandS-S0
```

| 产物 | 落点 |
|---|---|
| 模型权重 | `outputs/<EXP_ID>/final_checkpoint/` |
| 评估结果 | `results/sft/<EXP_ID>/eval_<域>_beam<B>[_n<N>].json` |
| **版本元数据** | `results/sft/<EXP_ID>/eval_<域>_beam<B>[_n<N>].meta.json` |
| **指标（HR / NDCG）** | `results/sft/<EXP_ID>/eval_<域>_beam<B>[_n<N>].metrics.json` |
| 日志 | `logs/sft/<EXP_ID>/` |

> 🔴 **`results/` 按阶段分层**：`results/sid/`、`results/sid_e5000/`、
> `results/alignment_methods/`、`results/fusion_rank.json` 都是 **SID 阶段**的产物
> （由 `rq/train_rqvae.py`、`rq/build_sid_dual.py`、`scripts/multimodal/*.py` 写）；
> **SFT 阶段一律收在 `results/sft/` 下**，两者互不混淆。日志同理走 `logs/sft/`。

- `RUN_TAG` 默认 `run0`，可任意取：`S0` / `S1` / `base-cmp` …
- 任务集后缀**只在非默认时出现**（默认四路全开不加后缀，Run-0 名字保持干净）
- `_n<N>` **只在 `MAX_SAMPLES>0` 时出现**，dry-run 结果不会与全量结果混
- 评估端**自动从 `MODEL_PATH` 反推 `EXP_ID`**，两边命名天然一致，不用手填

**`meta.json` 记什么**（这就是"一眼看出是哪个版本"的答案）：

```json
{ "exp_id": "IandS-run0",
  "domain": "IandS",
  "base_model": "outputs/IandS-run0/final_checkpoint",
  "base_model_has_sid_token_map": true,
  "registered_at_eval": false,
  "n_items": 25847, "num_beams": 50, "max_samples": 0, "max_new_tokens": 16,
  "git_commit": "c2326a7", "started_at": "2026-09-16T20:27:08+08:00" }
```

- `base_model_has_sid_token_map` = 训练产物标记（`sft.py:302` 落盘的 `sid_token_map.json`）
- `registered_at_eval` = 是否走了 dry-run 的现场注册（正常评估应为 `false`）

🔴 元数据**不写进 result json** —— `calc.py` 假设它是 `list[dict]`
（`json.load` 后直接 `for sample in test_data`），塞元数据会破坏解析。所以落在旁边的
`.meta.json`；`.metrics.json` 由 `scripts/sft/eval_report.py` 解析 `calc.py` 的 stdout 得到
—— **不改 calc.py，指标口径保持单一实现**。

**用法**（换版本只动环境变量，不改脚本）：

```bash
# 训练
bash sft_run0.sh                     # -> outputs/IandS-run0/            (EXP_ID=IandS-run0)
TASKS=T1 bash sft_run0.sh            # -> outputs/IandS-run0-T1/
RUN_TAG=S0 bash sft_run0.sh          # -> outputs/IandS-S0/

# 评估（EXP_ID 自动反推，不用手填）
bash evaluate_run0.sh                # 读 outputs/IandS-run0/final_checkpoint
MODEL_PATH=outputs/IandS-run0-T1/final_checkpoint bash evaluate_run0.sh
EXP_ID=my-exp MODEL_PATH=/abs/path/to/ckpt bash evaluate_run0.sh     # 完全显式

# 只看 HR / NDCG
cat results/sft/IandS-run0/eval_IandS_beam50.metrics.json
```

⚠️ **只报告 HR / NDCG，不报告 MRR**：生成式的候选集 = beam 内 SID，
`MRR ≈ 1/beam` 是结构常数，不携带排序质量信息（`EVAL_PROTOCOL §3.4` 已标 `n/a`）。
🔴 `--num_beams` 别调小 —— **beam 宽度就是 HR@K 的硬上限**（`beam_ceiling`）。

**评估脚本的健全性检查**（`evaluate_run0.sh` 前置）：

| 检查 | 判据 |
|---|---|
| 上游数据齐 | `test/*.csv` + `info/*.item_info.txt` 存在 |
| 模型目录存在 | 否则提示先 `bash sft_run0.sh` |
| **tokenizer 里真有 SID** | `grep '<a_0>' .../tokenizer.json`（**比"目录存在"强得多**）<br>没命中就说明指的不是训练产物；dry-run 请显式给 `SID_VOCAB_PATH=` |
| 训练/评估口径一致 | 跑 `probe_constrained_decoding.py`（§3.4，`SKIP_PROBE=1` 跳过） |

### 3.7 结果记录：`docs/SFT_EVAL_RESULTS.md`（入 git 的汇总表）

**为什么要它**：`results/` 与 `logs/` 都被 `.gitignore` 忽略 ⟹ **明细不入仓**。
重装环境、换机器、或隔几周回看，对照就丢了。所以另有一张**入 git 的汇总表**
（做法与 [baseline/RESULTS.md](../baseline/RESULTS.md) 一致：明细不入仓、汇总自动生成入仓）。

**流程**：

```bash
# ① 评估：自动落 .json（逐条预测）/ .meta.json（版本+时间+耗时）/ .metrics.json（HR/NDCG）
bash evaluate_run0.sh

# ② 汇总：扫 results/sft/ -> 重写 docs/SFT_EVAL_RESULTS.md
./.venv/Scripts/python.exe scripts/sft/collect_eval_results.py
```

`collect_eval_results.py` 扫描 `results/sft/*/eval_*.{meta,metrics}.json`，按同名主干配对后输出表格。

**每次评估记录什么**：

| 字段 | 来源 | 说明 |
|---|---|---|
| EXP_ID | 目录名 | 版本标识（§3.6） |
| 模型版本 | `meta.base_model` | 指向训练产物目录 |
| **推理时间** | `meta.started_at` | **这次评估是什么时候跑的**（`YYYY-MM-DD HH:MM`） |
| 样本 / beam | `metrics.n_evaluated` / `meta.num_beams` | 读 HR@K 前**必须先看 beam**（它是硬上限） |
| HR@1/5/10/20 | `metrics.HR` | 只列实际算出的 K（`calc.py` 按 beam 宽度裁剪） |
| NDCG@10 | `metrics.NDCG` | |
| 耗时 | `meta.eval_seconds` | `evaluate.py` 墙钟时长；括号内是**每样本**值，跨 batch/beam/机器可比 |
| commit | `meta.git_commit` | 代码版本，保证可追溯 |
| 备注 | 派生 | 「现场注册(dry-run)」/「非训练产物」/「抽样」等标签 |

⚠️ 表中**不报告 MRR**（生成式下 `≈1/beam` 是结构常数，见 §3.4）。
🔴 表头已写死两条口径红线：**beam 内排名 ≠ 全库排序**、**`HR@K` 的上限 = beam 宽度**。

> 表是**自动生成**的，别手改 —— 下次跑收集脚本会覆盖。要加字段就改
> `scripts/sft/collect_eval_results.py`。

### 3.8 本地冒烟：4GB 卡能不能跑起来（LoRA 双轨 + 实测）

**结论：能。** 全参跑不动（0.6B 全参 + Adam 状态约 24GB），但两条**冒烟路径**都能在本地跑通，
用来验证"训练链路是否完好"——在烧 3090 之前把问题挡掉。

| 方式 | 实际训练什么 | `[实测]` trainable | 用途 |
|---|---|---:|---|
| `FREEZE_LLM=True` | 只训 768 个新 SID token 的 embedding 行（grad mask） | 156M（26.2%） | 最省；它就是 S0 warmup 的配置（§6.4(1)） |
| `USE_LORA=True` | LoRA 挂 attention 投影 + `embed_tokens`/`lm_head` | 321M（35.0%） | `UPGRADE_PLAN §5.2` 双轨的**本地那一路**；§5.3 消融矩阵要用 |

```bash
# 冒烟：16 条样本 / 1 epoch / 16 步，约 1.5 分钟（3050Ti 4GB）
SAMPLE=16 MICRO_BATCH_SIZE=1 BATCH_SIZE=1 NUM_EPOCHS=1 CUTOFF_LEN=192 \
TASKS=T1 USE_LORA=True EVAL_FRAC=16 OUTPUT_DIR=outputs/_smoke_lora \
bash sft_run0.sh

# 评估该冒烟产物（可选，验证"训练 -> 评估"闭环）
EXP_ID=_smoke_lora_eval MODEL_PATH=outputs/_smoke_lora/final_checkpoint \
  MAX_SAMPLES=20 NUM_BEAMS=5 BATCH_SIZE=4 SKIP_PROBE=1 bash evaluate_run0.sh
```

`[实测]` LoRA 冒烟（2026-09-16）：

```
[LoRA] r=32 alpha=64 dropout=0.05 targets=['q_proj','k_proj','v_proj','o_proj']
trainable params: 321,366,016 || all params: 917,928,960 || trainable%: 35.0099
loss: 15.34 -> 15.27 -> 14.41 -> 12.84 -> 11.64 -> 10.34 -> 9.93 -> 9.10 -> 8.43 -> 7.57
      (16 步单调下降，无 OOM，exit=0，耗时 1m41s)

final_checkpoint/：完整权重 1.5G、**无 adapter_config.json 残留**、
  tokenizer=152437 / embedding 行=152437 / SID 编码 [151784,151976,152414] 全部对齐
  ⟹ evaluate.py（不做任何 peft 解析）**可直接加载**
```

**三个必须知道的实现点**：

1. 🔴 **`modules_to_save=["embed_tokens", "lm_head"]` 不是可选项**。SID 是**新增 token**，
   而 LoRA 只挂在 attention 投影上 —— 不显式纳入的话，768 个 SID token 的 embedding
   **会永远停在随机初始化**，模型学不会输出 SID（而 T1 的目标正是它们）。
   代价：可训参数从 ~9M 涨到 321M（tie_word_embeddings 下二者仍被各存一份）。
2. 🔴 **训练后必须 `merge_and_unload` 再保存**。LoRA 下 `trainer.model` 是 PeftModel，
   直接 `save_pretrained` 只落 adapter，而 `evaluate.py` 是用
   `AutoModelForCausalLM.from_pretrained` 原样加载的 ⟹ 会失败。已在 `sft.py` 保存段处理。
3. 🔴 **`fire` 会把命令行里的 `a,b,c` 解析成 tuple**（实测 `--tasks T1,T2a` 到手是
   `('T1','T2a')`）。所以任何"逗号分隔参数"都**不能**用 `str(v).split(",")`
   —— 那会得到 `["('T1'", " 'T2a')"]` 这种脏元素。已抽 `sft.py parse_csv_list()`
   统一兼容 str / tuple / list。⚠️ **默认的 `TASKS=T1,T2a,T2b,T3` 原本就是这么崩的**：
   训练直到 2026-09-16 才第一次真正启动，之前所有自检都没走到 `train()`。
   （回归用例原在 `scripts/sft/probe_task_switch.py`，该探针已于 2026-09-18 移除。）

**本地冒烟的两个额外坑（本机沙箱特有，云端没有）**：

- ⚠️ **`EVAL_FRAC` 要给绝对步数**。Trainer 配了 `save_total_limit=1`，每次保存都要删旧
  checkpoint（一次 60 个文件），而本机有 safe-delete 保护 ⟹ 被拦下并**中断训练**
  （实测两次 exit=1）。给 `EVAL_FRAC=<总步数>` 让它只 save 一次即可。
  注意 Trainer 的语义是 **`<1` = 占训练步数的比例、`>=1` = 绝对步数** ——
  传 `1.0` 等于"每 1 步"，不是"每 epoch"。
- ⚠️ **每个 checkpoint 1.8–3.1 GB**（`optimizer.pt` 是大头）。跑完记得清理 `outputs/_smoke_*`，
  否则几次冒烟就是十几 G。

---

### 3.9 Checkpoint 保存与 eval 节奏（2026-09-16 读源码 + 实测）

**结论：按步数存，不按 epoch 存。** `sft.py:470-476` 里 eval 与 save 全部用 `steps` 策略。

| 参数 | 值 | 说明 |
|---|---|---|
| `eval_strategy` / `save_strategy` | `steps` | 按步，不按 epoch |
| `eval_steps` / `save_steps` | `eval_frac`（**同一个变量**） | 语义见下 |
| `logging_steps` | 1 | 每步打 `{loss, grad_norm, learning_rate}` |
| `save_total_limit` | 1 | 运行时会被抬到 2（见「安全网」） |
| `load_best_model_at_end` | True | 训练结束回滚到最优 ckpt |
| `metric_for_best_model` | **`'loss'`**（自动默认） | 未传 `compute_metrics` ⟹ 按 `eval_loss` 选优；`greater_is_better=False` |
| `save_only_model` | False（默认） | ⟹ checkpoint 里含 **`optimizer.pt`**（1.8–3.1 GB/个的成因） |
| 早停 | `EarlyStoppingCallback(patience=3)` | 盯同一个 `eval_loss` |

**`eval_frac` 的语义（最容易踩）**：Trainer 的规则 = **`< 1` 当比例、`>= 1` 当绝对步数**
（`.venv/Lib/site-packages/transformers/trainer_callback.py:157-168` `TrainerState.compute_steps` → `ceil(max_steps × 比例)`）。
⚠️ **传 `1.0` 不是"每 epoch 一次"，而是"每 1 步"**。

`[实测]` 各场景的实际间隔（用真实 `TrainingArguments` + 真实 `compute_steps` 算，不是手算）：

| 场景 | 样本 | micro | gacc | 更新步/epoch | max_steps | **eval/save 间隔** | 存几次 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 云端默认 `TASKS=all` | 469,692 | 4 | 16 | 7,339 | 22,017 | **1,101** | ~20 |
| 云端 `TASKS=T1` | 208,999 | 4 | 16 | 3,266 | 9,798 | **490** | ~20 |
| 云端 `SAMPLE=5000` | 5,000 | 4 | 16 | 79 | 237 | **12** | ~20 |
| 本地冒烟 `SAMPLE=16` | 16 | 1 | 1 | 16 | 16 | **1** | 16 |

> `max_steps = ceil(epochs × ceil(len_dataloader / grad_accum))`（`.venv/Lib/site-packages/transformers/trainer.py:5682-5689`）。
> 本地冒烟那行间隔 = 1，正是实测被 safe-delete 拦下、中断训练的原因（每步存一个 1.8 GB ckpt）。

🔴 **安全网**：`save_total_limit=1` + `load_best_model_at_end=True` 本来会把**最优** ckpt 一起删掉；
`.venv/Lib/site-packages/transformers/trainer.py:4405-4413` 检测到这种情况后**自动把上限抬到 2** ⟹ 磁盘上最多留 2 个 checkpoint。

🔴 **落盘有两份**（`sft.py:491-504`，顺序不能反）：
1. `trainer.save_model(outputs/<EXP_ID>/)` → 根目录一份（含 tokenizer）
2. 再存 `outputs/<EXP_ID>/final_checkpoint/` → **`evaluate.py` 指的是这里**

⚠️ **LoRA 下根目录那份是 adapter，不是完整模型**：`trainer._save` 把 `PeftModel` 也算进
`supported_classes`（`.venv/Lib/site-packages/transformers/trainer.py:4311`）→ 走 `save_pretrained` → 只落 `adapter_config.json` +
`adapter_model.safetensors`。合并后的完整权重**只在 `final_checkpoint/`**。
⟹ **`--base_model` 永远指 `final_checkpoint/`。**

⚠️ **每次 eval 跑的是全量验证集**（IandS `valid` = 50,984 条 ⟹ 12,746 个 eval step/次）；
场景 A 会 eval ~20 次 ⟹ 累计约 25.5 万次前向。嫌慢就**调大 `EVAL_FRAC`**（间隔变大 = 次数变少）。

⚠️ 两者必须协调：`load_best_model_at_end=True` 且 `eval_steps`/`save_steps` 都 `>= 1` 时，
`save_steps` 必须是 `eval_steps` 的整数倍，否则 `TrainingArguments` 直接抛 `ValueError`
（`.venv/Lib/site-packages/transformers/training_args.py:1693-1700`）。本项目两者取**同一个变量**，天然满足。

---

### 3.10 本地 4GB 卡能训到什么程度（2026-09-16 实测）

**结论：能跑通、能出可用权重；但全量训练不可行 —— 与云端差 20 倍以上。**

`[实测]` 基准脚本 **`scripts/sft/bench_local_training.py`**（真实权重 + 与 `sft.py` 同款 LoRA +
真实 `SidSFTDataset` 出 batch，不靠估算器）。

环境：`RTX 3050 Ti Laptop` **4.00 GiB**（运行时 free 3.23 GiB）· `cutoff_len=320` · `micro_batch=1`。
实测序列长度 `min 90 / p50 100 / p90 180 / max 180` —— **远短于 320 ⟹ `cutoff_len` 对 T1 不构成瓶颈**。

| 优化器 | 峰值显存 | 是否换页 | s/步（batch=1） | T1 全量 3 epoch 外推 |
|---|---:|:---:|---:|---:|
| `adamw_torch`（**`sft.py` 现状**） | **4.26 GiB** | ⚠️ **是**（> 4.00） | **1.140** | **198.5 h（8.3 天）** |
| `adamw_bnb_8bit` | 3.05 GiB | 否 | 0.332 | 57.8 h（2.4 天） |
| `sgd`（仅诊断，无优化器状态） | 2.44 GiB | 否 | 0.279 | 48.5 h（2.0 天） |

🔴 **关键机制**：峰值 4.26 GiB **超过显卡物理显存 4.00 GiB** —— Windows WDDM 允许超额分配，
于是**落到共享内存（换页）**，表现为"不 OOM 但奇慢"。把优化器状态拿掉（SGD）峰值降到 2.44 GiB，
单步 1.140 → 0.279 s ⟹ **4.1 倍差距全在换页，不是算力**。
（根因是 LoRA 的 `modules_to_save=[embed_tokens, lm_head]` 让可训参数到 321 M，
AdamW fp32 状态 ≈ 2.57 GiB。）

**eval 是第二个瓶颈**：`sft.py:462` 把 `per_device_eval_batch_size` 绑死在 `micro_batch_size` 上，
micro=1 时**每次 eval 要跑 50,984 步（82 分钟）**；默认 20 次 eval ⟹ **+27.4 h**。

⟹ `[实测]` 合计：AdamW **225.7 h（9.4 天）**；换 8-bit Adam 也要 **85.2 h（3.5 天）**。
云端 3090（24 GiB / micro=4 / batch=64）是**小时级**。

**本地能做的两件事**：

1. **流程冒烟**（已验证）：`SAMPLE=16 MICRO_BATCH_SIZE=1 BATCH_SIZE=1 NUM_EPOCHS=1 CUTOFF_LEN=192
   USE_LORA=True EVAL_FRAC=16` → 1m41s / `exit=0`
2. **小样本真实训练**：`SAMPLE=2000 NUM_EPOCHS=1` → 约 11 min（8-bit Adam）/ 38 min（AdamW）

🔴 **本地跑必须跳过 eval**，否则 82 分钟的 eval 比训练本身还长：
把 `EVAL_FRAC` 给一个**大于总步数**的值即可（`steps` 策略 + `eval_steps > max_steps` ⟹ 一次都不 eval）。
安全性已核源码：`.venv/Lib/site-packages/transformers/trainer.py:2811` 是
`if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:` ——
从没 eval 过时 `best_model_checkpoint` 恒为 `None`（`.venv/Lib/site-packages/transformers/trainer_callback.py:109`），逻辑短路、
**不报错**，训练照常结束并落 `final_checkpoint/`。

**两个可选改进**（均未实施，按需启用）：

- 把 `optim` 变成参数（`sft.py:469` 现写死 `"adamw_torch"`）→ 本地可切 `adamw_bnb_8bit`，
  峰值 4.26 → 3.05 GiB，单步 1.140 → 0.332 s
- 把 `per_device_eval_batch_size` 与 `micro_batch_size` 解耦（`sft.py:462`）→ eval 从 82 min
  降到 ~10 min 量级（batch 放大 8 倍）

⚠️ 这两项只对本地有意义；**全量训练一律上云**（3090 不受 4 GiB 限制）。

> 🔴 **换卡 / 换精度前先读 [`PRECISION_GUIDE.md`](PRECISION_GUIDE.md)**：`--precision` 已参数化
> （`bf16` 默认 / `fp16` 给 V100 等无 bf16 的卡 / `fp32` 调试）。⚠️ **fp16 模式的显存约为 bf16 的 2 倍**
> （AMP 要求 fp32 主权重）——本节实测 4.26 GiB（bf16）vs 8.23 GiB（fp16）。

---

## 4. 体检实测数字（2026-09-14；原 `verify_sft_data.py` 与产物 `sft_verify.json` 已于 2026-09-18 移除）

| 检查项 | IandS | VG | 判定 |
|---|---:|---:|---|
| ① 768 个 SID token 是否各占 **1 个 token** | ✅ | ✅ | 抽 21 个全为单词元 |
| ② CSV ↔ index.json 往返一致 | 0 条不一致 | 0 条不一致 | ✅ |
| ③ T1 prompt 均值 / p95 / **max** | 130.7 / 175 / **175** | 125.1 / 175 / **175** | — |
| ③ T1 **total max**（含 target+EOS） | **179** | **179** | — |
| ③ T2 total max | 132 | 99 | — |
| ③ T3（同 T1 结构） | ≤179 | ≤179 | — |
| ③ T4 total max（**抽样值，已作废**） | ~~309~~ | ~~221~~ | ⚠️ 见下行 |
| ③ T4 total max（**全量真值**） | **391** | **362** | 决定 cutoff |
| ④ 首条样本 SID token 数 | 9 = 6 历史 + 3 目标 | 9 | ✅ |
| ⑤ MiniOneRec 三个 Dataset 类直接可用 | ✅ | — | 见下 |

**⑤ 兼容性实测**（`[实测]`）：用本仓 `data.py` 原样实例化，无需改一行：

```
SidSFTDataset      n=200  keys=['input_ids','attention_mask','labels']
SidItemFeatDataset n=200
FusionSeqRecDataset n=200
label 段解码 = ['<a_96>', '<b_200>', '<c_175>', '\n', '<|im_end|>']
```

**结论**：`sft.py` 可以直接跑，只需把 `sid_index_path` / `item_meta_path` / `train_file`
指向我们的新路径，并把 `cutoff_len` 从 512 降到 **400**（只训 T1/T2/T3 时 320 即可）。
注意 MiniOneRec 的截断是 `tokens[-max_len:]`，**从左侧砍**，超长样本会丢掉 instruction 头部。

### 4.1 两处必须改的配置（否则白烧显存）

| 项 | V0 | 本项目 | 理由 |
|---|---|---|---|
| `cutoff_len` | 512 | **400**（不带 T4 可 320） | `[实测]` 全量：T1 ≤180、T3 ≤277、**T4 ≤392**；512 有 40% 是纯 padding |
| `category` 参数 | `Industrial_and_Scientific` 等 5 个硬编码 | **IandS 单域实验无需改动**；上 VG 时要在 `sft.py:198` **和** `evaluate.py:56` **两处**都加 `"Video_Games": "video games"` | `[代码]` 两个脚本的 `category_dict` 都只有 5 个键，VG 会 `KeyError` |

> ⚠️ **修正一处先前的数字**：上表 ③ 曾报"T4 最长 309"，那是体检脚本**抽样** ≤n_probe 条的结果；
> 全量真实 max 是 **391（IandS）/ 362（VG）**，超 320 的分别有 18 / 3 条。
> 抽样在长度分布的长尾上不可靠 —— 定 `cutoff_len` 这种"取 max"的场合必须用全量。

### 4.2 超长 T4 **不用重生成数据集**，改 `cutoff_len` 即可

定 `cutoff_len` 之前先确认它在哪一层生效，否则会做无用功：

| 层 | 是否截断 | 证据 |
|---|---|---|
| `prepare_sft_data.py`（构造四任务） | ❌ 只按**字符**截（`--max_text_chars 512`），不管 token | 无 tokenizer |
| ~~`build_sft_prompts.py`（渲染明文）~~ | 🗑️ **脚本已删**（2026-09-18） | 它另有一条「从不截断」的事实，随脚本一并作废 |
| **`data.py:190-197`（训练端）** | ✅ `tokens[-max_len:]` —— **从左侧砍** | 唯一真正的截断点 |

所以数据集里存的是**明文**，长度是训练时才决定的 —— **改参数就够了，不用重生成**。

**代价也确实为零**，因为 padding 是动态的（`[实测]`）：

```
sft.py:422  DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True)
→ 模拟 batch（120 / 391 / 130 / 200）→ 输出 shape (4, 392)
```

`padding=True` 是 **pad 到 batch 内最长再取 8 的倍数**，不是 pad 到 `cutoff_len`。
所以 `cutoff_len` 320→400 **对 T1/T2/T3 毫无影响**（它们 max 179/161/277，本来就 <320，padding 长度由 batch 内真实最长决定）。

| 方案 | 18/3 条超长样本 | 代价 |
|---|---|---|
| **A. `cutoff_len=400`（推荐）** | ✅ 完整保留 | 几乎为零：仅含 T4 长样本的 batch 会 pad 到 ~392 |
| B. 重生成 T4、砍短 text | ❌ 信息永久丢失 | 且 `--max_text_chars` 默认值不改的话重跑会复原 |

→ **选 A**。重生成只在"想把 max 压到 320 以省显存"时才值得，而动态 padding 下这点收益可以忽略。

### 4.3 padding 方向：left 还是 right（**已改 `sft.py` 为 right**）

**left padding = 在序列开头（提示词前面）补 pad**，效果是全部样本**右端对齐**；right padding 反之。

| 场景 | 该用 | 原因 |
|---|---|---|
| **训练** | **right** | 真实 token 从位置 0 起算，与预训练时的位置分布一致 |
| **批量生成** | **left**（必须） | decoder-only 每步取 `logits[:, -1, :]`；只有右端对齐时最后一列才是"真实的下一个 token 位置"。right padding 会让模型从 pad 后面接着生成，直接崩 |

MiniOneRec 是把生成端的设置（`evaluate.py:160`）顺手复制到了训练端（`sft.py`）造成的，属于复制粘贴遗留。

**对 Qwen3-0.6B：left 与 right 数学等价，`[实测]` 三条依据**

1. **无绝对位置编码** —— config 里没有 `position_embedding_type`，纯 RoPE；
   `sliding_window=None`、`use_sliding_window=False`、`rope_scaling=None`
2. **`position_ids` 不看 attention_mask** —— `modeling_qwen3.py:382-383` 是
   `position_ids = cache_position.unsqueeze(0)` 即 `arange(L)`；且 `[实测]` `DataCollatorForSeq2Seq`
   （transformers 4.57.1）**不生成 `position_ids`**，输出只有 `input_ids / attention_mask / labels`
3. **RoPE 的 attention 只依赖相对距离** `(m-n)` → 位置整体平移不改变任何 attention 值

→ 所以 left padding 训练 Qwen3 **不会掉点**（MiniOneRec V0 能跑通也印证了）。

**但仍然是 bug，已改**：它是复制粘贴而非设计，且只在"纯 RoPE + 无 sliding window + 无 rope_scaling"
这个特定组合下才等价，任一条不满足就会错。改成 right 不影响数字（等价），只是去掉一个非标准疑点。

**§实测 附带查证的一个坑（结论：安全）**：`tokenizer.pad_token = tokenizer.eos_token`
→ `pad_token_id == eos_token_id == 151645`。担心 collator 会把**真 EOS 的 label 一起屏蔽掉**
（那样模型就学不会停）。实测两种 padding 下真 EOS 都保留：

```
left : 短样本 labels = [-100 ×20, <a_5>, <b_23>, <c_66>, 151645]   ← 真 EOS 在末位，保留 ✅
right: 短样本 labels = [-100 ×5,  <a_5>, <b_23>, <c_66>, 151645, -100 ×15]  ← 保留 ✅
```

`DataCollatorForSeq2Seq` 只用 `label_pad_token_id(-100)` **填充**、不做 `==` 替换，所以安全。
（对比：`DataCollatorForLanguageModeling` 会执行 `labels[labels == pad_token_id] = -100`，那个才真会出事。）

### 4.4 `eos_token` 在两版权重间不同（151645 vs 151643）—— **只有换 `-Base` 才需要这一节**

基座**默认** `Qwen/Qwen3-0.6B`（post-trained），`-Base` 作对照（选择依据见 `UPGRADE_PLAN §5.1.1`）。
`[实测]` 两份 tokenizer_config：

| | **`Qwen3-0.6B`（post-trained，默认）** | `Qwen3-0.6B-Base`（对照） |
|---|---|---|
| `eos_token` | **`<\|im_end\|>` = 151645**（与 chat_template 天然一致） | `<\|endoftext\|>` = 151643 |
| `pad_token` | `<\|endoftext\|>` = 151643 | `<\|endoftext\|>` = 151643 |
| `chat_template` | `<\|im_start\|>` / `<\|im_end\|>` | **语义等价**（4,116 vs 4,168 字符） |

**默认（post-trained）不需要任何处理**：本文档 §1~§4 里所有"EOS = 151645"的实测直接成立。

⚠️ **只有改用 `-Base` 时才要动这一节**：它的 `eos` 是 `<|endoftext|>`(151643)，与 ChatML 模板不一致。
好在链路里三处**都取 `tokenizer.eos_token_id`**，所以在 tokenizer 上设一次即可全链路同步：

| 位置 | 取值方式 | 换 Base 后 |
|---|---|---|
| `data.py` completion 末尾 EOS | `tokenizer.eos_token_id` | 自动跟随 |
| `evaluate.py:109-110` Trie `ID.append(tokenizer.eos_token_id)` | 同上 | 自动跟随 |
| `sft.py:233` `tokenizer.pad_token = tokenizer.eos_token` | 同上 | pad 会跟着变（无影响，pad 位被 -100 屏蔽） |

**推荐做法（TRL 官方口径）**：训练时**把 eos 显式设成与 chat_template 一致的 `<|im_end|>`**。
TRL `SFTTrainer` 文档原文：

> it is necessary to align the EOS token with the chat template to ensure the model's responses terminate correctly.
> ... for example, for `Qwen/Qwen2.5-1.5B`, one should set `eos_token="<|im_end|>"`.

落到代码，拿到 tokenizer 后加一行（`sft.py` 与 `evaluate.py` 各一处）：

```python
tokenizer.eos_token = "<|im_end|>"   # 与 chat_template 对齐；Base 默认值是 <|endoftext|>(151643)
```

> 另一选择是接受 Base 默认的 `<|endoftext|>(151643)`。但那样 ChatML 的 assistant 段会以一个
> **非模板终止符**结束，格式不自洽。**换 Base 的话选前者。**
>
> ⚠️ `[实测]` **两版 tokenizer 只差 `eos_token` 与 `chat_template` 两个键**；
> `<|im_end|>`(151645) / `<think>`(151667) / `</think>`(151668) **两版词表里都有** → 改 eos 是纯配置动作，不涉及加词。
> 另有 `[实测]`：官方 **2507（Instruct-only 非思考版）只覆盖 235B-A22B / 30B-A3B / 4B，无 0.6B**
> → 想靠"换官方非思考版"绕开 thinking，在 0.6B 档位不存在。

### 4.5 基座权重下载（`scripts/download_base_models.{ps1,sh}`）

定案：**student = `Qwen/Qwen3-0.6B`（post-trained）**，`Qwen3-0.6B-Base` 仅作探针对照、**默认不下载**。
`[实测]` 规格（2026-09-16 取自官方仓库文件清单）：

| 角色 | 仓库 | 体积 | 权重文件 |
|---|---|---|---|
| student | `Qwen/Qwen3-0.6B` | **1.50 GB** / 1,503,300,328 B | `model.safetensors`（**单文件，不分片**） | `f47f7117…6874b` |
| teacher | `Qwen/Qwen3-1.7B` | **4.06 GB** / 4,063,515,592 B | `model-00001-of-00002`(3,441,185,608 B) + `model-00002-of-00002`(622,329,984 B) + `index.json` | `169ad53e…30ed5` / `912becff…deff9` |

**`[实测]` 默认源已改为 ModelScope，不再走 `hf-mirror.com`**（2026-09-16 定）。
理由是一晚实测撞出来的：`hf-mirror.com` **在 TLS 层间歇性断流**，`/resolve/...` 与 `/api/...` 都会抛
`SSLError(SSL: UNEXPECTED_EOF_WHILE_READING)`（大文件 resolve 端点 5 次尝试挂了 4 次），
且 HEAD 偶尔不带 `X-Repo-Commit` 头 → `huggingface_hub` 抛
`FileMetadataError → LocalEntryNotFoundError`（`file_download.py:1568` / `:1661`）。
致命的是 `file_download.py:1600-1602` 会把裸 `SSLError` 直接抛出工作线程，
**任一 worker 抖一下，整个 snapshot 就中止** —— 这才是"Fetching 10 files 卡在 8/10"的机制
（**是链路问题，不是元数据格式问题**；我先怀疑是镜像绝对跳转丢 header，实测被否掉了）。

改走 `https://modelscope.cn/models/<repo>/resolve/master/<file>`：
`[实测]` 302 落到 **`cdn-lfs-cn-1.modelscope.cn`**（国内 CDN，不绕 CloudFront）、
支持 Range/206（**可续传**）、且与 HF 是**同一份字节** —— 三处哈希互相印证：

| 来源 | sha256（0.6B `model.safetensors`） |
|---|---|
| `hf-mirror` 的 `X-Linked-Etag` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| ModelScope 的 `X-Linked-Etag` | 同上 |
| ModelScope LFS 对象路径 `lfs-objects/f4/7f/7117…`（**按内容哈希寻址**） | 同上 |

> ⚠️ 诚实边界：`huggingface.co` 本网络返回 **502**，**官方站没能充当第三方独立来源**；
> 上面三处本质同源（都指向 HF 的 LFS 对象），严格说是"两次镜像 + 一次对象存储自证"。

**`[实测]` 速差（同一网络，2026-09-16）**：ModelScope 路径 **~5.5 MB/s**（110 s 落 607,877,910 B）；
`hf-mirror → cas-bridge.xethub.hf.co` 路径 **~0.7 MB/s** 且伴随断流，**差约 8 倍**。
> 与 `EXPERIMENT_LOG E-09`（沙箱内 118 kB/s vs 沙箱外 6.06 MB/s）并存：本次**在沙箱内**直连
> ModelScope 就跑到 5.5 MB/s，**未复现 118 kB/s 限速**。两种观测不算冲突，但也没被解释，
> 先如实并列记着；>100MB 的下载仍建议放沙箱外跑。

```powershell
# Windows / PowerShell（默认 student + ModelScope 源）
powershell -ExecutionPolicy Bypass -File scripts\download_base_models.ps1
powershell -ExecutionPolicy Bypass -File scripts\download_base_models.ps1 -Target all
powershell -ExecutionPolicy Bypass -File scripts\download_base_models.ps1 -Source hf   # 退回 huggingface_hub
powershell -ExecutionPolicy Bypass -File scripts\download_base_models.ps1 -DryRun      # 只看计划
```

```bash
# Linux / 上云 3090
bash scripts/download_base_models.sh                  # student
bash scripts/download_base_models.sh --target all     # + teacher
bash scripts/download_base_models.sh --source hf      # 退回 huggingface_hub（带重试）
bash scripts/download_base_models.sh --dry-run
```

两个脚本都是**增量 + 可续传**的：已存在且哈希一致的文件直接跳过，
半截文件用 `curl -C -` 续传（`hf` 路径靠 `.cache/huggingface/download/*.incomplete`），
重复执行成本很低。下完**逐个核对 sha256**（`-SkipHash` / `--skip-hash` 可跳过），
最后跑一次 config + tokenizer + `index.json` 一致性冒烟测试（不占显存）。
`-Source hf` 路径额外套了**外层重试循环**（默认 6 次，`HF_HUB_DISABLE_XET=1`、`--max-workers 2`）——
单次 TLS 抖动不再毁掉整个任务。

> ⚠️ **脚本内所有「MSYS 路径 → 原生 .exe」的传参都已规避**：git-bash 下 `$MODEL_DIR` 是
> `/d/Codings/...` 形式，直接交给原生 `curl.exe` 得到 `curl: (23) Failed to open the file`，
> 交给原生 `python.exe` 则被 `transformers` 当成 repo id（`HFValidationError: Repo id must be
> in the form 'repo_name' or 'namespace/repo_name'`）。因此 `curl_get()` / `sha256_at()` /
> 冒烟测试统一改成**子 shell `cd` 进目标目录 + 裸文件名**（→ `EXPERIMENT_LOG E-24`）。
> **改动此脚本后必须用绝对 `--model-dir` 复测**——相对路径会把这类 bug 完全盖住
> （我第一轮"验证通过"就是这么蒙过去的）。

**`[实测]` 已产出产物零返工（哈希证据）**：`models/Qwen3-0.6B/` 下已有的 5 个 tokenizer 文件，
sha256 与官方清单**逐个一致** ——

| 文件 | sha256（前 12 位） |
|---|---|
| `config.json` | `660db3b73d78` |
| `tokenizer.json` | `aeb13307a71a` |
| `tokenizer_config.json` | `d5d09f07b48c` |
| `vocab.json` | `ca10d7e9fb3e` |
| `merges.txt` | `8831e4f1a044` |

→ 本地这份就是从官方 post-trained 仓库下的，**本文档 §1~§4 所有基于它的实测（173 万条 prompt 长度、
768 个 SID token 单 token 验证、`cutoff_len=400`）对下载后的权重 100% 有效**。

> 另有 `[实测]`：`Qwen3-1.7B` 的 `tokenizer.json` / `tokenizer_config.json` 哈希与 0.6B **完全相同**
> → teacher 与 student **同词表**，logit 可直接对齐、无需词表映射（只有 `config.json` 不同 `1ddb5b89`，那是结构差异）。

**`[实测]` student 权重已下并校验通过（2026-09-16）**：

`models/Qwen3-0.6B/model.safetensors` = **1,503,300,328 B**，
sha256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` —— **与预期逐位一致**。
走 ModelScope 路径，`curl -C -` 从 607,877,910 B 续传到完整，全程约 4 min（≈5.5 MB/s）。
脚本冒烟测试输出：

| 项 | 实测值 | 与既有记录 |
|---|---|---|
| `vocab_size` | 151936 | 一致 |
| `hidden_size` | 1024 | 一致 |
| `num_hidden_layers` | 28 | 一致 |
| `eos_token_id` | 151645（`<|im_end|>`） | 一致 |

→ `models/Qwen3-0.6B/` 下 **10 个文件已齐**。**teacher（`Qwen3-1.7B`）尚未下载**，
需要时用 `-Target all` / `--target all`（约 4.06 GB）。
> `[实测]` ModelScope 下的 `config.json` sha256 = `660db3b73d78…`，与上表 HF 版**逐位一致**
> → **小文件也是同一份字节**，不只是权重。
> ⚠️ 但**小文件的 `X-Linked-Etag` 并不是内容 sha256**（0.6B `config.json` 的 etag 是
> `f5c3703b78ae…`）→ **只对 LFS 权重做哈希校验**才是对的做法，别拿 etag 当 sha256 比。

### 4.6 本项目对 MiniOneRec 训练脚本的**实际改动**（跑之前必看）

上文多处说"骨架沿用 MiniOneRec"，但**不是一行不改**。截至 2026-09-16 动过的全部如下
（刻意保持最小，原则是"能无误跑起来优先，不为了对齐而改"）：

| 文件 | 位置 | 改动 | 理由 / 依据 |
|---|---|---|---|
| `sft.py` | `:58` 新增 `SidVocabLoader` | 注册改读 `info/sid_vocab.json`（**码序**），并断言 `index` 的 token ⊆ 码表 | `[实测]` §6.4(4)：`TokenExtender` 的 `sorted()` 使 765/768 个 id 与码序不同 → M4 码本初始化会静默错位 |
| `sft.py` | `:189` 新增 `--sid_vocab_path` | 留空时从 `--sid_index_path` 自动推导 `<sft>/info/sid_vocab.json` | 少一个必填参数 |
| `sft.py` | `:302-306` 注册块 | 注册后把 `token→id` 落盘 `output_dir/sid_token_map.json` | 复现 / M4 / 评估端对齐 |
| `sft.py` | `:239` `padding_side` | `"left"` → `"right"` | §4.3（复制粘贴遗留；纯 RoPE 下数学等价） |
| `sft.py` | `:192` `torch_compile` | 硬编码 `True` → 参数 `--torch_compile`，**默认 `False`** | `[设计]` 动态 padding 下每 batch 宽度不同，`torch.compile` 会反复重编译。**未做实测对比**，先关保守 |
| `sft.py` | `:190` 新增 `--tasks`<br>`:147` `resolve_tasks` | 训练集由 `ConcatDataset` 三路硬拼 → 可选子集。<br>**默认 `T1,T2a,T2b,T3` 全开，Run-0 行为不变** | §3.3；为任务消融铺路，与 §6.5「每项只改一个变量」配套 |
| `data.py` | `:623-631` `EvalSidDataset.get_history` | 输入句式统一回训练端口径 | `[实测]` §3.4：原版此处与三个训练类**不一致**（共同前缀仅 49 token）→ train/eval prompt 漂移，会静默掉点 |
| `evaluate.py` | `:51` 新增 `--sid_vocab_path`<br>`:52` 新增 `--max_samples` | 现场注册 SID 词表（口径同 `sft.py:241-306`）+ 限制样本数，供 **dry-run / 未训练基座** 用。<br>**两者默认关闭，现有评估行为完全不变** | §3.5：evaluator 冒烟测试 + 随机下界锚点 |
| `evaluate_run0.sh` | — | 通用化：`EXP_ID` 命名规范 + 自动反推 + SID 健全性检查 | §3.6 —— 修掉「**所有版本结果写同一文件互相覆盖**」（原用 `basename(MODEL_PATH)`，而它恒为 `final_checkpoint`） |
| `sft_run0.sh` | — | 产物改落 `outputs/<EXP_ID>/`，日志进 `logs/<EXP_ID>/` | §3.6 命名统一 |
| `scripts/sft/eval_report.py` | — | **新增**：落 `*.meta.json`（版本元数据）与 `*.metrics.json`（HR/NDCG） | 不改 `calc.py` 的口径，只在外层解析其 stdout |
| `requirements-core.txt` | — | 补 `fire==0.7.1` | `[实测]` `fire.Fire(train)` 是入口（`sft.py:440`），缺它直接 `ModuleNotFoundError`；原 `requirements.txt:30` 有，裁剪版漏了 |

**刻意没改的**：
- `data.py` **三个训练类**的 Dataset 与提示词模板**未动** —— 即 §2.1「全部裸写」定版
  **尚未落到 `data.py`**，现状仍是 verbatim 带引号版（诚实边界见 §6.4(5)）。
  ⚠️ 但 `EvalSidDataset` 的输入句式**已改**（见上表与 §3.4）。
- `sft.py` 的单阶段 concat **默认配比**（T1:T2a:T2b:T3 = 44.5 : 5.5 : 5.5 : 44.5）保持原样 ——
  `--tasks` 默认全开，Run-0 仍是干净锚点；**只有显式传参才会变**。

**`[实测]` 注册端到端自检**（`scripts/sft/verify_run0_registration.py --domain IandS`）：

```
vocab=768  head=['<a_0>'..'<a_3>']   tail=['<c_252>'..'<c_255>']   layers={'a':256,'b':256,'c':256}
index coverage = used=768 / vocab=768 / missing=0
len(tokenizer) = 151669 -> 152437          id range = [151669, 152436]
码序 == id 连续 : True
'<a_115><b_51><c_233>' -> [151784, 151976, 152414]   (len=3)
'### Response:\n'      -> [14374, 5949, 510]         (len=3，与 evaluate.py:104 硬编码 prefix_index 一致)
T1 目标结构 512/512 通过 = [3 个 SID] + [\n, EOS]
```

**`[实测]` 任务开关路由自检**（原 `scripts/sft/probe_task_switch.py --domain IandS`，探针已移除）：

```
A 解析 : PASS   8 个用例（默认 / 单任务 / 双任务 / 非法键 / 空值 / 只给一半 -> WARN）
B 路由 : PASS
   T1  SidSFTDataset       目标 = SID    (3-SID 命中 3)    208,999 条
   T2  SidItemFeatDataset  目标 = SID    (title2sid 侧)     51,694 条（sid2title + title2sid）
   T3  FusionSeqRecDataset 目标 = TEXT   (title)           208,999 条
C 规模 : 默认 --tasks=T1,T2a,T2b,T3 合计 469,692 条
```

**`[实测]` 约束解码链路自检**（`scripts/sft/probe_constrained_decoding.py --domain IandS`）：

```
A prompt 一致性 : SidSFTDataset vs EvalSidDataset  逐 token 一致（len=85）
B Trie 形状     : 5 步  [256, 98, 1, 1, 1]
                    step3 只允许 \n  /  step4 只允许 EOS   <- 与训练 target 逐位对应
C prefix_index  : "### Response:\n" 三种 encode 路径均 = 3 token
结果: 全部通过
```

> 最后一行解释了一个容易误判的点：T1 的 label **不是 3 个 token**，而是
> `[a, b, c, \n, EOS]` 共 5 个。`\n` 不是脏数据 —— `LogitProcessor.py` 第 4 步（`count=3`）
> 命中 `hash([a,b,c])` 只放 `\n`，第 5 步才命中 `hash([a,b,c,\n])` 放 EOS，
> 与 `data.py:417` 的 `output = target_item + "\n"` 完全自洽。

---

## 5. 口径决策（写死，改动 = 作废数字）

### 5.1 碰撞桶：SID → 物品怎么映射（**本项目特有，MiniOneRec/TIGER 都没有这个问题**）

SID 定版是**语义桶**（不做 Sinkhorn 消解），所以一个 SID 可能对应 2~5 个物品
（`[实测]` I&S 碰撞物品 2,046 个 = 7.92%、最大桶 5；VG 见 `stats.json`）。
而 baseline 的候选是**单个物品**，所以必须定死映射规则：

| 口径 | 规则 | 用途 |
|---|---|---|
| **严格（主榜）** | 每个 SID 桶取 **1 个 representative**（训练频次最高，平局取最小 item_id），top-10 恰好 10 个物品 | **与 baseline 同构，用于 §闸门比较** |
| **宽松（上界）** | 命中 = 目标 ∈ 生成 SID 的整个桶 | 量化"语义桶"额外带来的收益/损失，单独一列报 |

- representative 的选取**只看训练频次、不看目标**，因此**无泄漏**。
- 映射表已落盘：`info/sid2items.json`（含 `representative` 字段）。
- 受影响样本比例 `[实测]`：I&S test **7.27%**、VG 见 `sft_verify.json` → 两个口径的差距是**个位数百分点**，
  但**主榜一律用严格口径**，否则"生成式 > sasrec"这句话会被质疑放水。

### 5.2 其余口径全部沿用 `EVAL_PROTOCOL.md`

- 划分 = plain LOO；history ≤ 20；候选 = 全库；屏蔽集 = history ∪ 重建的完整训练序列。
- 主指标 HR@10、选型 NDCG@10、必报 coverage@10 / gini@10 / MRR。
- 生成式专用：`beam_ceiling@K`、`exp(−valid_ce)`；**MRR 对生成式无意义**（标 n/a）。
- **三条闸门**（`EVAL_PROTOCOL §5`）：HR@10 > sasrec（I&S **0.0395** / VG **0.0971**）；
  > content_ann（0.0288 / 0.0237）；量级对齐论文（≥0.0422 / ≥0.0868）。

### 5.3 V0 锚点（HR@10 = 0.093）**不可比** —— 别为它锁死任何设计

2026-09-14 纠正。V0 与本项目的差异是**数据集级别的**，不是"换个基座"：

| | V0（MiniOneRec 复刻） | 本项目 v2 |
|---|---|---|
| 数据 | Amazon18 | Amazon23 |
| 划分 | **全局时间 8:1:1** | **plain LOO**（`EVAL_PROTOCOL §1.4`） |
| 基座 | Qwen2.5-0.5B | Qwen3-0.6B |

三个口径无一相同，HR@10 数字跨过去比较没有意义。**它只剩 sanity check 价值**
（验证管线没写错、量级不离谱）。

**连带推论**：我们的对照是自己的 4 个 baseline（`sasrec` / `gru4rec` / `twotower_id` / `content_ann`），
它们与 SFT 模型跑在**同一套 EvalSet** 上。所以**提示词格式怎么选都不影响与 baseline 的可比性**
——格式选型应该纯粹看"哪个让 SFT 更强"，不用再为保锚点妥协（→ §3.1 选 chatml 的依据）。

---

## 6. 训练顺序（课程学习，2026-09-14 定案 `[设计]`，待 Run-1 实测验证）

### 6.1 先给结论：不做"训完 T2 再切 T1"的硬串行

直觉「先让模型认识 SID，再学推荐」是对的，但**形态必须是"warmup → 混合 → 退火"，不能是两个独立训练硬切**。

| 反对硬串行的理由 | 依据 |
|---|---|
| 0.6B 容量小，硬切会灾难性遗忘 T2 | 小模型多任务遗忘是常识；无 replay 时尤其明显 |
| MiniOneRec 默认配比里 **T3 `seq2title` 占 44.5%，与 T1 同权重**，硬切会一起丢掉 | `[实测]` 见 §6.2 计数 |
| 文献里做 curriculum 的**全部是软过渡，没有硬切** | 见 §6.3 |

### 6.2 三阶段定义

| 阶段 | 任务 | step 占比 | 冻结 | LR | 目的 |
|---|---|---|---|---|---|
| **S0 warmup** | T2（sid2title + title2sid）+ T4（text2sid） | ~8%（≈1 epoch on 2N） | `freeze_LLM=True` | 1e-3 | 把 768 个新 token 拉进语义空间 |
| **S1 主训练** | T1 + T3 + T2 + T4（MiniOneRec 默认比例，见下） | ~80% | 全开 | 5e-4（沿用 V0） | 学序列模式，辅助任务防遗忘 |
| **S2 退火** | **只 T1** | ~12% | 全开 | 余弦 → 0 | 去掉辅助任务分布干扰，贴合评估口径 |

`[实测]` MiniOneRec `sft.py:340-368` 是 `ConcatDataset([SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset])`，
单阶段、不分先后。按我们产物算出的**实际配比**：

| 域 | T1 seq2sid | T2（2N，双向） | T3 seq2title | 合计 | T1 : T3 : T2 |
|---|---:|---:|---:|---:|---|
| IandS | 208,999 | 51,694 | 208,999 | 469,692 | **44.5% : 44.5% : 11.0%** |
| VG | 435,534 | 51,222 | 435,534 | 922,290 | **47.2% : 47.2% : 5.6%** |

⚠️ 注意 **T3 和 T1 一样重** —— MiniOneRec 实际上让一半训练信号去"生成标题"而不是 SID。
这是它和 TIGER（纯 seq2sid）最大的训练端差异，M4 消融必须单独测 T3 的去留。

### 6.3 文献证据（顺序这个事，论文其实意见不统一）

| 来源 | 做法 | 结论 |
|---|---|---|
| **LC-Rec**（ICDE'24，arXiv:2311.09049）官方 `run.sh` | **单阶段 6 任务混合**：`--tasks seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain`，4 epochs | ⚠️ **不是两阶段**。网上流传的"先对齐再推荐"是把它的**消融实验**（逐步加任务看增益）误当成训练 schedule |
| **PIT / OneRec-V2**（快手，arXiv:2602.08530） | curriculum：**warm-up 阶段用确定性一对一 SID↔item 映射**，等 User-to-Token loss 收敛到基线再切动态目标 | ✅ **支持"先对齐"**，且是工业级证据 |
| **MHL**（arXiv:2509.23649） | warm-up 随机 mask → 自适应熵引导 mask → 无 mask 微调 | ✅ 支持"warmup → 主 → 退火"三段式（但 curriculum 在 mask 策略上，不在任务集合上） |
| **Token-Weighted**（arXiv:2601.17787） | 三个 loss 用指数衰减 $e^{-ct}$ 过渡权重 | ✅ 支持**软过渡**，反对硬切 |

→ 综合：**warmup 有价值（PIT 的工业证据），但过渡要软、且主阶段必须保留辅助任务做 replay。**

### 6.4 `[实测]` 两个必须先知道的实现细节

**(1) `freeze_LLM=True` 已经就是 S0，不用自己写**（`sft.py:309-330`）：
全参数冻结 → 只解冻 `get_input_embeddings().weight` → **注册 grad hook 把前 `original_vocab_size` 行梯度清零**。
即：实际只有 768×1024 = 786,432 个参数在动。因为 Qwen3-0.6B 是 `tie_word_embeddings=true`
（`models/Qwen3-0.6B/config.json`），这个张量同时是 lm_head，所以"只训 embedding"在 tied 下语义正确。
⚠️ `sft_3090.sh` 默认 `FREEZE_LLM=False` —— **这个开关一直没被用过**。

**(2) `sid2title` 在碰撞桶上是一对多，但可以不管**：

| 域 | 冲突 SID 数 | 占 SID 空间 | 涉及样本 | 占 sid2title | 占全部 T2 |
|---|---:|---:|---:|---:|---:|
| IandS | 901 | 3.64% | 1,913 | **7.40%** | 3.70% |
| VG | 388 | 1.55% | 831 | **3.24%** | 1.62% |

实拍冲突样本（IandS `<a_249><b_229><c_206>`）→ `B&C Eagle B16-1 1-Inch` / `B16-2 2-Inch` / `B16-34 3/4-Inch`
同系列不同规格；VG → `Pokemon Red Version` / `Blue Version`。
**判定：不修**。这些是语义近义目标、前缀 token 相同，梯度方向大体一致，属 `sid_raw` 语义桶的设计后果
（与 §3.4.2 宽松口径同源）。MiniOneRec/TIGER 用 Sinkhorn 保证唯一性所以没这个现象。
若 Run-1 显示 T2 拖后腿，退路是"T2a 只在桶大小=1 的 SID 上构造"。

**(3) `[实测]` 新增 token 的 embedding 初始化走 `mean_resizing`，不是 `initializer_range`**

（2026-09-16，原探针 `scripts/sft/probe_vocab_registration.py`（**已移除**），Qwen3-0.6B fp32 CPU）

原推断（"新增 768 行按 `initializer_range=0.02` 初始化，若与预训练行 std 差一个量级则 S0 必须调 LR"）
**已被实测推翻一半**：

| 区间 | 行数 | std | 来源 |
|---|---:|---:|---|
| 旧行 `[0, 151936)` | 151,936 | **0.02920** | 预训练权重（resize 后逐位保留，已验证） |
| 旧 padding 区 `[151669, 151936)` | 267 | 0.00953 | ⚠️ 未被 resize 触及 |
| 真新行 `[151936, 152437)` | 501 | **0.00942** | `mean_resizing=True` |

- `resize_token_embeddings` 签名默认 **`mean_resizing: bool = True`**（`modeling_utils.py:3177`）
  → 新行按**旧行 mean + 协方差多元正态**采样（[vocab-expansion](https://nlp.stanford.edu/~johnhew/vocab-expansion.html)），
  并打一条 `logger.warning_once`。实测 `mean_resizing=False` 才给 std = 0.02001。
- 差距 **3.1 倍**，不是"一个量级" ⟹ **S0 的理由要改写**：不是"尺度不匹配"，
  而是 mean_resizing 的**设计后果就是压低新 token 的初始概率**（降低对已有 token 分布的 KL 扰动）。
  768 个 SID 全是 T1 的输出目标 → 初始 logit 偏小 ⇒ **warmup 的作用是"把它们的概率拉起来"**。
- 🔴 **僵尸区**：`len(tokenizer)` = **151669**，而 `config.vocab_size` = **151936**（= 1187 × 128），
  **gap = 267**。所以 `resize_token_embeddings(len(tokenizer))`（MiniOneRec 原始写法）下，
  新 token id 从 151669 起，**前 267 个（`<a_0>` … `<b_10>`）落在旧 padding 行**，未经 mean_resizing。
  两段 std 恰好接近（0.0095 vs 0.0094），**都无预训练语义 ⇒ 判定可接受**；
  M4 语义初始化整体覆盖 768 行时会自动消解。
- ⚠️ resize 后 `config.vocab_size` = **152437**，**不再是 128 的倍数**。纯性能项（tensor core），
  可选 `pad_to_multiple_of=128` → 152448。MiniOneRec 没做这一步。

**(4) `[实测]` 词表注册的 token 来源：不能用 `index/`，要用 `info/sid_vocab.json`**

MiniOneRec 的 `TokenExtender`（`sft.py:30-55`）是**从 `index.json` 现收现加**，且 `sorted()`（`:53`）排序。
照搬到本项目会同时踩两个坑：

| 坑 | 实测（原探针 `probe_vocab_registration.py`，**已移除**） |
|---|---|
| **顺序** | `sorted()` 是字典序（`<a_0>, <a_100>, … <a_109>, <a_10>`），与本项目 `sid_vocab.json` 的**码序**下 **765 / 768 个 token 的 id 不同** |
| **集合** | index 只含"被用到过"的码：**VG 只有 759 个**（缺 9 个 a 层死码 = **9/256 = 3.52%**，与 `SID_PIPELINE` 记的 VG L0 死码率**完全吻合**）；IandS 恰好 768 |

Run-0 自身是自洽的（id 只是重新编号），但：
- 🔴 **M4 码本语义初始化会静默错位** —— `codebook.npy` 是 `(3,256,32)`、按码序排列，
  `<a_k>` 必须落在 `cb[0][k]`；
- 🔴 VG 还会**少 9 行**，两域词表大小不一致。

**已定版（2026-09-16）**：注册一律读 `info/sid_vocab.json`（768，**码序**），并断言
`set(index 的 token) ⊆ set(vocab)`；**只有找不到 vocab 时才回退** MiniOneRec 的 `sorted()` 路径
（并打 WARN）。已落地 `sft.py:58 SidVocabLoader` + `:241-306` 注册块（推导 / 断言 / 注册 / 落盘），端到端自检见 §4.6。

**(5) ✅ `data.py` 的提示词漂移已修（2026-09-18）—— 但它同时改掉了 Run-0 的一个变量**

2026-09-16 记的漂移有两条：`sid2title` 带双引号、T1 的 target 带尾部 `\n`。现状：

| 位置 | 2026-09-16（漂移） | 现在 |
|---|---|---|
| `sid2title` 的 input | `What is the title of item "{sid}"?`（带引号） | **裸写**（引号版降级为 `verbatim` 格式，只用于校验） |
| T1 target | `target_item + "\n"`（当时判定为"多余"） | **保留 `\n`** —— 查明 Trie step3 就是"只允许 `\n`"，它是**设计**，不是残留 |
| 模板来源 | `data.py` 内联 f-string | `config/prompt_templates.json` + `prompt_templates.py`（单一真源） |

🔴 **代价必须记账**：`data.py` 的骨架同时从 `alpaca` 换成了 `chatml` ⟹ **Run-0 相对 V0 变了两个
变量（基座 + 骨架）**。这是"适配 Qwen3"的主动决定，但归因时不能假装只有基座变了。
若要把格式也变成单变量，退路是 `PROMPT_FORMAT=alpaca` 再跑一遍（见 §3.1）。

顺带解决：`info/prompt_templates.json` 自称"训练端直接读、防漂移"而**实际没接线**的问题——
现在 `data.py` 真的读了（经 `prompt_templates.py`），该机制首次生效。

### 6.5 执行队列（每项只改一个变量，否则数字归因不了）

| Run | 改什么 | 回答什么问题 |
|---|---|---|
| **Run-0 锚点** | 单阶段，原样复刻 MiniOneRec（`--tasks` 默认全开 = T1+T2a+T2b+T3，3 epoch，LR 5e-4，`cutoff_len` 320） | Qwen3-0.6B 相对 V0（Qwen2.5-0.5B, HR@10=0.093）值多少？ |
| **Run-1** | Run-0 + S0 warmup | warmup 有没有用？ |
| **Run-2** | Run-1 + S2 退火 | 退火有没有用？ |
| **Run-3** | 码本语义初始化（`codebook.npy`）替代 S0 | 能不能省掉 warmup？ |
| **Run-4~6**（辅助任务消融，可选） | 在 Run-0 基线上分别 `TASKS=T1` / `TASKS=T1,T2a` / `TASKS=T1,T3` | 每个辅助任务对 T1 主指标各贡献多少？（§3.3） |

🔴 **Run-0 必须先跑**：一次改基座 + 配比 + 顺序三个变量，出了数字不知道是谁的功劳。

> `--tasks`（§3.3）是这套队列的**执行工具**：Run-0~3 一律用默认全开，只有 Run-4 起的消融才传参；
> 每组输出目录自动带后缀（`outputs/sft_IandS_T1` 等），互不覆盖。
> ⚠️ 消融得到的**只有 T1 指标**（辅助任务无独立评测口径），见 §3.3 限制 3。

---

## 7. 待办 / 未做（诚实边界）

| 项 | 状态 | 说明 |
|---|---|---|
| LC-Rec 的 `itemsearch` / `preferenceobtain` | ⏸ | 需要额外的"自然语言意图"标注，本项目暂无；T4 已部分覆盖其对齐作用 |
| 用户侧 token | ⏸ | MiniOneRec 无 user token；CCFRec 等有。本项目用户数 5~9.5 万，加进去词表会再涨 60% |
| 语义初始化（M4） | ⏸ | `info/codebook.npy` 已备好 `(3,256,32)`，训练端还没接 |
| 课程学习（M4） | ⏸ | 方案已定案 → **§6**（S0 warmup / S1 混合 / S2 退火）；等 Run-0 锚点跑完再上 |
| raw 桶 vs 唯一化的端到端消融 | ⏸ | `sid_sk.npy` 已存档，切口径重跑即可 |
| Qwen3-0.6B 权重 | ✅ | `[实测]` 2026-09-16 已下并校验通过：1,503,300,328 B、sha256 `f47f7117…6874b` **逐位一致**（§4.5）。⚠️ **teacher `Qwen3-1.7B` 仍未下载**（4.06 GB） |
| SID 词表注册 | ✅ | `[实测]` 已定版 + 落地（§6.4(4) / §4.6）；自检脚本 `scripts/sft/verify_run0_registration.py --domain IandS` 全绿 |
| 分任务训练开关 | ✅ | `[实测]` `--tasks` 已落地（§3.3）：默认四路全开等价 MiniOneRec，合计 **469,692** 条（原 `probe_task_switch.py` 探针已移除） |
| 约束解码链路核验 | ✅ | `[实测]` §3.4：Trie 5 步 `[256,98,1,1,1]` 与训练 target 逐位对应；`prefix_index=3` 前提成立；**顺带修掉上游遗留的 train/eval prompt 不一致**。回归检查已接入 `evaluate_run0.sh`（`probe_constrained_decoding.py` 保留） |
| T4 `text2sid` 训练端接线 | ⏸ | `data.py` **无对应 Dataset 类**，要用需新写（原 `tasks/text2sid.jsonl` 中间产物已随渲染脚本一并删除） |
| **提示词模板单一真源** | ✅ | `[实测]` 2026-09-18 收敛（§3.1）：3 份实现 → `config/prompt_templates.json` + `prompt_templates.py`；7 个在用 Dataset 类全部改走真源，`data.py` 复验 BAD=0 |

### 7.1 已移除的开发期脚本（2026-09-18）

这些**不在训练/eval 链路上**，为减负删除；内容仍在 git 历史里，随时可取回：

```bash
git log --diff-filter=D --name-only --oneline -- 'scripts/**'   # 看被删清单
git show <commit>^:scripts/multimodal/probe_latent_rank.py      # 取回某个文件
```

| 已删 | 原用途 | 为什么可以删 |
|---|---|---|
| `scripts/data/build_sft_prompts.py` | 预渲染双格式明文 prompt | 产物零消费方；其唯一价值（格式消融）已由 `PROMPT_FORMAT` 开关取代 |
| `scripts/data/verify_sft_data.py` | 数据集体检 | 一次性；结论已固化在 §4 |
| `scripts/sft/probe_task_switch.py` | `--tasks` 路由自检 | 一次性；结论已固化在 §3.3 |
| `scripts/sft/probe_vocab_registration.py` | 词表注册实测 | 一次性；结论已固化在 §6.4(3)(4) |
| `scripts/sft/probe_rl_constraint_map.py` | RL 侧约束查表复刻 | 结论已固化在 `RL_PIPELINE §3` |
| `scripts/sft/probe_rl_memory.py` | 本地显存逐块账本 | 本地 4GB 专用；上云不需要（结论在 §3.8/§3.10） |
| `scripts/multimodal/probe_{alignment_methods,dataset_stats,fusion_rank,gate_twins,latent_rank,twin_sinkhorn}.py` | SID 阶段几何/数据诊断 | 结论已固化在 `SID_PIPELINE` / `DATASET` / `KNOWLEDGE_BASE` |

**保留的三个**（都在链路上）：`probe_constrained_decoding.py`（`evaluate_run0.sh:112-118` 的硬闸门）、
`verify_run0_registration.py`（Run-0 词表注册核验）、`baseline/scripts/probe_sequence_reconstruction.py`（baseline 复现树）。
