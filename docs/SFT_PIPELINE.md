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
| `cutoff_len` | **320**（T1/T2/T3 最长 179；含 T4 最长 309）→ 比 V0 的 512 省 ~40% | `[实测]` §4 ③ |
| 训练端 | **MiniOneRec `sft.py` + `data.py` 原样可用**，已实跑验证 | `[实测]` §4 ⑤ |
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

1. **可比性**：V0 锚点（README §3.4 的 0.5B SFT HR@10 = 0.093）就是 B 形态跑出来的。
   换成 A 会连"prompt 段算不算 loss"这个变量一起改，**M3 第一次跑就失去锚点**。
2. **样本效率**：B 只对 target 段计 loss，历史段是纯条件；A 对历史段也算 loss，
   在 0.6B 小模型 + 20 万样本上，A 会把容量浪费在"复述历史"上。
3. **与 C 的取舍**：C（IDGenRec 的文本化 ID）对**冷启动/零样本**更友好（新商品有文本就能出 ID），
   但本项目 SID 已定版为 RQ-VAE 离散码（`SID_PIPELINE.md`），换路线等于推倒重来。
   **C 的思想用 T4 任务部分吸收**（让模型学会"从富文本反查 SID"，见 §2）。

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

**MiniOneRec 的任务组合是 LC-Rec 的子集** `[代码]`（本仓 `sft.py:204-209` 实际 concat 的三个）：
`SidSFTDataset`(T1) + `SidItemFeatDataset`(T2) + `FusionSeqRecDataset`(T3)，**1:1:1 均匀混合**。

### 1.2 本项目相对 MiniOneRec 的两处改动

| 改动 | 内容 | 理由 |
|---|---|---|
| **新增 T4 `text2sid`** | `title + brand + categories + features` → SID | Amazon23 有 `features` 字段（要点列表），信息密度高于 description；MiniOneRec 只有 title。这是 LC-Rec `item2index` 的富文本变体 |
| **任务配比可配置** | 四任务**分别落盘**，训练端按配比组合 | M4 要消融 `1:1:1` / `2:1:1` / `1:1:2` / `+T4`（`UPGRADE_PLAN §5.3`）。写死就无法消融 |

**不改的**：T1/T2/T3 的提示词**逐字沿用 MiniOneRec**（含它那个拼写错误 `palyed` 也不改——
改了就不是同一条 prompt，V0 锚点失效）。

---

## 2. 四个任务的定义（模板落盘在 `info/prompt_templates.json`）

统一外层是 Alpaca 头（与 MiniOneRec 一致）：

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### User Input: 
{input}

### Response:
{output}
```

| 任务 | instruction | input（示例） | output | 落盘 |
|---|---|---|---|---|
| **T1 `seq2sid`** | `Can you predict the next possible item that the user may expect?` | `The user has interacted with items <a_147><b_81><c_146>, <a_148><b_184><c_140> in chronological order. Can you predict the next possible item that the user may expect?` | `<a_5><b_23><c_66>` | `{train,valid,test}/*.csv` |
| **T2a `sid2title`** | `Answer the question about item identification.` | `What is the title of item "<a_5><b_23><c_66>"?` | 商品标题 | `tasks/itemfeat.jsonl` |
| **T2b `title2sid`** | 同上 | `Which item has the title: 3Doodler "What Will You Create? Project Book?` | `<a_5><b_23><c_66>` | 同上 |
| **T3 `seq2title`** | `Can you recommend the next item for the user based on their interaction history?` | `The user has sequentially interacted with items <a_147>…, <a_148>…. Can you recommend the next item for him? Tell me the title of the item` | 商品标题 | `tasks/seq2title_*.jsonl` |
| **T4 `text2sid`** | `Answer the question about item identification.` | `An item can be described as follows: "3Doodler … \| Brand: WobbleWorks \| Categories: … \| Features: …". Which item is it describing?` | `<a_5><b_23><c_66>` | `tasks/text2sid.jsonl` |

**历史段的分隔**：物品内三层 token **直接拼接**（`<a_5><b_23><c_66>`），物品之间用 `", "` 分隔
——与 MiniOneRec `data.py::get_history` 一致，这样"一个物品 = 连续 3 个 token"，Trie 约束解码的层级才对得上。

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
├── info/prompt_templates.json  四任务模板（训练端直接读，防止两端漂移）
└── stats.json                  全部计数
```

**CSV 7 列**（`user_id, history_item_title, item_title, history_item_id, item_id, history_item_sid, item_sid`）
—— 后四个是 list 的 Python repr 字符串，因为 MiniOneRec 用 `eval(row[...])` 读 `[代码] data.py:404`。

### 复现命令

```bash
./.venv/Scripts/python.exe scripts/data/prepare_sft_data.py --domain IandS
./.venv/Scripts/python.exe scripts/data/prepare_sft_data.py --domain VG
./.venv/Scripts/python.exe scripts/data/verify_sft_data.py  --domain all   # 体检，见 §4
```

---

## 4. 体检实测数字（2026-09-14，`data/Amazon23/sft_verify.json`）

| 检查项 | IandS | VG | 判定 |
|---|---:|---:|---|
| ① 768 个 SID token 是否各占 **1 个 token** | ✅ | ✅ | 抽 21 个全为单词元 |
| ② CSV ↔ index.json 往返一致 | 0 条不一致 | 0 条不一致 | ✅ |
| ③ T1 prompt 均值 / p95 / **max** | 130.7 / 175 / **175** | 125.1 / 175 / **175** | — |
| ③ T1 **total max**（含 target+EOS） | **179** | **179** | — |
| ③ T2 total max | 132 | 99 | — |
| ③ T3（同 T1 结构） | ≤179 | ≤179 | — |
| ③ T4 total max | **309** | **221** | 决定 cutoff |
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
指向我们的新路径，并把 `cutoff_len` 从 512 降到 **320**。

### 4.1 两处必须改的配置（否则白烧显存）

| 项 | V0 | 本项目 | 理由 |
|---|---|---|---|
| `cutoff_len` | 512 | **320** | T1/T2/T3 ≤179、T4 ≤309 `[实测]`；512 有 40% 是纯 padding |
| `category` 参数 | `Industrial_and_Scientific` 等 5 个硬编码 | 需支持 `Video_Games` | `[代码] sft.py:120` 的 `category_dict` 只有 5 个键，VG 会 KeyError |

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

---

## 6. 待办 / 未做（诚实边界）

| 项 | 状态 | 说明 |
|---|---|---|
| LC-Rec 的 `itemsearch` / `preferenceobtain` | ⏸ | 需要额外的"自然语言意图"标注，本项目暂无；T4 已部分覆盖其对齐作用 |
| 用户侧 token | ⏸ | MiniOneRec 无 user token；CCFRec 等有。本项目用户数 5~9.5 万，加进去词表会再涨 60% |
| 语义初始化（M4） | ⏸ | `info/codebook.npy` 已备好 `(3,256,32)`，训练端还没接 |
| 课程学习（M4） | ⏸ | 计划：阶段1 只训 T2/T4（对齐）→ 阶段2 全混合 |
| raw 桶 vs 唯一化的端到端消融 | ⏸ | `sid_sk.npy` 已存档，切口径重跑即可 |
| Qwen3-0.6B 权重 | ⏸ | 本地只有 tokenizer（`models/Qwen3-0.6B/`），**权重未下载**（约 1.5GB，需沙箱外执行） |
