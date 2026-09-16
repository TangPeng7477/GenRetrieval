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
| 训练端 | MiniOneRec `sft.py` + `data.py` 为骨架，**本项目已改 3 处**（注册 / `torch_compile` / padding） | `[实测]` §3.2 · §4.6 |
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
| **T2a `sid2title`** | `Answer the question about item identification.` | `What is the title of item <a_5><b_23><c_66>?` | 商品标题 | `tasks/itemfeat.jsonl` |
| **T2b `title2sid`** | 同上 | `Which item has the title: 3Doodler "What Will You Create? Project Book?` | `<a_5><b_23><c_66>` | 同上 |
| **T3 `seq2title`** | `Can you recommend the next item for the user based on their interaction history?` | `The user has sequentially interacted with items <a_147>…, <a_148>…. Can you recommend the next item for him? Tell me the title of the item` | 商品标题 | `tasks/seq2title_*.jsonl` |
| **T4 `text2sid`** | `Answer the question about item identification.` | `An item can be described as follows: 3Doodler … \| Brand: WobbleWorks \| Categories: … \| Features: …. Which item is it describing?` | `<a_5><b_23><c_66>` | `tasks/text2sid.jsonl` |

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

### 3.1 明文 prompt 渲染产物（`scripts/data/build_sft_prompts.py`）

§3 的产物是"结构化中间态"（CSV / jsonl），训练端还要自己拼提示词。
`build_sft_prompts.py` 把它们**渲染成明文 prompt**，双格式各一套，供训练端直接读：

```
data/Amazon23/<域>/sft/prompts/
├── alpaca/  T1_seq2sid.{train,valid,test}.jsonl · T2_itemfeat.jsonl
│            T3_seq2title.{train,valid,test}.jsonl · T4_text2sid.jsonl · stats.json
├── chatml/  同上
└── stats.json
data/Amazon23/sft_prompts_verify.json    逐 token 对齐校验报告
```

每行 7 字段：`{task, split, prompt, completion, n_prompt_tok, n_compl_tok, meta}`
（`meta` 带 `user_id` / `item_id` / `n_hist`，方便后面做冷/热分桶）

| 口径 | 决定 |
|---|---|
| **主榜 Run-0 用 `chatml`** | Qwen3 原生格式，与预训练一致；`<\|im_start\|>assistant` 配 `<\|im_end\|>` 自洽 |
| `alpaca` 也产 | 作为"格式有没有影响"的单变量消融（**不是**为比 V0 —— 见 §5.3） |
| `completion` **不含 EOS** | 由训练端 `encode(eos=True)` 追加；`n_compl_tok` 也不计 EOS |
| `completion` **末尾无 `\n`** | MiniOneRec 原文有，但在 Trie 下必被 -inf 屏蔽，永远生成不出来 → 死权重 |
| 不落 token ids | 双域双格式约 2GB，只存长度；ids 由训练端现算 |

**逐 token 校验**（`--verify`）：用 MiniOneRec 自己的 `SidSFTDataset` / `SidItemFeatDataset` /
`FusionSeqRecDataset` 生成 ground-truth `input_ids`，与「verbatim 版」（带引号 + 带 `\n`、
即 MiniOneRec 逐字复刻）逐 id 比对，**必须 0 差异**；定版与 verbatim 的差异只允许发生在
「引号」和「尾部 `\n`」两处。报告落 `data/Amazon23/sft_prompts_verify.json`。

⚠️ 校验里的坑：三个 Dataset 的 `sample>0` 都是**随机采样**（`data.py:94` 的 `df.sample()`、
`data.py:723` 的 `random.sample`），所以必须用 `ds.data`（采样后）逐行构造对比，
拿自己 jsonl 的前 N 条去对会全错。

**全量长度实测**（`prompts/<fmt>/stats.json`，双域 × 双格式，2026-09-14 全量跑出）：

| 任务 | 样本数 I&S / VG | prompt 均值 | total_max（含 completion+EOS） | >320 |
|---|---:|---:|---:|---:|
| T1 `seq2sid` | 208,999 / 435,534 | 107.8 / 111.9 | **179 / 179** | 0 / 0 |
| T2a `sid2title` | 25,847 / 25,611 | 57.0 / 57.0 | **161 / 143** | 0 / 0 |
| T2b `title2sid` | 25,847 / 25,611 | 86.6 / 69.7 | **160 / 141** | 0 / 0 |
| T3 `seq2title` | 208,999 / 435,534 | 111.8 / 115.9 | **277 / 251** | 0 / 0 |
| T4 `text2sid` | 25,847 / 25,611 | 192.0 / 165.7 | **391 / 362** | **18 / 3** |

（alpaca 与 chatml 相差 1 token，来自 ChatML 的角色标记 ⇒ 长度结论不受格式影响。）

→ **`cutoff_len`：不带 T4 用 320；带 T4 用 400**。T4 只占 4.3% 样本，但把 max 从 277 顶到 392。
⚠️ MiniOneRec 是 `tokens[-max_len:]` 左侧截断，超长会砍掉 instruction 头部。

```bash
./.venv/Scripts/python.exe scripts/data/build_sft_prompts.py --domain all             # 全量渲染（约 18 分钟）
./.venv/Scripts/python.exe scripts/data/build_sft_prompts.py --domain all --verify    # 逐 token 校验
```

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
（`evaluate.py:72` 只做 `AutoTokenizer.from_pretrained(base_model)`），它依赖
`sft.py:386` 的 `tokenizer.save_pretrained(output_dir)` —— 即**训练产物自带扩展后的 tokenizer**。
指回原始基座的话，SID 会被切成碎片（`<a_1>` → 6 个 token），Trie 全挂。这是 MiniOneRec 的既有设计。

> ⚠️ **诚实边界（未接线项，不影响 Run-0 可跑）**：`tasks/*.jsonl` 与
> `prompts/{alpaca,chatml}/*.jsonl` **目前没有消费方**。训练端 T1/T2/T3 的数据由 `data.py`
> 三个 Dataset 类**从 `index/` + CSV 现场重建**（`sft.py:313` 的 `SidItemFeatDataset` 用
> `item.json` + `index.json` 现拼 sid2title/title2sid 对；`sft.py:315` 的 `FusionSeqRecDataset` 同）。
> 明文 prompt 渲染产物是给后续「格式消融 / 训练端直读明文」预留的，暂未接线。

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
| `category` 参数 | `Industrial_and_Scientific` 等 5 个硬编码 | **IandS 单域实验无需改动**；上 VG 时要在 `sft.py:169` **和** `evaluate.py:54` **两处**都加 `"Video_Games": "video games"` | `[代码]` 两个脚本的 `category_dict` 都只有 5 个键，VG 会 `KeyError` |

> ⚠️ **修正一处先前的数字**：§4 ③ 曾报"T4 最长 309"，那是 `verify_sft_data.py` **抽样** ≤n_probe 条的结果；
> 全量渲染后真实 max 是 **391（IandS）/ 362（VG）**，超 320 的分别有 18 / 3 条。
> 抽样在长度分布的长尾上不可靠 —— 定 `cutoff_len` 这种"取 max"的场合必须用全量。

### 4.2 超长 T4 **不用重生成数据集**，改 `cutoff_len` 即可

定 `cutoff_len` 之前先确认它在哪一层生效，否则会做无用功：

| 层 | 是否截断 | 证据 |
|---|---|---|
| `prepare_sft_data.py`（构造四任务） | ❌ 只按**字符**截（`--max_text_chars 512`），不管 token | 无 tokenizer |
| `build_sft_prompts.py`（渲染明文） | ❌ **只统计 `over_320` 计数，从不截断** | 脚本内无 `[:max_len]` |
| **`data.py:190-197`（训练端）** | ✅ `tokens[-max_len:]` —— **从左侧砍** | 唯一真正的截断点 |

所以数据集里存的是**明文**，长度是训练时才决定的 —— **改参数就够了，不用重生成**。

**代价也确实为零**，因为 padding 是动态的（`[实测]`）：

```
sft.py:266  DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True)
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

MiniOneRec 是把生成端的设置（`evaluate.py:140`）顺手复制到了训练端（`sft.py`）造成的，属于复制粘贴遗留。

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
| `evaluate.py:89-90` Trie `ID.append(tokenizer.eos_token_id)` | 同上 | 自动跟随 |
| `sft.py:155-156` `tokenizer.pad_token = tokenizer.eos_token` | 同上 | pad 会跟着变（无影响，pad 位被 -100 屏蔽） |

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
| `sft.py` | `:162` 新增 `--sid_vocab_path` | 留空时从 `--sid_index_path` 自动推导 `<sft>/info/sid_vocab.json` | 少一个必填参数 |
| `sft.py` | `:273-277` 注册块 | 注册后把 `token→id` 落盘 `output_dir/sid_token_map.json` | 复现 / M4 / 评估端对齐 |
| `sft.py` | `:210` `padding_side` | `"left"` → `"right"` | §4.3（复制粘贴遗留；纯 RoPE 下数学等价） |
| `sft.py` | `:371` `torch_compile` | 硬编码 `True` → 参数 `--torch_compile`，**默认 `False`** | `[设计]` 动态 padding 下每 batch 宽度不同，`torch.compile` 会反复重编译。**未做实测对比**，先关保守 |
| `requirements-core.txt` | — | 补 `fire==0.7.1` | `[实测]` `fire.Fire(train)` 是入口（`sft.py:391`），缺它直接 `ModuleNotFoundError`；原 `requirements.txt:30` 有，裁剪版漏了 |

**刻意没改的**：
- `data.py` 三个 Dataset 类与提示词模板**一字未动** —— 即 §2.1「全部裸写」定版**尚未落到 `data.py`**，
  现状仍是 verbatim 带引号版（诚实边界见 §6.4(5)）。
- `sft.py` 的单阶段 concat 配比（T1:T2:T3）保持原样，Run-0 才有一个干净锚点。

**`[实测]` 注册端到端自检**（`scripts/sft/verify_run0_registration.py --domain IandS`）：

```
vocab=768  head=['<a_0>'..'<a_3>']   tail=['<c_252>'..'<c_255>']   layers={'a':256,'b':256,'c':256}
index coverage = used=768 / vocab=768 / missing=0
len(tokenizer) = 151669 -> 152437          id range = [151669, 152436]
码序 == id 连续 : True
'<a_115><b_51><c_233>' -> [151784, 151976, 152414]   (len=3)
'### Response:\n'      -> [14374, 5949, 510]         (len=3，与 evaluate.py:84 硬编码 prefix_index 一致)
T1 目标结构 512/512 通过 = [3 个 SID] + [\n, EOS]
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

`[实测]` MiniOneRec `sft.py:202-214` 是 `ConcatDataset([SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset])`，
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

**(1) `freeze_LLM=True` 已经就是 S0，不用自己写**（`sft.py:173-195`）：
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

（2026-09-16，探针 `scripts/sft/probe_vocab_registration.py`，Qwen3-0.6B fp32 CPU）

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

MiniOneRec 的 `TokenExtender`（`sft.py:42-55`）是**从 `index.json` 现收现加**，且 `sorted()` 排序。
照搬到本项目会同时踩两个坑：

| 坑 | 实测（`scripts/sft/probe_vocab_registration.py`） |
|---|---|
| **顺序** | `sorted()` 是字典序（`<a_0>, <a_100>, … <a_109>, <a_10>`），与本项目 `sid_vocab.json` 的**码序**下 **765 / 768 个 token 的 id 不同** |
| **集合** | index 只含"被用到过"的码：**VG 只有 759 个**（缺 9 个 a 层死码 = **9/256 = 3.52%**，与 `SID_PIPELINE` 记的 VG L0 死码率**完全吻合**）；IandS 恰好 768 |

Run-0 自身是自洽的（id 只是重新编号），但：
- 🔴 **M4 码本语义初始化会静默错位** —— `codebook.npy` 是 `(3,256,32)`、按码序排列，
  `<a_k>` 必须落在 `cb[0][k]`；
- 🔴 VG 还会**少 9 行**，两域词表大小不一致。

**已定版（2026-09-16）**：注册一律读 `info/sid_vocab.json`（768，**码序**），并断言
`set(index 的 token) ⊆ set(vocab)`；**只有找不到 vocab 时才回退** MiniOneRec 的 `sorted()` 路径
（并打 WARN）。已落地 `sft.py:58 SidVocabLoader` + `:214-277` 注册块（推导 / 断言 / 注册 / 落盘），端到端自检见 §4.6。

**(5) 🔴 `data.py` 的提示词仍与 §2.1 定版**漂移**（2026-09-16 发现，**刻意未修**）**

| 位置 | 现状 | §2.1 定版 |
|---|---|---|
| `data.py:738` `SidItemFeatDataset.generate_prompt`（sid2title） | `What is the title of item "{sid}"?` —— **带双引号** | 全部裸写 |
| `data.py:735`（title2sid） | `Which item has the title: {title}?` —— 无引号 | 全部裸写 ✅ |
| `data.py:417` T1 的 `output` | `target_item + "\n"` —— **带尾部 `\n`** | completion 末尾不加 `\n` |

**为什么留着**：Run-0 要**只改「基座」一个变量**（`Qwen2.5-0.5B` → `Qwen3-0.6B`）。
提示词口径（裸写、去 `\n`）的改动留到 Run-1 之后单独消融，否则出了数字归因不干净。

🔴 **连带发现**：`info/prompt_templates.json` 自称"训练端直接读、防止两端漂移"，
但 **`grep data.py` 找不到任何读它的代码** —— 这个防漂移机制**实际上没接线**。
`prompts/{alpaca,chatml}/*.jsonl` 同理（§3.2 已标注"零消费方"）。
→ 真要防漂移，得把 `data.py` 的 `generate_prompt` 改成读 `prompt_templates.json`；**未做**。

### 6.5 执行队列（每项只改一个变量，否则数字归因不了）

| Run | 改什么 | 回答什么问题 |
|---|---|---|
| **Run-0 锚点** | 单阶段，原样复刻 MiniOneRec（T1+T2+T3 concat，3 epoch，LR 5e-4，`cutoff_len` 320） | Qwen3-0.6B 相对 V0（Qwen2.5-0.5B, HR@10=0.093）值多少？ |
| **Run-1** | Run-0 + S0 warmup | warmup 有没有用？ |
| **Run-2** | Run-1 + S2 退火 | 退火有没有用？ |
| **Run-3** | 码本语义初始化（`codebook.npy`）替代 S0 | 能不能省掉 warmup？ |

🔴 **Run-0 必须先跑**：一次改基座 + 配比 + 顺序三个变量，出了数字不知道是谁的功劳。

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
