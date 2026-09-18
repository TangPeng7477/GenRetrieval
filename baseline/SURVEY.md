# 经典 / 生成式召回 baseline 综述

> 目标：回答三个问题 —— ①这两个类目（Amazon Reviews 2023 **Industrial_and_Scientific** / **Video_Games**）
> 上大家都在用什么 baseline？②公开数字是多少？③哪些能在**一张 4GB 显存的笔记本卡**上本地复现？
>
> **每一条都要能追溯**，所以给数字加了两维标记：
> - **核验等级**：`[已核验]` = 我逐页打开原始来源、把表格数字抄下来过；`[二手]` = 来自检索汇总，**未逐篇复核**，只作线索。
> - **复现状态**：`[已复现]` = 本仓 `baseline/` 有实跑产物；`[未复现]` = 只登记不跑（附原因）。
>
> 相关：**评估协议定版见 `docs/EVAL_PROTOCOL.md`**（划分 / 指标 / 报数模板 / SFT 闸门）；
> 实测结果表见 `baseline/RESULTS.md`；口径与复现设置见 `baseline/README.md`。

---

## 0. 三句话结论

1. **这两个类目上唯一公开的同代锚点是 CCFRec（KDD'25, arXiv:2503.12183）与 MTGRec（SIGIR'25, arXiv:2504.04400）**
   各自实验表里的 I&S / VG 列 —— 两者都是 Amazon **2023** + leave-one-out + **全库排序** + K=5/10。
   而且 CCFRec 表 2 的数据集统计（Scientific 50,985 用户 / 25,848 商品 / 412,947 交互）
   **与本项目原始分片逐格相同**，说明用的是同一份原始数据，只是切分方式不同。
2. **TIGER / LC-Rec / LETTER / EAGER / RPG / GRID 的原始数字全部在 Amazon 2014 / 2018 上**，
   与 2023 版**不可横比**。MiniOneRec 的数字同理（本仓 V0 复刻记录用 2018，见 `docs/DATASET.md §9`）。
3. **最低算力的生成式召回形态 = "小 Transformer + SID 自回归"（TIGER 范式），不需要 LLM** ——
   这正是在 4GB 卡上能复现的那一档，本仓已实现为 `sid_gr`。

---

## 1. 口径先行：为什么不能直接抄论文数字

> 🔴 **本节已于 2026-09-19 按 v2 口径重写。** 2026-09-13 主榜从 v1（官方全局时间切分）切到
> **v2（plain LOO + sliding-window train）**，原表仍写 v1 的「官方全局时间切分 / 33% 冷启动」，
> 与 `EVAL_PROTOCOL.md §1.2/§1.3`、`docs/DATASET.md §7.2` 及实际 `*.stats.json` **打架**，现修正。

| 口径维度 | 论文常见设置 | 本项目设置（v2，`EVAL_PROTOCOL.md §1.3`） | 对数字的影响 |
|---|---|---|---|
| **数据集年份** | TIGER/LC-Rec/LETTER/EAGER/RPG/GRID = Amazon **2014 或 2018** | Amazon **2023**（官方 5core 分片） | 不可比。2023 版类目更大、更稀疏、上新更多 |
| **切分方式** | leave-one-out（每用户留最后 1 条做 test） | **`last_out`（plain LOO）** — 与论文**同协议** | ✅ 可**量级对齐**（不能直接横比，实现细节仍有差异） |
| **core 过滤** | 5core | **5core** | ✅ 同上 |
| **候选集** | 少数工作用 100 负采样 | **全库排序**（I&S 25,847 物品） | 负采样数字系统性偏高，不可混用 |
| **K 值 / beam** | K=5,10；生成式 beam 常为 10~50 | K=1,5,10 + MRR；生成式 beam=20 | 生成式的 HR@10 **天花板由 beam 决定**，必须连 beam 一起报 |
| **是否屏蔽已交互** | LOO 下用全序列，天然屏蔽 | **屏蔽 history ∪ 完整训练序列 ∪ valid 目标**（见 `EVAL_PROTOCOL.md §2.2`） | 不屏蔽会虚高；我们主动做严 |

⚠️ **v1 → v2 的代价（必须对外讲清）**：LOO + 5core 几乎保证 test 目标在 train 历史里出现过 ⟹
**冷启动命题从主榜消失**（v1 的 31.02% / 49.05% → v2 ≈ 0%）。
该命题改在 `MiniOneRec-style sliding-window time-split` 附录里单独跑（`EVAL_PROTOCOL.md §6` 待办）。

⚠️ **与官方参考实现（BLaIR / RecBole）不可横比**：它们用 **0core + timestamp + hist 50**，
候选空间是本项目的 **16.5×（I&S）/ 5.4×（VG）**。逐项对照见 §1.5.2 与 `EVAL_PROTOCOL.md §1.3`。

---

## 1.5 官方自己提供了什么？（2026-09-13 逐页 + 读源码核实）

> **一句话**：官方给的是**数据协议 + 参考实现**，**不是官方榜单**；
> 而且它**并列发布了两套切分协议，没有指定哪一套是"官方评测口径"**。

官方仓库 [`hyp1231/AmazonReviews2023`](https://github.com/hyp1231/AmazonReviews2023) 实际提供四样东西：

| 提供物 | 位置 | 内容 | 本项目 |
|---|---|---|---|
| **切分脚本 + 切分文件** | `benchmark_scripts/` | `{0core, 5core}/{rating_only, last_out, timestamp, last_out_w_his, timestamp_w_his}` | ✅ **直接下载它的 `5core/timestamp` 分片**（`DATASET.md §4`） |
| **序列推荐参考实现** | `seq_rec_results/` | 基于 **RecBole**：UniSRec / SASRecText 等文本序贯模型 + BLaIR 特征 | ⏳ 可做第三方交叉验证（§7 待办） |
| **商品检索参考实现** | `product_search_results/` | BM25 / BLaIR 复现脚本 | ➖ 任务不同（query→item，非序列召回） |
| **预训练语义编码器** | `blair/` | `hyp1231/blair-roberta-base` 等 checkpoint | ➖ 本项目用 Qwen3-Embedding + SigLIP |

**没有的东西（同样重要）**：

- ❌ **没有官方 leaderboard，也没有"官方数字表"**；
- ❌ **没有独立发布的评估脚本** —— 指标由 RecBole 内置 evaluator 提供，K 值写在仓库自带的 yaml 里；
- ❌ **没有指定 `last_out` 与 `timestamp` 哪一套是官方口径** —— 两套并列发布，由引用者自选。

> 官方论文（arXiv:2403.03952；v2 已改名为 *BLaIR: Benchmarking LLMs as Semantic Encoders*，ACL 2026）
> 报的是一套 unified benchmark（sequential rec + collaborative filtering + product search）。
> 但**官方仓库自己声明**：*"The original code has been refactored… statistics of the processed dataset,
> as well as the recommendation model results **could be slightly different from the numbers in our paper**."*
> → 连"官方论文数字"与"官方参考实现"都不保证逐格对齐，第三方更不该把它们当硬目标线。

### 1.5.1 两套官方切分协议（都是官方发布的）

| 协议 | 官方原文 | 谁在用 |
|---|---|---|
| `last_out` | *"For each user, the latest review will be used for testing, the second latest review will be used for validation, and all the remaining reviews are used for training."* | CCFRec / MTGRec 等绝大多数论文 |
| `timestamp` | *"we find two timestamps and split **all the reviews from Amazon Reviews 2023 dataset** in a ratio of 8 : 1 : 1"* —— train `(-∞, t₁)`、valid `[t₁, t₂)`、test `[t₂, +∞)`，**t₁ = 1628643414042、t₂ = 1658002729837**（毫秒，**跨全部类目共用同一对时间点**） | **本项目** |

官方对 `timestamp` 的定位（原文，**这是本项目选它的直接依据**）：

> *"Recommender systems in the real world only access interactions that occurred before a specific timestamp,
> and aim to predict future interactions. This strategy **aligns with real-world scenarios but is not widely used in research**.
> **Researchers are encouraged to experiment with this splitting strategy.**"*

→ **本项目选 timestamp 有官方背书**，且是官方自己承认"更贴近真实场景"的那一套，不是标新立异。
（也解释了为什么两个域的 test 比例不同：8:1:1 是**跨全数据集**的交互级比例，不是按类目配平的。）

### 1.5.2 三套口径并列（**决定"能不能横向比数字"的那张表**）

数据来源（2026-09-13 读官方源码/官方统计页，非估算）：
官方参考实现 = `seq_rec_results/dataset/process_amazon_2023.py` + `config/overall.yaml`；
0core 统计 = <https://amazon-reviews-2023.github.io/data_processing/0core.html>。

| 维度 | 主流论文（CCFRec / MTGRec） | 官方参考实现（BLaIR / RecBole） | **本项目** |
|---|---|---|---|
| core 过滤 | **5core** | **0core** | **5core** |
| 切分方式 | `last_out`（LOO，末条 test） | `timestamp`（t₁ / t₂ 官方写死） | **`last_out`（LOO）**（v2 切换） |
| I&S 商品空间 | 25,848 | **427,500** | 25,848 |
| VG 商品空间 | 25,612 | **137,200** | 25,612 |
| **相对本项目候选空间倍数** | 1×（I&S）/ 1×（VG） | **16.5×（I&S）/ 5.4×（VG）** | 1× |
| 样本形态 | 每用户 1 条（LOO） | 每行 = 一次交互 + 其历史（`_w_his`） | 自建滑窗（语义等价） |
| 历史截断 | 20 | **50**（`--max_his_len 50`） | 20（`--history_len 20`） |
| 候选集 | 全库 | **全库**（含仅 test 出现的冷启动物品） | **全库**（含冷启动物品） |
| 负采样 | 无 | `train_neg_sample_args: ~`（无） | 无 |
| 指标 | Recall@5/10、NDCG@5/10 | `[Recall, NDCG]` × `topk [10, 50]` | HR@1/5/10、NDCG@1/5/10、MRR、coverage@10、gini@10 |
| 模型选择 | — | `valid_metric: NDCG@10`、`stopping_step: 10` | valid NDCG@10 + early stop |
| beam | MTGRec 50 | — | `sid_gr` 20 |

三条必须一起讲的结论：

1. **三套口径两两都不能横比。** 本项目与官方参考实现**切分相同、core 不同**（候选空间差 5~16×）；
   与主流论文**core 相同、切分不同**（任务难度差一个量级）。没有"两两都对得上"的那一对。
2. **本项目选的这一套有明文依据**：官方发布的文件 + 官方鼓励的协议 + 最贴近线上（纯未来外推）。
   所以对外报数时，口径描述可以直接引用 `§1.5.1` 的官方原文，不需要自我辩护。
3. **想和任何一方对标，唯一办法是在同一份数据上重切再跑**（已列入 §7 待办）。
   在那之前，`RESULTS.md` 里所有数字的**唯一合法用法是组内排序**。

### 1.5.3 LOO 是怎么切的、本项目为什么切了它、又付出了什么代价

> **决策记录（2026-09-13，定版）**：本项目**主榜切到 LOO**（plain leave-one-out + sliding-window train）。
> 完整决策与代价见 `docs/EVAL_PROTOCOL.md §0 / §1.4`；本节是文献侧的事实核查。

#### ① LOO 在文献里到底怎么做？

学界惯例（TIGER / LETTER / LC-Rec / IDGenRec / CCFRec / MTGRec / EAGER / GRID 共 8 篇 Amazon 系论文，2026-09-13 逐篇核实原文）：

> *"**using the last item per user for test, second-to-last for validation, and the rest for training**."*（GRID §4，arXiv:2507.22224）
> *"we employ the **leave-one-out** strategy… the last interacted item is used as testing data, the second-last item is used as validation data"*（CCFRec §4.1.3，arXiv:2503.12183）
> 同样原文见 MTGRec §3.1.3（arXiv:2504.04400）、TIGER §4.1（arXiv:2305.05065）、LC-Rec §4.2（arXiv:2311.09049）。

→ 严格 LOO = 每用户 1 valid + 1 test + sliding-window train。

#### ② ⚠️ 但学界**还有两篇专门论文**论证 LOO 有问题

**A. Ji et al., TOIS 2023（arXiv:2010.11060）** —— 标题就叫《A Critical Study on Data Leakage in Recommender System Offline Evaluation》：

> *"**leave-one-out split and any other split that do not observe global timeline would lead to data leakage**…
> a model learns from the user-item interactions that are **not expected to be available at prediction time**…
> all models indeed recommend **future items that are not available at the time point of a test instance**…
> data leakage **does impact models' recommendation accuracy. Their relative performance orders thus become unpredictable** with different amount of leaked future data in training."*

实验：BPR / NeuMF / SASRec / LightGCN × MovieLens-25M / Yelp / Amazon-music / Amazon-electronic。
其提出的替代方案 = **"timeline scheme"** = **就是全局时间线切分** = v1 timestamp。

**B. Meng et al., RecSys 2020（arXiv:2007.13237）**：

> *"the splitting strategy employed is an **important confounding variable** that can **markedly alter the ranking** of state-of-the-art recommender systems, making much of the currently published literature **non-comparable**, even when the same datasets and metrics are used."*

→ 光换切分就能让 SOTA 排名翻天。

#### ③ 为什么切了它？（决策理由）

唯一站得住的理由：**与同类目 7/8 论文对齐**。TIGER / LETTER / LC-Rec / IDGenRec / CCFRec / MTGRec / EAGER / GRID 全是 LOO。
本项目目标是 SFT/RL 阶段的"对标对比"，**协议相同是最低要求**。
不切 LOO 的话，"我比 MTGRec 强多少"这句话永远说不清。

#### ④ 切 LOO 的可量化代价（**没得选就要接受**）

| 代价 | 量化 | 备注 |
|---|---|---|
| **冷启动命题从主榜消失** | 5core + LOO → test 目标几乎全部 ≥5 次交互，冷目标占比 **≈0**（v1 是 31% / 49%） | "内容派生 SID ⇒ 新商品天然可召回"这条最强发现**整列消失** |
| **数据泄漏** | LOO 让模型见过目标之前的近邻交互（Ji TOIS'23 论证） | 同协议下所有 baseline **共享**这份偏差，组内排序可比 |
| **不代表线上** | LOO 不像 `timestamp` 那样"纯未来外推" | 官方原文：timestamp *"aligns with real-world scenarios"* |
| **不可引 MiniOneRec 对比** | MiniOneRec 用的**也是全局时间切分**（arXiv:2510.24431 附录 B 原文 + 本地复刻项目**实测产物**核实，三条独立证据，详见 §1.5.4 附注） | MiniOneRec-style sliding-window time-split 改在附录单独跑（§6 待办） |

#### ⑤ v1 → v2 的差额（实测已验证，2026-09-13）

| 域 | v1 timestamp test 数 | v2 LOO test 数 | 倍数 |
|---|---:|---:|---:|
| I&S | 13,232 | **50,982** | **3.85×** |
| VG | 11,276 | **94,759** | **8.4×** |

> v2 的 50,985 / 94,762 与同类目论文 CCFRec / MTGRec / TIGER 报的 I&S / VG 用户数**完全相同**（统计表逐格一致），证实协议对齐无误。

**冷启动命题补救**：见 §6 待办（`MiniOneRec-style sliding-window time-split` 附录）。

### 1.5.4 GRID（CIKM'25）是怎么切的？（2026-09-13 读原文 + 读仓库源码核实）

> 证据来源：论文 arXiv HTML v1（<https://arxiv.org/html/2507.22224v1>）
> + 仓库 `snap-research/GRID` 的 raw 源码（yaml 配置 / `eval_metrics.py` / `pre_processing.py` / `interfaces.py`）。
> 引用为原文摘录，代码类事实为逐行读取。

**它是什么**：Snap Research《Generative Recommendation with Semantic IDs: **A Practitioner's Handbook**》，
CIKM 2025（34th ACM CIKM）。**定位不是纯综述** —— 它是"**开源框架 GRID + 系统性组件消融**"：
把 SID 生成（RQ-KMeans / RQ-VAE / RVQ）× 生成模型（TIGER）× 各类 trick 做成可替换模块，
然后逐个消融。所以它比综述**更可用**：消融结论可以直接指导实现选择。

**① 数据集：不是我们的类目。**

> *"We evaluate on **5-core filtered Amazon Beauty, Sports, and Toys datasets**
> (Hou et al., 2024; Rajput et al., 2023)…"*

即 **P5 / TIGER 血统的那三个类目**（仓库 README 明说数据来自 P5 论文，
`data/amazon_data/{beauty,sports,toys}`）——**不是 Amazon Reviews 2023，也没有 I&S / VG**。
→ 与本项目 §1 表的判断一致：**这类工作的数字对我们只有方法论价值，没有数字价值。**

**② 切分：LOO（第三篇确认）。**

> *"…**using the last item per user for test, second-to-last for validation, and the rest for training**."*

→ 加上 CCFRec / MTGRec，这是**第三篇**独立确认 LOO 在该领域的统治地位。
结合 §1.5.3，结论不变：**LOO 是该领域不可绕开的默认协议**，本项目 v2 已对齐（详见 §1.5.3 ③④ 的代价说明）。

**③ 🔴 真正要记的一条：GRID 的评估不是全库排序，是 beam 内排序。**

这是本轮最有价值的发现，也是它**数字不能当目标线**的硬原因。证据在源码：

```python
# src/components/eval_metrics.py :: SIDRetrievalEvaluator.__call__
batch_size, num_candidates, num_hierarchies = generated_ids.shape
matched_id_coord = torch.all((generated_ids == labels), dim=2).nonzero()
target = torch.zeros(batch_size, num_candidates).bool()   # 候选集 = 生成出来的 SID
target[matched_id_coord[:, 0], matched_id_coord[:, 1]] = True
preds = marginal_probs.reshape(-1)                        # 在候选集内按概率排
```

注释原文写着 *"check if the generated IDs **contain** the labels"*、*"we set the matched IDs to true
**if they are in the generated IDs**"* —— 候选集就是**模型自己生成的 SID 列表**，
而推理配置 `configs/experiment/tiger_inference_flat.yaml` 里 `top_k_for_generation: 10`。

> → **GRID 的 `Recall@10` ≈ "目标有没有被生成进候选 + 在候选内排第几"，也就是我们 `RESULTS.md` 里的
> `beam_ceiling@K` 概念，而不是全库排序的 HR@10。** 论文自己也承认这是可调设计：
> *"The inference generation is conducted through beam search with KV-cache, with tunable hyper-parameters
> such as **beam width, whether search is restricted to valid SIDs**, etc."*；§4.2 还专门做了
> constrained vs. unconstrained beam search 的消融。

⚠️ **因此"GRID 的 Recall@10 = x"这句话必须连候选集一起读**，否则会和全库排序的数字直接混淆。
本项目 `sid_gr` 报的是**全库排序 HR@10（beam 只用于产出候选）+ 单列 `beam_ceiling`**，
口径比 GRID 更严，两者不可横比。

**④ 配置细节（可直接抄的工程参数）**

| 项 | GRID 取值 | 出处 |
|---|---|---|
| `sequence_length` | **120** | `tiger_train_flat.yaml` |
| `min_sequence_length` | 1 | 同上（`interfaces.py` 默认 10，配置覆盖为 1） |
| 批大小 | train 32 / eval 8 | 同上 |
| 训练预算 | `max_steps 320000`、`max_epochs 10` | 同上 |
| 模型选择 | `monitor: val/recall@5`、`patience: 10` | `callbacks.model_checkpoint` / `early_stopping` |
| 指标 | Recall@K / NDCG@K，`top_k_list: [5, 10]` | `eval.evaluator` |
| 结果稳定性 | *"All results are averaged over **5 runs with different seeds**"* | §4 Setup |
| 骨干 | **T5 encoder-decoder，各 4 层**、`d_model 128`、`vocab_size 256`、dropout 0.15、weight tying | `model:` 段 |
| 优化 | Adam `lr 1e-3`、`weight_decay 1e-4`、`accumulate_grad_batches 16` | `optim:` 段 |
| 码本 | 3 × 256（README 示例 `num_hierarchies=3, codebook_width=256`） | README Quick Start |
| 去重 | TIGER 式**追加 1 位** ⇒ 下游按 4 层训练 | README 注释 *"we add 1 for num_hierarchies because… we appended one additional digit to de-duplicate"* |

> 对照：本项目骨干是小 Transformer（`sid_gr` 约 1.1M）+ 3×256 码本 + `sequence_length 20`
> + **不做去重**（见 ⑥）。**码本 3×256 与 GRID 恰好同构**，这让我们和它的组件消融结论有可比性。

**⑤ 每轮 32 万步、5 seed 平均** —— 说明它的消融结论**统计上比单 seed 论文更可信**，
引用其结论时不必像引用单 seed 结果那样打折。

**⑥ ✅ 它的去重结论，独立背书了本项目的"不做唯一化"决策。**

> *"De-duplicating SIDs is essential for accurate retrieval. We compare two strategies:
> TIGER's method, which **appends a digit to SIDs to resolve collisions** ("With De-dup."),
> and a simpler approach of **randomly selecting an item when SIDs conflict**…"*
>
> *"both perform comparably, with **TIGER's strategy having a slight edge**. However, TIGER's approach
> **increases sequence length and decoding complexity**, and its requirement for **global SID distribution
> knowledge is impractical for large item sets**."*

→ 这正是本项目 `docs/SID_PIPELINE.md §3.3` 决策的**第二个独立来源**（第一个是 Snap 的 SIGIR'26
生产实践论文）：**追加码位换唯一性 = 用解码长度 + 全局分布依赖去换一点点收益，大规模物品集上不划算。**
GRID 是用 5 seed + 逐项消融得出这条的，比单篇经验更硬。

**⑦ 顺带：GRID 的 `RetrievalEvaluator` 默认 500 负采样。**
```python
should_sample_negatives_from_vocab: bool = True, num_negatives: int = 500,
```
这是**评估 SID/embedding 检索质量**那条路径用的，**不是** TIGER 生成评估（后者走 `SIDRetrievalEvaluator`）。
引用 GRID 数字时务必分清走的是哪个 evaluator。

**一句话总结**：GRID 切分 = **LOO**；类目 = **Beauty/Sports/Toys（2018 血统）**；
评估 = **beam 内排序（非全库）**。
→ 对"该不该用 LOO"没提供新论据（第三篇 LOO），但对"**怎么讲生成式数字**"和"**要不要做 SID 去重**"
提供了两条可直接引用的硬证据。

---

## 1.6 生成式召回论文到底怎么划分数据集？（2026-09-13 逐篇读原文核实）

| # | 论文 | 会议 / 年 | 数据集 | **切分** | 评估候选 | 指标（K） | history 上限 |
|---|---|---|---|---|---|---|---|
| 1 | **TIGER** | NeurIPS'23 | Amazon Reviews **1996–2014**：Beauty / Sports / Toys | 🔴 **LOO** | beam 生成（未写宽度） | R@5/10、N@5/10 | **20** |
| 2 | **LC-Rec** | ICDE'24 | Amazon'18：Instruments / Arts / Games | 🔴 **LOO** | **全库排序**，beam=**20** | HR@1/5/10、N@1/5/10 | **20** |
| 3 | **LETTER** | CIKM'24 | Amazon'18：Instruments / Beauty + Yelp22 | 🔴 **LOO** | Trie 约束生成 | R@5/10、N@5/10 | **20** |
| 4 | **EAGER** | KDD'24 | Amazon'18：Beauty / Sports / Toys + Yelp'19 | 🔴 **LOO** | beam=**100** | R@5/10/20、N@5/10/20 | **20** |
| 5 | **IDGenRec** | SIGIR'24 | Amazon：Sports / Beauty / Toys + Yelp | 🔴 **LOO** | **全库排序**（明文无负采样） | HR@5/10、N@5/10 | 未写 |
| 6 | **CCFRec** | KDD'25 | Amazon'23：**I&S / VG** | 🔴 **LOO** | 全库排序 | R@5/10、N@5/10 | 20 |
| 7 | **MTGRec** | SIGIR'25 | Amazon'23：**I&S / VG** | 🔴 **LOO** | beam=**50** | R@5/10、N@5/10 | 20 |
| 8 | **GRID** | CIKM'25 | Amazon'18（P5 血统）：Beauty / Sports / Toys | 🔴 **LOO** | **beam 内**（`top_k_for_generation: 10`） | R@10 | 120 |
| 9 | **MiniOneRec** | 2025 | Amazon'18：I&S / Office | 🟡 **全局时间 8:1:1** | 约束 beam | R@K、N@K | **10** |
| 10 | **Google SID** | RecSys'24 | YouTube 工业（视频） | 🟡 **时间**：前 N 天训练 / 第 **N+1** 天测 | 全库（AUC） | CTR、**CTR/1D** | — |
| 11 | **Temporal Cold-Items** | arXiv **2607.21101**（2026-07） | Amazon Beauty/Sports/Toys + 4 个 DyTAG | 🟡 **绝对时间滑窗协议** | — | R@5/20、N@20、**Hit@20** | — |
| 12 | **Cold-Start Reproducibility** | arXiv **2603.29845**（2026-03） | Amazon-Toys / MicroLens / Steam | 🟡 **时间 90/10** | full catalog | R@10、N@10 | — |
| 13 | **HSTU** | ICML'24 | ML-1M / ML-20M + Amazon **Books** | ⚠️ **full shuffle + multi-epoch**（既非 LOO 也非时间） | 全库 | HR@10/50/200、N@20/200 | — |
| 14 | **OneRec** | 2025（快手） | 工业内部（非公开） | ⚠️ 随机抽样 test case + RM 打分 | — | swt/vtr/wtr/ltr | 256 |

### 1.6.1 读出来的四件事

**① Amazon 系学术基准 8/8 全是 LOO。** TIGER / LC-Rec / LETTER / EAGER / IDGenRec / CCFRec / MTGRec / GRID —— 无例外。
所以"生成式召回用 LOO"这个印象是对的，但它是**社区惯例**（为了互相可比），不是**方法论结论**。

**② 用时间切分的四篇，目的都不是"更严格"，而是"命题本身要求"。**

- Google SID（RecSys'24）：*"evaluate the model's performance using AUC for CTR for the data from **(N+1)-th day** … We further slice the metric on items **introduced on the (N+1)-th day**. We refer to this as **CTR/1D** … evaluate the model's ability to **generalize over time** due to data distribution shifts and **cold-start items**."*
- arXiv:2607.21101：*"Unlike leave-one-out evaluation, which splits each user sequence independently, we **partition the entire interaction stream by absolute time** … This design ensures that the model is always **trained on past interactions and evaluated on future interactions**."*
- arXiv:2603.29845：*"we sort all interactions **chronologically** and use the **first 90%** for training. The remaining 10% are used for validation and testing."*
- MiniOneRec：全局时间 8:1:1（见 §1.5.3）。

**四篇都是为了回答"新物品 / 未来物品能不能被召回"。** 没有一篇是纯粹为了"更严谨"而切。

**③ 🔴 我们的"冷 / 热分桶"，与 arXiv:2607.21101 **逐字同构**。**

那篇的定义：

> *"A test target item i is considered **seen** if it appeared in the training period: `i ∈ I_seen ⟺ i ∈ I_train`, and **cold (unseen)** otherwise."*

我们的定义（`EVAL_PROTOCOL §3.3`）：**按"目标商品是否在 train 段出现过"把 test 样本拆成两组**。
→ 一模一样。加上它的测试集构造（绝对时间窗 τ_train < τ_val < τ_test）与我们的 t₁/t₂ 同构，
→ **本项目的主榜协议有三篇独立文献同构背书**（2607.21101 / 2603.29845 / Google RecSys'24）。

它还做了**比我们更细的 token 级分桶**（可抄）：

| 桶 | 定义 |
|---|---|
| `all-token-seen` | 冷物品，但 SID 的**每个** token 都在训练期出现过 |
| `any-token-unseen` | 冷物品，且**至少一个** token 未出现过 |
| `prefix-seen(ℓ)` | 冷物品，但其前 ℓ 个 token 组成的前缀在训练期出现过 |

**④ 🔴 它的一条结论，直接决定我们后续 SFT 的预期。**

> *"**TIGER's unseen Recall@20 is nearly zero on most datasets.** SASRec shows a similar pattern, with unseen Recall@20 below 0.001 on Beauty, Sports, Toys, WeiboDaily, and WeiboTech."*
> *"For TIGER, the near-zero unseen performance suggests that SID-based generation is **severely constrained outside the training item universe**."*

它评的是 **生成**（TIGER 必须逐 token 生成出正确 SID）；我们评的是 **检索**（`sid_prefix` 桶内直接召回）。
两者**不矛盾，互补**：

- 它证明：**生成器**对训练物品空间之外的目标近乎无能为力；
- 我们的 `sid_prefix` 证明：**同一份 SID 里**，冷目标的信号是存在的（冷目标 HR@10 = **0.0385 / 0.0405**，达热目标的 **80% / 94%**）。

→ 合起来正好落回 `EVAL_PROTOCOL §3.4` 那句话：**信息在 SID 里，输在生成器。**
→ **预期管理**：若后续 SFT 模型对冷目标也接近 0，那是**符合文献的**，不是实现 bug；
   要突破必须动**解码/打分接口**，而不是换 SID。该文的三个控制变体正是这个方向（`TIGER-SID` / `TIGER-Scorer` / `TIGER-Edge`，其中 Scorer = **以打分替代精确解码**）。

> ⚠️ **引用保留（必须一起讲）**：arXiv:2607.21101 目前是**预印本 + 公开评审**，评审报告指出两处硬伤——
> ① `all-token-seen` 用**位置无关**的 token 集合定义，忽略了 RQ-VAE 各层 token 的位置相关性 → 该分类的结论被削弱；
> ② 它称 *"Compared with LOO, TIGER suffers substantial degradation"*，但**全文未给出 LOO 的具体数字**，该对比**不可验证**。
> 因此它可以支撑"**时间协议更有意义**"这个方向判断，但**不能引用它的具体数值**。

### 1.6.2 对本项目的直接含义

| 决策 | 依据 |
|---|---|
| **主榜切到 LOO**（2026-09-13，完整记录见 `EVAL_PROTOCOL §1.4`） | 与 7/8 同类目论文（TIGER / LETTER / LC-Rec / IDGenRec / CCFRec / MTGRec / EAGER / GRID）**同协议**；可与论文量级对齐 |
| **冷启动命题补救**：MiniOneRec-style sliding-window time-split 附录（§7 待办） | 恢复 v1 的 31%/49% 冷启动占比与"内容派生 SID ⇒ 新商品天然可召回"这条最强发现 |
| `history` 上限 20 是**主流选择**（TIGER/LC-Rec/LETTER/EAGER/CCFRec/MTGRec 一致） | 本项目 20 ✅；GRID 用 120、MiniOneRec 用 10 属两端 |
| 冷目标 HR 的**闸门不能设高**（MiniOneRec 附录里要设） | 文献一致结论：生成式对 unseen 目标近乎失效（TIGER unseen R@20 ≈ 0） |
| **值得抄的增量**：token 级分桶（MiniOneRec 附录里做） | 比二值冷/热更细，能直接区分"粗桶选错"与"后期细化失败" → 已列入 §7 |

---

## 2. 经典（非生成式）baseline 清单

| 方法 | 论文 / 会议 | 开源 | I&S/VG 上有公开数字？ | 单张 4GB 卡可跑？ | 本项目 |
|---|---|---|---|---|---|
| **PopRec** | 启发式（无论文） | RecBole `Pop` | 未查到 | ✅ CPU 级 | **[已复现]** 零训练 |
| **ItemKNN** | 启发式 | RecBole `ItemKNN` | 未查到 | ✅ CPU 级 | **[已复现]** 零训练 |
| **BPR-MF** | Rendle 2012, arXiv:1205.2618 | 多实现 | 未查到（非序列） | ✅ | **[已复现]** |
| **GRU4Rec** | Hidasi 2016, arXiv:1511.06939 | hidasib/GRU4Rec | ✅ CCFRec 表 3 | ✅ | **[已复现]** |
| **SASRec** | Kang & McAuley, ICDM 2018, arXiv:1808.09781 | kang205/SASRec | ✅ CCFRec 表 3 | ✅ | **[已复现]** |
| **BERT4Rec** | Sun et al., CIKM 2019, arXiv:1904.06690 | FeiSun/BERT4Rec | ✅ CCFRec 表 3 | ✅ | **[已复现]** |
| **FMLP-Rec** | Zhou et al., WWW 2022, arXiv:2202.04212 | RUCAIBox/FMLP-Rec | ✅ CCFRec 表 3（该类目最好之一） | ✅ | [未复现] 需实现滤波层，本轮先控范围 |
| **S³-Rec** | Zhou et al., CIKM 2020, arXiv:2008.07873 | RUCAIBox/CIKM2020-S3Rec | ✅ CCFRec 表 3 | ✅ | [未复现] 需 4 个自监督预训练任务 |
| **DuoRec** | Qiu et al., WSDM 2022, arXiv:2112.09059 | StarsDict/DuoRec | ✅ CCFRec 表 3 | ✅ | [未复现] 需对比学习双分支 |
| **UniSRec** | Hou et al., KDD 2022, arXiv:2206.05941 | RUCAIBox/UniSRec | ✅ CCFRec 表 3（该类目最强经典） | ⚠️ 需 BERT 编码文本 | [未复现] 依赖预训练文本编码权重 |
| **CL4SRec / ICLRec** | arXiv:2010.14395 / 2202.08664 | HKUDS/SSLRec、salesforce/ICLRec | 未查到（原论文在 Beauty/Sports/Yelp） | ✅ | [未复现] |
| **LightGCN** | He et al., SIGIR 2020, arXiv:2002.02126 | kuandeng/LightGCN | 未查到（图方法，非序列） | ✅ | [未复现] |
| **RecFormer** | Li et al., KDD 2023, arXiv:2305.13731 | 复现实现较多 | 有（Amazon 2023 Scientific/Games）`[二手]` | ❌ Longformer-base，需 ≥8GB | [未复现] 超本地显存 |
| **RecBole / RecBole 2.0** | arXiv:2206.07351 | RUCAIBox/RecBole | 非方法本身，提供统一协议 | ✅ 视模型而定 | [未采用] 见 §5 决策说明 |

---

## 3. 生成式召回 baseline 清单

| 方法 | 论文 / 年份 | SID 构造 | 生成模型 | I&S/VG 上有数字？ | 算力 | 本项目 |
|---|---|---|---|---|---|---|
| **TIGER** | Rajput et al., NeurIPS 2023, arXiv:2305.05065 `[已核验]` | RQ-VAE 3×256 + **第 4 位消冲突码** | T5-small | 原论文无（Amazon 2014：Beauty / Sports and Outdoors / Toys and Games）；**MTGRec 在 I&S/VG 上复现了它** | 单卡数小时 | **[已复现]** 以 `sid_gr` 形态本地复现 |
| **MTGRec** | Zheng et al., SIGIR 2025, arXiv:2504.04400 `[已核验]` | RQ-VAE 多标识符（相邻 epoch ckpt 当"语义相关的另一个 tokenizer"） | T5 系 | ✅ **I&S + VG**（目前这两个类目上公开的最好生成式数字） | 单卡 | [未复现] 论文未给代码链接，且需要多 tokenizer |
| **LETTER** | Wang et al., CIKM 2024, arXiv:2405.07314 `[已核验]` | RQ-VAE + 协同对比 + 多样性正则 | 实例化于 TIGER(T5) / LC-Rec(LLaMA) | 原论文无（**Amazon 2018** Instruments / Beauty + Yelp22）；MTGRec 在 I&S/VG 上跑了它 | 小 | [未复现] 切分 = **LOO**，约束生成（Trie），history 20 — 见 §1.6 |
| **LC-Rec** | Zheng et al., ICDE 2024, **arXiv:2311.09049** `[已核验]` | RQ-VAE 4×256 + **USM 均匀语义映射消冲突** | LLaMA-7B + LoRA（**全库排序**，beam=20） | 原论文无（**Amazon 2018** Instruments / Arts / Games：HR@10 **0.1220 / 0.1266 / 0.1174**） | 多卡 | [未复现] 超本地显存。切分 = **LOO**，指标 HR@1/5/10 — 见 §1.6 |
| **EAGER** | Wang et al., KDD 2024, arXiv:2406.14017 `[二手]` | 语义树状 ID | T5 双流 | 原论文无 | 未明确 | [未复现] |
| **RPG** | Facebook, KDD 2025, arXiv:2506.05781 `[二手]` | OPQ（最长 64 token） | 小 Transformer（并行 MTP） | 原论文无（2018 Sports/Beauty/Toys/CDs） | 轻量 | [未复现] |
| **HSTU / generative-recommenders** | Meta, ICML 2024, arXiv:2402.17152 `[已核验]` | 非 SID（ID 类，可挂 SID） | HSTU pointwise attention | 原论文无（**ML-1M / ML-20M + Amazon Books**，未标年份）；工业级规模 | 工业级 | [未复现] ⚠️ 其公共数据集实验用 **full shuffle + multi-epoch**（既非 LOO 也非时间切分），指标 HR@10/50/200、全库 — 见 §1.6 |
| **OneRec** | 快手, arXiv:2502.18965 `[二手]` | RQ-KMeans 3 层 | Enc-Dec + MoE(Llama3) | 内部数据 | 工业级 | [未复现] |
| **MiniOneRec** | arXiv:2510.24431 `[二手]` | RQ-VAE 3 层 / RQ-KMeans | Qwen2.5 0.5B–7B | 报 Industrial / Office，但属 **Amazon 2018** 口径（本仓 V0 复刻记录） | 4–8× A100/H100 | [未复现] 是**本项目的下游目标**（M3~M5），不是 baseline |
| **GRID** | Ju et al., **CIKM 2025**, arXiv:2507.22224 `[已核验]`（仓库 `github.com/snap-research/GRID`，本仓 `refs/GRID` 有镜像） | RQ-KMeans / RQ-VAE / RVQ（**3×256**，与本研究同构） | TIGER（**T5 4 层 d128**，vocab 256） | **2018 P5：beauty / sports / toys**，非 Amazon 2023 | 单 GPU | [未复现] 配置可直接参考；🔴 **其评估是 beam 内排序、非全库**（`SIDRetrievalEvaluator`，`top_k_for_generation: 10`）——**数字不可与本表其他行混比**，详见 **§1.5.4** |
| **Snapchat 语义 ID 生产实践** | arXiv:2604.03949（SIGIR'26 Industry）`[二手]` | RQ-VAE（STE + 多模态融合） | 生产 GR 栈 | 内部数据 | 工业级 | [未复现] 本项目"语义桶"决策的依据来源 |
| **ETEGRec** | SIGIR 2025, arXiv:2409.05546 `[二手]` | RQ-VAE（端到端可学习） | T5 双 Enc-Dec | 称在 Amazon 2023 上超 TIGER/LETTER（类目与数字未核） | 小 | [未复现] |
| **SpecGR** | arXiv:2410.02939 `[二手]` | SID（TIGER 作验证器） | 检索式 drafter + 生成式验证器 | ✅ VG（但只报 @50，协议不同） | 小 | [未复现] |

**最低算力的生成式形态**（本案选用的）：TIGER 范式本身不依赖 LLM ——
RQ-VAE 把物品量化成 3~4 个 token，再用一个**从零训练的小 Transformer**（T5-small 约 13M，
本仓 `sid_gr` 约 1.1M）做自回归生成 + 前缀树约束解码。单卡分钟~小时级。
LLM 那一档（LC-Rec / MiniOneRec）是"用 LLM 的语义先验换效果"，属于**下一阶段的升级**，
不作为 baseline 的起点。

---

## 4. 公开数字锚点（逐页核验后抄录）

> 🔴 **先说结论：这两张表的切分与本项目不一致，数字只能当"量级参照"，不能当"要打败的目标线"。**
> 两篇论文用的都是 **leave-one-out（LOO）**（每用户最后 1 条作 test、倒数第 2 条作 valid、其余作 train），
> 本项目用的是**官方全局时间切分**（train ≤ 2021-08-11，test 从 2022-07-16 起，纯未来外推）。
> 两者**原始数据完全相同**（同一份官方 5core/timestamp 分片，见 §4.1 末尾的统计对照），
> 但 **test 集不是同一批样本**。逐项差异与量化对照见 **§4.3**。

### 4.1 经典方法：CCFRec (KDD'25) Table 3 `[已核验]`
来源：<https://arxiv.org/html/2503.12183v2>（Table 3）
协议：Amazon 2023 四子集 · **leave-one-out** · **全库排序** · K=5,10 · 单正样本 ⇒ **R@K ≡ HR@K**

**Industrial_and_Scientific（表中列名 Scientific）**
| 方法 | R@5 | **R@10 = HR@10** | N@5 | N@10 |
|---|---|---|---|---|
| GRU4Rec | 0.0230 | 0.0374 | 0.0148 | 0.0194 |
| BERT4Rec | 0.0186 | 0.0296 | 0.0119 | 0.0155 |
| SASRec | 0.0259 | **0.0412** | 0.0150 | **0.0199** |
| FMLP-Rec | 0.0269 | 0.0422 | 0.0155 | 0.0204 |
| S³-Rec | 0.0263 | 0.0418 | 0.0171 | 0.0219 |
| DuoRec | 0.0245 | 0.0379 | 0.0166 | 0.0209 |
| UniSRecT | 0.0296 | 0.0469 | 0.0191 | 0.0246 |
| UniSRecID+T | 0.0286 | 0.0457 | 0.0157 | 0.0214 |

**Video_Games（表中列名 Game）**
| 方法 | R@5 | **R@10 = HR@10** | N@5 | N@10 |
|---|---|---|---|---|
| GRU4Rec | 0.0530 | 0.0820 | 0.0350 | 0.0443 |
| BERT4Rec | 0.0460 | 0.0735 | 0.0298 | 0.0386 |
| SASRec | 0.0535 | **0.0847** | 0.0331 | **0.0438** |
| FMLP-Rec | 0.0528 | 0.0857 | 0.0338 | 0.0444 |
| S³-Rec | 0.0485 | 0.0769 | 0.0315 | 0.0406 |
| DuoRec | 0.0559 | 0.0844 | 0.0378 | 0.0469 |
| UniSRecT | 0.0587 | 0.0925 | 0.0372 | 0.0480 |
| UniSRecID+T | 0.0563 | 0.0921 | 0.0347 | 0.0459 |

> 同页 Table 2 的数据集统计：Scientific **50,985 用户 / 25,848 商品 / 412,947 交互**（稀疏度 99.969%，平均序列长 8.10）；
> Game **94,762 / 25,612 / 814,586**（99.966%，8.60）。
> 与本项目 `docs/DATASET.md §4.2` 的**原始分片统计完全一致** → 同一份官方分片，唯一差异是切分方式。

### 4.2 生成式方法：MTGRec (SIGIR'25) Table 2 `[已核验]`
来源：<https://arxiv.org/html/2504.04400v3>（Table 2）
协议：Amazon 2023 三子集 · **leave-one-out** · **全库排序** · K=5,10 · **beam=50**

| 数据集 | 方法 | R@5 | **R@10 = HR@10** | N@5 | N@10 |
|---|---|---|---|---|---|
| I&S | TIGER | 0.0264 | 0.0422 | 0.0175 | 0.0226 |
| I&S | LETTER | 0.0279 | 0.0435 | 0.0182 | 0.0232 |
| I&S | TIGER++ | 0.0289 | 0.0450 | 0.0190 | 0.0241 |
| I&S | **MTGRec** | **0.0322** | **0.0506** | **0.0212** | **0.0271** |
| VG | TIGER | 0.0559 | 0.0868 | 0.0366 | 0.0467 |
| VG | LETTER | 0.0563 | 0.0877 | 0.0372 | 0.0473 |
| VG | TIGER++ | 0.0580 | 0.0914 | 0.0377 | 0.0485 |
| VG | **MTGRec** | **0.0621** | **0.0956** | **0.0410** | **0.0517** |

> **这两张表合起来是本项目唯一可用的同代参照系**。注意 TIGER 在 I&S 的 0.0422 与 SASRec 的 0.0412 几乎持平——
> 这正面印证了本项目把 `sid_prefix`（零训练 SID 检索）和 `sid_gr`（小 Transformer 生成）
> 一起做对照的必要性：**生成式范式的增益不能靠"跟弱 baseline 比"来证明**。

### 4.3 切分差异逐项对照（**这一节决定第四节能不能用**）

**来源核对**（2026-09-13 重读 arXiv HTML 原文，非凭记忆）：
- CCFRec §4.1.3 原文：*"we employ the **leave-one-out** strategy for dataset splitting… the last interacted item is used as testing data, the second-last item is used as validation data, and all remaining items are used for training."*
- MTGRec §3.1.3 原文：*"we apply the **leave-one-out** strategy to split training, validation, and test sets… Additionally, the beam size of autoregressive decoding is set to 50."*

| 维度 | 论文（CCFRec / MTGRec） | 本项目（v2） | 是否一致 |
|---|---|---|---|
| **原始数据** | Amazon 2023 官方 5core 分片 | 同一份官方分片 | ✅ **一致** |
| 原始统计 I&S | 50,985 用户 / 25,848 商品 / 412,947 交互 | 同（`DATASET.md §4.2`） | ✅ 逐格相同 |
| 原始统计 VG | 94,762 / 25,612 / 814,586 | 同 | ✅ 逐格相同 |
| **切分方式** | **leave-one-out**（末条 test、倒二 valid） | **leave-one-out**（同；`scripts/data/prepare_amazon23_loo.py`） | ✅ **一致** |
| **I&S test 样本数** | **50,985**（= 用户数，每用户 1 条） | **50,982**（重跑后实测；差 3 条 = 无法构造 test 的用户） | ✅ 一致 |
| **VG test 样本数** | **94,762**（= 用户数） | **94,759**（重跑后实测；差 3 条） | ✅ 一致 |
| **test 目标冷启动占比** | ≈0（推断） | ≈0（LOO + 5core 决定，**冷启动命题改在 MiniOneRec-style 附录跑**，§7 待办） | ✅ 一致（都不测冷启动） |
| 商品数 | 25,848 / 25,612 | 25,848 / 25,612（v2 LOO 不过滤无 title 物品，跟论文对齐） | ✅ **一致** |
| 指标 | full ranking + R@K / NDCG@K，K=5,10 | 同 + HR@1、MRR、coverage；**单正样本 ⇒ R@K ≡ HR@K** | ✅ 一致 |
| 序列长度上限 | 20 | 20（`--history_len 20`） | ✅ 一致 |
| beam | MTGRec **50** | `sid_gr` **20**（GRID = 10） | ❌ 不一致（已列入 §7 beam 消融） |

**结论（v2，三条）**：

1. **本项目与同类目 7/8 论文协议完全相同**：同一份 5core 数据、同一份 LOO 切分、同样的 hist 长度、同样的全库排序候选集 →
   绝对数字可以做**量级对齐**（不再只是"目标线参考"，是"能直接对照"）。
2. **唯一不一致的仍是 beam 宽度**：MTGRec=50 / TIGER=20 / GRID=10，本项目 sid_gr=20；`beam_ceiling` 是必报的解码上限，
   让读者自己换算到任意 beam 下的成绩。
3. **冷启动命题** 不再是主榜的核心指标（LOO 下冷目标 ≈0），
   改在 MiniOneRec-style sliding-window time-split 附录单独跑（§7 待办），恢复 v1 的 31% / 49% 占比 + sid_prefix 冷目标 80% / 94% 这条原始发现。
3. **因此不要写"本项目的 sid_prefix 已经超过文献 SASRec"这类话。** 两个数字
   （0.0449 vs 0.0412）落在 10% 以内，是**两套完全不同的协议下的巧合**，不构成任何证据。
   本项目内部唯一有意义的比较是 `RESULTS.md` 里**同一份切分**的各模型排序。

> **想真正对标 MTGRec 的 I&S 0.0506 / VG 0.0956，必须先对齐切分**：
> 在**同一份原始 5core 数据**上按 LOO 重切一轮再跑（已列入 §7 待办）。
> 在那之前，第四节只能当"我们没跑偏"的体检表。

### 4.4 冷/热目标拆分：为什么内容类基线能赢 ID 类（**实测**）

### 4.4 v1 时代的冷/热诊断（LOO 下不适用，仅作历史记录）

> **2026-09-13 切到 LOO 后本节内容失效**，冷启动命题改在 MiniOneRec-style sliding-window time-split 附录里跑（§7 待办）。
> 这里保留是为了不让"内容派生 SID ⇒ 新商品天然可召回"这条原始发现断掉引用链——它在 v1 timestamp 协议下是成立的，三条实测结论是真的。

冷/热标注口径：`cold = (test 目标物品没在全量 train 交互里出现过)`。

| 域 | 模型 | 全部 HR@10 | 热目标 HR@10 | **冷目标 HR@10** | 热/冷 |
|---|---|---|---|---|---|
| IandS | pop | 0.0094 | 0.0136 | **0.0000** | ∞ |
| IandS | itemknn | 0.0043 | 0.0048 | 0.0032 | 1.52× |
| IandS | content_ann | 0.0092 | 0.0107 | 0.0058 | 1.84× |
| IandS | **sid_prefix** | 0.0449 | 0.0478 | **0.0385** | 1.24× |
| VG | pop | 0.0051 | 0.0099 | **0.0000** | ∞ |
| VG | itemknn | 0.0067 | 0.0120 | 0.0013 | 9.49× |
| VG | content_ann | 0.0114 | 0.0164 | 0.0061 | 2.66× |
| VG | **sid_prefix** | 0.0418 | 0.0430 | **0.0405** | 1.06× |

三条 v1 结论（**已不在 LOO 主榜下适用，但仍然是事实**）：

1. **纯流行度对冷目标完全失明**：两域 `pop` 的冷目标 HR@10 都**恰好是 0.0000**。
   这就是"ID-based 范式在 1/3 冷启动测试集上的结构性天花板"的具体数字。
2. **`sid_prefix` 几乎免疫冷启动**：冷目标 HR@10 达到热目标的 **80%（I&S）/ 94%（VG）**。
   → 本项目"用内容量化出 SID，于是新商品天然可召回"这条论据，
   第一次有了**可复现的实测数字**，而不只是原理陈述。
3. ⚠️ **但它只能解释差距的一部分，不能解释全部 —— 剩余部分尚未归因。**
   `sid_prefix` 在**热目标**上仍远高于已训练的 SASRec
   （I&S 0.0478 vs SASRec 全样本 0.0157 / 热目标约 0.0227）。
   最可疑的解释是 `DATASET.md §8.5` 的"同品多 listing"（孪生表示）：
   候选若与历史物品共享**完整 3 码 SID**（同桶孪生），`sid_prefix` 会直接给最高分。
   **下一个诊断**：统计 `sid_prefix` 命中里"来自同桶孪生"的比例，把
   "近似重复商品的召回"与"真正跨物品的语义召回"分开。
   **在跑完这个诊断之前，不要把 sid_prefix 的领先写成已定论。**

---

## 5. 五个容易踩的坑（都是本项目实际遇到的）

1. **2018 的数字长得像 2023 的，但完全不能混。** MiniOneRec 在 "Industrial" 上报的 HR@10 是 0.15 量级，
   比 2023 版高 3~4 倍——不是方法强 3 倍，是**数据集和切分都不同**（本仓 V0 复刻记录为 Amazon 2018，7 倍更小的物品库）。
2. **LOO vs 全局时间切分。** 本项目 test 段有 **33.06%（I&S）/ 33.15%（VG）** 的目标商品在 train 中从未出现
   （`docs/DATASET.md §8.3`）。这类样本对 ID-based 方法近乎不可解，会系统性压低所有 baseline 的绝对值。
3. **LOO 下 `Recall@K ≡ HR@K`**（每用户只有 1 个正样本）。看到论文报 R@10 可以直接当 HR@10 用，但要确认它确实是 LOO。
4. **生成式的 HR@K 是 beam-limited 的，而且这个坑学界自己也在踩。** beam=20 时最多只有 20 个候选 SID
   参与排序，HR@10 的上限被 beam 卡住。所以本仓额外报 **`beam_ceiling`**（目标是否出现在生成的 beam 里），
   把"模型不会生成"与"beam 太窄"分开。
   ⚠️ 更极端的例子是 **GRID（CIKM'25）**：它的 `SIDRetrievalEvaluator` 直接**在模型生成的候选集里排序**
   （`top_k_for_generation: 10`），所以它的 `Recall@10` 本质是 beam 内召回，
   **连全库排序都不是**（§1.5.4 ③）。→ **看到生成式论文的 R@10，先问"候选集是什么"。**
5. **碰撞处理路线不同，会影响可比性。** TIGER 给共享前 3 码的物品**追加第 4 位唯一码**（SID 长度 4）；
   本项目**不做唯一化**，同码物品成"语义桶"整桶召回（决策依据见 `docs/SID_PIPELINE.md` §1.4 决策表 + §3.6）。
   两者都能跑，但"ICR=1.0 + 4 token"与"ICR≈0.96 + 3 token"的端到端代价不可直接对等。
   ✅ **GRID（CIKM'25）用 5 seed 逐项消融独立验证了这条判断**：两种去重策略
   *"perform comparably, with TIGER's strategy having a slight edge"*，但 TIGER 式追加码位
   *"increases sequence length and decoding complexity"* 且依赖 *"global SID distribution knowledge
   … impractical for large item sets"* → 与本项目"不做唯一化"的取舍同向（§1.5.4 ⑥）。

---

## 6. 本轮为什么只本地复现这 9 个

| 决策 | 理由 |
|---|---|
| 复现 `pop / itemknn / bprmf / gru4rec / sasrec` | 5 个经典档，覆盖"启发式 → CF → 序列"三级；全部能在 4GB 卡上几十分钟内跑完。**`bert4rec` 已移出**（单轮 364~675 s 是 `sasrec` 的 25 倍，且未超过它，见 `run.py` registry 注释） |
| 复现 `content_ann / sid_prefix` | **零训练**，但恰恰是本项目最关键的对照：前者问"多模态融合向量本身值多少"，后者问"SID 本身的信息量值多少" |
| 复现 `sid_gr` | 生成式召回的最低成本形态（TIGER 范式），是"生成式 vs 检索式"的直接对照 |
| 暂不跑 `FMLP-Rec / S³-Rec / DuoRec / UniSRec / RecFormer` | 前四个要各自加模块（滤波层 / 自监督 / 对比分支 / 预训练文本编码器），边际信息量低于"先拿到同口径的骨架基线"；RecFormer 超显存 |
| 暂不接 RecBole | 本仓数据已是 RecBole atomic 格式（`.inter`），但 RecBole 会引入第二套评估口径与依赖树；本轮自己实现可以**保证与后续 SFT/RL 的 HR/NDCG 口径逐字一致**。注意：**官方参考实现本身就是 RecBole**（§1.5），所以"用 RecBole 交叉验证"是官方支持的路径，已列入 §7 待办 |

---

## 7. 待办（诚实列出）

- [ ] **借官方参考实现做第三方交叉验证**（优先级最高）。官方 `seq_rec_results/` 已给全套 RecBole 配置 +
  处理脚本（§1.5），两条可选路线：
  **(a) 复现官方口径** —— 拉 `0core_timestamp_w_his_{domain}`，跑 UniSRec / SASRecText，
  拿到"官方协议下的参考数字"，用来确认我们的实现没有整体跑偏一个量级；
  **(b) 灌入本项目数据** —— 把 `5core/timestamp` 的 `.inter` 转成 RecBole atomic 格式跑 SASRec，
  与自研 `sasrec` **逐格对表**。这才是"自研基线是否偏弱"的决定性检验。
  > 动机：`sid_prefix` 在**热目标**上仍高于自研 SASRec（§4.4），头号嫌疑是自研序列基线偏弱。
- [ ] **第二条交叉验证路径：GRID 开源框架（§1.5.4）**。与 RecBole 互补 —— GRID 走的正是本项目这条路
  （RQ-VAE / RQ-KMeans **3×256** + TIGER 自回归），所以它验的是 **`sid_gr`**，而 RecBole 验的是序列模型。
  可选：(a) 跑通 GRID 自带配置、复现它 beauty/sports/toys 的数字以校准我们的实现水平；
  (b) 把本项目 SID + 交互序列灌进 GRID，与 `sid_gr` 对表。
  ⚠️ **对表前必须统一口径**：GRID 的评估是 beam 内排序（`top_k_for_generation: 10`），
  而 `sid_gr` 是全库排序，不统一则数字不可比。
- [ ] §4.4 的孪生诊断：统计 `sid_prefix` 命中里"来自同桶孪生（共享完整 3 码 SID）"的比例，
  把"近似重复商品召回"与"真正跨物品的语义召回"分开。**跑完之前不要把 sid_prefix 的领先当定论。**
- [ ] 补 `FMLP-Rec`（CCFRec 表里经典档最强之一）与 `UniSRec`（该类目最强经典）。
- [ ] `sid_gr` 的 beam 宽度消融（10 / 20 / 50），把"生成能力"与"beam 限制"彻底分离。
- [ ] 与 MTGRec 的 I&S 0.0506 / VG 0.0956 做**同口径**对比：v2 已切到 LOO，**直接可量级对照**（重跑完成后做）；与 CCFRec 的 SASRec 0.0412 / 0.0847 同样；
  → ⏸ **正在执行**（2026-09-13，决策记录见 `docs/EVAL_PROTOCOL.md §1.4`）：v2 LOO 已切，16 组 baseline 重跑中（task OAx33E），跑完即可量级对齐。
  → "自研实现是否偏弱"另一条路 = 官方 RecBole 参考实现对表（已列入待办）。
- [ ] **token 级冷度分桶（抄自 arXiv:2607.21101，见 §1.6.1 ③）**：比现在的二值冷/热更细 ——
  `all-token-seen`（冷物品，但 SID 每个 token 训练期都出现过）/ `any-token-unseen`（至少一个 token 没见过）/
  `prefix-seen(ℓ)`（前 ℓ 个 token 的前缀在训练期出现过）。
  **价值**：能直接区分"**粗桶阶段就选错了**"与"**粗桶对了但后期细化失败**" —— 这是当前二值分桶看不出来的。
  零训练组即可先跑（看 `sid_prefix` 的冷目标命中落在哪个桶里），成本极低。
- [ ] **给冷目标 HR 设预期区间（而非越高越好）**：文献一致结论是生成式对 unseen 目标近乎失效
  （TIGER unseen R@20 ≈ 0，§1.6.1 ④）。后续 SFT 的闸门**不能**要求"冷目标也要高"，
  否则会为了一个文献上已知的天花板去调参；该做的是把瓶颈定位在解码/打分接口（同 §5 闸门 3）。
