# GenRetrieval — 多模态生成式召回系统 · 技术文档

> **这份文档是干什么的**：把整个项目从「数据 → 表征 → SID → SFT → RL → 评估」串成一条能讲清楚的链路，供**复习**与**面试**使用。
> 每个结论都标了来源：`[实测]` = 本机跑出来的真实数字；`[设计]` = 方案已定但实验待跑；`[V0]` = 引自 MiniOneRec 复刻版的既有结果。
> 配套文档：数据集完整档案 → `docs/DATASET.md`｜升级方案 → `docs/UPGRADE_PLAN.md`｜实验与踩坑 → `docs/EXPERIMENT_LOG.md`｜V0 原文档（归档）→ `docs/V0_MINIONEREC_TECH_DOC.md`

---

## 📋 一、项目概述

### 1.1 这个项目在做什么

一句话：**把推荐从「向量检索」改成「语言生成」，并让 item 的语义 ID 来自多模态内容而非纯文本。**

传统推荐（双塔 / 协同过滤）的做法是：给用户和商品各学一个稠密向量，算内积取 Top-K。本项目走另一条路：

```
用户历史（一串已购商品的 SID）
        │
        ▼
   LLM 自回归生成   ──→  下一个商品的 SID（3 个 token）
        │
        ▼
   SID → item_id 映射表  ──→  推荐列表
```

**为什么商品要用 SID 而不是直接生成 item_id？**
item 空间是 25,847（本项目）/ 数亿（工业级），作为词表太大且无语义结构。SID（Semantic ID）用**残差量化**把每个 item 的内容向量压成 3 层、每层 256 个码字的离散编码：

```
item "AmScope 显微镜套装"  ──Qwen3-Embedding──→  e_text ∈ R^1024
                                                     │
item 图像                ──SigLIP──────────→  e_img  ∈ R^768
                                                     │
                                              融合 e ∈ R^d
                                                     │
                             ┌───────────────────────┘
                             ▼
                   RQ-VAE / RQ-KMeans（残差量化）
                             │
                             ▼
                    <a_137><b_62><c_205>   ← 3 个 token，空间 256³ = 16.7M
```

于是「推荐」变成「让 LLM 生成 3 个 token」，词表只需新增 768 个 SID token（3 × 256），且**语义相近的商品共享 SID 前缀**（层级聚类结构），LLM 在生成时天然获得"同类商品"的泛化能力。

### 1.2 与 MiniOneRec（V0）的关系

本项目是 **MiniOneRec 的复刻 + 五个方向升级**。V0 的代码完整保留在仓库里（`git` 历史可回溯，原始提交 `ac0d0b9`），随时可重跑作对照。

| 模块 | V0（MiniOneRec 复刻） | 本项目（GenRetrieval） | 状态 |
|---|---|---|---|
| **数据** | Amazon 2018 I&S，3,686 items，纯文本 | **Amazon 2023 I&S，25,847 items，+图像** | `[实测]` 已完成 |
| **Item 表征** | Qwen text2emb（title + description） | **Qwen3-Embedding-0.6B 文本 + SigLIP 图像 → 门控融合** | `[设计]` 脚本就绪 |
| **SID** | FAISS RQ-KMeans 3×256 + Sinkhorn | **RQ-VAE 为默认**，RQ-KMeans / FAISS-RQ 作挑战者，三件套评估择优 | `[设计]` |
| **基座模型** | Qwen2.5-0.5B-Instruct | **Qwen3-0.6B**（teacher 用 Qwen3-1.7B） | `[设计]` |
| **SFT** | 全参、新 token 随机初始化、3 任务混合 | **SID token 语义初始化 + 课程学习 + QLoRA/全参双轨** | `[设计]` |
| **RL** | GRPO + 二值/NDCG/semantic/sasrec 奖励 | **路线 A：分层奖励；路线 B：OPD 在线策略蒸馏** | `[设计]` |
| **评估** | 端到端 HR/NDCG（SID 改动只能盲判） | **+ SID 质量三件套（ICR / 重构 MSE / LCP）** | `[设计]` |

### 1.3 三条核心增量（面试讲的三个点）

1. **多模态融合 SID** —— 图像引入文本没有的外观/风格信息，用「共现对比学习」训练门控融合层，使融合向量直接优化"推荐友好度"而不是随便对齐两个模态。融合质量用两个**不依赖端到端训练**的代理指标量化：

   - **共现检索 recall@k**（决定 SID 质量的那个）：`R@K = 1/Q · Σ_{q∈Q} 1[ Top-K(q) ∩ Gold(q) ≠ ∅ ]`
     —— 用融合向量在全库做余弦检索、排除自身，看 top-k 里有没有留出的共现伙伴；
     随机基线 `≈ avg_pos · K / N`。
   - **模态互检索 recall@k**：文本→图像、图像→文本（两侧维度不同，先岭回归学线性映射再检索），
     衡量两个模态是否线性可对齐。

   两者的完整口径与坑见 **§4.4**。
2. **SID token 语义初始化** —— V0 里新增的 SID token embedding 是**随机初始化**（`sft.py:169-170` 只做了 `add_tokens` + `resize_token_embeddings`），模型要花大量步数学"这些符号是什么"。本项目把 codebook 向量投影到 LLM 隐空间来初始化，让 token embedding 一出生就带着商品的语义。
3. **OPD 替代/增强 GRPO** —— 二值奖励在 SID 生成上是**极度稀疏**的（3 个 token 全对才给 1.0），GRPO 组内归一化后大量 group 优势全 0、无梯度。路线 A 用分层奖励（exact + 语义部分分 + 前缀命中 + 组内 NDCG）缓解；路线 B 用 teacher（Qwen3-1.7B）在线蒸馏提供**稠密**梯度信号，把 KL-to-ref 换成 KL-to-teacher。

---

## 🏗️ 二、技术架构

### 2.1 整体 Pipeline

```
┌──────────────────────────── 离线阶段（本项目 M1~M2） ────────────────────────────┐
│                                                                                  │
│  Amazon 2023 metadata (1.13GB)                   官方 5core/timestamp 分片        │
│         │                                                  │                     │
│         ├─ title / features / categories ──→ Qwen3-Emb ──→ e_text (N,1024)         │
│         ├─ images[] (100% 覆盖) ──下载 25,824 张──→ SigLIP ──→ e_img  (N,768)      │
│         │                                                  │                     │
│         └───────────────────────────────────────────→ 门控融合 ──→ e_fused        │
│                                                                   │              │
│                                          RQ-VAE / RQ-KMeans ◄─────┘              │
│                                                   │                              │
│                                    ┌──────────────┴───────────────┐              │
│                                    ▼                              ▼              │
│                          index.json (SID 映射)        SID 质量三件套评估           │
│                          <a_i><b_j><c_k>              ICR / MSE / LCP            │
└──────────────────────────────────────────────────────────┬───────────────────────┘
                                                           │
┌─────────────────────────── 在线训练阶段（M3~M5） ──────────┴───────────────────────┐
│                                                                                  │
│  ① SFT   3 任务混合训练（SidSFT + SidItemFeat + FusionSeqRec）                     │
│          + SID token 语义初始化 + 课程学习                                          │
│          Qwen3-0.6B ｜ 本地 4GB 卡 QLoRA ／ 云 3090 全参                            │
│                                    │                                             │
│  ② RL    路线 A：GRPO + 分层奖励    路线 B：OPD（KL-to-teacher, Qwen3-1.7B）        │
│                                    │                                             │
│  ③ 评估  Constrained Beam Search（前缀 hash 约束，保证 100% 合法 SID）             │
│          → HR@K / NDCG@K ｜ 消融矩阵 R0~R3                                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流动（具体到文件）

```
raw/Industrial_and_Scientific_5core_timestamp.{train,valid,test}.csv   (23.9 MB)
raw/meta_Industrial_and_Scientific.jsonl                              (1.13 GB)
                    │
                    │  scripts/data/prepare_amazon23.py
                    ▼
data/Amazon23/IandS/
   ├─ IandS.item.json       25,847 个商品：title/features/categories/brand/price/images
   ├─ IandS.item2id         asin → item_idx
   ├─ IandS.{train,valid,test}.inter    user_id \t "历史 item_id 列表" \t 目标 item_id
   └─ IandS.stats.json      统计摘要（进实验记录的数字来源）
                    │
                    │  scripts/multimodal/{encode_text,encode_image,fuse_embeddings}.py
                    ▼
data/Amazon23/IandS/emb/
   ├─ emb_text_title+features+category.npy   (N, 1024) L2 归一化
   ├─ emb_image_siglip.npy                   (N, 768)  L2 归一化（缺图=零向量）
   ├─ emb_image_mask.npy                     (N,) bool，标记 26 个无图物品
   ├─ emb_fused_gate.npy                     融合向量 → 下一步的量化输入
   └─ fusion_report.json                     两个代理指标的评估结果
                    │
                    │  rq/（RQ-VAE 或 rqkmeans_faiss.py）
                    ▼
   index.json  {item_id: ["<a_5>","<b_12>","<c_33>"], ...}
      +  rq/eval_sid.py  → ICR / 重构 MSE / LCP / prefix cohesion
                    │
                    │  convert_dataset.py → SFT/RL 训练样本
                    ▼
   {prompt: "用户历史 SID 序列", response: "目标 SID"}
```

### 2.3 目录结构

```
GenRetrieval/
├── docs/
│   ├── DATASET.md                    # 数据集完整档案 ← 复习数据用
│   ├── UPGRADE_PLAN.md               # 升级方案
│   ├── EXPERIMENT_LOG.md             # 实验记录 + 踩坑（含失败实验）
│   ├── V0_MINIONEREC_TECH_DOC.md     # V0 原技术文档（归档）
│   └── V0_MINIONEREC_RESUME.md       # V0 简历/STAR（归档）
├── scripts/
│   ├── data/{download_amazon23.sh, prepare_amazon23.py}
│   └── multimodal/{download_images.py, encode_text.py, encode_image.py, fuse_embeddings.py}
├── rq/                               # SID 构建与评估
│   ├── models/rqvae.py               # RQ-VAE 模型（从上游补回，见踩坑 E-01）
│   ├── rqkmeans_faiss.py             # FAISS 残差量化 + Sinkhorn（V0 沿用）
│   └── eval_sid.py                   # [待建] SID 三件套评估
├── sft.py / rl.py / minionerec_trainer.py   # V0 训练入口（原地升级）
├── convert_dataset.py / data.py             # 样本构造
├── LogitProcessor.py                        # 约束解码（前缀 hash）
└── data/Amazon23/                            # 数据产物（不进 git）
```

---

## 📊 三、数据集（摘要）

> 完整的统计、字段覆盖率、切分口径、踩坑与面试洞察见 **`docs/DATASET.md`**。这里只放面试最常被追问的部分。

### 3.0 核心设定：双域并行（不是合并）

本项目**用两个类目、各自独立成集、并行跑完整 pipeline**（编码 → SID → SFT → RL → 评估），**不做跨域合并**。

| | 域 A：**IandS**（工业器材） | 域 B：**VG**（电子游戏） |
|---|---:|---:|
| 商品数 | 25,847 | **25,612**（−0.9%） |
| 用户数 | 46,341 | 91,540 |
| 原始交互 | 412,947 | 814,586 |
| 单商品均交互 | 16.0 | **31.8** |
| 5★ 占比 | 72.30% | 63.84% |
| 域语义结构 | 参数/型号/规格驱动 | IP/题材/封面驱动 |
| 图像在决策中的角色 | 补充性 | **主导性** |

**为什么这么做（面试高频追问）**：单类目上的增益无法区分「方法真的有效」和「这个类目恰好合适」。两个**域距离最远、规模几乎相同**（差 0.9%，硬件与超参无需调整）的类目上都能复现，才算证明**方法本身**有效。判据事先写死在 `docs/DATASET.md §13`，避免事后找解释。

### 3.1 为什么从 2018 换到 2023

| V0 的硬伤 | 本项目如何解决 |
|---|---|
| 数据停留在 2016-10 ~ 2018-11 | 换成 2023 版，时间跨度延伸到 **2023-08** |
| 只有 title + description，**无图像** | 商品图 **25,824 张，有效覆盖率 99.91%** |
| item 仅 3,686 个，SID 空间只用 0.02% | item **25,847 个（7.0×）**，SID 利用率提到 0.15% |
| 单类目，泛化性无法自证 | **双域并行**（IandS + VG），见 §3.0 |

### 3.2 实测规模（`[实测]`）

| 指标 | IandS（域 A） | VG（域 B） |
|---|---:|---:|
| 数据集 | Amazon Reviews 2023 · 官方 5core 三段分片 | 同 |
| 商品数 | **25,848** | **25,612** |
| 用户数 | **50,985** | **94,762** |
| 原始交互（去重后） | **412,947** | **814,586** |
| **v2 LOO** train / valid / test 样本 | **208,999 / 50,984 / 50,982** | **435,534 / 94,761 / 94,759** |
| 历史长度上限 | 20 | 20 |
| 图像覆盖率 | **99.91%**（URL 侧 100%） | **99.84%**（URL 侧 100%） |
| 切分协议 | **plain LOO + sliding-window train**（v2，2026-09-13） | 同 |
| 与同类目论文对齐 | TIGER / LETTER / LC-Rec / IDGenRec / CCFRec / MTGRec / EAGER / GRID **7/8 同协议** | 同 |

### 3.3 四个必须能讲出来的数据特性（以 IandS 为例）

**① 评分极度偏斜 → 按隐式反馈处理**
5★ 占 **72.30%**（298,567 / 412,947），均值 4.4。（VG 为 63.84%，同样偏斜。）rating 不能当回归目标，本项目 `min_rating=0.0` 全保留，有交互即正向，与工业主流一致。

**② 长尾 + 用户极稀疏 → 这是选 LLM 范式的理由**

| 维度 | min | p50 | mean | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| 商品交互数 | 5 | 9 | 16.0 | 29 | 120 | **2,133** |
| 用户交互数 | 5 | **6** | 8.1 | 13 | 31 | 204 |

用户交互**中位数只有 6**，头部商品 vs 中位商品差 **238×**。ID-embedding 类模型在这种稀疏度下训练信号严重不足 → 用 LLM 的世界知识 + 商品文本/图像内容来补。

**③ 33.06% 的测试目标在训练集从未出现 → 真冷启动**

```
test 目标商品 12,164 个 → 其中 4,021 个（33.06%）未在 train 出现
```

ID-based 方法**结构上无法推荐**未见过的 item；SID 从内容量化而来，新商品只要有 title/图像就能编码 → **这是本项目选生成式召回的核心论据**。

**④ 时间漂移** train 跨越 2001~2021（20 年），test 是 2022-07~2023-08。可直接支撑"时序泛化"类消融实验。

### 3.4 与 V0 数据的对比

| 维度 | V0（2018） | 本项目（2023） | 倍数 |
|---|---|---|---|
| 商品数 | 3,686 | **25,847** | 7.0× |
| train 样本 | 36,259 | **251,576** | 6.9× |
| 商品字段 | title, description | + features/brand/categories/price/**images** | — |
| 切分 | **全局时间切分**（滑窗样本级 8:1:1） | **全局时间切分**（官方交互分片，每用户 valid/test 各 1 条） | 同族，粒度更严 |
| 类目数 | 1 | **2（独立成集并行）** | — |

> ⚠️ V0 的绝对指标与本项目**不可直接比较**（数据、切分都不同）。公平对比必须在本项目数据上重跑 V0 配置。

---

## 🧬 四、Item 表征：多模态编码 `[设计]`

### 4.1 文本侧：Qwen3-Embedding-0.6B

**为什么换掉 V0 的 text2emb？** 因为在 esci-ai-search 项目上有实测对照：把 `bge-base-en-v1.5`（MTEB-en 63.5）换成 `Qwen3-Embedding-0.6B`（MTEB-en 70.7）后，SID 的 **LCP +20%**、**prefix cohesion ×1.58~2.29**。SID 质量对上游 embedding 极敏感，所以这一步必须先做对。

**字段组合消融**（`--fields`）—— 因为 2023 metadata 的 `description` 覆盖率只有 **57.5%**，不能当必需项：

| 组合 | 覆盖率 | 说明 |
|---|---|---|
| `title` | 100% | 单字段基线 |
| `title+features` | 92.2% | features 是结构化卖点，信息密度高于 description |
| `title+features+category` | 97.3% | **默认**，再加品牌与类目路径 |

输出：`emb_text_<fields>.npy`，`(25847, 1024)` float32，**L2 归一化**。

### 4.2 图像侧：SigLIP-base-patch16-224

- 768 维，fp16 显存约 0.4GB，4GB 卡轻松跑；
- 选 SigLIP 而非 CLIP：同为小模型量级，SigLIP 的 sigmoid 对比损失在小 batch 下更稳，检索性价比更好；
- **无图/坏图 → 置零向量 + `mask=False`**，不丢物品，由门控自适应降权（实测只有 26 个，属于兜底路径）。

**最容易踩的坑**：图像与文本向量必须**行序严格一致**。脚本统一按 `sorted(int(idx))` 排序，否则融合会把商品 A 的图像接到商品 B 的文本上，而且**不会报错**，只会让指标莫名变差。

### 4.3 融合：三档模式

| 模式 | 做法 | 定位 |
|---|---|---|
| `text` | 纯文本向量 | 单模态基线（等价"不做多模态"） |
| `concat` | `[e_text ; e_img]` → L2 归一化（可 PCA 降到 1024） | 无训练目标的朴素基线 |
| `gate` | `e = g ⊙ W_t·e_t + (1-g) ⊙ W_i·e_i`，`g = sigmoid(MLP([e_t; e_i]))` | **主方案** |

**门控层的训练目标很关键**：不是简单对齐两模态，而是用 **InfoNCE 共现对比** —— 正样本 = 在同一用户行为序列中共现的商品对（从 `train.inter` 抽，上限 20 万对）。这样融合向量**直接优化"推荐友好度"**，而不是优化一个与下游无关的对齐目标。

### 4.4 融合质量的独立评估（不依赖端到端训练）

> 教训来源：V0 里 SID 改了只能靠端到端 HR/NDCG 盲判，而端到端实验很贵，导致"改一个地方要等几小时才知道好不好"。

| 指标 | 含义与公式 | 期望 |
|---|---|---|
| **模态互检索 recall@k** | 文本→图像、图像→文本：`R@K = 1/Q · Σ_q 1[ Top-K(q) ∩ Gold(q) ≠ ∅ ]`，Gold 是配对样本；两侧维度不同（1024 / 768），先在拟合子集解岭回归线性映射再检索 | 衡量两模态语义是否对齐；太低说明图像向量质量差 |
| **共现检索 recall@k** | 同式，但 Gold = 留出共现伙伴（与训练对互斥，否则会背答案）；随机基线 `R_rand@K ≈ avg_pos · K / N` | **真正决定 SID 质量**的代理指标 |

两个指标都是**秒级**出结果，可以在跑昂贵的 SFT 之前先筛掉坏方案。
⚠️ 绝对值随留出集的 `avg_pos` 浮动，**只能在同一留出集内横比**。

---

## 🔢 五、语义 ID 构建 `[设计]`

### 5.1 三条技术路线（不预设结论，用评估择优）

| 路线 | 实现 | 优势 | 风险 |
|---|---|---|---|
| **RQ-VAE（默认）** | `rq/models/rqvae.py`（上游补回） | 有重构损失，量化误差可控；学术界 SID 主流，可比性强 | 需训 encoder/decoder，且**码本坍塌**风险 |
| **RQ-KMeans** | `rqkmeans_plus.py` | 训练快（无梯度），无坍塌 | 无重构目标，量化误差不可控 |
| **FAISS-RQ + Sinkhorn** | `rqkmeans_faiss.py`（V0 沿用） | 在 esci 121 万向量上实测 119s 训完；**last-layer Sinkhorn** 版本 ICR 92.1%、LCP 19.21% | 依赖 FAISS，Sinkhorn 参数需调 |

**决策方式**：三者在同一份融合向量上跑，先过 §5.3 的三件套，再进 SFT 做端到端确认。**不预设 RQ-VAE 一定赢**（V0 用的是 RQ-KMeans，esci 项目上证过 FAISS-RQ + Sinkhorn 的 ICR/LCP 都很好）。

### 5.2 码本坍塌（Codebook Collapse）与 SID 碰撞

**坍塌**：只有分布在密集区的原型被选中 → 有梯度 → 继续存活；稀疏区的原型永远不被选中 → 梯度为零 → "饿死"。V0 的实测症状是第一层 codebook 利用率只有 48/256。

**碰撞**：多个 item 量化到完全相同的 SID → 生成式召回无法区分它们，召回上限被直接锁死。

**解法（Sinkhorn）**：把量化看成"分配问题"，用 Sinkhorn 归一化让每个码字被分配到大致等量的样本，即**均匀先验**。

关键经验（来自 esci 项目实测）：**只对最后一层做 Sinkhorn**。逐层都做（`all`）虽然把 ICR 做到 100%，但 **LCP 反而更差** —— 因为前两层的语义层级结构被"抹平"了。这是一个非常典型的"指标好看但结构变坏"的案例，面试时可以讲"为什么我没有把 ICR 当唯一目标"。

### 5.3 SID 质量评估三件套（本项目新增）

三个指标的**定义与公式**（与 `rq/eval_sid.py` 实现一致，完整版见 `docs/SID_PIPELINE.md §0.1`）：

| 层 | 指标（公式） | 回答什么问题 |
|---|---|---|
| **uniqueness** | **ICR 唯一码率** `ICR = 不同 SID 元组数 / N`，碰撞率 `= 1 − ICR`；**per-layer entropy** `H_l = −Σ_k p_{l,k} log p_{l,k}`（归一化 `H_l / log K`，K = 256），**死码率** `dead_l = 1 − 该层用过的码数 / K` | SID 是否唯一？码本用得均不均匀？ |
| **fidelity** | **重构 MSE** `MSE = 1/(N·D) · Σ_i ‖ x̂_i − x_i ‖²`，其中 `x̂_i = Dec( Σ_l C_l[c_{i,l}] )`；配套 `R² = 1 − MSE / Var(x)`；逐层累积版（只用前 k 层）回答"该用几层" | 量化丢了多少信息？ |
| **retrieval structure** | **LCP** `lcp(i,j) = Σ_{l=0}^{L-1} Π_{t≤l} 1[c_{i,t} = c_{j,t}]`；**LCP ratio** = 近邻对均值 ÷ 随机对均值（随机基线 ≡ 1）；**prefix cohesion** = 同前缀组内两两余弦均值 ÷ 随机分组（置换检验）同值 | SID 前缀是否真的语义聚类？ |

**为什么必须有这一层**：SID 是整条链路的地基。地基坏了，后面 SFT/RL 再训也只能在一个错的天花板下优化。三个指标各自便宜、秒级、可解释，且能互相制衡（ICR 高 ≠ SID 好，见上面的 Sinkhorn 案例）。

**消融矩阵**：`{纯文本, 图像, 融合} × {无 Sinkhorn, last-layer Sinkhorn} × {3×256（默认）, 2×1024, 4×64}`。

### 5.4 SID token 语义初始化（核心创新点）

**V0 的现状**（`sft.py:169-170`）：`tokenizer.add_tokens(new_tokens)` + `model.resize_token_embeddings(len(tokenizer))` —— 新增的 768 个 SID token embedding 是**随机初始化**的。模型必须先学会"这些符号代表什么"，再学"怎么用它们做推荐"，相当于把两个任务串行，浪费大量训练步数。

**本项目的做法**：SID 第 `l` 层 token `<a_i>` 的初始化向量由 codebook 向量投影到 LLM 隐空间得到：

```
E_init[<a_i>] = W · C_l[i] + b ,     W ∈ R^{d_llm × d_emb}
```

`W, b` 用 **ridge regression** 在全体 item 上拟合，锚点有两种取法（消融）：

| 版本 | 锚点 | 思路 |
|---|---|---|
| v1 | item title 经 LLM 编码的 mean-pooling 隐状态（冻结模型） | 让 SID token ≈ 该符号在语言模型里的"自然含义" |
| v2 | item 融合向量的 **RQ 重构向量** `Σ_l C_l[codes_l]` | 直接对齐"（SID token 组合）≈ item 语义" |

**为什么这个实验性价比极高**：`sft.py` 已支持 `freeze_LLM=True`，冻结 LLM + 只训 token embedding + 语义初始化 = 极小成本，**本地 4GB 卡就能跑**，但能给出一个干净的结论（初始化方式对收敛速度与最终 HR@10 的影响）。

---

## 🎯 六、SFT 监督微调 `[设计]`

### 6.1 三个训练任务（沿用 V0 设计，数据换成新的）

| 任务 | 样本形式 | 目的 |
|---|---|---|
| **SidSFTDataset** | 用户历史 SID 序列 → 下一个 SID | 主任务：序列推荐 |
| **SidItemFeatDataset** | SID → 商品标题 / 标题 → SID（各半） | 把 SID 与自然语言语义绑起来 |
| **FusionSeqRecDataset** | 历史 SID + 商品特征 → 下一个 SID | 让模型同时用上内容特征 |

### 6.2 本项目的三处升级

1. **SID token 语义初始化**（§5.4）—— 替代随机初始化；
2. **课程学习**：先做 title↔SID 的双向对齐（简单、信号密集），再做序列预测（难、信号稀疏）。理由：序列任务在稀疏数据上早期梯度很差，先建立 token 语义再去学序列规律会更稳；
3. **双轨训练**：本地 4GB 走 **QLoRA**（`bitsandbytes` 4bit + LoRA）做快速消融；云 3090 走**全参**出正式数字。两轨的配置差异必须记录，否则结论不可比。

### 6.3 训练配置（V0 参照，`[V0]`）

`batch_size=64, 3 epochs, lr=5e-4, bf16`；tokenizer 动态添加 SID token + 9 个特殊 token（评分、上下文类型等）。

---

## 🎮 七、强化学习：双路线 `[设计]`

### 7.1 V0 的问题（为什么必须改）

V0 的 GRPO 奖励（`rl.py`）有四种：`rule`（二值命中）、`ndcg_rule`（组内 NDCG 排序）、`semantic`（语义相似度）、`sasrec`（SASRec 打分）。核心问题是 **`rule` 奖励太稀疏**：

SID 是 3 个 token 的序列，**必须全部命中才给 1.0**，任何一位错都是 0。在 256³ 的空间里，早期模型几乎不可能全中 → 大量 group 的奖励**全为 0** → GRPO 组内归一化后 `advantage = (r - mean) / std` 分母为 0、优势全 0 → **该 batch 无梯度**，白跑。

### 7.2 路线 A：分层奖励设计

把"对/错"拆成**连续的部分分**，让"接近正确"也有梯度：

| 奖励项 | 定义 | 作用 |
|---|---|---|
| **exact** | 三层全中 = 1.0 | 主目标 |
| **semantic（部分分）** | 仅前 1 层命中 / 前 2 层命中 → 阶梯分 | 缓解稀疏：前缀正确说明模型抓到了"大类" |
| **prefix hit** | 命中 SID 前缀对应的**真实商品集合**（不只是精确 SID） | 修正量化的"意外碰撞"带来的误判 |
| **组内 NDCG** | 生成候选按 SID 前缀层级编码成排序，算组内 NDCG | 提供**相对排序**信号，天然避免全零 |

关键设计思想：**"奖励的稠密程度"要匹配"任务的稀疏程度"**。SID 生成是稀疏任务（16.7M 空间里的一个点），二值奖励的信息量太低，必须人为注入结构化的部分分。

### 7.3 路线 B：OPD（Online Policy Distillation）

**思路**：用更大的 teacher（Qwen3-1.7B，同一词表家族）在**在线**生成的轨迹上提供 token 级监督，把 GRPO 目标里的 `KL(π_θ ‖ π_ref)` 换成 `KL(π_θ ‖ π_teacher)`。

| | V0 的 GRPO | 本项目 OPD |
|---|---|---|
| 参考模型作用 | `π_ref`（SFT 初始化后的自己）只做**约束**，防跑偏 | `π_teacher`（Qwen3-1.7B）做**监督**，提供正向信号 |
| 信号密度 | 稀疏（序列级奖励） | **稠密（token 级分布）** |
| 能超越 SFT 上限？ | 受 `π_ref` 上限约束 | 受 teacher 上限约束（teacher 更强 → 上限更高） |

**风险**：学生被 teacher 的上限锁死；若 teacher 本身在推荐任务上不强，蒸馏反而有害。缓解：λ 退火调度（早期强蒸馏、后期让 RL 奖励主导）、或换更大 teacher（3090 场景可上 4B）。

### 7.4 实验矩阵（归因隔离）

| 编号 | 配置 | 要回答的问题 |
|---|---|---|
| **R0** | SFT only（无 RL） | baseline |
| **R1** | GRPO + 原二值奖励 | V0 配置在新数据上的表现 |
| **R2** | GRPO + 分层奖励（路线 A） | 稠密奖励带来多少增益？ |
| **R3** | OPD（路线 B） | 蒸馏 vs RL 哪个更适合 SID 生成？ |
| （可选 R4） | 分层奖励 + OPD 联合 | 两者是否互补 |

**这个矩阵的价值**：R2 和 R3 分别隔离了「奖励稠密化」和「监督信号来源」两个变量，能明确回答"性能提升到底来自哪里"，而不是把多个改动混在一起报一个总数。

---

## 📈 八、评估体系

### 8.1 端到端指标（沿用 V0）

- **HR@K**（命中率）：`HR@K = 1/U · Σ_u 1[ rank_u ≤ K ]` —— 真实下一个物品是否落进 top-K。
- **NDCG@K**（折损累计增益）：`NDCG@K = 1/U · Σ_u 1[ rank_u ≤ K ] / log₂(rank_u + 1)`
  —— 单正样本（next-item）⇒ IDCG = 1，所以 DCG 就是 `1/log₂(rank+1)`，未命中记 0；
  它比 HR 多一份"排得越靠前越好"的排序敏感性。
- K = 1, 3, 5, 10；beam search = 10。实现见 `utility.py: calculate_hit`。
- 解码用 **Constrained Beam Search**：预计算「前缀 token 序列 → 有效后继 token」的 hash 字典，每步把非法 token 的 logit 置 `-inf`，保证 **100% 生成合法 SID**
- 单次 `generate()` 同时生成 K 个候选，再 reshape 成每样本 K 个

### 8.2 分层评估（本项目新增，见 §4.4 / §5.3）

| 层级 | 指标 | 什么时候跑 |
|---|---|---|
| Item 表征 | 模态互检索 recall@k、共现检索 recall@k | 每次换 embedding/融合方式 |
| SID 质量 | ICR、重构 MSE、LCP、prefix cohesion | 每次换量化方案/层数 |
| 端到端 | HR@K、NDCG@K | 阶段收口时 |

**为什么要分层**：端到端实验贵（小时级），SID 与表征实验便宜（秒~分钟级）。分层后可以把大部分试错放在便宜的一层，只在关键决策点支付端到端成本。

### 8.3 V0 基线数字（`[V0]`，仅作历史参照）

| 阶段 | HR@10 | NDCG@10 |
|---|---|---|
| SFT | 0.093 | 0.060 |
| RL epoch1 | 0.103 | 0.070 |
| RL epoch2 | **0.109** | **0.074** |
| 官方 1.5B 权重 | 0.154 | 0.116 |

> GRPO 相对 SFT：HR@10 **+17%**、NDCG@10 **+23%**。
> ⚠️ 这些数字来自 **Amazon 2018 数据**（全局时间切分、滑窗样本级），与本项目的新数据**不可比**。本项目必须在新数据上重跑 V0 配置作为真正的 baseline（对应 R1）。

---

## 🛠️ 九、工程与踩坑（面试的"落地能力"素材）

### 9.1 环境与网络（实测踩坑，编号与 `docs/EXPERIMENT_LOG.md` §4 一致）

| # | 坑 | 事实 | 解法 |
|---|---|---|---|
| E-03 | `huggingface.co` 国内直连超时 | http=000 / 12s | 全走 `hf-mirror.com`（200 / 1.1s） |
| E-04 | `conda create` 在本机沙箱直接失败 | safe-delete 拦截 → `WinError 183` | 改用**项目内 `.venv`**（py3.11.7） |
| E-05 | 清华 / 中科大 pip 源 403 | `Could not find a version ... from versions: none`（**不是包不存在，是源拒绝服务**） | 换**阿里云** `-i https://mirrors.aliyun.com/pypi/simple/` + 官方源兜底。别信配置文件里配的源，要实测 |
| E-09 | **沙箱限速 51 倍** | 同一 URL：沙箱内 118 kB/s vs 沙箱外 **6.06 MB/s** | >100MB 的下载一律在沙箱外执行 |
| E-10 | `ps -W` 的 PID 不是 Windows PID | `taskkill` 报"找不到进程" | 用第 4 列 **WINPID** |
| E-11 | **上游仓库 `rq/models/` 目录整个缺失** | 4 个脚本都 `from models.rqvae import RQVAE`，RQ-VAE 路线开箱即崩 | 从上游取回 5 个文件，diff 全部 81 个 blob 确认对齐。教训：接手他人仓库先做**依赖完整性检查** |
| E-12 | `datasets==4.2.0` 与 `pyarrow==17` 依赖冲突 | `ResolutionImpossible`（像是"某两个包不共存"，实为版本配对错误） | 查 PyPI 元数据 → datasets 4.2.0 **要求 pyarrow>=21.0.0**。教训：**冲突先查上游包元数据，别靠猜** |
| E-13 | pip 重装 torch 卡在"删除旧版本" | 日志停住、CPU 时间不涨，疑似死锁 | 定位 `.venv/.../~orch`（卸载中转目录，4.1GB / 10,057 文件）手动清理。排查手法：**`ls -lat` 看最后被写入的文件**比看日志准 |

### 9.2 环境版本（实测确定）

| 包 | 版本 | 说明 |
|---|---|---|
| torch | **2.6.0+cu118** | 必须 ≥2.6：`minionerec_trainer.py` 直接 import trl 0.24 的内部符号（`selective_log_softmax` 等） |
| transformers | 4.57.1 | Qwen3 支持下限是 4.51 |
| trl | 0.24.0 | 自定义 GRPO 训练器的硬依赖 |
| datasets / pyarrow | 4.2.0 / **21.0.0** | 版本配对，不能随意降 pyarrow |
| numpy / pandas / scipy | 1.26.3 / 2.2.2 / 1.14.0 | 与 torch 2.6 兼容 |

硬件：**RTX 3050 Ti Laptop 4GB**（本地迭代）+ 云 3090（正式训练）。本地必须关掉 `torch_compile`。

---

## ❓ 十、面试问答

### Q1｜为什么用生成式召回，而不是双塔 / SASRec？

**S**：数据是 5-core 的 I&S，用户交互中位数只有 6 次，商品中位数 9 次、头部却有 2133 次。

**T**：在这么稀疏的行为信号下，做出比 ID-embedding 方法更好的召回。

**A**：
1. 分析瓶颈：ID-embedding 方法（SASRec / 双塔）要靠共现学 embedding，中位用户只有 6 次交互，尾部商品几乎没有训练信号；
2. 换表征来源：把商品用**内容**（title/features/图像）编码成 SID，LLM 用它的语言理解能力去建模"什么样的历史 → 什么样的下一个商品"；
3. 用数据验证动机：**test 里 33.06% 的目标商品在 train 完全没出现**，这是 ID 方法的结构性盲区，而 SID 从内容量化而来，天然可泛化。

**R**：方案在数据层面有明确论据支撑；分层评估体系保证每一步改动可归因（结果待 R0~R3 实验补全）。

---

### Q2｜SID 是什么？为什么不直接用 item_id？

**S**：item 空间 25,847（工业级 10⁸），直接生成 item_id 等于让 LLM 在一个巨大且无结构的词表上做分类。

**T**：设计一个既能压缩空间、又保留语义结构的离散表示。

**A**：
1. 用**残差量化**把 item 的融合向量压成 3 层编码：第一层粗粒度聚类（大类），后续层逐层残差修正 → `256³ = 16.7M` 的表示空间；
2. 好处 1：词表只需加 768 个 token（3×256），而不是 25,847 个；
3. 好处 2：**语义相近的商品共享前缀**，LLM 生成前缀正确就能召回整个语义簇，泛化性强；
4. 好处 3：**新商品只要有内容就能编码 SID**，无需重训（解决冷启动）；
5. 代价：量化有信息损失 → 所以必须用 ICR / 重构 MSE / LCP 三件套监控 SID 质量。

**R**：SID 空间利用率从 V0 的 0.02% 提到 0.15%，且支撑了 33% 冷启动场景的可召回性。

---

### Q3｜SID 碰撞（Collision）和码本坍塌是什么？怎么解决？

**S**：多个 item 量化到同一个 SID，或 codebook 里大量码字从未被使用（V0 实测第一层只用 48/256）。

**T**：在不破坏语义层级结构的前提下，让码本被均匀使用、降低碰撞。

**A**：
1. **诊断**：度量 ICR（碰撞率倒数）、per-layer entropy、码本利用率；
2. **手段**：Sinkhorn 归一化 —— 把量化视为带容量约束的分配问题，用 Sinkhorn 迭代求解，使每个码字获得的样本量大致均衡；
3. **关键取舍**：**只对最后一层做 Sinkhorn**。实测（esci 项目 121 万向量）逐层都做虽然 ICR = 100%，但 **LCP 反而更差** —— 前两层的语义层级被"抹平"了；
4. 结论：**ICR 高 ≠ SID 好**，必须同时看结构指标（LCP / prefix cohesion）。

**R**：确定 `last-layer Sinkhorn` 为默认配置，ICR 92.1%、LCP 19.21%，3 tokens；`all` 版本存档作对照。

---

### Q4｜GRPO 的奖励为什么设计成二值？稀疏性怎么处理？

**S**：SID 是 3-token 序列，二值 reward 要求三层全中，早期几乎不可能命中。

**T**：让奖励信号"稠密"起来，使早期训练也有梯度。

**A**：
1. 分析失效模式：group 内 reward 全为 0 → `advantage = (r - mean)/std` 分母 0 → 优势全 0 → **整批无梯度**；
2. 分层奖励：exact（全中）+ semantic 部分分（前 1 层 / 前 2 层命中，阶梯给分）+ prefix hit（命中前缀对应的真实商品集合）+ 组内 NDCG（相对排序）；
3. 核心思想：**奖励的稠密程度要匹配任务的稀疏程度**；
4. 同时保留 NDCG 类"相对奖励"，它天然不会全零。

**R**：得到 R1（二值）vs R2（分层）的对照实验设计，可量化稠密化带来的增益。

---

### Q5｜OPD 和 GRPO 的区别是什么？为什么考虑 OPD？

**S**：GRPO 里的参考模型 `π_ref` 只起"别跑太偏"的约束作用，不提供任何正向知识。

**T**：找到比"稀疏序列奖励"更高效的学习信号来源。

**A**：
1. GRPO 的 KL 项是 `KL(π_θ ‖ π_ref)`，`π_ref` 就是 SFT 后的自己 → **上限被自己锁住**；
2. OPD 把参考模型换成更强的 teacher（Qwen3-1.7B），KL 变成 `KL(π_θ ‖ π_teacher)` → 提供 **token 级稠密监督**；
3. 风险与缓解：teacher 上限锁定风险 → λ 退火（早期蒸馏、后期 RL 主导）、或换更大 teacher；
4. 用 R0~R3 矩阵做归因，不把多个变量混在一起。

**R**：形成"稠密奖励 × 蒸馏监督"两条独立路线的对照实验设计。

---

### Q6｜为什么给 SID token 做语义初始化？

**S**：V0 里新增的 768 个 SID token embedding 是**随机初始化**的。

**T**：缩短 SFT 学到"这些符号是什么"的过程。

**A**：
1. 现状分析：`sft.py` 只做 `add_tokens` + `resize_token_embeddings`，新 token 从随机开始 → 模型要先学"符号含义"再学"序列规律"，两个任务串行；
2. 方案：用 codebook 向量经 ridge regression 投影到 LLM 隐空间，`E_init[<a_i>] = W·C_l[i] + b`；
3. 锚点两种取法消融：v1 = title 的 LLM 隐状态（符号的自然含义）；v2 = RQ 重构向量（保证"token 组合 ≈ item 语义"）；
4. 工程优势：配合 `freeze_LLM=True`，只训 token embedding，**4GB 卡就能跑**，实验成本极低。

**R**：给出一个低成本、可归因的对照实验（随机初始化 vs 语义初始化）。

---

### Q7｜什么是 LCP？为什么它比 ICR 更能反映 SID 质量？

**S**（Situation）：两个 SID 方案的 ICR 分别是 92% 和 100%，看起来后者更好。

**T**：判断哪个方案的 SID 真的更有利于生成式召回。

**A**：
1. ICR 只衡量"唯一性"——不碰撞不代表结构合理；
2. **LCP（最长公共前缀）**衡量的是"语义相近的商品是否共享前缀"。生成式召回依赖前缀泛化：模型只要生成对前 1~2 层，就能命中整个语义簇；
3. 实测反例：`all` 逐层 Sinkhorn 把 ICR 做到 100%，但 LCP 反而下降 —— 因为均匀化抹平了层级结构；
4. 所以 SID 评估必须三层齐看：uniqueness（ICR）、fidelity（重构 MSE）、structure（LCP / prefix cohesion）。

**R**：形成三层评估框架并定 `last-layer Sinkhorn` 为默认，避免"指标好看但结构变坏"的陷阱。

---

### Q8｜多模态是怎么真正用起来的？怎么证明图像有用？

**S**：把图像编码后简单 concat 到文本向量上，未必能带来收益，甚至可能引入噪声。

**T**：让图像信息以"对推荐有用"的方式进入 SID。

**A**：
1. 不用朴素 concat 作主方案，而是**门控融合** `e = g ⊙ W_t e_t + (1-g) ⊙ W_i e_i`，让模型自己决定每个样本上两个模态的权重；
2. 门控的学习目标用 **InfoNCE 共现对比**（正样本 = 用户序列中共现的商品对），使融合向量直接优化"推荐友好度"；
3. **证明有用**用两个不依赖端到端训练的代理指标：模态互检索 recall@k（两模态是否对齐）、共现检索 recall@k（融合后是否更利于共现检索，附随机基线）；
4. 全流程可消融：`text` / `concat` / `gate` 三档 × `{无 Sinkhorn, last-layer Sinkhorn}`，最终由端到端 HR/NDCG 确认。

**R**：图像覆盖率 99.91% 提供了实验前提；评估链路可在分钟级完成筛选，避免在坏方案上烧端到端训练。

---

### Q9｜这个项目里你觉得最有价值的工程决策是什么？

**A**（可按实际情况挑）：
1. **分层评估体系**：把"贵"的端到端实验与"便宜"的表征/SID 实验分开，大部分试错在秒~分钟级完成，只在关键节点付端到端成本；
2. **不预设 SID 方案**：RQ-VAE / RQ-KMeans / FAISS-RQ 三条路线同时跑三件套择优，避免"因为用了某个方法所以它最好"的确认偏误；
3. **R0~R3 归因矩阵**：把"奖励稠密化"和"监督来源"两个变量隔离，结论可解释；
4. **数据管线务实**：用官方 5core 分片替代自跑 k-core，省掉 2.2GB 下载；图像下载做断点续传 + 魔数校验，26 分钟拿到 99.91% 覆盖。

---

## 📝 十一、简历 bullet（可直接用）

```markdown
多模态生成式召回系统（GenRetrieval）｜个人项目
- 基于 LLM 生成商品语义 ID（SID）的生成式召回范式，覆盖 数据构建 → 多模态表征 →
  SID 量化 → SFT → RL 全流程；数据规模 25.8k 商品 / 251.6k 训练样本（Amazon 2023 I&S）
- 构建多模态 item 表征：Qwen3-Embedding-0.6B（文本）+ SigLIP（图像，25.8k 张，
  覆盖率 99.91%）门控融合，以 InfoNCE 共现对比目标训练，并用互检索 / 共现检索 recall@k
  作为不依赖端到端训练的融合质量代理指标
- 设计 SID 质量三层评估框架（uniqueness: ICR ｜ fidelity: 重构 MSE ｜ structure: LCP +
  prefix cohesion），量化对比 RQ-VAE / RQ-KMeans / FAISS-RQ+Sinkhorn 三条量化路线，
  实证发现逐层 Sinkhorn 虽将 ICR 提到 100% 但 LCP 反而下降，据此确定 last-layer Sinkhorn 为默认
- 提出 SID token 语义初始化（codebook 向量经 ridge regression 投影到 LLM 隐空间），
  替代随机初始化，配合 freeze-LLM 可在单卡低成本验证收敛收益
- 针对二值奖励稀疏（3-token 全中才给分、group 内优势全 0 无梯度）设计分层奖励
  （exact + 语义部分分 + 前缀命中 + 组内 NDCG），并引入 OPD 在线策略蒸馏
  （KL-to-teacher 替代 KL-to-ref）提供 token 级稠密监督，以 R0~R3 矩阵做归因隔离
```

---

## 📚 十二、参考资料

| 主题 | 链接 |
|---|---|
| MiniOneRec（V0 来源） | https://github.com/AkaliKong/MiniOneRec ｜ arXiv:2510.24431 |
| Amazon Reviews 2023 数据集 | https://amazon-reviews-2023.github.io/ ｜ arXiv:2403.03952 |
| GRPO / DeepSeekMath | https://huggingface.co/papers/2402.03300 |
| Qwen3-Embedding | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B |
| SigLIP | https://huggingface.co/google/siglip-base-patch16-224 |
| 项目内文档 | `docs/DATASET.md`（数据）· `docs/UPGRADE_PLAN.md`（方案）· `docs/EXPERIMENT_LOG.md`（实验与踩坑） |

---

## 🔖 十三、一页速记

```
【是什么】LLM 生成商品 SID 的生成式召回；SID 来自多模态内容（文本+图像）的残差量化
【数据】  Amazon 2023 I&S ｜ 25,847 items ｜ 50,985 users ｜ v2 LOO train/valid/test = 208,999/50,984/50,982
          图像 25,824 张（99.91%）｜ 5★ 占 72.3% → 隐式反馈 ｜ 用户交互中位数 6（稀疏）
          33.06% 测试目标未在训练集出现（冷启动 → 生成式 vs CF 的核心论据）
          全局时间切分无泄漏（train ≤2021-08 < test 2022-07）
【五升级】①数据 2023+图像 ②多模态门控融合 SID ③Qwen3-0.6B ④SID token 语义初始化+课程学习
          ⑤RL 双路线：分层奖励 / OPD
【评估】  分层：表征（互检索/共现检索）→ SID（ICR/MSE/LCP）→ 端到端（HR@K/NDCG@K）
【V0 参照】SFT HR@10 0.093 → RL 0.109（+17%）｜ NDCG@10 0.060 → 0.074（+23%）｜官方 1.5B 0.154
          ⚠️ 2018 数据 + 滑窗样本级切分，与本项目不可直接比
【硬技能】 残差量化 / Sinkhorn / GRPO / 在线蒸馏 / 约束解码 / 多模态融合 / 稀疏奖励设计
【工程】 沙箱限速 51×（>100MB 出沙箱）｜ hf-mirror｜ 阿里云 pip 源｜ 项目内 .venv（conda 被沙箱拦）
         torch 2.6.0+cu118（trl 0.24 硬依赖）｜ datasets 4.2.0 必须配 pyarrow 21
```
