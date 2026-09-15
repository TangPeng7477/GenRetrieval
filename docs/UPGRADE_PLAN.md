# GenRetrieval 升级方案：MiniOneRec → 多模态生成式召回 2.0

> 状态：**已确认**（决策结果见 §13），**执行中**（2026-09-13 更新）
> 日期：2026-09-11（2026-09-13 刷新执行状态）
> 基线锚点：本项目复刻版 MiniOneRec（原 `D:\Codings\cs\Rec\MiniOneRec_oneGPU_preject`，已完整复制到本目录）

### 执行状态（2026-09-13）

| 里程碑 | 状态 | 交付 |
|---|---|---|
| M1 数据（Amazon23 I&S + VG，图文 100%/99.9% 可得） | ✅ 完成 | `docs/DATASET.md` |
| M1 多模态编码 + 融合 | ✅ 完成 | 定版 `gate_e60`：**I&S R@10 0.1100（vs 纯文本 +32.5%）｜VG 0.2113（+41.2%）**；指标定义见 §4.5 |
| M2 SID 构建（RQ-VAE） | ✅ 完成 | **定版配方 = gate + `--init_samples 8192` + 5000 轮 + `sid_raw`（语义桶）**，两域各跑一遍 |
| M2 SID 质量评估 | ✅ 完成 | 三件套双域：**I&S ICR 0.9582 / LCP 222.1 / R² 0.6530 / L0 死码 0%**；**VG ICR 0.9744 / LCP 163.6 / R² 0.8691 / L0 死码 3.52%**。VG 为跨域复验，未重做消融（公式见 §4.5；结论见 `SID_PIPELINE.md` §3.8） |
| M3/M4 基座与 SFT | ⏳ 待启动 | — |
| M5 RL | ⏳ 待启动 | — |

**两处相对原方案的调整**：

1. **碰撞不再消解**（原计划默认走 Sinkhorn）：碰撞组实拍是同系列规格变体，且 Snap 论文证明
   唯一性 ~70% 以上即平台期、其线上就是"一码多品 + 启发式消歧"→ 交付 `sid_raw`，`sid_sk` 存档。
2. **不做 raw vs sk 的 SFT 端到端消融**（原 §4.1 择优规则里的"小规模 SFT 一锤定音"）：
   时间与成本有限，决策依据改为离线结构指标 + 工业界先例，已在文档中标注为"未经端到端验证"。

👉 **SID 阶段的完整实验与结论整理见 `docs/SID_PIPELINE.md`**（新读者入口）。

---

## 0. TL;DR

在单卡可跑的 MiniOneRec 框架上做五个方向的系统性升级，全部保留 V0 作为公平对照锚点：

| 模块 | V0 现状 | 升级方向 |
|---|---|---|
| **数据** | Amazon 2018 I&S / Office（5-core，纯文本，~10k items）**（已按决策移除）** | **Amazon Reviews 2023** I&S 类目 + 商品图像 → 多模态 |
| **Item Embedding** | Qwen text2emb（title+description） | Qwen3-Embedding 文本向量 + SigLIP 图像向量 → 门控融合 |
| **SID 构建** | RQ-VAE / FAISS RQ 3×256 + Sinkhorn | **RQ-VAE 为默认**（RQ-KMeans/FAISS-RQ 作挑战者，三件套 + 端到端择优）+ 融合向量 + 归一化 + **新 SID token embedding 语义初始化** |
| **基座模型** | Qwen2.5-0.5B-Instruct | **Qwen3-0.6B**（同词表家族，迁移成本低）；teacher 用 Qwen3-1.7B |
| **SFT** | 全参、随机初始化新 token、任务混合 | 语义初始化 + 课程学习 + 任务配比消融；本地 4GB 卡走 QLoRA |
| **RL** | GRPO + 二值 rule / ndcg ranking / semantic / sasrec | 双路线：**A) 分层奖励设计**；**B) OPD 在线策略蒸馏**（KL-to-teacher 替代 KL-to-ref） |

核心增量创新点（面试可讲的三个）：
1. **多模态融合 SID**：图像+文本融合向量做残差量化，并用 ICR/LCP 三层评估体系验证融合有效性；
2. **SID token 语义初始化**：用 codebook 向量投影到 LLM 隐空间初始化新 token embedding，替代随机初始化（预期加速收敛、提升 SFT 上限）；
3. **OPD（Online Policy Distillation）替代/增强 GRPO**：teacher（Qwen3-1.7B）在线蒸馏提供稠密梯度信号，解决二值奖励稀疏问题。

---

## 1. V0 现状盘点（升级前基线）

### 1.1 现有 pipeline（复刻版实测，RTX 3090）

```
Amazon 2018 (5-core) ──> item.json (title/description)
        │
        ▼ text2emb (Qwen embedding, title+description)
emb-*.npy ──> 量化器（两套并存）:
        │      A) RQ-VAE (rq/models/rqvae.py + rq/trainer.py)   ← 本项目默认
        │      B) FAISS ResidualQuantizer 3×256 + optional Sinkhorn (rqkmeans_faiss.py)
        ▼
index.json (<a_i><b_j><c_k> 三层 SID)
        │
        ▼ convert_dataset.py ──> train/valid/test CSV (SID 序列化)
        │
        ▼ SFT (Qwen2.5-0.5B, 全参, 3任务混合: SidSFT + SidItemFeat + FusionSeqRec)
        │
        ▼ GRPO RL (minionerec_trainer, rule/ranking/semantic/sasrec 奖励)
        │
        ▼ evaluate (约束解码 + beam=10) ──> HR/NDCG
```

### 1.2 V0 指标（Industrial_and_Scientific, beam=10）

| 配置 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 |
|---|---|---|---|---|---|
| 0.5B SFT | 0.036 | 0.057 | 0.068 | 0.093 | 0.060 |
| 0.5B RL-epoch2 | 0.047 | 0.070 | 0.083 | 0.109 | 0.074 |
| 官方 1.5B 权重 | 0.085 | 0.113 | 0.133 | 0.154 | 0.116 |

### 1.3 已识别的问题（升级动机）

| # | 问题 | 证据/来源 | 对应升级 |
|---|---|---|---|
| P1 | 数据陈旧（2018，7 年前）且纯文本，商品外观/风格信息完全丢失 | 2018 数据只有 title/description | §3 |
| P2 | 新 SID token 随机初始化，SFT 需大量步数学"这些 token 是什么" | sft.py L160-170 只 add_tokens + resize | §5.3 |
| P3 | RL 奖励稀疏：rule 奖励二值（全对才有 1.0），大量 group 全 0 无梯度 | rl.py rule_reward，GRPO 组内归一化后优势全 0 | §6.1 |
| P4 | KL-to-ref 只防遗忘，不提供正向监督信号；1.5B teacher 的知识没被利用 | GRPOConfig beta 参数 | §6.2 |
| P5 | 本地硬件 3050 Ti 4GB 跑不动 0.5B 全参 SFT（原项目在 3090 上跑的） | VRAM 实测 ~3.46GB 可用 | §7、§12 |
| P6 | 无 SID 质量独立评估（ICR/LCP），SID 改动只能靠端到端指标盲判 | esci 项目已建好三层评估框架，可迁移 | §4.4 |
| P7 | **副本缺失 RQ-VAE 实现**：`rq/rqvae.py`、`rq/generate_indices.py`、`rq/rqkmeans_plus.py` 均 `from models.rqvae import RQVAE`，但 `rq/models/` 整个目录在复制时不存在 → RQ-VAE 路线开箱即崩 | 4 处 import 全部失败 | 已从上游 `AkaliKong/MiniOneRec` 恢复 `rq/models/{rqvae,rq,vq,layers,generate_indices}.py` |

### 1.4 本次复制时补齐的上游文件（2026-09-11）

| 文件 | 原因 |
|---|---|
| `rq/models/rqvae.py`、`rq/models/rq.py`、`rq/models/vq.py`、`rq/models/layers.py`、`rq/models/generate_indices.py` | 原副本缺失，RQ-VAE 路线（用户选定的默认方案）依赖 |
| `ts_rec_data.py`、`ts_rec_sft.py`、`ts_rec_sft.sh`、`ts_rec_data/IandS.description_keywords.json` | 上游新增的 text-summary 变体 pipeline，作 SFT 改进参考 |
| `tests/` | 上游测试文件（最小覆盖） |

> 其余上游与本地差异仅为数据文件；代码侧已 1:1 对齐上游 `main` 分支。

---

## 2. 升级路线总览

```
V0 基线（已复制，随时可重跑）
 │
 ├─ M1  数据升级     Amazon 2023 I&S（官方 5core 分片）+ 图像下载 + 多模态编码（文本+图像→融合向量）
 ├─ M2  SID 升级     融合向量 → RQ-VAE（默认） vs RQ-KMeans / FAISS-RQ 对照 + 三件套评估（ICR/LCP/MSE）
 ├─ M3  基座迁移     Qwen3-0.6B 跑通 SFT/RL 全流程 + 在新数据上重建 V0 锚点
 ├─ M4  SFT 优化     语义初始化 + 课程学习 + 任务配比 + LoRA/QLoRA 消融
 ├─ M5  RL 升级      路线A（分层奖励）与路线B（OPD）双线对比
 └─ M6  总报告       全量指标对比表 + 消融 + 成本/延迟
```

每个里程碑结束写入 `docs/EXPERIMENT_LOG.md`（实验记录与踩坑文档，已建好模板）。

---

## 3. M1：数据集升级（多模态）

### 3.1 候选数据集对比

| 数据集 | 年份 | 多模态 | 规模 | 优点 | 缺点 | 推荐 |
|---|---|---|---|---|---|---|
| **Amazon Reviews 2023**（McAuley-Lab） | 2023 | ✅ 图像（URL 在 metadata） | 按类目，几十万~百万级 | 与 V0 同源可对照；metadata 含图像 URL；esci 项目已有 Amazon 处理经验 | 图像需自行下载，有失效链接 | ⭐⭐⭐ 首选 |
| MicroLens-100k | 2023 | ✅ 视频/封面 | 10 万级视频推荐 | 真多模态（帧级） | 视频处理重、领域差异大（短视频 vs 电商）、冷门难对标 | 备选 |
| Amazon ESCI（2022） | 2022 | ❌ 查询侧文本 | 121 万商品 | 已在 esci-ai-search 用熟 | 无图像；是搜索不是序列推荐；两项目撞车 | 不选 |
| Tenrec / Taobao MM | 2023 | 部分 | 大 | 工业界数据 | 无公开图像 or 申请门槛 | 不选 |

**结论：Amazon Reviews 2023 子类目。** 与 V0 同为 Amazon（纵向可比），且 metadata 的 `images.hi_res` 字段直接给图像 URL。

### 3.2 类目：已定 Industrial_and_Scientific（2023 版）

决策理由：与 V0 同类目 → 纵向对照最干净（能说清"升级带来的增益"而不是"换了类目所以数字变了"）。

**M1 数据源实测（2026-09-11 验证，走 `hf-mirror.com`，官方站不可达）**：

| 文件 | 体积 | 用途 | 是否下载 |
|---|---|---|---|
| `benchmark/5core/timestamp/{train,valid,test}.csv` | 17.3MB / 2.6MB / 4.1MB | **官方 5-core 切分**（`user_id, parent_asin, rating, timestamp`），免自跑 k-core | ✅ 已下 |
| `raw/meta_categories/meta_Industrial_and_Scientific.jsonl` | **1130.0MB（实测）** | 商品 metadata，**图像 URL 唯一来源**；427,564 行，命中目标物品 25,848（100%） | ✅ 已下载 |
| `raw/review_categories/Industrial_and_Scientific.jsonl` | 2238.3MB | 全量评论（含评论文本） | ❌ 不需要（除非要自跑 k-core / 用评论文本做文本向量） |
| `raw_meta_.../full-0000x-of-00002.parquet` | 217.4 + 187.9MB | 同上 metadata 的 parquet 版（压缩更小，需 pyarrow） | ⏸ 备选（若 jsonl 解析太慢） |

策略调整（相对初版方案）：**用官方 5core 切分代替自跑 k-core** —— 省掉 2.2GB 下载 + k-core 迭代，且官方切分是文献通用口径（可比性更好）。规模控制不再靠"截断到 5 万 items"，改为：**先用全量 I&S 2023 跑通，若 items 规模超出本地承受范围（>10 万）再按交互频次截断**。

### 3.3 多模态编码 pipeline（新增 `scripts/multimodal/`）

```
metadata.jsonl ─┬─ title/brand/categories/features/store ──> Qwen3-Embedding-0.6B ──> e_text ∈ R^1024
               └─ images[0].hi_res ──下载──> SigLIP-base-patch16-224 ──> e_img ∈ R^768
                                                     │
                                                     ▼
                                    门控融合: e = g ⊙ W_t·e_text + (1-g) ⊙ W_i·e_img
                                    （g、W_t、W_i 用对比目标学习；
                                      消融基线：直接 concat + MLP / 纯文本）
                                                     │
                                                     ▼ L2 归一化 ──> M2 的量化器输入
```

要点：
- **图像下载容错**：并发 + 断点续传 + 失效 URL 回退纯文本（图像向量置 0，门控自适应降权）。预期失效 10~20%，方案必须把"纯文本回退"作为一等公民。
- **模型选型**：SigLIP-base（而非 CLIP）——小模型里检索性价比最优一档，4GB 卡可跑。文本侧沿用 esci 项目已验证的 Qwen3-Embedding-0.6B（MTEB 70.7，本地 `embqwen` venv 可复用，torch 2.5.1+cu121）。
- **文本字段比 V0 多**：V0 只用 title+description；2023 metadata 还有 `features`（要点列表，信息密度高于 description）、`categories`、`store/brand`、`details`。这里要做**字段组合消融**（title / title+features / title+features+category），因为文本向量是 SID 的上游，影响比 SFT 超参大得多。
- **融合层先简后繁**：v1 = concat+MLP → v2 = 门控融合。融合质量用"文本↔图像互检索 recall@k"作为独立代理指标，否则多模态有没有用只能靠端到端盲判（同 P6 教训）。

### 3.4 交付物（M1 执行清单）

| # | 动作 | 脚本 | 状态 |
|---|---|---|---|
| 1 | 下载官方 5core 切分 + metadata | `scripts/data/download_amazon23.sh` | ✅ 切分已下 / metadata 下载中 |
| 2 | 构建 item.json（title/features/brand/categories/images）+ 交互序列 | `scripts/data/prepare_amazon23.py` | ⏳ 待写 |
| 3 | 图像并发下载 + 失效记录 | `scripts/multimodal/download_images.py` | ⏳ 待写 |
| 4 | 文本向量（Qwen3-Embedding）+ 图像向量（SigLIP） | `scripts/multimodal/encode_{text,image}.py` | ⏳ 待写 |
| 5 | 融合 + 互检索质量评估 | `scripts/multimodal/fuse_embeddings.py` | ⏳ 待写 |
| 6 | 复用/改造 `data/amazon23_data_process.py`（其已采集 images 字段，可部分复用其 metadata 解析逻辑） | — | ⏳ 待改 |

---

## 4. M2：SID 构建优化

### 4.1 路线选择：RQ-VAE 默认，KMeans 作挑战者（按决策）

**默认走 RQ-VAE**（`rq/models/rqvae.py` + `rq/trainer.py`，已从上游恢复），因为：
- 它带**编码器/解码器**（MLP：in_dim→2048→1024→512→256→128→64→e_dim），能端到端优化 reconstruction，量化保真度上限高于"KMeans + 残差"；
- codebook 是**可学习的**（+ commitment loss + kmeans 初始化），能适应融合后的多模态空间；
- 原仓库的 `generate_indices*.py` 就是 RQ-VAE 路线，改动量最小。

**同时保留对照**（用手上的两个现成实现，成本极低）：
- `rq/rqkmeans_plus.py`（RQ-KMeans 变体）
- `rq/rqkmeans_faiss.py`（FAISS ResidualQuantizer + Sinkhorn uniform mapping，esci 项目验证过：119s 训完 121 万向量，last-layer Sinkhorn → ICR 92.1% / LCP 19.21%）

**择优规则（先离线、后端到端）**：三者用 §4.4 三件套评分（fidelity 优先、结构指标次之、uniqueness 设硬门槛），**只有三件套胜出的才进 SFT**；若三件套打平，再用小规模 SFT（10% 数据）比 NDCG@10 一锤定音。谁好就用谁，**不做预设结论**。

### 4.2 关键超参（RQ-VAE 默认档，来自上游脚本）

| 参数 | 默认 | 说明 / 消融计划 |
|---|---|---|
| `num_emb_list` | `[256,256,256]` | 3 层 SID。消融 `[256,256]`（2 层，序列更短）与 `[256,256,256,256]` |
| `e_dim` | 32 | codebook 向量维度。**消融 64/128**——32 维承载多模态信息可能不足，这是 RQ-VAE 最容易吃亏的点 |
| `layers` | `[2048,1024,512,256,128,64]` | 编解码 MLP 宽度 |
| `kmeans_init` | True | 避免 codebook 死码 |
| `sk_epsilons` | `[0,0,0]`（**Sinkhorn 关闭**） | 打开 = 逐层 Sinkhorn 均衡；但 esci 实测逐层 Sinkhorn 会伤 LCP → **只在最后一层做，或作为消融项**（复用 `rqkmeans_faiss.py` 的 `sinkhorn_balance_level`） |
| `beta` (commitment) | 0.25 | — |
| `quant_loss_weight` | 1.0 | 消融 0.5 / 2.0 |
| `epochs` / `batch_size` / `lr` | 5000 / 2048 / 1e-3 | 早停看 collision rate |

### 4.3 输入侧升级

1. **多模态融合向量**（§3.3）替换纯文本向量；
2. **L2 归一化**（esci 上隐式受益，这次显式化）——注意 RQ-VAE 的 MSE 重建在归一化空间上做，需在 eval 时反归一化换算真实 MSE；
3. **文本字段组合消融**（title / +features / +features+category），这是比 SID 超参更上游的杠杆。

### 4.4 升级：新 SID token embedding 语义初始化（关键创新点）

V0 现状（P2）：`tokenizer.add_tokens()` 后新 token embedding 随机初始化，模型要花大量 SFT 步数学"SID token 长什么样"。

方案：SID 第 l 层 token `<a_i>` 的初始化向量 = 该层 codebook 向量 `C_l[i] ∈ R^{e_dim}` 投影到 LLM 隐空间：

```
E_init[<a_i>] = W · C_l[i] + b,   W ∈ R^{d_llm × e_dim}
```

`W, b` 用 ridge regression 在全体 items 上拟合。锚点（回归目标）取法（消融）：
- **v1（简单）**：锚点 = 该 item title 的 LLM 词向量（冻结模型，取 title token embedding 均值）；
- **v2（对齐量化结构）**：锚点 = 该 item 的 RQ-VAE **解码器重建向量** `x̂_i = Dec(Σ_l C_l[codes_l])` → 让"SID token 组合 ≈ item 语义"在同一个空间对齐；
- **v3（残差感知，进阶）**：每层单独拟合——第 l 层 token 的目标是"第 l 层的残差方向在 LLM 空间的投影"，让前缀天然携带粗到细的语义。**v3 是新东西，值得写成方法论**。

实现落点：新增 `sid_token_init.py`，在 `sft.py` 的 TokenExtender 之后调用；配 `freeze_LLM=True`（现有开关）= 极低成本消融，**本地 4GB 卡就能跑**（只训新 token embedding）。

### 4.5 SID 质量评估三件套（从 esci 项目迁移，新增 `rq/eval_sid.py`）

| 层 | 指标（定义与公式） | 含义 |
|---|---|---|
| uniqueness | **ICR** `= 不同 SID 元组数 / N`，碰撞率 `= 1 − ICR`；**per-layer entropy** `H_l = −Σ_k p_{l,k} log p_{l,k}`（归一化 `H_l / log K`，K = 256）、死码率 `dead_l = 1 − 该层用过的码数 / K` | SID 是否唯一、是否死码 |
| fidelity | **reconstruction MSE** `= 1/(N·D) · Σ_i ‖ x̂_i − x_i ‖²`，`x̂_i = Dec(Σ_l C_l[c_{i,l}])`；**R²** `= 1 − MSE / Var(x)` | 量化损失（RQ-VAE / KMeans 直接可比） |
| retrieval structure | **LCP** `lcp(i,j) = Σ_{l=0}^{L-1} Π_{t≤l} 1[c_{i,t} = c_{j,t}]`，**LCP ratio** = 近邻对均值 ÷ 随机对均值（基线 ≡ 1）；**prefix cohesion** = 同前缀组内两两余弦均值 ÷ 随机分组同值 | SID 前缀是否语义聚类 |

> 融合侧的共现检索 `R@K = 1/Q · Σ_q 1[Top-K(q) ∩ Gold(q) ≠ ∅]`（随机基线 `≈ avg_pos · K / N`）、
> 端到端 `HR@K / NDCG@K` 的定义与公式见 **§8** 与 `docs/SID_PIPELINE.md §0.1`。

**消融矩阵**（M2 交付）：{纯文本, 融合} × {RQ-VAE, RQ-KMeans, FAISS-RQ(+last-layer Sinkhorn)} × {2/3/4 层}，其中 {纯文本 × RQ-VAE × 3 层} 为 V0' 锚点。每次 SID 变更先过三件套，再进昂贵的 SFT。

### 4.6 待验证消融：SID 源 embedding 换成统一多模态空间（Qwen3-VL-Embedding）

**动因**：当前双塔的两条腿来自**两个互不相通的嵌入空间**——这是门控融合之所以复杂、跨模态评估必须绕道的根因。

| 现状（实测） | 值 |
|---|---|
| 文本向量 | `Qwen3-Embedding-0.6B` → **1024 维** |
| 图像向量 | `google/siglip-base-patch16-224` → **768 维** |
| 跨模态可比性 | **点积无定义**。`fuse_embeddings.py::cross_modal_align` 注释原文：“text 是 1024 维、image 是 768 维，分属两个独立的嵌入空间，点积没有定义” |
| 现有绕行 | 先在拟合子集解岭回归学线性映射，再到留出集算 recall（text→image R@10 = I&S 0.6217 / VG 0.6363） |

**候选**：`Qwen3-VL-Embedding`（阿里通义 **2026-01-08** 发布，Apache-2.0，论文 arXiv:2601.04720）

| 项 | 事实 |
|---|---|
| 基座 | **Qwen3-VL**（不是 Qwen3-Embedding）；双塔，取最后一层 `[EOS]` 隐态 |
| 训练 | 多阶段：大规模对比预训练做跨模态对齐 → reranker 蒸馏 → MRL |
| 空间 | 文本 / 图像 / 视觉文档 / 视频 → **同一语义空间** |
| 维度 | 2B→2048、8B→4096；**MRL 支持 2048→{256,512,1024,2048}**，可截 1024 与现有 RQ-VAE 输入兼容 |
| 规格 | 2B：28 层 / 32K tokens / bf16 约 4GB（int8 ~2GB，int4 ~1GB） |
| 基准 | MMEB-v2：2B **73.2**、8B **77.8**（发布时第一）；MMTEB 纯文本检索 2B **68.1** |

🔴 **必须先纠正的前提**：它**不是**和 `Qwen3-Embedding` 对齐训练出来的，两者是 2026-01 与 2025-06 两个独立系列。官方明确说 VL 版在纯文本 MMTEB 上比**同规模**纯文本版“有少许差距”。
**推论：统一空间只在 Qwen3-VL-Embedding 内部成立。只换图像侧、保留 `Qwen3-Embedding-0.6B` 做文本，两边照旧不对齐**——收益只剩“图像特征更好”一项，且融合层仍须重训。

**预期收益（三重，置信度递减）**

1. **图像侧表征**：SigLIP-base（ViT-B 量级、2023 年 CLIP 式对比模型）→ Qwen3-VL 的 ViT（2B 量级）。**域匹配度高**：Amazon 商品图大量是包装盒白底图、带型号规格文字，Qwen3-VL 系强在 OCR 与布局理解，SigLIP-base 对此近乎无能为力。
2. **文本侧参数量**：0.6B → 2B。
3. **架构简化（最值得期待）**：同空间后图文天然可比，门控网络（1792→512→1024）里“学跨空间映射”那部分参数**可省**，融合退化为加权/拼接，甚至直接用混合模态输入得单一向量。

**成本与风险（诚实）**

| 项 | 评估 |
|---|---|
| 显存 | 2B bf16 ≈ 4GB，**本地 3050 Ti 4GB 跑不了**；int8/int4 可跑但 embedding 对低位量化敏感（社区共识 Q6_K 为下限）→ **建议上云 3090 编码**（一次性离线任务，符合既有约定） |
| 规模 | 51,458 商品 × 图文各一遍 ≈ 10 万次编码；沿用已下载的本地图片，无需重下 |
| 🔴 孪生不解决 | VG 那 133 组逐位重复的融合向量，根源是同品多 listing **主图 URL 相同**；相同输入 → 相同输出，换编码器不影响，死码与碰撞该在还在 |
| 🔴 域适配未知 | MMEB 是自然图像/文档图像，本项目是白底商品图。**“榜单分高”不等于“在这批图上更好”，必须实测** |
| 🔴 窗口在关闭 | SFT 数据集已按现 SID 生成，换 SID 源 → SID 变 → 数据集须重生成。现在（SFT 未开训）是成本最低窗口；但**不建议同时动两个变量**（raw/sk 端到端消融尚未跑，会混淆归因） |

**消融设计（按建议执行序）**

| 编号 | 内容 | 成本 | 通过判据（可证伪） |
|---|---|---|---|
| **V4-A** | **跨模态对齐先行验证**：抽 3000 对，用 Qwen3-VL-Embedding-2B 编码图文，**直接点积**算 text→image R@10 | 极低（分钟级，纯离线、不动现有 pipeline） | **> 0.66**（即高于现有岭回归后的 0.6217/0.6363，且无需学映射）。**不通过则 V4-B/C/D 全部免做** |
| **V4-B** | **两侧全替换 + SID 三层评估**：MRL 截 1024，重跑融合 → SID，双域对照 raw ICR / 重建 R² / L0 死码 / holdout R@10 | 中（10 万次编码 + 现 SID 流程 68 min 级） | holdout R@10 不低于现 `gate_e60`（I&S 0.1100 / VG 0.2113）**且** raw ICR ≥ 现（0.9582 / 0.9744） |
| **V4-C** | **单换图像侧**（归因用）：SigLIP → Qwen3-VL-Embedding 的 image-only 编码 | 低 | 逻辑上不解决对齐，仅用于量化“图像特征质量”单独贡献多少 |
| **V4-D** | **混合模态单向量**：直接把 `{text, image}` 喂进同一模型得单一向量，**彻底取消门控网络** | 中 | 与 V4-B 对比，验证融合层是否真的可以省掉 |

**建议执行序**：**V4-A 先做**（几分钟给出方向性答案，完全不影响现有 pipeline）→ 通过后再排 **V4-B**，且**排在 SFT 基线跑通之后**，让 SFT 保留一个“旧 SID”锚点、换 SID 前后可比。V4-C/D 视 V4-A 结果再定。

> 判据口径提醒：V4-A 用的是与 `fusion_report.json → modality_alignment` 完全相同的构造（同一批 pair、同一 n_query），
> 唯一差别是 Qwen3-VL-Embedding 下**不再需要岭回归**，可直接点积。这是全流程里最灵敏、最早能读出信号的指标。

![当前双塔异空间 vs Qwen3-VL-Embedding 统一空间](figures/dual_tower_vs_unified_space.png)

### 4.6.1 文献地图：图像-文本对齐的四条技术路线（检索于 2026-09-15）

> 检索先给一个反直觉的结论：**「有没有做图文对齐训练的模型」这个问题，本项目其实已经在用模型回答了**——
> 图像侧的 `google/siglip-base-patch16-224` 就是**拿图文对比对齐当训练目标**训出来的（CLIP 系，见路线 ①）。
> 所以真正缺的**不是「对齐模型」，而是「与我们的文本嵌入空间同源的对齐模型」**。
> 下面按「对齐发生在哪一层」分四条路线，各自给代表工作、事实与成本。

#### 路线 ①：CLIP 式双塔对比对齐（**本项目图像侧已在用**）

| 工作 | 时点 | 对齐目标 | 事实 |
|---|---|---|---|
| CLIP | 2021 | softmax 对比 | 开山，图文共享空间 |
| SigLIP | 2023 | **sigmoid** 损失 | 免去 batch 内伪竞争；`SigLIP-SO400M` 成 LLaVA-OneVision / DeepSeek-VL2 的默认视觉塔 |
| **SigLIP 2** | 2025-02 | sigmoid + LocCa + SILC/TIPS 自蒸馏与掩码 | 百语种训练，密集特征更强 → 被 **Qwen3-VL / Gemma 3** 采用 |
| **MetaCLIP 2** | 2025-07 | 全球 300+ 语种原生图文对 | ViT-H/14 打破「多语言诅咒」：ImageNet 81.3%、XM3600 I→T 64.3% |

🔴 **这条纠正很重要**：`SigLIP-base` 早已是图文对齐的产物，所以 §4.6 说的「两条腿互不相通」
**不是图像侧没对齐，而是它对齐到的文本塔不是 `Qwen3-Embedding-0.6B`**
（SigLIP 的文本塔是 77-token 上限的小编码器，语义容量与被换掉的那条腿不是一个量级）。
**推论：把 SigLIP 换成「更会做图文对齐」的同族模型（SigLIP 2 / MetaCLIP 2）不解决本项目的问题**——
问题在「和谁对齐」，不在「对齐得好不好」。
（另一种零训练解法是用同一个 CLIP 模型的**文本塔 + 图像塔**，天然对齐；代价是文本侧退回小编码器，SID 质量恐难接受。未见有人在本任务上验证，可作 V4-C 的补充对照。）

#### 路线 ②：VLM 统一空间嵌入（当前 SOTA 主流）

把 VLM 改造成 embedder：文本/图像/视频/视觉文档走**同一次前向**，取末层特殊 token 隐态 → 一个向量。
通用基准 **MMEB-v2**（78 个任务：图像 / 视频 / 视觉文档）。

| 模型 | 规模 | MMEB-v2 | 时点 | 备注 |
|---|---:|---:|---|---|
| **WeMM-Embedding-9B** | 9.41B | **80.6** | 2026-08 | 腾讯微信视觉；MMEB-v2 榜首快照（2026-08-24） |
| DME-Large | — | 80.2 | 2026-08 | 抖音（闭源） |
| **WeMM-Embedding-4B** | 4.54B | 79.2 | 2026-08 | |
| DME-Medium | 9.4B | 78.4 | 2026-08 | |
| **WeMM-Embedding-2B** | 2.21B | **77.9** | 2026-08 | **2B 反超 `Qwen3-VL-Embedding-8B`**；MRL 64~4096 维；MMEB-v3 56.0 |
| Qwen3-VL-Embedding-8B | 8.14B | 77.8 | 2026-01 | §4.6 原候选 |
| DME-Small | 2.21B | 74.8 | 2026-08 | |
| Qwen3-VL-Embedding-2B | 2.13B | 73.3 | 2026-01 | §4.6 原候选（2B 档） |
| UEmbed-9B (dense) | 9B | 71.8 | 2026-08 | 阿里，**Apache-2.0、纯公开数据**；dense+sparse 单次前向 |
| UEmbed-4B (dense) | 4B | 70.4 | 2026-08 | |
| e5-omni-7B | 7B | 67.8 | 2026-01 | 见路线 ④ 的方法论 |
| UEmbed-2B (dense) | 2B | 66.5 | 2026-08 | |
| UME-R1-7B | 8.29B | 64.1 | ICLR 2026 | 厦大 + **腾讯微信**；**生成式嵌入 + RL** |
| UME-R1-2B | 2.21B | 59.7 | ICLR 2026 | |
| gme-Qwen2-VL-7B | 8.29B | 57.3 | 2024-12 | 阿里 GME，统一多模态嵌入的开篇之一 |
| VLM2Vec-V2-2B | 2.2B | 55.4~56.2 | 2025 | TIGER-Lab，MMEB 基准的提出方 |
| VLM2Vec-V1-7B | 8.29B | 47.0 | 2024-10 | |

**对本项目的含义**

1. 候选池比 §4.6 当初写的更宽，且**格局变了**：2B 档里 **WeMM-Embedding-2B（77.9）明显优于 `Qwen3-VL-Embedding-2B`（73.3）**，甚至超过后者的 8B 版。
2. WeMM-Embedding 出自**腾讯微信视觉团队**，**已在微信视频号/公众号/电商/微信搜索的 14 项 A/B 中取得一致提升并全量上线**——这是「能不能进生产」最有力的外部证据（也是面试可讲的一手锚点）。
3. **UME-R1 的方法论与本项目主线同源**：把 embedding 放进「生成 + RL」范式（生成式嵌入 + 相似度反馈的可验证奖励），ICLR 2026 且与腾讯微信合作。本项目 M5 正在做 RL 升级，这是可直接引用的邻居工作。
4. `[设计]` 若最终走路线 ②，**V4-B 的候选应从「Qwen3-VL-Embedding」改为「WeMM-Embedding-2B vs Qwen3-VL-Embedding-2B」二选一或并列**，两者都需实测（榜单分高 ≠ 白底商品图更好，见 §4.6 风险表）。

#### 路线 ③：把新模态对齐到**既有的、冻结的**文本嵌入空间（← 直击本项目痛点）

路线 ② 的统一空间是**模型内部**的，换就得两侧一起换。这条路线目标不同：
**保留现有文本嵌入几何（不重训、不重建索引），只新增一个对齐的图像塔。**

| 工作 | 时点 | 做法 | 事实 |
|---|---|---|---|
| ImageBind | 2023 | 以**图像**为中心绑定六模态 | 跨模态 emergent，但绑到图像空间 |
| **LanguageBind** | ICLR 2024 | 以**语言**为中心：**冻结语言编码器**，对比学习把视频/音频/深度/红外映射进语言空间 | 15 个 zero-shot 检索基准 SOTA；「language as the bind」的范式定义者 |
| **jina-embeddings-v5-omni** | 2026-05（SIGIR 2026，arXiv 2605.08384） | **frozen-tower composition**：文本骨干 + 全部任务 LoRA 适配器**冻结**，只训跨模态投影器 | 🔴 **只训 0.35% 参数**（Small 约 5.5M）；**文本输出与 `jina-embeddings-v5-text` 逐位相同（bit-identical）**，**升级无需重建索引**；1024 维 MRL→32；视觉塔 SigLIP2-So400m |

**对本项目的含义（本次检索最有价值的一条）**

1. 「**保留文本腿几何 + 外挂对齐的图像塔**」被证明**可行且极便宜**——Jina 只用 0.35% 参数就拿到与更大统一模型相近的效果。
2. 它**直接解掉 §4.6 那条「窗口在关闭」的焦虑**：若文本侧向量逐位不变，文本侧 SID 就是稳定的，**SFT 数据集不必重生成**，变的只是融合向量。代价比「两侧全换 → 全链路重生成」低一个量级。
3. 验收方式极简且是**二元的**：训完对文本输入断言逐位相等（`np.array_equal`），无阈值、无噪声。

#### 路线 ④：推荐/电商场景「量化之前的对齐」（与本项目 pipeline 最同构）

这一族不问「用哪个通用嵌入模型」，而问「**在喂给 RQ-VAE 之前，图文该怎么对齐**」。

| 工作 | 出处 | 与本项目的关系 | 关键数字 |
|---|---|---|---|
| **MMQ**（多模态混合量化） | 阿里国际 Lazada，**WSDM 2026**（arXiv 2508.15281） | 与我们的 `gate 融合 + RQ-VAE` 是**同一问题的两个解法**：共享-特定混合专家 + 专家正交约束 + **行为感知微调**（Soft Index + STE 把下游行为梯度回传到 tokenizer） | 召回/精排均超同类 SOTA；**Lazada 线上 A/B：REV +1.29% / GMV +2.61% / ROI +1.18%** |
| **When Text-as-Vision Meets Semantic IDs** | arXiv 2601.14697 | 🔴 **依据最强**：不动模型，把商品描述**渲染成图**、用 OCR 模型（DeepSeek-OCR 主 / Donut / TrOCR）编码——视觉塔输出天然与图像嵌入几何兼容，**直接绕开 modality gap 而不是修复它** | 4 个 Amazon 类目（含 **Scientific / Instruments**，与本项目 IandS 重叠）；Scientific 单模态 NDCG@5 **+8.06%**；多模态早融合 **TIGER(O+I) Recall@5/N@5 +13%**；1024→256 分辨率压缩仍鲁棒 |
| **TGQ-Former** | **KDD 2026**（arXiv 2605.17366） | 针对电商图的促销贴片 / 背景噪声，用结构化 metadata 引导视觉 token 提取；**其 connector 设计可对照我们的门控融合** | H@100 +3.03%（另一快照写 6.04%，⚠️ 回原文核对） |
| **DeepInterestGR / CMSA** | arXiv 2604.20861 | 用 VLM 把非文本模态**对齐到统一文本语义空间** + LLM 挖 intent + RL 质量奖励；明确把「Modality Distortion」列为三大瓶颈之一 | NDCG@5 最高 **+15.1%** |
| VL-CLIP | Walmart，RecSys 2025（arXiv 2507.17080） | 视觉定位（Grounding DINO 抠主体）+ LLM 改写文本；**「不去噪就对齐」的反面教材** | 千万级商品上线；**CTR +18.6% / ATC +15.5% / GMV +4.0%** |
| Factorized Transport Alignment | **Etsy**，WSDM 2026（arXiv 2512.18117） | 多视图（非主图 + 辅助文本）用最优传输的轻量近似对齐，推理时融合为单向量 | 1M listings；R@500 **+7.9%** |
| **e5-omni** | 人大 + Würzburg + 早稻田，**2026-01**（arXiv 2601.03666） | 🔴 **「显式对齐」的方法论论文**：指出「继承 VLM 的隐式对齐」不够用，给出三件套（下详） | MMEB-v2：7B 67.8 / 3B 63.6 |
| PixRec | arXiv 2601.06458（2026-01） | 🔴 **同源数据**：**Amazon Reviews + 商品图片**做序列推荐，双塔 + 混合训练目标 | 报告 top-rank 提升 3×、top-10 +40%（⚠️ 摘要原文含排版乱码，回原文核对） |

**e5-omni 的三件套值得单列**——本项目当前的对齐做法是「**事后**岭回归」，它给的是「**训练时**对齐」：

| 组件 | 解决什么 | 与本项目现状对照 |
|---|---|---|
| 模态感知温度校准 | 相似度 logits 的锐度随模态变化，单一全局温度造成对比梯度失衡 | 现有 InfoNCE 用单一温度 |
| 可控负样本课程 + 去偏 | 混合模态 batch 的负样本难度分布失衡、假负样本拖累 | 现有实现是共现对上的对称 InfoNCE，**无课程、无去偏** |
| **batch whitening + 协方差正则** | 跨模态 embedding 的一阶/二阶统计量不匹配 → 排序不稳 | 🔴 现有做法是**事后在拟合子集上解岭回归**（`cross_modal_align`）；whitening 相当于把这件事挪进训练，且不做线性映射假设 |

### 4.6.2 由文献新增的三条消融（V4-E / V4-F / V4-G）

§4.6 原有的 V4-A/B/C/D 全是「**换模型**」路线。文献显示还有三条**不改主干**的路线，成本低一个量级：

| 编号 | 内容 | 文献依据 | 成本 | 通过判据（可证伪） |
|---|---|---|---|---|
| **V4-E** | **显式对齐（不换模型）**：在现有 `fuse_embeddings.py` 的 InfoNCE 上加 e5-omni 三件套（模态温度校准 / 去偏负样本课程 / batch whitening + 协方差正则），看 `cross_modal_align` 的 R@10 能否逼近「同空间直接点积」的水平 | e5-omni（arXiv 2601.03666） | **极低**：只改损失函数，重跑 `gate` 60 轮 ≈ 50 min | text→image R@10 **> 0.66**（高于现岭回归 0.6217 / 0.6363），**且** `gate_e60` 的 holdout R@10 不下降 |
| **V4-F** | **frozen-tower：锁死文本腿，只新增对齐的图像投影塔**。冻结 `Qwen3-Embedding-0.6B`，训一个把图像特征投影进 1024 维文本空间的轻量塔（参数按 jina-v5-omni 的量级，≪1%） | jina-embeddings-v5-omni（arXiv 2605.08384） | 低 | ① **文本侧向量逐位不变**（`np.array_equal` 断言，二元判据）；② text→image R@10 > 0.66；③ raw ICR ≥ 现（I&S 0.9582 / VG 0.9744） |
| **V4-G** | **Text-as-Vision**：title（+description）渲染成图，用 OCR-VLM 编码作为文本侧，**绕开 modality gap 而非修复它** | arXiv 2601.14697（Instruments/Scientific 上 +8%~+13%） | 中（需引 OCR 模型 + 重跑全链路） | raw ICR / 重建 R² 不低于现文本塔，且跨模态几何错配指标显著下降 |

**优先级建议**：**V4-A**（分钟级，先读方向）→ **V4-E**（50 min，不改主干）→ **V4-F**（低成本，且解掉「SFT 窗口」焦虑）→ **V4-B/C/D**（换模型，代价最高）。

**V4-G 单独排队**：它改的是「文本侧表示形式」，会同时改掉 T2（`sid↔title`）与 T4（`text2sid`）两个任务里 title 的语义（见 `SFT_PIPELINE §2 四个任务的定义`），与 SFT 数据集构造耦合，**建议排在 SFT 基线跑通之后**。

> ⚠️ **引用纪律**：本节所有数字（MMEB-v2 榜单、各论文增益）均为**文献值**，非本项目实测，
> 按 `baseline/SURVEY.md` 的规矩只作「量级参照」，**不得与 `RESULTS.md` 的内部排序横比**。
> 另有两处已在表内标注「回原文核对」（TGQ-Former 的 3.03%/6.04%、PixRec 的 3×/+40%），
> 原因是检索到的二手摘要在这些数字上互相冲突或存在排版乱码——**二手转述不是引用**。

---

## 5. M3/M4：基座模型与 SFT 优化

### 5.1 基座：Qwen2.5-0.5B-Instruct → Qwen3-0.6B

| 项 | Qwen2.5-0.5B | Qwen3-0.6B |
|---|---|---|
| 发布 | 2024-09 | 2025-04 |
| 词表 | ~151k BPE（同源） | ~151k BPE（同源，新 token 注入流程不变） |
| 隐维 | 896 | 1024 |
| 训练数据 | 18T | 36T |
| 选择理由 | — | 同家族最小迁移成本；数据质量翻倍；社区 trl/transformers 支持成熟 |

teacher 候选：**Qwen3-1.7B**（OPD 用，见 §6.2；同 tokenizer 保证 logit 对齐，无需处理词表映射）。备选 Qwen3-4B（显存压力大，仅 3090+ 场景）。

风险：trl GRPOTrainer 对 Qwen3 的兼容性（thinking 模式模板差异）→ 统一用 non-thinking 模式 + 锁版本（transformers==4.51+/trl==0.15+，以实测为准，记录进实验文档）。

### 5.2 SFT 训练策略升级

| 项 | V0 | 升级 | 预期收益 |
|---|---|---|---|
| 新 token 初始化 | 随机 | §4.3 语义初始化（v1/v2 消融） | 收敛加速 + 上限提升 |
| 课程 | 3 任务均匀 concat | 阶段1 只训 SidItemFeat（title↔SID 对齐）→ 阶段2 全混合 | 先学会"读 SID"再学"序列预测" |
| 任务配比 | 均匀 | 消融：SidSFT:ItemFeat:FusionSeqRec = 1:1:1 / 2:1:1 / 1:1:2 | 数据效率 |
| 显存 | 全参（3090 专用） | 全参(3090) / **QLoRA r=32(本地 4GB)** 双轨 | 本地可迭代 |
| 采样 | cosine + warmup20 | 沿用（V0 的调度已含 min lr=0.1·peak 的 floor，不动） | — |
| 早停 | patience=3 | 沿用 | — |

注意：V0 `sft.py` 里 `torch_compile=True` 在 Windows + 4GB 卡上大概率开不了（编译缓存爆磁盘 + 首步极慢），本地脚本需关掉——进踩坑记录。

### 5.3 SFT 消融矩阵（M4 交付）

固定数据 + SID，跑：{随机 init, 语义 init-v1, 语义 init-v2} × {全参, LoRA}，主指标 NDCG@10 + 收敛步数（到 90% 最终性能所需 step 数）。

---

## 6. M5：RL 升级（双路线）

### 6.1 路线 A：分层奖励设计（改 `rl.py` reward 函数）

V0 的问题（P3）：rule 奖励二值，组内全错时 advantage 全 0，白烧采样预算；semantic 奖励（余弦）虽然稠密但单独用时无"正确"概念。

**升级为分层组合奖励**：

```
r = 1.0  · exact_match(completion, target)                     # 正确性（保留）
  + α    · cos(e_completion, e_target)                        # 语义部分分（稠密）
  + β    · prefix_hit(completion, target)                      # 前缀命中（层粒度，0~1）
  + γ    · beam_ndcg(组内排名)                                 # 组内排序塑形
```

- `prefix_hit`：SID 三层逐层比对，第 1 层命中记 1/3……（约束解码下非法 token 已被 mask，奖励只需区分"合法但错"的层次）；
- `α/β/γ` 网格：{0.2, 0.5} × {0.1, 0.3} × {0, 0.1}，共 8 组，先在小数据（10%）上筛；
- **去流行度偏置**（针对 Amazon 长尾）：对高频 item 的 exact 奖励乘 `1/sqrt(pop)` 截断系数，消融开关；
- 所有 reward 函数向量化（V0 的 for 循环 + dict 查询在 GRPO 里是瓶颈之一，顺手修）。

### 6.2 路线 B：OPD（Online Policy Distillation）

动机（P4）：GRPO 的 `beta` 项是 KL-to-ref（只会拉住不跑偏），而 teacher 的知识是稠密信号。OPD 把 KL 目标从 ref 换成在线 teacher：

```
L = L_GRPO(student; 分层奖励)
    + λ · KL(π_student ∥ π_teacher)     # 替代 KL(π_student ∥ π_ref)
```

- **teacher**：Qwen3-1.7B，先用 student 同款 SFT 数据训一版（或在 3090 上全参 SFT 一次性产出），之后 RL 阶段冻结；
- **student**：Qwen3-0.6B；
- 实现落点：`minionerec_trainer.py`（已自定义 ReReTrainer，加 teacher forward + KL 替换是最小侵入改动）；GRPOConfig 的 `beta` 复用为 λ，`sync_ref_model` 路径改挂 teacher；
- **成本**：teacher 前向 ≈ 显存 +2× student。4GB 本地卡跑不动在线 OPD → OPD 实验默认在 3090 做；本地卡只做路线 A（teacher-free）；
- **风险**：teacher 太弱会把 student 拉向 teacher 的局部最优 → λ 用 cosine 退火（先强后弱，后期让奖励主导）；teacher 与 student 容量差 3 倍是甜点区（差太多蒸馏不动，差太少没信息量）。

### 6.3 RL 实验矩阵（M5 交付，固定 SFT 最优 checkpoint）

| 实验 | 算法 | 奖励 | KL 对象 | 价值 |
|---|---|---|---|---|
| R0 | GRPO | rule | ref | V0 对照 |
| R1 | GRPO | 分层 | ref | 隔离"奖励设计"贡献 |
| R2 | GRPO | rule | **teacher (OPD)** | 隔离"OPD"贡献 |
| R3 | GRPO | 分层 | **teacher (OPD)** | 全量升级 |
| R4（可选） | + DAPO / GSPO 开关 | 最优奖励 | 最优 KL | V0 代码已有 dapo/gspo flag，白捡的消融 |

主指标 NDCG@10；副指标：组内 reward 方差（衡量信号稀疏度是否改善）、KL 曲线（是否 mode collapse）。

---

## 7. 硬件与算力预算（两套环境）

| 阶段 | 本地 3050 Ti 4GB / 17GB RAM | 3090（租/云） |
|---|---|---|
| 图像下载+SigLIP 编码 | ✅（batch 32 fp16） | ✅ 更快 |
| Qwen3-Embedding 文本编码 | ✅（复用 `embqwen` venv） | ✅ |
| RQ-VAE / RQ-KMeans / FAISS-RQ 训练（CPU 或小卡） | ✅（py3.11 `aisearch` env，分钟级~小时级） | ✅ 更快 |
| SID 三件套评估 | ✅（CPU） | ✅ |
| SFT 0.6B | QLoRA r=32, micro_bs=1~2 | 全参 bf16 |
| RL 路线 A | num_generations=4, bs=2（勉强，需实测） | num_generations=16 |
| RL 路线 B (OPD) | ❌ | ✅（teacher 前向需 ~10GB 额外） |
| 评估 beam=10 | ✅（慢） | ✅ |

策略：**本地做 SID/消融/路线 A 的快速迭代，3090 跑正式数字**（与 esci 项目相同的双环境节奏）。

---

## 8. 评估体系（M6）

- 主指标（定义与公式，实现见 `utility.py: calculate_hit`）：
  - **HR@K** = `1/U · Σ_u 1[ rank_u ≤ K ]` —— 真实下一个物品是否进 top-K；
  - **NDCG@K** = `1/U · Σ_u 1[ rank_u ≤ K ] / log₂(rank_u + 1)` —— 单正样本 ⇒ IDCG = 1；
  - **MRR** = `1/U · Σ_u 1 / rank_u`（未命中记 0）—— 对"第一个命中项的位置"更敏感；
  - K = 1/5/10，beam=10，约束解码，与 V0 完全同口径；
- 新增：
  - **样本效率曲线**：达到 V0 最终 NDCG@10 所需 step 数（衡量语义初始化/OPD 的训练效率收益）；
  - **覆盖率/多样性**：推荐分布的 catalog coverage、gini 系数（去流行度偏置奖励的效果验证）；
  - **多模态归因**：无图像 item（下载失败子集）上的分桶指标——若融合有效，有图 item 应显著高于无图 item；
  - **延迟**：SFT/RL 模型单 query 推理耗时（P95）。
- 公平性：所有新配置与"V0'"在**同一份 2023 数据**上重跑（V0 的 2018 数字只作历史参考，不进对比表；2018 数据已从工作区移除，见 §13 决策 4）。

---

## 9. 里程碑与产出

| 里程碑 | 内容 | 产出 | 验收标准 |
|---|---|---|---|
| M1 | 数据+多模态编码 | 新数据集 + 融合向量 + 下载/编码脚本 | 互检索 recall@k > 单模态基线 |
| M2 | SID 优化 | SID + `eval_sid.py` 报告 | 三件套全量指标 ≥ 纯文本 SID |
| M3 | 基座迁移 + 锚点 | Qwen3-0.6B 全流程跑通 + V0' 锚点数字 | SFT 不塌（NDCG@10 ≥ V0' 预期带内） |
| M4 | SFT 消融 | §5.3 矩阵结果 | 语义初始化有显著收益（否则诚实记录无效） |
| M5 | RL 双路线 | §6.3 R0~R3 对比 | R1/R2 至少一个显著优于 R0 |
| M6 | 总报告 | 升级前后全表 + 消融 + 踩坑复盘 | 写入 EXPERIMENT_LOG.md + README |

每个里程碑完成即写入 `docs/EXPERIMENT_LOG.md`（含失败实验——尤其重要，踩坑记录是这份文档一半的价值）。

---

## 10. 风险与回退

| 风险 | 概率 | 影响 | 回退 |
|---|---|---|---|
| 图像 URL 大量失效（>40%） | 中 | 多模态退化 | 融合向量图像位补零 + 门控降权，纯文本退化为 V0'，项目仍成立 |
| **RQ-VAE `e_dim=32` 承载不了多模态信息**（最可能的 SID 侧坑） | 中 | 量化保真度低、碰撞率高 | 三件套直接暴露（MSE/ICR）→ 升 e_dim 到 64/128 重训；再不行切 RQ-KMeans 路线 |
| RQ-VAE 5000 epochs 训练慢 / codebook 死码 | 中 | M2 延期 | 早停按 collision rate；`kmeans_init=True` 已有；减少 epochs + 提高 quant_loss_weight |
| 2023 I&S items 规模远超本地承受（>10 万） | 中 | 编码/训练慢 | 按交互频次截断到 ≤5 万 items（保留长尾消融对照组） |
| Qwen3 + trl 版本兼容问题 | 中 | M3 延期 | 锁 transformers/trl 版本；最坏回 Qwen2.5-0.5B（其余升级不受影响） |
| 语义初始化无收益 | 中 | M4 结论变负 | 保留实验记录，转为"我们试过 X，原因是 Y"的面试素材 |
| OPD 被 teacher 上限锁死 | 中 | R2/R3 不及 R1 | λ 退火调度 / 换 4B teacher（3090 场景） |
| 4GB 本地 RL OOM | 高 | 本地不能迭代 RL | 本地只跑 SFT 消融 + 路线 A 小规模；RL 正式实验全部上云 |
| hf-mirror 单点依赖（官方站不可达） | 高 | 数据下不下来 | 已实测 mirror 可用；metadata 下完即本地留档，后续不再依赖网络 |

---

## 11. 新目录结构规划（增量）

```
GenRetrieval/
├── docs/
│   ├── UPGRADE_PLAN.md          # 本文档
│   └── EXPERIMENT_LOG.md         # 实验记录与踩坑
├── data/
│   ├── Amazon23/raw/             # M1：官方 5core 分片 + metadata（下载物，不入 git）
│   └── Amazon23/IandS/           # M1：item.json / inter.json / train|valid|test CSV
├── scripts/data/                 # M1 新增
│   ├── download_amazon23.sh
│   └── prepare_amazon23.py
├── scripts/multimodal/           # M1 新增
│   ├── download_images.py
│   ├── encode_text.py / encode_image.py
│   └── fuse_embeddings.py
├── rq/
│   ├── models/                   # ✅ 已从上游恢复（rqvae/rq/vq/layers/generate_indices）
│   ├── eval_sid.py               # M2 新增：SID 三件套评估
│   └── rqkmeans_faiss.py         # 沿用（对照路线）
├── sid_token_init.py             # M4 新增：语义初始化
├── sft.py / rl.py / minionerec_trainer.py   # 原地升级
├── ts_rec_*.py                   # 上游新变体（SFT 参考）
└── ...（V0 文件全部保留，git 历史可回溯）
```

**数据不进 git**：`data/Amazon23/` 下的大文件（1GB metadata、图像、.npy）走 `.gitignore`，只提交脚本与小的样本/统计摘要。

---

## 12. 环境与版本规划

| 环境 | 用途 | 关键依赖（实测确定） |
|---|---|---|
| **`.venv`（项目内，py3.11.7）** | 全部阶段：文本/图像编码、RQ-VAE、SID 评估、SFT/RL | **torch 2.6.0+cu118** / torchvision 0.21.0+cu118 / transformers 4.57.1 / trl 0.24.0 / peft 0.14.0 / accelerate 1.10.1 / datasets 4.2.0 / numpy 1.26.4 / pandas 2.2.2 / scipy 1.13.1 / pyarrow 17.0.0 / faiss-cpu 1.8.0 / pot |
| 系统 python（`C:\Users\z\.workbuddy\binaries\python\versions\3.13.12`） | 数据下载/准备等零依赖脚本 | 仅标准库 |

> **栈版本 = 与 `requirements.txt` 完全一致**（torch 2.6.0 / transformers 4.57.1 / trl 0.24.0）。这不是随意选的：`minionerec_trainer.py` 直接 import trl 的内部符号（`trl.trainer.utils.selective_log_softmax`、`trl.models.unwrap_model_for_generation`…），trl 0.16 与 0.24 之间有差异 → **必须保持 0.24**，因此 torch 不能停在 2.0.0。
> torch wheel 已离线保存在 `.wheels/torch-2.6.0+cu118-cp311-cp311-win_amd64.whl`（2.73GB），换机器/重装无需再下。
> **下载性能铁律**：沙箱把网络限速到 ~110kB/s（实测沙箱外 6-7MB/s，差 51 倍）→ 所有 >100MB 下载必须在沙箱外跑。
> 踩坑全记录：conda 沙箱失败（E-04）、清华/中科大镜像 403（E-05）、torch 源慢（E-06）、日志混写（E-07）、pandas 3 与 numpy 冲突（E-08）、**沙箱限速（E-09）**、WINPID（E-10）→ 详见 EXPERIMENT_LOG。

---

## 13. 决策记录（2026-09-11 已确认）

| # | 决策点 | 决定 | 影响 |
|---|---|---|---|
| 1 | 数据类目 | **① Industrial_and_Scientific 2023 版** | 与 V0 纵向对照最干净（§3.2） |
| 2 | 硬件形态 | **① 本地 4GB 迭代 + 云 3090 出正式数字** | SFT 双轨（QLoRA/全参）；OPD 只在云上跑（§7、§12） |
| 3 | RL 路线 | **① 双线并行**（R1 隔离奖励设计 / R2 隔离 OPD） | §6.3 实验矩阵 R0~R3 全跑 |
| 4 | 2018 老数据 | **不用** —— 已从工作区移除（`git rm data/Amazon`） | 数据脚本保留；如需恢复：`git checkout ac0d0b9 -- data/Amazon`（原提交内有完整备份） |
| 5 | SID 路线（本轮追加） | **RQ-VAE 为默认**；RQ-KMeans / FAISS-RQ(+last-layer Sinkhorn) 作挑战者，**用 §4.5 三件套 + 端到端择优，不预设结论** | §4.1；并触发补齐上游缺失的 `rq/models/`（§1.4） |

> 决策 5 的落地顺序：先 RQ-VAE（默认）跑通 3 层 × 256 + e_dim 32/64 两档，同时跑 RQ-KMeans 与 FAISS-RQ 对照，三件套结果出来后写进 EXPERIMENT_LOG 再决定主路线。

---

## 附：与 esci-ai-search 项目的关系

本项目（GenRetrieval）= **序列生成式推荐/召回**（query 是行为序列）；esci-ai-search = **查询-商品搜索排序**。两者共享：Qwen3-Embedding 编码资产（`embqwen` venv）、FAISS RQ + last-layer Sinkhorn 的 SID 代码经验、三层 SID 评估框架。面试叙事上互为补充：一个讲生成式召回，一个讲搜索排序。
