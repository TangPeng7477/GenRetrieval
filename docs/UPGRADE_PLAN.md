# GenRetrieval 升级方案：MiniOneRec → 多模态生成式召回 2.0

> 状态：**已确认**（决策结果见 §13），**执行中**（2026-09-13 更新）
> 日期：2026-09-11（2026-09-13 刷新执行状态）
> 基线锚点：本项目复刻版 MiniOneRec（原 `D:\Codings\cs\Rec\MiniOneRec_oneGPU_preject`，已完整复制到本目录）

### 执行状态（2026-09-13）

| 里程碑 | 状态 | 交付 |
|---|---|---|
| M1 数据（Amazon23 I&S + VG，图文 100%/99.9% 可得） | ✅ 完成 | `docs/DATASET.md` |
| M1 多模态编码 + 融合 | ✅ 完成 | 定版 `gate_e60`（R@10 0.1100，vs 纯文本 +33%；指标定义见 §4.5） |
| M2 SID 构建（RQ-VAE） | ✅ 完成 | **定版配方 = gate + `--init_samples 8192` + 5000 轮 + `sid_raw`（语义桶）** |
| M2 SID 质量评估 | ✅ 完成 | 三件套（ICR 0.9582 / LCP 222.1 / R² 0.6530 / L0 死码 0%；公式见 §4.5） |
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
