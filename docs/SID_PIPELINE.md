# 语义 ID（SID）构建：方法参考、实现细节与实验全记录

> 定位：**SID 阶段的唯一入口**。把原先分散的三份文档
> （`SID_PIPELINE.md` 流程结论 / `MULTIMODAL_SID_SURVEY.md` 工业调研 / `SID_TRAINING_RECIPES.md` 开源配方）
> 合并成一份，按「参考方法 → 本项目实现与知识点 → 探索过程 → 边界与下一步」组织。
>
> - **第一部分**是别人怎么做（大致介绍，含代码级配方，用于定位与对标）；
> - **第二部分**是我们实际用的模型与知识点（详细，可当实现说明书）；
> - **第三部分**是我们自己踩出来的过程与数字（详细，含失败实验）；
> - 机理推导与 FAQ 另见 `docs/KNOWLEDGE_BASE.md`；流水账见 `docs/EXPERIMENT_LOG.md`（本地，不上传）。
>
> 最后更新：2026-09-13 · 域：Amazon23 `IandS`（Industrial_and_Scientific，25,847 items）

---

## 0. TL;DR：定版配方与最终指标

**默认配方（一条命令可复现）**

```
Amazon23 I&S 官方 5core/timestamp 分片
  → 文本 Qwen3-Embedding-0.6B (1024d) + 图像 SigLIP-base (768d)
  → gate 门控融合（共现 InfoNCE，60 轮）
  → RQ-VAE 3 层 × 256 码 × 32 维，--init_samples 8192，5000 轮（bs 2048，lr 1e-3）
  → 直接导出 sid_raw（纯 argmin，不做 Sinkhorn 碰撞消解）
```

```bash
RESULTS_ROOT=results/sid_e5000 INIT_SAMPLES=8192 \
  bash scripts/multimodal/run_sid_exp.sh IandS 5000 "gate"
```

**最终 SID 质量（`gate__init8192`，N=25,847）**

| 指标 | 值 | 说明 |
|---|---|---|
| ICR（唯一码比例） | **0.9582** | 965 组 / 2,046 物品（7.9%）共享码，桶均 2.1、最大 5 |
| LCP ratio | **222.1** | 语义近邻对的 SID 前缀长度 / 随机对，random 的 222 倍 |
| prefix-1 内聚比 | **2.086** | 同前缀组内向量相似度 / 随机基线 |
| 重建 R² | **0.6530** | decoder 重建，量化噪声约 34.7% |
| 第 0 层死码率 | **0%**（256/256 满用） | 塌缩问题在该配置下不存在 |

**为什么不再做 Sinkhorn 碰撞消解**（2026-09-13 定版，理由三条）

1. **碰撞组不是噪声，是语义簇**：实拍最大 3 组全是同品牌同系列规格变体（O 圈 / 自攻螺丝 /
   拉紧带），组内两两 cos 0.92~0.95，随机对基线 0.18 —— 生成式召回整桶返回后交给排序，
   正是工业界形态（YouTube PLUM：code → 桶 → ranker）。
2. **消解不彻底，而彻底化的收益没有证据**：Sinkhorn 只能把 ICR 从 0.9582 提到 0.9997，
   残留 14 物品 / 7 组全部是**逐位相同的重复 embedding**（确定性规则数学上不可分）；
   且消解要改写 6.54% 物品的码，LCP 222.1→209.4、R² 0.6530→0.6504 —— **换唯一性、丢结构**。
3. **工业界证据：唯一性本身不是硬指标**。Snap《Semantic IDs for Recommender Systems at
   Snapchat》（SIGIR'26 Industry Track, arXiv 2604.03949，官方代码 = `refs/GRID`）Table 5：
   唯一性 92.95%→70.58%，Amazon Beauty 的 GR Recall@10 只从 6.1 掉到 6.0 —— **~70% 以上即平台期**；
   原文："Uniqueness should not be evaluated as a gold standard"。其 Table 4 的线上 A/B 更是
   直接用"Top 10 SIDs、每码映射 100 个物品、relevance-guided 消歧"（view +0.57%、share +4.39%）。

> 口径声明：本项目**不做** raw / sk 的 SFT 端到端消融（成本有限）。上述决策依据是
> 离线结构指标 + 工业界先例，属"有依据的设计选择"，不是端到端验证过的结论。
> `sid_sk` 仍随每次导出一并产出（`sid_sk.npy`），随时可切回唯一化口径。

---

# 第一部分 · 参考方法：别人怎么做（大致介绍）

## 1.1 共同骨架与工业落地全景

2026 年工业界的多模态 SID 已收敛出一条**共同骨架**：

```
多模态 embedding（各自冻结的 encoder）
  → 融合成单一向量
  → RQ（残差量化）成层级离散码
  → LLM 自回归生成
```

但**融合层的位置**和**碰撞处理策略**上各家给出了不同答案 —— 这两处也正是我们做取舍的地方。

| 公司 | 系统 | 场景 | 关键贡献 |
|---|---|---|---|
| Google/YouTube | **PLUM** (arXiv:2510.07784) | Shorts / 长视频 | **SID-v2**：多模态融合 + 多分辨率码本 + 共现对比正则；线上 Panel CTR **+4.96%** |
| 阿里/淘宝 | **FORGE** (arXiv:2509.20904) | 猜你喜欢 | SID 构建策略系统性基准；**两个免训练评估指标**；线上 **+0.35% 交易量** |
| 快手 | **AdaSID** (arXiv:2604.23522) | 电商短视频 | **自适应碰撞处理**（两阶段）；线上 **GMV +0.98%** |
| 快手 | **OneRec** 系列 | 4 亿+ DAU 全场景 | 首次工业级替代级联；V2 转 decoder-only，训练算力 **-94%** |
| Snap | **GRID** (CIKM'25) | 学术基准 | SID 组件系统性 ablation 框架，官方代码 `refs/GRID` |
| Meta | HSTU / LIGER | ranking + retrieval | 推荐领域首个 Scaling Law（1.5T 参数，A/B +12.4%） |
| 京东 / 腾讯 / 阿里国际 | OxygenREC / GPR / Masked Diffusion GR | 多场景 | 指令跟随式 GR / 多场景 GR / 掩码扩散生成 |

## 1.2 四种主流范式（简述）

### PLUM / YouTube —— "融合即投影 + 对比正则进量化器"

SID-v2 四项增强：① 多模态 embedding 各自 encoder 后 concat 再投影成单向量；
② **多分辨率码本**（第 level 层码本 = `2048 / 2^(level-1)` → 2048 / 1024 / 512，浅层强区分、深层编残差）；
③ Progressive Masking 强制层级可解释；④ **共现对比正则 `L_con` 直接加进 RQ-VAE 训练目标**，
把协同信号注入 tokenization 过程。线上 Shorts Panel CTR +4.96%，CPT 约 260B tokens。

> 对我们的意义：PLUM 把对比损失加在**量化器**上；我们是"先训融合网络、冻结后再训 RQ-VAE"的
> **两阶段分离**。这是明确的升级点（见第五部分）。

### FORGE / 淘宝 —— "SID 构建策略基准 + 免训练评估"

系统性研究 SID 构建自由度：默认 **3 层 × 8192** 最优；`1024_4096_32768` 组合能提升 HR@1000；
RQ-VAE 是最合适的 tokenizer。提出两个**不需要跑 GR 训练**的评估指标：
`embedding hitrate` + `Gini coefficient`（衡量码本使用均衡度）。线上 +0.35% 交易量，已全量部署。

> 对我们的意义：我们 `rq/eval_sid.py` 的三件套目标与之一致；Gini 可直接量化
> "Sinkhorn 是否把物品摊得太均匀"，值得作为第 4 个指标补进来。

### AdaSID / 快手 —— "自适应碰撞处理"（★ 直接命中我们的实测问题）

论文批评现有方法"依赖固定重叠正则（fixed overlap regularization），
不能自适应区分某个重叠应该被抑制还是保留"。两阶段框架：

**Stage 1 — 语义自适应重叠放宽**：对冲突物品对 `(i,j)`，看连续多模态 embedding 的 cos 相似度
`sim_ij`，若 `sim_ij ≥ η` 则**判定语义兼容、不惩罚**。阈值**深度感知**（冲突越深要求越严）：

| 冲突层级 | 阈值 η |
|---|---|
| 1 级 | 0.18 |
| 2 级 | 0.24 |
| 3 级 | 0.30 |

**Stage 2 — 自适应压力分配**：未放行的冲突，惩罚权重 = 空间拥挤度 `a_ij`（1~3）
× 时间衰减 `λ_col(τ) = 1 - (1-λ_min)·τ`；总损失
`L_col_ada = Σ [(1 - g_ij) × a_ij × λ_col(τ) × margin_loss]`。

离线上 Toys + TIGER：vs RQ-VAE 基线 **Recall@5 +42.6%**；线上快手电商 GMV **+0.98%**。

> 这正是我们实测到的现象：**Sinkhorn 无差别消解 = 论文批评的 fixed overlap regularization**。
> 我们最终的"语义桶"决策与 AdaSID 的结论同源，只是我们选了更省事的形态（整桶保留 + 下游排序消歧）。

### OneRec / 快手 —— "统一生成 + 多模态 tokenization"

把召回、排序、解释统一进单一生成模型；item 用 hierarchical quantization（RQ-KMeans 或 RQ-VAE）
压成 "itemic tokens"；V2 转纯 decoder-only "lazy" 架构 → 训练算力 -94%，可扩到 8B+；
RL 用 GRPO / ECPO / DPO。

## 1.3 五家开源实现的真实训练配方（读代码核对，非论文转述）

### 三个训练阶段必须分开谈

```
① 融合投影（本项目自加）     → 维度对齐，10^1 轮量级
② SID tokenizer（码本）      → 量化，按路线差 300 倍
③ 生成式模型（SFT / RL）     → LLM 微调，按 step 算，几个 epoch
```

### 各家配方

| 项目 | 数据集 | tokenizer | 层×码本 | tokenizer 训练量 | 融合方式 | 生成器训练量 |
|---|---|---|---|---|---|---|
| **MiniOneRec** | Amazon23 I&S | RQ-VAE | 3×256 | **5,000~10,000 轮**（bs 20480 / lr 1e-3 / warmup 50） | 无（纯文本） | SFT ≤10 ep（lr 3e-4, 8×H100），GRPO 2 ep，beam 16 |
| **ETEGRec** (SIGIR'25) | Amazon23 sci/inst/game | RQ-VAE 端到端 | 3×256, e_dim 128 | **10,000 轮**（bs 1024, wd 1e-4, linear sched） | 无（SASRec 协同） | `--cycle=2` tokenizer↔推荐器交替；`lr_rec=5e-3` / `lr_id=1e-4` |
| **MQL4GRec** (ICLR'25) | Amazon18 | **RQ-VAE ×2（文本/图像各一个）** | **4×256** | **500 轮**（bs 2048, eval_step 2） | **不融合，双 SID** | pretrain 30 ep + finetune 200 ep，5 个跨模态任务 |
| **GRID** (Snap, CIKM'25) | Amazon P5 | **RQ-KMeans** | 3×256 | **30 步/层**（SGD lr=0.5，只更新质心） | 无 | TIGER 320,000 步 |
| GRID | Amazon P5 | RQ-VAE | 3×256, latent 64 | **3,000 步**（**Adagrad** lr 1e-3，BN→L2 归一化） | 无 | 同上 |
| **MMGRec** (arXiv'24) | 多模态 | Graph RQ-VAE | — | — | **concat + GCN 注入 CF** | R@10 0.1269 |
| **本项目** | Amazon23 I&S + VG | RQ-VAE（同 MiniOneRec） | 3×256, e_dim 32 | **5,000 轮**（bs 2048, lr 1e-3, AdamW） | **concat/mlp/gate → 单向量** | 待 SFT |

**GRID 的 RQ-VAE config 值得单独抄**（`configs/experiment/rqvae_train_flat.yaml`）：
encoder MLP `input_dim → [768,256,128] → 64`、decoder 镜像、`BetaQuantizationLoss(beta=0.25)`、
**Adagrad lr=0.001**（不是 Adam）、`WarmupLinear(warmup_steps=1000)`、`max_steps=3000`、
`init_buffer_size=3072`（MiniBatchKMeans, KMeans++, max_iter=1000）、量化前 **BatchNorm → L2 归一化**。

### ★ Sinkhorn 该放在哪：MiniOneRec 与 MQL4GRec 是两派

`sk_epsilons` 两家都有，但**插入位置完全不同**：

| | 训练期 | 导出期 |
|---|---|---|
| **MiniOneRec** | `sk_epsilons=[0,0,0]` **完全不开** | 前 L-1 层强制 0.0、**仅最后一层 0.003**，while 循环 ≤20 轮、**只对碰撞组重编码** |
| **MQL4GRec** | `sk_epsilons=[0,0,0,0.003]` **训练期就带** | 直接导出，无后处理 |
| **本项目** | **纯 argmin**（= MiniOneRec 口径，`--train_use_sk` 默认 False） | raw = 纯 argmin；sk = 仅末层、仅碰撞组、eps 0.003、iters 50、batch 64、≤20 轮 patience 3 |

判断是否真在跑 Sinkhorn，只看 `vq.py` 这一个条件：

```python
if not use_sk or self.sk_epsilon <= 0:
    indices = torch.argmin(d, dim=-1)      # 最近邻
else:
    Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
    indices = torch.argmax(Q, dim=-1)      # 最优传输分配
```

注意 `RQVAE.forward(use_sk=True)` 与 `get_indices(use_sk=False)` 的**默认值是反的**，
所以"训练期开不开"完全取决于 trainer 调哪个入口。

## 1.4 我们抄了什么、改了什么、拒绝了什么

| 决策 | 来源 | 我们的处理 |
|---|---|---|
| 导出期 **只动最后一层** | MiniOneRec `generate_indices.py:106-110` | ✅ 抄。保前缀语义结构（Trie beam / LCP 的命根子） |
| 导出期 **只修碰撞组**、迭代到收敛（上限 20 轮） | 同上 | ✅ 抄。全量跑 Sinkhorn 会把不撞的 item 也挪位，白白损失保真度 |
| 训练期 **不开** Sinkhorn | MiniOneRec | ✅ 采纳（MQL4GRec 派经实测被支配，见 §3.3） |
| k-means 初始化 | 上游是"训练首 batch" | 🔧 **改成独立 init pass（8192 样本）**：L0 死码 32.6%→0% |
| 逐层训练 `train_layer_wise` | GRID（仅 RQ-KMeans 路线启用） | ❌ 不适用（我们 RQ-VAE 端到端） |
| 量化前 BN→L2 归一化 | GRID | ⚠️ 未接，可选精修 |
| 共现对比损失下沉到量化器 `L_con` | PLUM | ⚠️ 未接（我们只加在融合层），列为高 ROI 改进 |
| 语义自适应碰撞 | AdaSID | ⚠️ 未接；我们改用更省事的"整桶保留 + 下游排序消歧" |
| 多分辨率码本（2048/1024/512） | PLUM | ⚠️ 未接；仍是均匀 256×3 |

---

# 第二部分 · 本项目实际用的模型与知识点（详细）

## 2.1 流水线、产物与耗时

| 阶段 | 脚本 | 产物 | 耗时（实测） |
|---|---|---|---|
| 数据准备 | `scripts/data/prepare_amazon23.py` | `data/Amazon23/IandS/{item.json, item2id, *.inter, stats.json}` | 分钟级 |
| 图像下载 | `scripts/multimodal/download_images.py` | `images/*.jpg` 25,824 张 / 2.7GB | 26 min（32 线程） |
| 文本编码 | `scripts/multimodal/encode_text.py` | `emb/emb_text_title-features-category.npy` (N,1024) | ~4 min（token 预算装箱后 101 item/s） |
| 图像编码 | `scripts/multimodal/encode_image.py` | `emb/emb_image_siglip.npy` (N,768) + `mask` | ~20 min |
| 融合训练 | `scripts/multimodal/fuse_embeddings.py` / `fuse_long.sh` | `emb/long/emb_fused_gate_e60.npy` (N,1024) | 60 轮 ≈ 20 min |
| RQ-VAE 训练 | `rq/train_rqvae.py`（`run_sid_exp.sh` 编排） | `ckpt/selected_model.pth` + `train_metrics.json` | **3,644 s（5000 轮）** |
| SID 导出 | `rq/build_sid_dual.py` | `sid_raw.*` / `sid_sk.*` | 9 s |
| 质量评估 | `rq/eval_sid.py` | `eval_raw.json` / `eval_sk.json` | ~30 s |

⚠️ 两域（I&S / VG）**各自独立跑，不合并、不共享码本**；本文所有数字均为 I&S。

## 2.2 数据

| 项 | I&S |
|---|---|
| items / users | 25,847 / 46,341 |
| train / valid / test | 251,576 / 16,074 / 13,232 |
| history_len 上限 | 20（train p50=3、test p50=5） |
| 图像 URL 覆盖 → 下载有效 | 100% → **99.91%**（25,824 张） |
| 时间泄漏 | 无（max train 2021-08-11 < min test 2022-07-17，全局时间切分） |

切分用官方 `benchmark/5core/timestamp` 分片（23.9MB），省 2.2GB 全量评论下载与自跑 k-core。
valid/test 中"无 train 历史"的用户必须丢弃（1,856 / 4,207），否则指标口径不可解释。

## 2.3 文本塔：Qwen3-Embedding-0.6B

| 项目 | 值（★ = 本机 `config.json` 实测读出） |
|---|---|
| 出身 | 阿里 Qwen 团队，2025-06 发布，Apache 2.0 |
| 架构 | ★ `Qwen3ForCausalLM` —— **本质是 decoder-only 因果 LM**，经 LoRA 微调成双塔 embedding |
| 层数 / 宽度 | ★ 28 层 / hidden 1024，intermediate 3072 |
| 注意力 | ★ GQA：16 Q 头 / 8 KV 头；★ `head_dim=128` |
| 词表 / 上下文 | ★ 151,669；★ 32,768（RoPE θ=1e6）；★ `tie_word_embeddings=true`，★ bf16 |
| 输出 | 1024 维（MRL 支持 32~1024 截断） |
| 池化 | **last-token pooling**：取最后一层 `[EOS]` 位置隐状态（需**左 padding**） |
| 官方评测 | MTEB (Eng) **70.70** / MMTEB 64.33 / MTEB-Code 75.41 |
| 本机实测 | dim=1024，I&S 文本均长 711 字符 / token 均 173（中位 175，p95 256）；显存 1.19GB |

**三个容易被忽略的设计点**

1. **`head_dim` 与 `hidden_size / num_heads` 解耦**：★ 实测 `16 × 128 = 2048 ≠ hidden 1024`。
   Qwen3 刻意把 head_dim 固定 128，写自定义 attention 或估 FLOPs 时按老公式会算错一倍。
2. **它是因果 LM，不是双塔编码器**：embedding 只是"把最后一层隐状态抠出来"——
   这也解释了为什么必须 last-token pooling（因果注意力下只有末位 token 能看见全部前文）。
3. **指令感知**：query 侧加 `Instruct: {任务描述}\nQuery: {query}`，文档侧不加；
   官方称不加指令掉 1%~5%。我们**只做物品侧编码、不加指令**（合理）；
   将来若要编码用户 query，必须走 `get_detailed_instruct`。

**三阶段训练范式**（可直接复用到面试）：① 用 Qwen3-32B **合成**弱监督文本对做大规模对比预训练
→ ② 高质量标注数据监督训练 → ③ **Slerp 球面插值模型融合**。去掉任一阶段都会掉点。

**为什么停在 0.6B**：4B 是 MTEB 74.60 / 2560 维（fp16 权重 8GB）、8B 是 75.22 / 4096 维（16GB），
**4GB 卡上只有 0.6B 跑得动**。诚实的差距：0.6B 的 70.70 低于最佳竞品 73.30。
真正的解法是"轻量域适应"或在云端 3090 上换 4B 重编码——那也是唯一能验证
"编码器规模是否是瓶颈"的实验设计。

## 2.4 图像塔：SigLIP-base-patch16-224

| 项目 | 值 |
|---|---|
| 出身 | Google DeepMind，ICCV 2023 Oral，arXiv:2303.15343 |
| 视觉塔 | ViT-B/16，12 层，hidden 768，patch 16×16，输入 224×224 → 196 patches |
| 训练数据 | WebLI（仅英文图文对） |
| 官方零样本 / 检索 | ImageNet 73.4%；COCO R@1：T→I 49.7 / I→T 67.5（576-patch） |
| 输出 | 768 维，已 L2 归一化，两塔共享空间 |
| 本机实测 | 覆盖 25,824/25,847 = **99.91%**，858s，显存 0.42GB |

**核心创新 —— 逐对 sigmoid 损失（替代 CLIP 的 softmax InfoNCE）**：

- CLIP 的 softmax 要在整批相似度矩阵上做**全局归一化**，需 all-gather 同步、显存 O(B²)、损失与 batch 强耦合；
- SigLIP 把对齐转成**所有图文对组合上的二分类**（对角线为正、非对角线为负），逐对 sigmoid CE 求和，
  **不需要看到全 batch** → 可分块计算，显存降到 O(b²)；
- 再加**可学习温度 t 与偏置 b**（初始化 log 10 / −10）缓解 1:N−1 极端不平衡——消融显示这个 bias **至关重要**；
- 副产品结论：对比学习的 batch 收益 **32k 就饱和**。

**对我们的意义（诚实版）**：sigmoid 的卖点是**训练效率**，而我们纯推理（冻结、不微调），
这个优势用不到。选它的真实理由是：开箱即用的强图文对齐 + 只占 0.42GB + 768 维省下游开销。
⚠️ 局限：SigLIP 学的是**通用英文网络图文**，不是电商商品域（白底图 + 包装盒 + 文字特写分布有偏），
我们**没有做任何域适应**——这是流水线里一个未被验证的假设。

## 2.5 融合：四模式 + 共现 InfoNCE

同一留出集（**80,079 对，avg_pos 7.88，随机基线 R@10 = 0.003**）：

| 模式 | R@10 | R@50 | R@100 | vs text | 参数量 |
|---|---|---|---|---|---|
| text（单模态基线） | 0.0830 | 0.1783 | 0.2280 | — | 0 |
| concat（PCA 线性） | 0.0883 | 0.1790 | 0.2347 | +6% | 0 |
| mlp_e60 | 0.1013 | 0.2693 | 0.3623 | +22% | 1.44M |
| **gate_e60** | **0.1100** | **0.2787** | **0.3820** | **+33%** | 3.28M |

**结论：监督融合才是多模态增益来源，线性拼接几乎白给。**
gate 的乘性门控让模型自己学"每个物品信文本还是信图像"，缺图商品（23 个）自动偏文本。

⚠️ 公平性边界：gate 参数量是 mlp 的 2.27×，且 mlp 有 LN+Dropout 而 gate 分支没有 →
现有证据只支持"gate_e60 是最好的可用配置"，**不支持"门控机制优于 MLP"的机制性结论**。

**轮次行为（`fuse_long.sh IandS 60 5`，12 个留出点）**：

| epoch | 5 | 10 | 15 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|---|
| R@10 | 0.0855 | 0.0960 | **0.0995** | 0.0920 | 0.0990 | **0.1080** | 0.0955 | 0.0985 |
| R@50 | 0.2135 | 0.2355 | 0.2330 | 0.2360 | 0.2395 | 0.2445 | 0.2495 | **0.2515** |
| R@100 | 0.2955 | 0.3245 | 0.3215 | 0.3220 | 0.3285 | 0.3375 | 0.3405 | **0.3430** |

- **R@10 在 e10 后即饱和**：e10=0.096 → e60=0.0985，中间无趋势震荡（e40 的 0.108 是噪声）。
- **R@50/R@100 到 e60 仍单调改善**（+17.8% / +16.1%），无拐点。
- loss 一路单调降（11.29 → 10.22），**无法用来判断何时停**。

→ 面向 top-10 精度取 **10~15 轮**；面向长尾覆盖与码本语义结构取 **40~60 轮**。
正式配方取 **e60**（下游是 RQ-VAE，更看重整体语义结构）。

**评估必须用与训练互斥的留出对**：实测训练对自评有 **13.6 倍水分**（0.610 vs 0.045），
不切分则所有 epoch 结论都不可信。

## 2.6 RQ-VAE：结构、损失与超参

```
x (1024d 融合向量)
  → Encoder MLP  [2048, 1024, 512, 256, 128, 64] → z (32d)
  → 残差量化 3 层：每层 256 个码 × 32 维
       r0 = z                    c0 = argmin_k ||r0 - e_k||     r1 = r0 - e_{c0}
       c1 = argmin_k ||r1 - e_k||     r2 = r1 - e_{c1}
       c2 = argmin_k ||r2 - e_k||
  → ẑ = Σ_l e_{c_l}（straight-through，梯度直接复制到 encoder）
  → Decoder MLP（镜像）→ x̂
```

**损失**

```
L = ||x - x̂||²                                   # 重建
  + Σ_l ||sg[z_l] - e_{c_l}||²                   # codebook loss（只训码本）
  + β · Σ_l ||z_l - sg[e_{c_l}]||²               # commitment loss（只训 encoder），β = 0.25
```

**超参（`rq/train_rqvae.py`）**：AdamW / lr 1e-3（warmup 后 constant）/ grad_clip 1.0 /
bs 2048 / `kmeans_init=True` / `num_emb_list=[256,256,256]` / `e_dim=32` / seed 2024 /
`--select_ckpt collision --burn_in_frac 0.1`。

**三个要点**

1. **straight-through**：量化不可导，梯度从 decoder 直接穿过量化层回传 encoder，
   码本则靠 codebook loss 单独更新（用 `sg` 切断另一侧）。
2. **commitment β=0.25** 是 TIGER/ETEGRec/MiniOneRec 的共识值；它约束 encoder 别把 latent 拉得离码本太远。
3. **为什么先压到 32 维再量化**：高维直接量化碰撞率高得多——RQ-KMeans 在 1024 维直接量化的
   raw ICR 只有 0.8395，而 RQ-VAE 压到 32 维后有 0.9582（§3.4）。

## 2.7 残差量化与 k-means 预初始化（我们的关键改造）

**逐层 k-means 初始化**：每层码本在训练首 batch 上跑 k-means 得到初始质心。
上游实现用的是**训练第一个 batch**（2048 条），样本覆盖面不足 → 第 0 层码本严重不均。

**我们的改造 —— 独立 init pass（`--init_samples N`）**：训练前随机抽 N 条样本过一次 encoder，
在 latent 上逐层 k-means：

```
抽 N 条 → encoder → latent (N,32)
  L0: k-means(latent)          → 256 质心；残差 r0 = latent - q0
  L1: k-means(r0)              → 256 质心；残差 r1 = r0 - q1
  L2: k-means(r1)              → 256 质心
写入各层 embedding.weight，置 initted=True，训练循环 bs 保持 2048 不变
```

实测（N=8192）：三层均 256/256 非零码，init 耗时 **7.9~9s**，
逐层残差能量 ‖·‖ = **1.1354 / 0.8720 / 0.6955**（单调衰减 ⟹ 各层量化预算逐层收敛）。

**⚠️ 踩坑：k-means 不可复现**。原 `layers.py` 的 `KMeans(...)` 没设 `random_state`、且 `shuffle=True`，
实测同一输入两次聚类质心 max|diff| = **1.605e-01**，而元素量级仅 1.661e-01（**96.6%**）
—— 这正是"第 0 层用码数在 seed 间波动 59~85"的根因。已改为 `random_state=0, n_init=10`，
复测 L0 **bit-level 一致**（L1/L2 残留 1.9e-9 属 CPU float32 多线程归约噪声，可忽略）。

**⚠️ 踩坑：init 样本量不是越大越好**。8192 是甜点（L0 死码 0%、LCP 222.1）；
全量 25,847 反而略差（死码 8.6%、LCP 212.7）且 init 成本 28.5s。排序：**8192 ≥ full ≫ 首 batch**。

## 2.8 码本塌缩与死码

**现象**：训练中大量码从未被选中（死码），尤其**第 0 层**（gate__init0 时 L0 死码 32.6%）。

**机理**：梯度只流向被选中的码 → 被选中的码不断向样本靠拢 → 更容易被选中（**富者愈富的正反馈**）。
第 0 层残差能量最大（1.1354 vs 0.8720 vs 0.6955），可选码的"地盘"最悬殊，塌缩最狠。

**四种通用修法（学术界）** —— 我们一条都没接，靠 init pass 在本配置下绕开了：

| 方法 | 做法 | 代价 |
|---|---|---|
| EMA 更新 | 码本用指数滑动平均而非梯度 | 需维护额外状态 |
| diversity loss | 显式加"码本使用熵"正则 | 多一个超参 |
| dead code reset | 定期把死码重置到样本密集区 | 需监控与触发策略 |
| random last levels | 深层码本随机初始化不训 | 牺牲深层表达力 |

**我们的选择**：init8192 在本配置下把 L0 死码打到 **0%**（ep500 即满用并保持到 5000 轮），
零额外超参、零训练期代价。但要诚实：**这是"绕开"不是"修复"**，
换数据集/换码本规模是否仍成立未验证。

## 2.9 Sinkhorn 碰撞消解：算法、能力与边界

**算法**：把 [B, K] 距离矩阵 `d` 转成分配问题，用 Sinkhorn 迭代求**近似双随机矩阵** Q，
再 `argmax(Q)` 得到分配：

```python
Q = sinkhorn_algorithm(d, eps=0.003, iters=50)   # 行列交替归一化
indices = torch.argmax(Q, dim=-1)
```

**为什么能消碰撞**：双随机约束让每个码在一批里"被用且只用一次"，迫使拥挤码本腾位。

**⚠️ 关键坑：batch 必须 ≤ K**。Sinkhorn 的 [B,K] 双随机约束下，若 B > K，
每码期望 B/K 个物品 → **结构性碰撞**。我们最初导出 batch=2048（≫ K=256），
ICR 只有 0.935；改到 **batch=64**（MiniOneRec 实际值）后 ICR 直接到 **0.9997**。

**为什么消不彻底（实测）**：残留下来的 14 物品 / 7 组 **100% 是逐位相同的重复 embedding**。
对相同输入，任何确定性规则（argmin 也好、argmax-of-Sinkhorn 也好）必给相同输出，
**数学上不可分**。实测调参完全无效：

| 扫描 | 结果 |
|---|---|
| eps 0.003 → 1.0 | 最多消 1 组，MSE 代价 +0.11（比正常消解大 4~5 个量级）；eps≥0.3 反而 Q 趋均匀、argmax 退化、碰撞回升 |
| iters 50 → 20000 | 同样只消 1 组 |

**"该全碰撞却只剩 7 组"的机理**：sid_raw 里 30 组孪生 embedding **100% 全碰撞**（两套量化器都验证）。
Sinkhorn 消解循环每轮把碰撞名单按 64 一桶重新分组——**同桶孪生必然同码**（列归一化对等比缩放的行无效），
**异桶则上下文不同、可能拆开一个**，拆开后即退出名单、结果固化。29/30 组索引差 ≥64，
所以这是一场**确定性但任意的抽签**：RQ-VAE 抽中 23/30、rqkmeans 21/30；
两套 run 的残留名单几乎不重叠（仅 2 组相同），证实是抽签而非几何规律。

## 2.10 训练动力学：三相结构与"为什么不能按 loss 选 ckpt"

![RQ-VAE 训练曲线三相结构](figures/train_curve_3phase.svg)

| 阶段 | 轮次 | 现象 |
|---|---|---|
| 一 · 码本未激活 | ep1~100 | recon 主导下滑（9.74→5.53），rq ≈ 0（码本还没接管） |
| 二 · 码本接管 | ep100~500 | recon 继续降、rq 起飞（0.44→2.43），**总损失触底 5.96（ep100 附近）** |
| 三 · 记账位移 | ep500~5000 | recon 缓慢降（4.16→3.43），rq 继续涨（2.43→5.42），**总损失单调回升 6.59→8.85** |

**读数**：重建 **9.74 → 3.43（−65%，全程单调下降）**；量化 **0.05 → 5.42（单调上升）**。

**含义**：后段总损失回升是 **rate-distortion 的记账位移**——encoder 把 latent 拉得更远
（离码本更远 ⇒ rq 变大）以换取更好的重建，**不是过拟合**。
→ 选 ckpt 必须按**碰撞率**（默认 `--select_ckpt collision --burn_in_frac 0.1`，本 run 选中 ep4950），
**不能按总损失**（那样会在 ep100 就停）。

另外还有一个早期现象值得知道：**"塌缩谷"** —— ep2~3 碰撞率会崩到 ~1.0（warmup 早期 encoder 大幅漂移、
码本还没跟上的固有过渡态），与 init 方式无关。这也是 `burn_in_frac` 要跳过前 10% 的原因之一。

## 2.11 SID 质量三件套（`rq/eval_sid.py`）

| 维度 | 指标 | 定版读法 |
|---|---|---|
| **uniqueness** | ICR、碰撞率、per-layer 死码率、最大冲突组 | 语义桶口径下 ICR 是"桶化程度"，不再是硬门槛 |
| **fidelity** | 逐层累积重建 MSE / R² / cosine | 同时报"实际交付 codes"与"纯 argmin 上限"，差值 = 消解代价 |
| **retrieval structure** | LCP ratio、prefix cohesion vs random | **选型主指标**（前缀结构决定 beam search 行为） |

- **LCP ratio** = 语义近邻对的 SID 最长公共前缀长度 ÷ 随机对的同值。random=1 是基线。
- **prefix cohesion** = 同前缀组内向量平均相似度 ÷ 随机基线。
- ⚠️ 深前缀层（prefix-2/3）组大小≈1 时 cohesion 会标"样本不足/退化，不可信"，别拿它下结论。

## 2.12 对照路线：RQ-KMeans（MiniOneRec 原版 FAISS-RQ）

`rq/rqkmeans_faiss.py` 直接调用 FAISS `ResidualQuantizer`（3 层 × 256 码、beam=1、
**在 1024 维融合向量上直接量化**，无 encoder/decoder）。它代表"无梯度"路线：
GRID 的 RQ-KMeans 只要 **30 步/层**，我们全量跑完只要 **6 秒**。

---

# 第三部分 · 探索过程全记录

## 3.1 阶段一：编码器与融合（2026-09-11~12）

1. 两个塔都冻结只推理；**4GB 卡上两塔必须串行**（并行会显存 thrashing、进度归零）。
2. 融合四模式消融 + 60 轮留出曲线 → 定 **gate_e60**（R@10 0.1100，+33%）。
3. 关键认知：**留出对与训练对必须互斥**，否则 13.6 倍水分让所有 epoch 结论失效。

## 3.2 阶段二：1500 轮三模式（2026-09-12）

固定导出 batch=64 后跑 text / mlp / gate 三模式 × 1500 轮。

| 模式 | ICR | LCP ratio | R² | 结论 |
|---|---|---|---|---|
| text | — | — | 0.852 | 纯文本基线，唯一性/保真度都好但语义结构与协同信号弱 |
| mlp | 0.9901 | 90.5 | 0.7140 | "省事的安全默认"，但 LCP 远差于 gate |
| gate | 0.9505 | 151.4 | 0.475 | 轮数明显不足（R² 只有 0.475） |

→ 触发"轮次太少"的判断，去查五家开源实现的真实配方（第一部分 §1.3），
确认 RQ-VAE 端到端路线的标准量级是 **5000~10000 轮**。

## 3.3 阶段三：5000 轮五组消融（2026-09-12，14:50→20:20，5.5h）

固定 seed=2024、k-means `random_state=0`，全部 **raw 口径**：

| run | ICR | LCP ratio | R² | prefix-1 内聚 | L0 死码率 | 结论 |
|---|---|---|---|---|---|---|
| text（5000 轮） | 0.8383 | 117.5 | **0.8698** | 1.548 | 65.5% | 纯文本基线 |
| gate__init0 | 0.9505 | 151.4 | 0.6399 | 2.003 | 32.6% | 上游默认（首 batch 初始化） |
| **gate__init8192** | **0.9582** | **222.1** | **0.6530** | **2.086** | **0%** | ✅ **定版** |
| gate__initfull | 0.9567 | 212.7 | 0.6468 | 2.075 | 8.6% | 全量初始化反而略输 |
| gate__trainsk | 0.9537 | 205.5 | 0.6427 | 2.050 | 4.3% | 训练期 Sinkhorn，被 init8192 支配 |

**四条结论**

1. **轮数收益集中在 gate**：1500→5000 轮，gate R² 0.475→0.640（**+0.165**），text 只 +0.017
   （已接近 3×256 的表示上限）。
2. **`--init_samples 8192` 是甜点**：L0 在 ep500 就满用并保持到 5000 轮，LCP 151→222；
   全量 25,847 反而略差（死码 22 个、LCP 低 10），init 成本 28.5s vs 9s。
3. **训练期 Sinkhorn（MQL4GRec 配方）不采用**：首次让自变量真生效后确认它**有效**
   （L0 死码 32.6%→4.3%），但耗时 +44%（4996s vs 3473s），且被零成本的 init8192 全面支配。
4. **选 ckpt 必须按碰撞率**（见 §2.10）。

## 3.4 阶段四：RQ-KMeans 对照（2026-09-12）

量化器**原样调用** MiniOneRec 复刻代码，碰撞消解用**同款** Sinkhorn（同一 embedding、同一 eval）：

| 指标 | RQ-VAE(gate+init8192) raw→sk | RQ-KMeans(MiniOneRec) raw→sk |
|---|---|---|
| ICR | 0.9582 → 0.9997 | 0.8395 → 0.9996 |
| LCP ratio | 222.1 → 209.4 | 175.8 → 163.3 |
| 保真度 R² | 0.6530 → 0.6504（decoder 重建） | 0.4460 → 0.3797（直接量化） |
| Sinkhorn 改码比例 | 6.54% | **27.06%** |
| 耗时 | 3,644 s | **6 s** |

**结论：维持 RQ-VAE。** raw ICR 差 12 个点（0.84 vs 0.96）就是"先压到 32 维 latent 再量化"的价值；
Sinkhorn 后唯一性打平，但 rqkmeans 要改写 27% 物品的码，LCP 与保真度全面落后。
⚠️ 两者 R² 口径不同（decoder 重建 vs 直接量化），只能作参考性对照。

## 3.5 阶段五：碰撞溯源（2026-09-12~13）

**碰撞账本（`gate__init8192` sid_raw）**：965 组 / 2,046 物品（7.9%），
组大小 {2:872, 3:73, 4:17, 5:3}，桶均 2.1、最大 5。

**实拍三个最大组（全是同系列规格变体）**

| 组 | 内容 | 组内两两 cos | 随机对基线 |
|---|---|---|---|
| (172,166,152) | Mr O-Ring 硅胶 O 圈 ×5（同 70A 硬度，只差内径/外径） | 0.9456 | 0.1814 |
| (179,149,164) | Fastenere 自攻螺丝 ×5（同盘头/驱动，只差 #4~#10、长度） | 0.9196 | 0.1814 |
| (25,27,170) | CLUTCH 拉紧带 ×5（只差 16"/20"、2~4 件装） | 0.9359 | 0.1814 |

**残留碰撞的溯源**：

| 层 | 重复情况 |
|---|---|
| 数据源 | Amazon **同品多 listing**：30 组 60 个物品 title+features+categories 逐字相同、主图 URL 相同、**ASIN 不同**（9/30 组价格也不同） |
| 文本编码层 | 40 组 / 82 个逐位重复（0.32%）—— 输入字符串逐字相同，编码器忠实输出，**不是转码 bug** |
| 融合层 | 图像分支**救回 10 组**（同文不同图）；剩 30 组连图都一样 → 融合无能为力 |

→ **"残留碰撞 ≡ 重复 embedding" 严格成立**：Sinkhorn 后残留的 14 物品 / 7 组 100% 是孪生组，
非孪生碰撞一个不剩。

**一个 nuance（诚实边界）**：碰撞组 ≠ "最相似集合"的精确边界。碰撞物品的**组外**最近邻
100% 共享 prefix-1、但只有 31% 共享 prefix-2；O 圈组所在的 prefix-2 家族桶有 **12 个**成员
（碰撞的只有 5 个）。**真正的语义家族在前缀层，比碰撞组大一圈**；
而 Sinkhorn 拆开后前缀完全不变（(172,166,152) → (172,166,{13,30,127,152,209})）
——**消解只影响末层的"家族内区分"，两种方案的家族结构完全一样**。

## 3.6 阶段六：决策定版 —— 语义桶（2026-09-13）

| | sid_sk（唯一化） | **sid_raw（语义桶，定版）** |
|---|---|---|
| ICR | 0.9997 | 0.9582（桶均 2.1、最大 5） |
| LCP ratio | 209.4 | **222.1** |
| 重建 R² | 0.6504 | **0.6530** |
| 导出侧 | 需 Sinkhorn 迭代（改码 6.54%） | 纯 argmin，一步到位 |
| SFT 训练目标 | 每物品唯一 | 同一 SID 可出现在 1~5 个物品样本（学"SID↔商品族"，无标签冲突） |
| 推断侧 | SID→1 物品 | SID→整桶（≤5）；同码同路径，beam/Trie 不受影响 |
| 先例 | TIGER / MQL4GRec / ETEGRec | YouTube PLUM；Snap 线上 A/B（每码 100 品 + relevance-guided） |

**裁决依据**三条见 §0。决策的哲学是"**要么不做，要么做彻底**"：
既然 Sinkhorn 也到不了 1.0（残留全是数学上不可分的重复 embedding），
那就不如把碰撞当语义桶、把消歧责任交给下游排序。

> 已明确**不做** raw vs sk 的 SFT 端到端消融（时间成本有限），
> 并在所有文档中标注为"未经端到端验证的设计选择"。

## 3.7 踩坑清单（按发现顺序）

| # | 坑 | 后果 / 修法 |
|---|---|---|
| 1 | `train_rqvae.py` 把 `use_sk=False` **硬编码** | `--sk_epsilons` 从未生效 → 一份"训练期 Sinkhorn 有效"的结论**作废**（自变量没变，差异只是 k-means 随机性）。已加 `--train_use_sk` 开关 |
| 2 | 导出 Sinkhorn **batch=2048 ≫ K=256** | 结构性碰撞，ICR 只有 0.935；改 **batch=64** → 0.9997 |
| 3 | k-means **无 `random_state`** | 同输入两次质心差 96.6% 量级；L0 用码数在 seed 间波动 59~85。已加 `random_state=0, n_init=10` |
| 4 | 融合评估用**训练对自评** | 13.6 倍水分（0.610 vs 0.045）。改用互斥留出对 |
| 5 | 4GB 卡上**两塔并行编码** | 显存 thrashing、进度归零。必须串行 |
| 6 | 用**总 loss** 选 ckpt | 会在 ep100 就停（后段 loss 回升是记账位移）。改用碰撞率 + burn-in 10% |
| 7 | 早期结论说"本项目走 FAISS RQ-KMeans" | **勘误**：把 esci-ai-search 项目的路线串到本项目上了。本项目是 **RQ-VAE 端到端** |
| 8 | 合成数据微测（6000 物品 / 码本 64×64） | 数学上不可能唯一，消解空转 20 轮、保真度 −51%。**教训：码本总容量必须远大于物品数**，该数字不作为正式结论 |

---

# 第四部分 · 面试话术（可直接用）

**Q: 你们的 SID 是怎么构建的？**

> 文本走 Qwen3-Embedding-0.6B、图像走 SigLIP，先做融合消融——对比了纯文本、PCA 线性、
> MLP、门控融合四种，用共现对比损失（同用户行为序列内共现的物品对做 InfoNCE）做监督，
> 这样融合向量优化的是"推荐友好度"而不是单纯的模态对齐，R@10 从 0.083 提到 0.110（+33%，随机基线 0.003）。
> 融合后统一 1024 维进 RQ-VAE：压到 32 维 latent、3 层 ×256 码，训练 5000 轮。
> 关键改造是**独立 k-means 预初始化**（8192 样本），把第 0 层死码从 32.6% 打到 0%。

**Q: SID 质量怎么评估？**

> 三层免训练评估：uniqueness（ICR / 碰撞率 / per-layer 死码）、
> fidelity（逐层累积重建 MSE / R²）、retrieval structure（LCP + prefix cohesion vs 随机基线）。
> 选型主指标是 LCP ratio——前缀结构直接决定 beam search 的行为。
> 这个思路和阿里 FORGE 一致，他们也强调 SID 评估不该依赖昂贵的 GR 训练。

**Q: 碰撞怎么处理？**（★ 最有料的一问）

> 我们一开始照 TIGER 系做 Sinkhorn 末层消解，把 ICR 从 0.958 提到 0.9997。
> 但实测发现两件事：一是消解要改写 6.5% 物品的码，LCP 从 222 掉到 209、R² 也掉——**用结构换唯一性**；
> 二是它根本到不了 1.0，残留的 7 组全是**逐位相同的重复 embedding**（Amazon 同品多 listing），
> 确定性规则数学上分不开，调 eps/iters 完全无效。
> 于是我们反过来问：碰撞组到底是什么？实拍三个最大组——O 圈、自攻螺丝、拉紧带，
> 全是同品牌同系列只差规格，**组内 cos 0.92~0.95，随机对只有 0.18**。
> 那就**不消解**：一个码 = 一个语义簇（桶均 2.1、最大 5），整桶召回交给排序消歧。
> 这正好对上 Snap 的论文——他们 Table 5 显示唯一性 92.95% 掉到 70.58%，Recall@10 只从 6.1 到 6.0，
> 明确写"唯一性不该被当作黄金标准"；他们线上 A/B 更是直接每码映射 100 个物品做 relevance-guided 消歧。
> 快手的 AdaSID 也从另一面证明：固定重叠正则会强行拆散语义上本该共享的物品。

**Q: 训练时遇到过什么反直觉的现象？**

> 总损失在 ep100 触底后**一路回升到 ep5000**，但重建损失全程单调下降。
> 这不是过拟合，是 rate-distortion 的**记账位移**：encoder 把 latent 拉得离码本更远
> （量化损失从 0.05 涨到 5.42），换来更好的重建（9.74 → 3.43，−65%）。
> 所以我们要按**碰撞率**选 checkpoint，而不是按 loss——按 loss 会在 ep100 就停。

---

# 第五部分 · 尚未做的验证与下一步

## 5.1 诚实边界

| # | 缺口 | 影响 |
|---|---|---|
| 1 | **raw vs sk 无 SFT 端到端消融** | 语义桶决策靠离线指标 + 工业界先例，未端到端验证 |
| 2 | 只用单 seed（2024） | 码本利用率在 seed 间曾波动 ±26 个码；定版已固定 seed，稳健性未多 seed 复核 |
| 3 | mlp 模式未跑 5000 轮 | 三模式对比仍是 1500 轮口径（mlp 仅此一份） |
| 4 | 第 0 层塌缩的**机制性**修法未落地 | init8192 在本配置下**消除了**塌缩，但 EMA / diversity loss / dead code reset 一条都没接 |
| 5 | 融合侧 gate vs mlp 参数量不对等 | "门控优于 MLP"的机制性结论不可靠 |
| 6 | 重复 embedding（30 组）未在上游去重 | 同时是残留碰撞与评估口径的噪声源；去重会改 25,847 基准，需整体权衡 |
| 7 | 两个编码器均未做域适应 | SigLIP 是通用英文图文、Qwen3-Embedding 是通用语义，电商商品域未验证 |
| 8 | 0.6B 编码器低于最佳竞品 3 个点 | 受 4GB 卡限制；"编码器是否是瓶颈"这一假设未验证 |

## 5.2 按 ROI 排序的下一步

| 优先级 | 动作 | 依据 |
|---|---|---|
| 【高】 | 把共现对比损失下沉到量化器（`L = L_recon + L_rq + λ·L_con`） | PLUM 做法；共现对抽取代码已有，改造成本低；预期 LCP / cohesion 提升 |
| 【高】 | 碰撞改语义自适应（AdaSID 两阶段：深度感知阈值 0.18/0.24/0.30） | 我们已量化出消解代价；这是最可能做出"自己的数字"的改进点 |
| 【中】 | `eval_sid.py` 补 FORGE 的 Gini coefficient | 能量化"码本使用均衡度"，与语义自适应形成闭环验证 |
| 【中】 | 码本结构实验（PLUM 递减 2048/1024/512 vs FORGE 3×8192） | 两家结论相反（递减 vs 递增），这个自由度**尚无定论** |
| 【低】 | 上游按 title+features+categories 去重 | 正解，但会改 25,847 基准口径，需整体权衡 |
| 【低】 | 训练范式补齐 Pre-train → SFT → RL | UPGRADE_PLAN 已规划，方向对齐工业界 |

---

## 参考来源

| 工作 | arXiv / 来源 | 关键数字 |
|---|---|---|
| PLUM (YouTube/Google) | arXiv:2510.07784 | Panel CTR +4.96%，260B tokens CPT，词表 13.2x |
| FORGE (阿里/淘宝) | arXiv:2509.20904 | 交易量 +0.35%，AL-GR 140 亿交互 |
| AdaSID (快手) | arXiv:2604.23522 | GMV +0.98%，Recall@5 +42.6% (vs RQ-VAE) |
| OneRec 系列 (快手) | 2502.18965 / 2508.20900 / 2510.11639 / 2512.24762 | V2 训练算力 -94% |
| Snap Semantic IDs | arXiv:2604.03949（SIGIR'26 Industry） | 唯一性 ~70% 平台期；线上每码 100 品 + relevance-guided |
| TIGER | NeurIPS'23 | 层级 SID + Trie beam search 的开创工作 |
| MiniOneRec | github.com/weepon/MiniOneRec | RQ-VAE 5000~10000 轮；导出期"只动末层 + 只修碰撞组" |
| ETEGRec (SIGIR'25) | github.com/RUCAIBox/ETEGRec | 10000 轮；`--cycle=2` 交替优化 |
| MQL4GRec (ICLR'25) | github.com/N-A-E-S/MQL4GRec | 双 SID 不融合；训练期末层 Sinkhorn |
| GRID (Snap, CIKM'25) | github.com/snap-research/GRID（本地镜像 `refs/GRID/`） | RQ-KMeans 30 步/层；RQ-VAE 3000 步 Adagrad |
| MMGRec | github.com/hanliu95/MMGRec | concat + GCN 注入 CF |
| HSTU (Meta) | 2402.17152 | 推荐 Scaling Law，A/B +12.4% |
| SigLIP | arXiv:2303.15343 | 逐对 sigmoid 损失，ImageNet 73.4% |
| Qwen3-Embedding | 阿里 Qwen，2025-06 | MTEB (Eng) 70.70 |
