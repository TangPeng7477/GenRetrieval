# SID 构建全流程：实验与结论

> 定位：**SID（语义 ID）构建阶段的单一结论入口**。只放全量数据的正式实验与定版结论，
> 冒烟/合成数据/小规模探针一律不收（已归档到 `scripts/multimodal/_archive/`）。
> 机理推导、踩坑、FAQ 见 `docs/KNOWLEDGE_BASE.md`；流水账见 `docs/EXPERIMENT_LOG.md`。
>
> 最后更新：2026-09-13 · 域：Amazon23 `IandS`（Industrial_and_Scientific）

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

## 1. 流水线、产物与耗时

| 阶段 | 脚本 | 产物 | 耗时（实测） |
|---|---|---|---|
| 数据准备 | `scripts/data/prepare_amazon23.py` | `data/Amazon23/IandS/{item.json, item2id, *.inter, stats.json}` | 分钟级 |
| 图像下载 | `scripts/multimodal/download_images.py` | `images/*.jpg` 25,824 张 / 2.7GB | 26 min（32 线程） |
| 文本编码 | `scripts/multimodal/encode_text.py` | `emb/emb_text_title-features-category.npy` (N,1024) | ~4 min（token 预算装箱后 101 item/s） |
| 图像编码 | `scripts/multimodal/encode_image.py` | `emb/emb_image_siglip.npy` (N,768) + `emb_image_mask.npy` | ~20 min |
| 融合训练 | `scripts/multimodal/fuse_embeddings.py` / `fuse_long.sh` | `emb/long/emb_fused_gate_e60.npy` (N,1024) | 60 轮 ≈ 20 min |
| RQ-VAE 训练 | `rq/train_rqvae.py`（`run_sid_exp.sh` 编排） | `ckpt/selected_model.pth` + `train_metrics.json` | **3,644 s（5000 轮）** |
| SID 导出 | `rq/build_sid_dual.py` | `sid_raw.*` / `sid_sk.*` | 9 s |
| 质量评估 | `rq/eval_sid.py` | `eval_raw.json` / `eval_sk.json` | ~30 s |

⚠️ 两域（I&S / VG）**各自独立跑，不合并、不共享码本**；本文所有数字均为 I&S。

---

## 2. 分阶段实验与结论

### 2.1 数据（详见 `docs/DATASET.md`）

| 项 | I&S |
|---|---|
| items / users | 25,847 / 46,341 |
| train / valid / test | 251,576 / 16,074 / 13,232 |
| history_len 上限 | 20（train p50=3、test p50=5） |
| 图像 URL 覆盖 → 下载有效 | 100% → **99.91%**（25,824 张） |
| 时间泄漏 | 无（max train 2021-08-11 < min test 2022-07-17，全局时间切分） |

切分用官方 `benchmark/5core/timestamp` 分片（23.9MB），省 2.2GB 全量评论下载与自跑 k-core。
valid/test 中"无 train 历史"的用户必须丢弃（1,856 / 4,207），否则指标口径不可解释。

### 2.2 两个编码器（均冻结，只推理）

| 塔 | 模型 | 维度 | 显存 | 关键注意 |
|---|---|---|---|---|
| 文本 | Qwen3-Embedding-0.6B | 1024 | 1.19GB | 是**因果 LM**，必须 last-token pooling；编物品侧不加指令 |
| 图像 | SigLIP-base-patch16-224 | 768 | 0.42GB | 4GB 卡上**两塔必须串行**跑（并行会显存 thrashing，进度归零） |

### 2.3 融合四模式（同一留出集 80,079 对，avg_pos 7.88）

| 模式 | R@10 | R@50 | R@100 | vs text | 参数量 |
|---|---|---|---|---|---|
| text（单模态基线） | 0.0830 | 0.1783 | 0.2280 | — | 0 |
| concat（PCA 线性） | 0.0883 | 0.1790 | 0.2347 | +6% | 0 |
| mlp_e60 | 0.1013 | 0.2693 | 0.3623 | +22% | 1.44M |
| **gate_e60** | **0.1100** | **0.2787** | **0.3820** | **+33%** | 3.28M |

随机基线 R@10 = 0.003。**结论：监督融合才是多模态增益来源，线性拼接几乎白给。**

⚠️ 公平性边界：gate 参数量是 mlp 的 2.27×，且 mlp 有 LN+Dropout 而 gate 分支没有 →
现有证据只支持"gate_e60 是最好的可用配置"，**不支持"门控机制优于 MLP"的机制性结论**。

轮次行为：R@10 在 e10~15 后饱和；R@50/R@100 到 e60 仍单调上升（+17.8% / +16.1%）。
面向 top-10 取 10~15 轮即可，面向长尾/码本语义结构取 40~60 轮 → 正式配方用 **e60**。

### 2.4 RQ-VAE 训练（5000 轮五组，`results/sid_e5000/IandS/`）

固定 seed=2024、k-means `random_state=0`，全部 **raw 口径**读数：

| run | ICR | LCP ratio | R² | prefix-1 内聚 | L0 死码率 | 结论 |
|---|---|---|---|---|---|---|
| text（5000 轮） | 0.8383 | 117.5 | **0.8698** | 1.548 | 65.5% | 纯文本基线 |
| gate__init0 | 0.9505 | 151.4 | 0.6399 | 2.003 | 32.6% | 上游默认（首 batch 初始化） |
| **gate__init8192** | **0.9582** | **222.1** | **0.6530** | **2.086** | **0%** | ✅ **定版** |
| gate__initfull | 0.9567 | 212.7 | 0.6468 | 2.075 | 8.6% | 全量初始化反而略输 |
| gate__trainsk | 0.9537 | 205.5 | 0.6427 | 2.050 | 4.3% | 训练期 Sinkhorn，被 init8192 支配 |

（mlp 模式只在 1500 轮口径下跑过：ICR 0.9901 / LCP 90.5 / R² 0.7140 —— 它是"省事的安全默认"，
但语义结构远差于 gate，未进 5000 轮。）

**四条结论**

1. **轮数收益集中在 gate**：1500→5000 轮，gate R² 0.475→0.640（**+0.165**），text 只 +0.017
   （已接近 3×256 的表示上限）。5000 轮才吃到大半收益，但继续加轮只对 R² 有边际意义。
2. **`--init_samples 8192` 是甜点**：L0 在 ep500 就满用 256/256 并保持到 5000 轮，
   LCP 151→222；全量 25,847 反而略差（死码 22 个、LCP 低 10），且 init 成本 28.5s vs 9s。
   排序稳健：**8192 ≥ full ≫ 首batch**。
3. **训练期 Sinkhorn（MQL4GRec 配方）不采用**：首次让自变量真生效后确认它有效
   （L0 死码 32.6%→4.3%），但耗时 +44%（4996s vs 3473s），且被零成本的 init8192 全面支配。
4. **选 ckpt 必须按碰撞率、不能按 loss**：总 loss 在 ep120 触底后一路回升，而 R² 与唯一性
   持续变好（回升的是 commitment 项）。默认 `--select_ckpt collision --burn_in_frac 0.1`。

### 2.5 碰撞：从"必须消解"到"语义桶"（2026-09-13 定版）

**碰撞账本（`gate__init8192` sid_raw）**：965 组 / 2,046 物品（7.9%），
组大小 {2:872, 3:73, 4:17, 5:3}，桶均 2.1、最大 5。

**实拍三个最大组（全部是同系列规格变体）**

| 组 | 内容 | 组内两两 cos | 随机对基线 |
|---|---|---|---|
| (172,166,152) | Mr O-Ring 硅胶 O 圈 ×5（同 70A 硬度，只差内径/外径） | 0.9456 | 0.1814 |
| (179,149,164) | Fastenere 自攻螺丝 ×5（同盘头/驱动，只差 #4~#10、长度） | 0.9196 | 0.1814 |
| (25,27,170) | CLUTCH 拉紧带 ×5（只差 16"/20"、2~4 件装） | 0.9359 | 0.1814 |

**残留碰撞的溯源（解释了"为什么消解也消不干净"）**

| 层 | 重复情况 |
|---|---|
| 数据源 | Amazon **同品多 listing**：30 组 60 个物品 title+features+categories 逐字相同、主图 URL 相同、ASIN 不同（9/30 组价格也不同） |
| 文本编码层 | 40 组 / 82 个逐位重复（0.32%）—— 输入字符串逐字相同，编码器忠实输出，**不是转码 bug** |
| 融合层 | 图像分支**救回 10 组**（同文不同图）；剩 30 组连图都一样 → 融合无能为力 |

→ Sinkhorn 消解后残留的 14 物品 / 7 组 **100% 是这些孪生组**（非孪生碰撞一个不剩）。
机制：确定性分配规则对逐位相同的输入必给相同输出；能拆开 23/30 组靠的是分桶时
兄弟落进不同 batch 的"上下文抽签"——**是运气，不是收敛**。

**方案对比与裁决**

| | sid_sk（唯一化） | **sid_raw（语义桶，定版）** |
|---|---|---|
| ICR | 0.9997 | 0.9582（桶均 2.1、最大 5） |
| LCP ratio | 209.4 | **222.1** |
| 重建 R² | 0.6504 | **0.6530** |
| 导出侧 | 需 Sinkhorn 迭代（改码 6.54%） | 纯 argmin，一步到位 |
| SFT 训练目标 | 每物品唯一 | 同一 SID 可出现在 1~5 个物品样本（学"SID↔商品族"，无标签冲突） |
| 推断侧 | SID→1 物品 | SID→整桶（≤5）；同码同路径，beam/Trie 不受影响 |
| 先例 | TIGER / MQL4GRec / ETEGRec | YouTube PLUM；Snap 线上 A/B（每码 100 品 + relevance-guided） |

### 2.6 对照：RQ-KMeans（MiniOneRec 原版代码，同一 emb、同一 Sinkhorn）

| 指标 | RQ-VAE(gate+init8192) raw→sk | RQ-KMeans(MiniOneRec) raw→sk |
|---|---|---|
| ICR | 0.9582 → 0.9997 | 0.8395 → 0.9996 |
| LCP ratio | 222.1 → 209.4 | 175.8 → 163.3 |
| 保真度 R² | 0.6530 → 0.6504（decoder 重建） | 0.4460 → 0.3797（直接量化） |
| Sinkhorn 改码比例 | 6.54% | **27.06%** |
| 耗时 | 3,644 s | **6 s** |

**结论：维持 RQ-VAE。** raw ICR 差 12 个点（0.84 vs 0.96）就是 RQ-VAE"先压到 32 维 latent
再量化"的价值；Sinkhorn 后唯一性打平，但 rqkmeans 要改写 27% 物品的码，LCP 与保真度全面落后。
⚠️ 两者 R² 口径不同（decoder 重建 vs 直接量化），只能作参考性对照。

---

## 3. SID 质量三件套（`rq/eval_sid.py`）

| 维度 | 指标 | 定版读法 |
|---|---|---|
| **uniqueness** | ICR、碰撞率、per-layer 死码率、最大冲突组 | 语义桶口径下 ICR 是"桶化程度"，不再是硬门槛 |
| **fidelity** | 逐层累积重建 MSE / R² / cosine | 同时报"实际交付 codes"与"纯 argmin 上限"，差值 = 消解代价 |
| **retrieval structure** | LCP ratio、prefix cohesion vs random | **选型主指标**（前缀结构决定 beam search 行为） |

⚠️ 深前缀层（prefix-2/3）组大小≈1 时 cohesion 会标"样本不足/退化，不可信"，别拿它下结论。

---

## 4. 尚未做的验证（诚实边界）

| # | 缺口 | 影响 |
|---|---|---|
| 1 | **raw vs sk 无 SFT 端到端消融** | 语义桶决策靠离线指标 + 工业界先例，未端到端验证；若后续有余力，这是第一优先 |
| 2 | 只用单 seed（2024） | 码本利用率在 seed 间曾波动 ±26 个码；定版 run 已固定 seed，但稳健性未多 seed 复核 |
| 3 | mlp 模式未跑 5000 轮 | 三模式对比仍是 1500 轮口径（mlp 仅此一份） |
| 4 | 第 0 层塌缩的**机制性**修法未落地 | init8192 是在本配置下**消除了**塌缩（L0 满用），但 EMA / diversity loss / dead code reset 三条通用机制一条都没接 |
| 5 | 融合侧 gate vs mlp 参数量不对等 | 机制性结论不可靠（见 §2.3） |
| 6 | 重复 embedding（30 组）未在上游去重 | 它们同时是 SID 残留碰撞与评估口径的噪声源；去重会改 25,847 基准，需整体权衡 |
