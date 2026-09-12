# GenRetrieval — 多模态生成式召回（MiniOneRec 升级版 2.0）

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu118-red.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)

在 [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) 开源框架上做的系统性升级：
**Amazon Reviews 2023 + 图文多模态 → 门控融合 → RQ-VAE 语义 ID（SID）→ LLM 生成式召回**。

> **当前进度（2026-09-13）**：**数据与 SID 构建阶段已在两个域上定版**
> —— 域 A `Industrial_and_Scientific`（25,847 商品）与域 B `Video_Games`（25,611 商品）
> 各跑完完整 pipeline（M1 / M2 完成），SFT / RL 阶段（M3~M5）待上云启动。
> 本 README 讲"项目是什么、SID 怎么定版的、怎么复现"；
> 完整实验与结论见 **[docs/SID_PIPELINE.md](docs/SID_PIPELINE.md)**。

---

## 1. 它解决什么问题

生成式召回把"检索"改写成"生成"：LLM 直接生成下一个物品的**语义 ID 序列**。
SID 的质量决定了这件事的上限——同一个 SID 下的物品应当语义相近，前缀应当形成层次聚类。

本项目的三个增量点：

1. **多模态融合 SID**：文本（Qwen3-Embedding-0.6B）+ 图像（SigLIP）经**门控融合**成单向量，
   再送 RQ-VAE 做 3 层残差量化。融合阶段用**用户行为共现**做 InfoNCE 监督，
   使"会被一起买的物品在向量空间里靠近"——这正是 SID 需要的性质。
2. **SID 质量三层评估体系**（ICR / 保真度 / 检索结构，指标定义与公式见 §3.0），所有选型用数字裁决，不靠直觉。
3. **碰撞即语义桶**：不做 Sinkhorn 碰撞消解，同码物品整桶召回，消歧交给下游排序
   （依据：碰撞组实拍 + Snap《Semantic IDs for Recommender Systems at Snapchat》的线上实践）。

---

## 2. 流水线与定版配方

```
Amazon23 官方 5core/timestamp 分片（I&S / VG 两域独立，不合并）
   │
   ├─ 文本 ──→ Qwen3-Embedding-0.6B ────→ (N, 1024)
   └─ 图像 ──→ SigLIP-base-patch16-224 ─→ (N, 768)
                    │
                    ↓  门控融合 gate（共现 InfoNCE，60 轮）
              (N, 1024)  emb_fused_gate_e60
                    │
                    ↓  RQ-VAE：3 层 × 256 码 × 32 维，--init_samples 8192，5000 轮
              SID codes (N, 3)
                    │
                    ↓  导出（同一 ckpt 两版，互不覆盖）
            sid_raw【默认交付：语义桶】 / sid_sk【存档：Sinkhorn 唯一化】
                    │
                    ↓  SFT → RL（MiniOneRec 路线，待上云）
              生成式召回
```

**定版配方**：`gate` 融合 + `--init_samples 8192` + 5000 轮 + **交付 `sid_raw`**。

```bash
# 一条命令复现 SID（I&S，本地 4GB 卡约 1.5h，含编码与融合约 1h）
bash scripts/multimodal/encode_all.sh IandS
bash scripts/multimodal/fuse_long.sh IandS 60 5
RESULTS_ROOT=results/sid_e5000 INIT_SAMPLES=8192 \
  bash scripts/multimodal/run_sid_exp.sh IandS 5000 "gate"
# 域 B 复验（VG：融合 60 轮 + SID 2 组，约 2h）
bash scripts/multimodal/run_vg_sid.sh
```

---

## 3. 关键结果（双域：I&S N=25,847 ｜ VG N=25,611）

### 3.0 指标定义与公式（各指标首次出现处给出；完整版见 [docs/SID_PIPELINE.md §0.1](docs/SID_PIPELINE.md)）

| 指标 | 定义与公式 | I&S 定版读数 | VG 复验读数 |
|---|---|---|---|
| **Recall@K**（共现检索，融合阶段代理指标） | `R@K = 1/Q · Σ_q 1[ Top-K(q) ∩ Gold(q) ≠ ∅ ]`：按融合向量余弦取 Top-K（排除自身），Gold = 留出共现伙伴，Q = 3,000 个 query；随机基线 `≈ avg_pos · K / N` | 0.1100（基线 0.003） | 0.2113（基线 0.0066） |
| **ICR**（唯一码率）**← 跨域选型主判据** | `ICR = 不同 SID 元组数 / N`，碰撞率 `= 1 − ICR` | 0.9582 | 0.9744 |
| **LCP ratio**（前缀语义保持） | `mean_NN lcp(i,j) ÷ mean_随机对 lcp(i,j)`；`lcp` = 两个 SID 的最长公共前缀层数；随机基线恒为 1 | 222.1 | 163.6 |
| **prefix cohesion**（前缀内聚比） | 同前缀组内两两余弦均值 ÷ 随机分组（5 次置换）的同值 | prefix-1 = 2.086 | prefix-1 = 1.585 |
| **重建 MSE / R²**（量化保真度）**← 跨域选型主判据** | `MSE = 1/(N·D) · Σ‖x̂ − x‖²`，`R² = 1 − MSE / Var(x)` | R² = 0.6530 | R² = 0.8691 |
| **L0 死码率** | `dead_l = 1 − 第 l 层被用到的码数 / K`，K = 256 | 0% | 3.52%（L1/L2 为 0） |
| **HR@K / NDCG@K**（端到端，SFT/RL 阶段） | `HR@K = 1/U · Σ_u 1[rank_u ≤ K]`；`NDCG@K = 1/U · Σ_u 1[rank_u ≤ K] / log₂(rank_u + 1)`（单正样本 ⇒ IDCG = 1） | 见 §3.4 V0 基线 | 待 SFT |

> ⚠️ **主判据不是 LCP**：第二域复验时发现 LCP 这一族指标**跨域会排序反转**
> （VG 上 RQ-KMeans 的 LCP ratio 206.3 反超 RQ-VAE 的 163.6，而它的 raw ICR 落后 8.6 个点）。
> 原因是随机基线只有 ~0.005，ratio 被极小分母放大；且 VG 上前缀结构已饱和、失去区分度。
> 所以跨域选型改用 **raw ICR + 重建 R²**（这两项两域排序完全一致）。推导见
> [SID_PIPELINE §2.11](docs/SID_PIPELINE.md)。

### 3.1 融合：监督融合才是增益来源（两域各用互斥留出对评估）

| 模式 | I&S R@10 | I&S R@100 | vs 纯文本 | VG R@10 | VG R@100 | vs 纯文本 |
|---|---:|---:|---:|---:|---:|---:|
| text（单模态基线） | 0.0830 | 0.2280 | — | 0.1497 | 0.3957 | — |
| concat（PCA 线性） | 0.0883 | 0.2347 | +6.4% | 0.1587 | 0.4493 | +6.0% |
| mlp | 0.1013 | 0.3623 | +22.1% | 0.1917 | 0.6063 | +28.1% |
| **gate（定版）** | **0.1100** | **0.3820** | **+32.5%** | **0.2113** | **0.6337** | **+41.2%** |
| 随机基线 | 0.00305 | 0.03048 | — | 0.00659 | 0.06589 | — |
| 留出对 / query 数 | 80,079 / 3,000 | | | 186,275 / 3,000 | | |
| `avg_pos` | 7.88 | | | 16.88 | | |

- 随机基线按 `R_rand@K ≈ avg_pos · K / N` 算（定义见 §3.0）：I&S `7.88×10/25,847 = 0.00305`，
  VG `16.88×10/25,611 = 0.00659`。**两域基线差 2.16× 全部来自 `avg_pos`，所以跨域只能比"相对 text 的增益"。**
- **VG 上的增益更大**（gate/text +41.2% vs +32.5%，gate/mlp +10.3% vs +8.6%），
  且模态互检索也更强（text→image R@10 **0.6363** vs 0.6217，image→text **0.5837** vs 0.5377）。
  这与开跑前的预期一致——游戏的封面是决策主导信息，工业品的图只是补充。

**为什么跑四个模式而不是只比 text / gate**：四档构成"融合复杂度阶梯"——
`text`（无融合）→ `concat`（无参拼接）→ `mlp`（参数化变换）→ `gate`（输入自适应门控）。
`mlp` 与 `gate` 参数量相同、训练轮数相同、损失相同，唯一区别是"融合权重是否由输入决定"，
因此 `gate 0.1100 vs mlp 0.1013（+8.6%）` 才能把增益归因到**门控机制本身**，
而不是"只要上 MLP 就能学"。少了这一档，+33% 里有多少来自"有参数"就说不清。
（这也是本阶段敢多试的原因：一次融合几十分钟，比下游一次 RQ-VAE 便宜一个量级。）

### 3.2 SID 消融：定版 `gate + init8192 + 5000 轮`（6 组，≈5.4 GPU 小时）

这 6 组**不是并列罗列，而是两组独立消融**——变量分别是"输入表示"与"量化器配置"：

| run | 唯一变量 | ICR | LCP ratio | 重建 R² | L0 死码率 | 耗时 |
|---|---|---|---|---|---|---|
| text | **输入**：纯文本 embedding（无融合） | 0.8383 | 117.5 | 0.8698 | 65.5% | 58 min |
| gate + 首 batch 初始化 | **init 采样数** = 首个 batch | 0.9505 | 151.4 | 0.6399 | 32.6% | 58 min |
| **gate + init8192（定版）** | **init 采样数** = 8192 | **0.9582** | **222.1** | **0.6530** | **0%** | 61 min |
| gate + 全量初始化 | **init 采样数** = 全量 N | 0.9567 | 212.7 | 0.6468 | 8.6% | 62 min |
| gate + 训练期 Sinkhorn | **训练期是否开 Sinkhorn** | 0.9537 | 205.5 | 0.6427 | 4.3% | 83 min |
| rqkmeans（FAISS-RQ） | **量化器路线**（见下） | 0.8395 | 175.8 | — | 0% | **6 s** |

为什么只消融这些：`init` 与"训练期 Sinkhorn"是唯二能改变结论的变量
（死码 32.6%→0%、LCP 151→222）；而 `256×3` 码本、32 维 latent、`beta=0.25` 沿用上游已验证配方，
改它们属于调参而非选型，在 4GB 卡上单次 1 小时，不划算。
`rqkmeans` 只要 6 秒却能裁掉"要不要用 RQ-VAE"这个路线问题，是全场性价比最高的对照。

- **LCP ratio** = 语义近邻对的 SID 前缀长度 ÷ 随机对（random 的 222 倍，公式见 §3.0）→ 前缀层次结构强。
- `rqkmeans` 两点补充：① 它的 Sinkhorn **改码比例 27.06%**（RQ-VAE 只需 6.54%）；
  ② 它的 MSE 是 emb 空间直接量化（无 decoder），与 RQ-VAE 的 decoder 重建**不可横比**，故表中 R² 留空。
  综合裁决：维持 RQ-VAE。
- `mlp` 融合向量的 SID 在早期 1500 轮阶段跑过（`results/sid/IandS/mlp/`），
  定版阶段没再跑——gate 已在融合阶段胜出，再跑一遍属于重复验证，不划算。

训练曲线（三条线分离即"rate-distortion 记账位移"，**总损失后期回升但重建仍在降**，
所以按碰撞率而非总损失选 ckpt）：

![RQ-VAE 训练曲线三相结构](docs/figures/train_curve_3phase.svg)

### 3.3 为什么不消解碰撞（语义桶）

- 碰撞账本：965 组 / 2,046 物品（7.9%），**桶均 2.1、最大 5**。
- 实拍最大三组全是同系列规格变体（O 圈 / 自攻螺丝 / 拉紧带），**组内 cos 0.92~0.95**（随机对 0.18）。
- Sinkhorn 只能到 ICR 0.9997，残留 7 组**全部是逐位相同的重复 embedding**（Amazon 同品多 listing），
  确定性规则数学上不可分；代价是改写 6.54% 物品的码、LCP 222.1→209.4。
- 工业界证据：Snap（SIGIR'26, arXiv 2604.03949，官方代码 = `refs/GRID`）Table 5 显示
  唯一性 92.95%→70.58% 时 Recall@10 仅 6.1→6.0（~70% 以上即平台期），
  其线上 A/B 用的就是"每码映射 100 个物品 + relevance-guided 消歧"。

⚠️ **诚实边界**：未做 raw vs sk 的 SFT 端到端消融（成本有限）。这是"离线指标 + 工业界先例"
支撑的设计选择，不是端到端验证过的结论。`sid_sk.npy` 随每次导出产出，可随时切回唯一化口径。

### 3.4 V0 基线（复刻版 MiniOneRec，RTX 3090 实测，beam=10）

| 配置 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|------|------|------|------|-------|--------|---------|
| 0.5B SFT | 0.036 | 0.057 | 0.068 | 0.093 | 0.052 | 0.060 |
| 0.5B RL-epoch2 | 0.047 | 0.070 | 0.083 | 0.109 | 0.066 | 0.074 |
| 官方 1.5B 权重 | 0.085 | 0.113 | 0.133 | **0.154** | 0.109 | **0.116** |

这是后续 SFT / RL 阶段要超越的锚点。

### 3.5 VG 第二域复验：配方跨域可迁移（2026-09-13）

**为什么只跑 2 组**：I&S 的 6 组是**选型**（配方在那里定）；VG 只回答"换一个分布还成立吗"，
再跑 6 组消融 = 重复已知结论。所以 VG 只跑 `RQ-VAE`（定版配方）+ `RQ-KMeans`（路线对照）。

```bash
bash scripts/multimodal/fuse_long.sh VG 60 5   # 融合 60 轮（旧产物是 3 轮口径，不可比，故重跑）
bash scripts/multimodal/run_vg_sid.sh          # SID 2 组 + 三件套 + 对比表
```

| 指标 | **VG RQ-VAE** | VG RQ-KMeans | I&S RQ-VAE（参照） | I&S RQ-KMeans（参照） |
|---|---:|---:|---:|---:|
| **raw ICR** | **0.9744** | 0.8882 | 0.9582 | 0.8395 |
| 最大冲突组 | 7 | 12 | 5 | 14 |
| L0 死码率 | **3.52%** ⚠️ | 0% | 0% | 0% |
| LCP ratio | 163.6 | 206.3 | 222.1 | 175.8 |
| 重建 R²（口径不同，仅参考） | **0.8691** | 0.7770 | 0.6530 | 0.4460 |
| Sinkhorn 后 ICR | 0.9954 | 0.9959 | 0.9997 | 0.9996 |
| Sinkhorn 改码比例 | **3.93%** | 18.10% | 6.54% | 27.06% |
| 耗时 | 4,015 s | **5.5 s** | 3,644 s | **6 s** |

**验收判据（开跑前写死，含未达标项）**：ICR ≥ 0.94 → **0.9744 ✅**；
L0 死码 = 0% → **3.52% ⚠️ 未达标**（9/256 个码弃用，L1/L2 为 0；L0 熵 0.960 说明不是塌缩，
但按预设标准仍记未达标）；LCP ratio ≥ 150 → **163.6 ✅**；R² ≥ 0.6 → **0.8691 ✅**。
逃逸条件（死码 > 5% 或 LCP < 100 才补跑 VG init 消融）**未触发 → 不补跑**。

**三个新发现**：

1. **LCP 族指标跨域会排序反转**——VG 上 RQ-KMeans 的 LCP ratio 反超 RQ-VAE，
   而它的 raw ICR 落后 8.6 个点。→ 跨域主判据改为 **raw ICR + 重建 R²**（详见 §3.0 警示）。
2. **"融合后 ICR 上限"不是流水线天花板**——它只约束纯 argmin；Sinkhorn 靠 batch 分桶能突破
   （VG 0.9954 > 理论值 0.9936）。
3. **非孪生碰撞可以被 100% 清干净**——VG 的 Sinkhorn 残余碰撞 119 份**逐份都在孪生组内**
   （I&S 残余 7 份同理），"残留碰撞 ≡ 重复 embedding"在双域成立。

### 3.6 SID 产物落盘位置

两域目录同构，`results/sid_e5000/<IandS|VG>/<配置>/`：

| 内容 | 文件 |
|---|---|
| **交付 SID**（语义桶） | `gate__init8192/sid_raw.npy` — `(N,3)` int，纯 argmin |
| 唯一化存档 | `gate__init8192/sid_sk.npy` — 末层 Sinkhorn 版，可随时切回 |
| 模型权重 / 训练日志 | `gate__init8192/ckpt/selected_model.pth` ｜ `train.log`、`train_metrics.json` |
| 三件套评估 / 汇总 | `gate__init8192/eval_raw.json`、`eval_sk.json` ｜ `summary.json` |
| RQ-KMeans 对照 | `rqkmeans/`（同结构） |
| 路线对比表 | `compare_rqkmeans.md` |
| 上游融合向量 | `data/Amazon23/<IandS|VG>/emb/long/emb_fused_gate_e60.npy` |

---

## 4. 快速开始

> 完整版（环境分步、数据准备、故障排除）见 **[docs/QUICKSTART.md](docs/QUICKSTART.md)**。

| 项目 | 云端训练（正式） | 本地迭代（SID 阶段） |
|------|------------------|----------------------|
| GPU | RTX 3090 24GB | RTX 3050 Ti 4GB |
| Python / CUDA | 3.11 / 11.8 | 3.11 / 11.8 |

```bash
# 环境（云平台 / Linux）
bash scripts/setup_env.sh                      # Windows 用 scripts\setup_env.ps1
# 数据
bash scripts/data/download_amazon23.sh
python scripts/data/prepare_amazon23.py --category Industrial_and_Scientific --short IandS
python scripts/multimodal/download_images.py --short IandS
# 编码 → 融合 → SID（详见 QUICKSTART §2.5）
bash scripts/multimodal/encode_all.sh IandS
bash scripts/multimodal/fuse_long.sh IandS 60 5
RESULTS_ROOT=results/sid_e5000 INIT_SAMPLES=8192 bash scripts/multimodal/run_sid_exp.sh IandS 5000 "gate"
# 域 B：Video_Games（同一套命令，只换 --category/--short；SID 只跑定版 2 组，不重做消融）
python scripts/data/prepare_amazon23.py --category Video_Games --short VG
python scripts/multimodal/download_images.py --short VG
bash scripts/multimodal/encode_all.sh VG
bash scripts/multimodal/run_vg_sid.sh
# SFT / RL（上云 3090，待启动）
bash sft_3090.sh
MODEL_PATH=./outputs/sft_IandS_3090/final_checkpoint bash evaluate_3090.sh
```

> ⚠️ 不要直接 `pip install -r requirements.txt`（含 `torchrec`/`fbgemm_gpu`/`deepspeed` 等装不上或冗余项），
> 用裁剪后的 `requirements-core.txt`。
> ⚠️ >100MB 的下载（torch / 数据集 / 模型权重）不要在受限沙箱内执行，实测限速可达 51 倍。

---

## 5. 代码地图

### 5.1 核心链路（本项目实际运行，★ = 主干）

```
scripts/data/download_amazon23.sh        官方 5core/timestamp 分片 + meta（hf-mirror）
        ↓
scripts/data/prepare_amazon23.py         → item2id / *.inter / stats.json / domain 文件
        ↓
scripts/multimodal/download_images.py    并发抓图（I&S 覆盖 100%，VG 99.84%）
scripts/multimodal/encode_text.py        Qwen3-Embedding-0.6B      → (N, 1024)
scripts/multimodal/encode_image.py       SigLIP-base-patch16-224   → (N,  768)
        ↓   （scripts/multimodal/encode_all.sh 串行调度上面三步）
scripts/multimodal/fuse_embeddings.py    ★ 四模式融合 + 共现 InfoNCE + 留出对 R@K
        ↓   （fuse_long.sh = 全量对 60 轮，定版口径）
rq/train_rqvae.py                        ★ RQ-VAE 训练（kmeans init，按碰撞率选 ckpt）
        ↓
rq/build_sid_dual.py                     ★ 导出 sid_raw（默认交付）/ sid_sk（存档）
rq/build_sid_rqkmeans.py                 ★ RQ-KMeans 对照（MiniOneRec 原版 FAISS-RQ）
        ↓
rq/eval_sid.py                           ★ 三件套评估
rq/diag_sinkhorn_residual.py               残留碰撞诊断（eps × iters 双扫描）
        ↓
scripts/multimodal/{compare_sid_modes, compare_rqkmeans, make_sid_summary}.py
```

- **编排**：`scripts/multimodal/run_sid_exp.sh`（I&S 实验矩阵）、`run_vg_sid.sh`（VG 全流程）
- **诊断**：`scripts/multimodal/diag_collision.py`（碰撞组溯源 / 孪生 embedding 定位）、
  `probe_gate_twins.py`（门控是否用上图像：A/B/C 三类细分）、
  `probe_twin_sinkhorn.py`（孪生组在 raw/sk 下的存活账本）、
  `probe_dataset_stats.py`（双域字段覆盖率 / 评分分布 / 长尾 / 冷启动）
- **环境**：`scripts/setup_env.sh` | `setup_env.ps1`、`download_models.sh`、`tools/hf_repair_cache.py`
- **被直接 import 的上游文件**（复制自 MiniOneRec 但在用）：`rq/datasets.py`（EmbDataset）、
  `rq/rqkmeans_faiss.py`（FAISS-RQ 量化器）、`rq/models/{rqvae,rq,vq,layers}.py`

### 5.2 未启用（复刻自 MiniOneRec，本项目未运行）

| 类别 | 文件 |
|---|---|
| V0 训练链路 | `data.py` `sasrec.py` `sft.py` `rl.py` `evaluate.py` `minionerec_trainer.py` `LogitProcessor.py` `SASRecModules_ori.py` `utility.py` `convert_dataset.py` `split.py` `merge.py` `calc.py` `data_test.py` `sinkhorn_demo.py` + 各 `.sh` / `_3090.sh` |
| 上游分支 | `convert_dataset_gpr.py` `sft_gpr.py` `rl_gpr.py` `ts_rec_data.py` `ts_rec_sft.py` `ts_rec_data/` `config/zero2_opt.yaml` |
| 旧数据管线 | `data/amazon18_data_process.py` `data/amazon23_data_process.py` `data/process.py` |
| 另一条索引路线 | `rq/rqvae.py`(原版) `rq/trainer.py` `rq/utils.py` `rq/rqkmeans_constrained.py` `rq/rqkmeans_plus.py` `rq/generate_indices*.py` `rq/text2emb/` |

**判据**：`models/` 无权重、`logs/` 无训练日志、`results/` 无对应产物。
§3.4 的 V0 数字来自原 MiniOneRec 项目（RTX 3090 实测），在本仓仅作待超越的锚点。

## 6. 文档地图

| 文档 | 内容 |
|---|---|
| **[docs/SID_PIPELINE.md](docs/SID_PIPELINE.md)** | **SID 唯一入口**：参考方法综述 + 本项目实现与知识点 + 探索过程全记录 + 定版配方（新读者从这进） |
| [docs/KNOWLEDGE_BASE.md](docs/KNOWLEDGE_BASE.md) | 知识点与机理总览（RQ-VAE / Sinkhorn / 死码 / **指标口径总表（含公式）** + FAQ） |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 从零跑通（环境 → 数据 → SID → 训练） |
| [docs/UPGRADE_PLAN.md](docs/UPGRADE_PLAN.md) | 升级方案与里程碑（M1~M6） |
| [docs/DATASET.md](docs/DATASET.md) | Amazon23 数据集档案与切分口径 |
| — | `docs/EXPERIMENT_LOG.md`（实验流水账）为本地文档，按要求未上传 |
| [docs/V0_MINIONEREC_TECH_DOC.md](docs/V0_MINIONEREC_TECH_DOC.md) | V0 复刻版技术文档（归档） |

---

## 7. 引用

```bibtex
@misc{MiniOneRec,
    title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation},
    author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
    year={2025}, eprint={2510.24431}, archivePrefix={arXiv}, primaryClass={cs.IR},
}

@article{SnapSID2026,
    title={Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices},
    author={Ju, Clark Mingxuan and Zhao, Tong and Neves, Leonardo and others},
    journal={arXiv preprint arXiv:2604.03949}, year={2026},
    note={SIGIR 2026 Industry Track; 官方代码 github.com/snap-research/GRID（本仓 refs/GRID 镜像）},
}
```
