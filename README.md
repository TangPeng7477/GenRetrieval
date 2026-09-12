# GenRetrieval — 多模态生成式召回（MiniOneRec 升级版 2.0）

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu118-red.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)

在 [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) 开源框架上做的系统性升级：
**Amazon Reviews 2023 + 图文多模态 → 门控融合 → RQ-VAE 语义 ID（SID）→ LLM 生成式召回**。

> **当前进度（2026-09-13）**：**数据与 SID 构建阶段已定版**（M1 / M2 完成），
> SFT / RL 阶段（M3~M5）待上云启动。
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
```

---

## 3. 关键结果（I&S，N=25,847）

### 3.0 指标定义与公式（各指标首次出现处给出；完整版见 [docs/SID_PIPELINE.md §0.1](docs/SID_PIPELINE.md)）

| 指标 | 定义与公式 | 本项目定版读数 |
|---|---|---|
| **Recall@K**（共现检索，融合阶段代理指标） | `R@K = 1/Q · Σ_q 1[ Top-K(q) ∩ Gold(q) ≠ ∅ ]`：按融合向量余弦取 Top-K（排除自身），Gold = 留出共现伙伴，Q = 3,000 个 query；随机基线 `≈ avg_pos · K / N` | R@10 = 0.1100（随机基线 0.003） |
| **ICR**（唯一码率） | `ICR = 不同 SID 元组数 / N`，碰撞率 `= 1 − ICR` | 0.9582 |
| **LCP ratio**（前缀语义保持，选型主指标） | `mean_NN lcp(i,j) ÷ mean_随机对 lcp(i,j)`；`lcp` = 两个 SID 的最长公共前缀层数；随机基线恒为 1 | 222.1 |
| **prefix cohesion**（前缀内聚比） | 同前缀组内两两余弦均值 ÷ 随机分组（5 次置换）的同值 | prefix-1 = 2.086 |
| **重建 MSE / R²**（量化保真度） | `MSE = 1/(N·D) · Σ‖x̂ − x‖²`，`R² = 1 − MSE / Var(x)` | R² = 0.6530 |
| **L0 死码率** | `dead_l = 1 − 第 l 层被用到的码数 / K`，K = 256 | 0% |
| **HR@K / NDCG@K**（端到端，SFT/RL 阶段） | `HR@K = 1/U · Σ_u 1[rank_u ≤ K]`；`NDCG@K = 1/U · Σ_u 1[rank_u ≤ K] / log₂(rank_u + 1)`（单正样本 ⇒ IDCG = 1） | 见 §3.4 V0 基线 |

### 3.1 融合：监督融合才是增益来源（同一留出集 80,079 对）

| 模式 | R@10 | R@50 | R@100 | vs 纯文本 |
|---|---|---|---|---|
| text（单模态基线） | 0.0830 | 0.1783 | 0.2280 | — |
| concat（PCA 线性） | 0.0883 | 0.1790 | 0.2347 | +6% |
| mlp | 0.1013 | 0.2693 | 0.3623 | +22% |
| **gate（定版）** | **0.1100** | **0.2787** | **0.3820** | **+33%** |

随机基线 `R_rand@K ≈ avg_pos · K / N = 7.88 × 10 / 25,847 = 0.003`（定义见 §3.0）。

### 3.2 SID：定版配置 `gate + init8192 + 5000 轮`

| run | ICR | LCP ratio | 重建 R² | L0 死码率 |
|---|---|---|---|---|
| text（纯文本 5000 轮） | 0.8383 | 117.5 | 0.8698 | 65.5% |
| gate + 首 batch 初始化 | 0.9505 | 151.4 | 0.6399 | 32.6% |
| **gate + init8192（定版）** | **0.9582** | **222.1** | **0.6530** | **0%** |
| gate + 全量初始化 | 0.9567 | 212.7 | 0.6468 | 8.6% |
| gate + 训练期 Sinkhorn | 0.9537 | 205.5 | 0.6427 | 4.3% |

- **LCP ratio** = 语义近邻对的 SID 前缀长度 ÷ 随机对（random 的 222 倍，公式见 §3.0）→ 前缀层次结构强。
- 对照 **RQ-KMeans（MiniOneRec 原版 FAISS-RQ，同一 embedding、同一 Sinkhorn）**：
  raw ICR 0.8395、LCP 175.8、改码 27.06%（RQ-VAE 只需 6.54%）→ 维持 RQ-VAE。

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
# SFT / RL（上云 3090，待启动）
bash sft_3090.sh
MODEL_PATH=./outputs/sft_IandS_3090/final_checkpoint bash evaluate_3090.sh
```

> ⚠️ 不要直接 `pip install -r requirements.txt`（含 `torchrec`/`fbgemm_gpu`/`deepspeed` 等装不上或冗余项），
> 用裁剪后的 `requirements-core.txt`。
> ⚠️ >100MB 的下载（torch / 数据集 / 模型权重）不要在受限沙箱内执行，实测限速可达 51 倍。

---

## 5. 项目结构

```
├── rq/                              # SID：RQ-VAE 训练 / 导出 / 评估
│   ├── train_rqvae.py               #   训练（--init_samples、按碰撞率选 ckpt）
│   ├── build_sid_dual.py            #   双版导出（sid_raw 默认 / sid_sk 存档）
│   ├── eval_sid.py                  #   三件套评估
│   ├── build_sid_rqkmeans.py        #   RQ-KMeans（MiniOneRec 原版）对照
│   └── models/                      #   rqvae / rq / vq / layers
├── scripts/
│   ├── data/                        # 数据集下载与预处理
│   └── multimodal/                  # 图文编码、融合、SID 实验编排
├── data/Amazon23/{IandS,VG}/        # 处理产物（raw/、images/ 不入 git）
├── results/sid_e5000/IandS/         # 5000 轮五组实验 + rqkmeans 对照
├── sft_3090.sh / rl_3090.sh / evaluate_3090.sh
└── docs/                            # 文档（见下）
```

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
