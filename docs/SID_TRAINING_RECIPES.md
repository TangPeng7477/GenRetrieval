# SID 训练配方对照：五家开源实现的真实超参

> 调研时间：2026-09-12
> 触发问题：本项目融合投影只训 3 轮、RQ-VAE 只跑了合成冒烟 80 轮，怀疑"轮次太少"
> 调研方式：**全部读开源代码本体**（不是论文/博客二手转述），逐条标出文件路径
>
> 🔴 **勘误（2026-09-12，用户指出后回代码核对）**：本文件初版把本项目误标为「FAISS ResidualQuantizer（k-means）」路线。
> **错误**。本项目的 SID tokenizer 是 **RQ-VAE 端到端**（`rq/models/rqvae.py` 有 encoder + `ResidualVectorQuantizer` + decoder；
> `rq/build_sid.py:44` `from models.rqvae import RQVAE`；默认入口 `rq/rqvae.sh --epochs 10000`）。
> FAISS `ResidualQuantizer` 只在 `rq/rqkmeans_faiss.py`（**备选脚本**）和 `rq/eval_sid.py`（仅 `IndexFlatIP` 检索）出现。
> 误判来源：把 **esci-ai-search 项目**（`D:\Codings\ai-search`）的 FAISS RQ 路线串到了本项目上。

---

## 0. 结论先行（TL;DR）

**"SID 要训上千轮"这句话只对 RQ-VAE 一条路线成立，而且跨度极大（30 步 ~ 10000 轮，差 300 倍）。**

| 路线 | 代表实现 | 训练量 | 有 encoder/decoder 参与梯度？ |
|---|---|---|---|
| **RQ-KMeans / RQ 残差聚类** | GRID（Snap, CIKM'25） | **30 步/层**（3 层 = 90 步） | ❌ 只迭代质心 |
| **RQ-VAE（端到端）** | GRID / MQL4GRec / MiniOneRec / ETEGRec / **本项目** | 3,000 步 ~ 10,000 轮 | ✅ 真在训网络 |
| **融合投影（本项目自加）** | 学术界无对应物 | 本项目：60 轮（≈169k 步） | ✅ 但 R@10 十轮即饱和、R@50 到 60 轮仍未饱和 |

三条硬结论：

1. **本项目走的正是第二行（RQ-VAE 端到端）**，标准量级 **5000~10000 轮**。
   → **用户"轮次太少"的判断成立。** 我们仓里唯一的 ckpt 是合成数据冒烟（`.tmp/sidtest/emb.npy`、码本 `[64,64]`、80 轮），
   **真实 I&S / VG 的 RQ-VAE 还没开始训**。这是 M2 当前真正缺的一块。
2. **融合投影阶段学术界没有先例**（各家要么不融合、要么把融合塞进 RQ 一起训）。我们的 60 轮是自创参数，而实测 holdout 在 **10 轮就饱和**（详见 §5）——这一段确实可以砍。
3. **"上 k 轮"的正确落点是 RQ-VAE 本体**（阶段②），不是融合投影、也不是 SFT。SFT/RL 按 step 算，3 轮 / 2 轮是正常量级。

---

## 1. 三个训练阶段必须分开谈

很多讨论把这三件事混成一个"轮次"，量级完全不同：

```
① 融合投影（本项目自加）        → 维度对齐，10^1 轮量级
② SID tokenizer（码本）        → 量化，按路线差 300 倍
③ 生成式模型（SFT / RL）        → LLM 微调，按 step 算，几个 epoch
```

---

## 2. 各家配方（代码出处）

### 2.1 MiniOneRec（中科大 LDS + AlphaLab，2025-11 开源）

仓库：https://github.com/weepon/MiniOneRec

| 阶段 | 参数 | 出处 |
|---|---|---|
| RQ-VAE | `--lr 1e-3 --epochs 10000 --batch_size 20480` | README §3.1.1 命令行 |
| RQ-VAE（仓库默认） | `epochs=5000`, `warmup_epochs=50` | `rq/rqvae.py` argparse 默认值 |
| SFT | 8×H100，每卡 128，**最多 10 epochs**，early stopping `patience=1`，lr **3e-4** cosine | 论文 §4 TRAINING |
| RL (GRPO) | **2 epochs**，beam width 16，KL β 不变 | 论文 §4 |

> 注意仓库默认（5000）与 README（10000）**不一致**；`rq/trainer.py` 里 `data_num = len(data_loader)`，
> 所以 `max_steps = epochs × batch数`。25,847 items / bs 20480 → 2 batch → 5000 轮 ≈ 10,000 步。
> **README 的 10000 轮也才 ~2 万步**，不是一个夸张的数字。

### 2.2 ETEGRec（人大 + 快手，SIGIR'25）

仓库：https://github.com/RUCAIBox/ETEGRec ｜ 数据集：**Amazon 2023**（scientific / instrument / game）

`RQVAE/run_pretrain.sh`：
```bash
--lr 1e-3 --epochs 10000 --batch_size 1024 --weight_decay 1e-4
--e_dim 128 --quant_loss_weight 1.0 --beta 0.25
--num_emb_list 256 256 256 --sk_epsilons 0.0 0.0 0.0
--layers 512 256 --kmeans_init True --lr_scheduler_type linear
```

`run.sh`（主训练，**端到端交替优化**）：
```bash
--lr_rec=0.005 --lr_id=0.0001 --cycle=2
--rec_kl_loss=0.0001 --rec_dec_cl_loss=0.0003
--id_kl_loss=0.0001  --id_dec_cl_loss=0.0003
```

**`--cycle=2` 就是"tokenizer ↔ 推荐器交替 2 轮"**——这是全库里最接近我们"融合投影再训 RQ"这种两段式耦合的参考。
注意 `lr_id`（tokenizer 侧）= **1e-4**，比 `lr_rec`（推荐器侧）小 50 倍：**tokenizer 必须小步走，否则码本塌。**

### 2.3 MQL4GRec（ICLR'25，唯一"多模态双 SID"开源）

仓库：https://github.com/N-A-E-S/MQL4GRec ｜ 数据集：Amazon 2018（Instruments / Arts / Games…）

`index/scripts/run.sh`（**文本与图像各训一个独立 RQVAE**）：
```bash
--num_emb_list 256 256 256 256      # 4 层！不是 3 层
--sk_epsilons 0.0 0.0 0.0 0.003     # 训练期最后一层走 Sinkhorn 分配（前 3 层为 0 = argmin）
--batch_size 2048 --epochs 500 --eval_step 2
```

`scripts/pretrain.sh`：`per_device_batch_size=1024`, `lr=1e-3`, `epochs=30`, `weight_decay=0.01`, 4 卡, `max_his_len=20`
`scripts/finetune.sh`：`lr=5e-4`, `epochs=200`, `patient=10`, `max_his_len=20`, `num_beams=20`, 5 个任务
（`seqrec,seqimage,item2image,image2item,fusionseqrec`）

> **这是和我们最像的多模态开源实现，但它的多模态处理方式与我们的设计相反**：
> 它**不融合**——文本 SID 和图像 SID 各自独立量化，再让 LLM 学跨模态映射任务（item2image / image2item）。
> 我们的 concat/mlp/gate 是**把两模态压成单向量再量化**（YouTube PLUM 路线）。
> 两条路线的 A/B 值得后面补——这是本项目一个明确的、可发论文的空白点。

### 2.3.1 Sinkhorn 该放在哪：MiniOneRec 与 MQL4GRec 是两派（关键）

`sk_epsilons` 这个参数两家都有，但**插入位置完全不同**，别混为一谈。

| | 训练期 | 导出期 | 代码出处 |
|---|---|---|---|
| **MiniOneRec** | `sk_epsilons=[0,0,0]` **完全不开** | 前 L-1 层强制 `sk_epsilon=0.0`、**仅最后一层 0.003**，且 **while 循环 ≤20 轮、只对碰撞组重编码** | `rq/rqvae.py:35` 默认值；`rq/generate_indices.py:92`（首轮 `use_sk=False`）、`:106-110`（分层置 epsilon）、`:114-133`（迭代修复） |
| **MQL4GRec** | `sk_epsilons=[0,0,0,0.003]` **训练期就带** | 直接导出，**无后处理** | `index/scripts/run.sh`；`index/trainer.py` 的 `self.model(data)` |
| **本项目** | **纯 argmin**（= MiniOneRec 口径） | raw = 纯 argmin；sk = 迭代修复碰撞组（对齐 MiniOneRec，末层 0.003，`--max_rounds`） | `rq/train_rqvae.py`（`use_sk=args.train_use_sk`，默认 False）；`rq/build_sid_dual.py:169/:200-207` |

**判断训练期到底有没有跑 Sinkhorn，只看一个条件**（`vq.py:74`）：

```python
if not use_sk or self.sk_epsilon <= 0:
    indices = torch.argmin(d, dim=-1)     # 最近邻
else:
    Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
    indices = torch.argmax(Q, dim=-1)     # 最优传输分配
```

即 **`use_sk=True` 且 `sk_epsilon>0` 两者同时成立**才走 Sinkhorn。
而 `RQVAE.forward(use_sk=True)` / `get_indices(use_sk=False)` —— **两个入口的默认值是反的**，
所以「训练期开不开」完全取决于 trainer 调的是哪个：

| 实现 | 训练期调用 | 结果 |
|---|---|---|
| MiniOneRec `rq/trainer.py` | `self.model(data)` → `forward(use_sk=True)` | `sk=[0,0,0]` → 全 argmin |
| MQL4GRec `index/trainer.py` | `self.model(data)` → `forward(use_sk=True)` | `sk=[0,0,0,0.003]` → **末层真跑 Sinkhorn** |
| 本项目 `rq/train_rqvae.py` | `model(d, use_sk=args.train_use_sk)` | 默认 False → 全 argmin（可控） |

MiniOneRec 导出期的逻辑值得抄，它是**定向修复**而不是全量重编码：

```python
# rq/generate_indices.py:106-133（简化）
for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon = 0.0                      # 前两层一律关掉
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003 # 只有最后一层开

while True:
    if tt >= 20 or check_collision(all_indices_str):
        break
    for collision_items in get_collision_item(all_indices_str):
        d = data[collision_items].to(device)
        indices = model.get_indices(d, use_sk=True)   # 只重编"撞车"的那几个 item
        for item, index in zip(collision_items, indices):
            all_indices[item] = ...
    tt += 1
```

三个容易踩的点：
1. **只动最后一层**——前两层保持 argmin，保证前缀（prefix）语义结构不被破坏，这对后续 Trie 束搜索 / LCP 指标是命根子。
2. **只修碰撞组**——全量跑 Sinkhorn 会把本来不撞的 item 也挪位，白白牺牲重建保真度。
3. **迭代到收敛（上限 20 轮）**——单轮不一定清零碰撞，MiniOneRec 是循环重编直到 `check_collision` 通过。

> 🔴 **历史坑（2026-09-12 修正，务必记住）**：
> `train_rqvae.py` 此前把 `use_sk=False` **硬编码**进训练循环，于是 `run_sid_exp.sh` 里传的
> `TRAIN_SK="0.0 0.0 0.003"` **完全没生效** —— 实际跑的一直是 MiniOneRec 口径（全 argmin）。
> 由此导致一份**无效对照**：早前"训练期 Sinkhorn 把第 0 层码数从 59/256 拉到 85/256、ICR 0.760→0.795"
> 的结论**不成立**（自变量根本没变），那个差异只能来自 k-means 初始化的随机性。
> 现已新增 `--train_use_sk` 开关（默认 False = MiniOneRec 口径），
> **要做"训练期 Sinkhorn"这个消融，必须显式加 `--train_use_sk`，光传 `--sk_epsilons` 没用**。

### 2.4 GRID（Snap Research，CIKM'25 "Practitioner's Handbook"）★最有价值

仓库：https://github.com/snap-research/GRID ｜ 本地镜像：`refs/GRID/`（已 gitignore）
数据集：Amazon（P5 预处理：beauty / sports / toys）

这是**专门做 SID 组件系统性 ablation 的统一框架**，三种 tokenizer × 一种生成器可自由组合。

**① RQ-VAE 路线** — `configs/experiment/rqvae_train_flat.yaml`
```yaml
encoder: MLP  input_dim → [768, 256, 128] → 64     # latent 只有 64 维
decoder: MLP  64 → [128, 256, 768] → input_dim
quantization_layer: n_clusters=256, n_features=64
loss: BetaQuantizationLoss(beta=0.25) + reconstruction_loss_weight 1.0
optimizer: Adagrad lr=0.001, weight_decay=0.0      # ← Adagrad，不是 Adam！
scheduler: WarmupLinear(warmup_steps=1000, min_ratio=0.01)
trainer.max_steps: 3000                             # ← 只有 3000 步
batch_size_per_device: 2048
normalization_layer: BatchNorm1d → L2 normalize     # 归一化后再量化
init_buffer_size: 3072  (MiniBatchKMeans, KMeans++ init, max_iter=1000)
```

**② RQ-KMeans 路线** — `configs/experiment/rkmeans_train_flat.yaml`
```yaml
# 没有 encoder / decoder，直接在 embedding 空间量化
train_layer_wise: true        # 逐层训
normalize_residuals: true     # 残差每层归一化
optimizer: SGD lr=0.5         # ← 高得离谱，因为其实只更新质心
trainer.max_steps: 30         # ← 30 步！3 层 = 每层 10 步
```

**③ TIGER 生成器** — `configs/experiment/tiger_train_flat.yaml`
```yaml
T5 enc-dec: vocab 256, d_model 128, 4 层, dropout 0.15
batch_size_per_device: 32, accumulate_grad_batches: 16   # 有效 batch 512
trainer.max_steps: 320000
optimizer: Adam lr=1e-3, weight_decay=1e-4
EarlyStopping(monitor=val/recall@5, patience=10)
sequence_length: 120
```

**GRID 的 `train_layer_wise` 实现很有意思**（`src/modules/clustering/residual_quantization.py:459`）：
```python
eff_n_layers = self.n_layers + 1 if self.reconstruction_loss_function is not None else self.n_layers
self.steps_per_layer = total_steps // eff_n_layers   # 总步数按层均分
```
即"逐层训 N 步，训完一层冻一层"。⚠️ **这个策略只在 RQ-KMeans 路线启用**——`rqvae_train_flat.yaml` 里对应字段是 `train_layer_wise: false` / `normalize_residuals: false`。我们的 RQ-VAE 是端到端整体训练，**不适用**。

### 2.5 MMGRec（arXiv 2024，多模态融合进 RQ-VAE 的学术实现）

仓库：https://github.com/hanliu95/MMGRec

融合路径（`data_pro.py` → `model_train.py`）：
```
concat(visual, aural, text) → 线性映射到 d 维  ┐
                                              ├→ 拼接 → GCN 聚合(用户-商品二分图) → RQ-VAE 量化
随机初始化 CF embedding            ────────────┘
```
报告的 R@10 = 0.1269 / NDCG@10 = 0.0802。

> 这是**"多模态 + 协同信号一起吃进 RQ-VAE"** 的开源样本，和我们 `fuse_embeddings.py` 的
> concat 模式最接近；差别是它多了 GCN 把 CF 信号显式注入，而我们是用共现对比损失隐式注入。

---

## 3. 横向对照总表

| 项目 | 数据集 | tokenizer 路线 | 层×码本 | tokenizer 训练量 | 融合方式 | 生成器训练量 |
|---|---|---|---|---|---|---|
| MiniOneRec | Amazon23 I&S | RQ-VAE | 3×256 | 5,000~10,000 轮 | 无（纯文本） | SFT ≤10 ep, GRPO 2 ep |
| ETEGRec | **Amazon23** sci/inst/game | RQ-VAE 端到端 | 3×256, e_dim 128 | 10,000 轮 | 无（SASRec 协同） | cycle=2 交替 |
| MQL4GRec | Amazon18 | RQ-VAE ×2 | **4×256** | 500 轮 | **不融合，双 SID** | pretrain 30 ep + finetune 200 ep |
| **GRID** | Amazon P5 | **RQ-KMeans** | 3×256 | **30 步/层** | 无 | 320,000 步 |
| GRID | Amazon P5 | RQ-VAE | 3×256, latent 64 | **3,000 步** | 无 | 同上 |
| MMGRec | 多模态 | Graph RQ-VAE | — | — | **concat + GCN** | — |
| **本项目** | Amazon23 I&S + VG | **RQ-VAE（同 MiniOneRec）** | 256×3（默认） | **尚未正式训练**（仅合成冒烟 80 轮；`rq/rqvae.sh` 默认 10000 轮） | **concat/mlp/gate → 单向量** | 待 SFT |

---

## 4. 对本项目的三条具体建议

**① 融合投影的轮次要按指标定，不能一句"越多越好"。**
60 轮 = 60 × (720,716 / 256) ≈ **169,000 步**，是 GRID 整个 RQ-VAE 训练的 **56 倍**。
实测（§5）：**R@10 在 10 轮后就平台了**，但 **R@50/R@100 到 60 轮仍在单调上升（+17.8% / +16.1%）**。
→ 面向 top-10 精度取 **10~15 轮 + early stopping**；面向长尾覆盖/码本语义结构取 **40~60 轮**。
无论如何，**继续加轮次都不是提升 R@10 的正确手段**（该调的是 hidden 宽度与对比损失温度）。

**② RQ-VAE 本体才是真正缺训练的一段（勘误后修正）。**
我们的 SID tokenizer 已是端到端 RQ-VAE（encoder + 量化器 + decoder，AdamW，`rq/rqvae.sh` 默认 10000 轮），
但**真实数据（I&S / VG 融合向量）一次都没训过**——目前只有合成冒烟 ckpt。
对标量级：MiniOneRec 5000~10000 轮、ETEGRec 10000 轮（bs 1024）、MQL4GRec 500 轮（bs 2048）、GRID RQ-VAE 3000 步。
→ **M2 下一步：对每个融合模式（text/concat/mlp/gate）× 每个域各训一个正式 RQ-VAE ckpt**，
25,847 items / bs 2048 → 13 batch/轮，5000 轮 = 65k 步，本地 4GB 卡可跑（模型仅 ~70MB）。

**③ 碰撞消解策略维持现状（最后一层 Sinkhorn），但要按模式分别调。**
逐层 Sinkhorn 会伤 LCP（esci 项目实测），`build_sid.py` 只开最后一层是对的。
 GRID 的逐层策略属于无梯度的 RKMeans 路线，对 RQ-VAE 不适用。
可选精修手段（来自 GRID 的启发，按需 A/B）：encoder 输出端加 BatchNorm→L2 归一化（`normalize_layer`），
latent 压到 64 维（我们默认 e_dim 32，已更低）。

---

## 5. 与本项目实测数据的交叉验证

`fuse_long.sh IandS 60 5` 的留出曲线（每 5 轮在互不重叠的留出对上打点，完整 12 个点）：

| epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loss | 11.291 | 10.913 | 10.738 | 10.623 | 10.535 | 10.464 | 10.406 | 10.358 | 10.315 | 10.278 | 10.246 | **10.215** |
| **R@10** | 0.0855 | 0.0960 | **0.0995** | 0.0920 | 0.0885 | 0.0990 | 0.1030 | **0.1080** | 0.0960 | 0.0955 | 0.1035 | 0.0985 |
| **R@50** | 0.2135 | 0.2355 | 0.2330 | 0.2360 | 0.2370 | 0.2395 | 0.2410 | 0.2445 | 0.2425 | 0.2495 | 0.2530 | **0.2515** |
| **R@100** | 0.2955 | 0.3245 | 0.3215 | 0.3220 | 0.3275 | 0.3245 | 0.3285 | 0.3375 | 0.3340 | 0.3405 | 0.3365 | **0.3430** |

**两个指标的行为完全不同，不能一句话概括：**

- **R@10 在 e10 后即饱和**：e10=0.096 → e60=0.0985，中间在 0.085~0.108 之间**无趋势震荡**。
  峰值出现在 e40（0.108），但 e45 立刻掉回 0.096 —— 这是噪声，不是可复现的改进。
- **R@50 / R@100 仍在单调改善**：0.2135→0.2515（**+17.8%**）、0.2955→0.3430（**+16.1%**），
  且到 e60 都没有拐点。

loss 则一路单调降（11.29 → 10.22），**无法用它判断何时停**（这正是我们需要留出集的原因）。

**结论**：轮次该给多少，取决于下游用 SID 做什么——
- 若关心 **top-10 精度**（生成式召回的最终命中）：**10~15 轮足够**，继续训只增成本。
- 若关心 **长尾覆盖 / 码本整体语义结构**（R@50、R@100，直接影响 SID 的层次语义质量）：
  **可以训到 40~60 轮**，收益还没枯竭。

> 一个待验证的猜想：R@10 饱和而 R@50 不饱和，可能是 MLP 把大量容量花在了"拉开中间段相似度"上，
> 而 top-10 那批最难的判别边界早已学完。若成立，**加宽 hidden 或改对比损失的 temperature**
> 比加轮次更能推动 R@10。这条留作 M2 的消融。

### 5.1 gate 60 轮 + 四模式最终对比（2026-09-12 04:07 完成）

**gate 曲线**（与 mlp 同形态：R@10 早饱和、R@50/100 长尾仍在涨）：

| epoch | 5 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| R@10 | 0.0990 | 0.1030 | 0.1080 | 0.1110 | 0.1070 | 0.1105 | 0.1095 |
| R@50 | 0.2260 | 0.2400 | 0.2510 | 0.2645 | 0.2700 | 0.2650 | 0.2730 |
| R@100 | 0.3015 | 0.3205 | 0.3515 | 0.3680 | 0.3800 | 0.3730 | **0.3800** |

**四模式在同一留出集（80,079 对，avg_pos=7.88，随机基线 R@10=0.003）上的最终对比**：

| 模式 | R@10 | R@50 | R@100 | vs text |
|---|---|---|---|---|
| text（单模态基线） | 0.0830 | 0.1783 | 0.2280 | — |
| concat（PCA 线性） | 0.0883 | 0.1790 | 0.2347 | +6% / +0.4% / +3% |
| mlp_e60 | 0.1013 | 0.2693 | 0.3623 | +22% / +51% / +59% |
| **gate_e60** | **0.1100** | **0.2787** | **0.3820** | **+33% / +56% / +68%** |

- **监督融合（共现对比）是多模态增益的来源**：concat 线性拼接几乎白给（+6%），
  mlp/gate 的 InfoNCE 才真正把共现信号压进向量。
- **gate > mlp**：乘性门控让模型自己学每个物品"信文本还是信图像"，缺图商品（23 个）自动偏文本。
- 训练对 vs 留出对的水分仍在（gate 0.795 vs 0.110，约 7 倍），印证留出评估的必要性。
- 跨模态线性可对齐性：text→image R@10 0.622（岭回归映射后），两模态信息互补性 OK。

**→ 正式配方定为 gate_e60**（长尾指标未枯竭，全量训练对 720,716），
RQ-VAE 将以 `emb_fused_gate_e60.npy` 为输入开始正式训练。

---

## 6. 参考链接

| 项目 | 仓库 | 关键配置文件 |
|---|---|---|
| MiniOneRec | github.com/weepon/MiniOneRec | `rq/rqvae.py`, README §3.1.1 |
| ETEGRec | github.com/RUCAIBox/ETEGRec | `RQVAE/run_pretrain.sh`, `run.sh` |
| MQL4GRec | github.com/N-A-E-S/MQL4GRec | `index/scripts/run.sh`, `scripts/{pre,finetune}.sh` |
| GRID | github.com/snap-research/GRID | `configs/experiment/{rqvae,rkmeans,tiger}_train_flat.yaml` |
| MMGRec | github.com/hanliu95/MMGRec | `data_pro.py`, `model_train.py` |
| 论文索引 | github.com/HKBU-LAGAS/Awesome-Item-ID-Gen-RecSys | SID 综述 + 全量论文清单 |
