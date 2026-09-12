# GenRetrieval 知识点总览

> 定位：本项目**所有用到的知识点**的单一入口。每个结论都标注代码出处或实测数字，
> 不写"应该/可能"。机制解释、踩坑与 FAQ 留在这里，本文负责串起来；
> **工业界调研与开源配方、SID 探索过程全记录见 `docs/SID_PIPELINE.md`**（新读者从那里进，
> 原 `MULTIMODAL_SID_SURVEY.md` / `SID_TRAINING_RECIPES.md` 已并入）；
> 流水账见 `docs/EXPERIMENT_LOG.md`（本地，未上传）。
>
> 最后更新：2026-09-13

---

## 0. 全局流水线

```
Amazon23 原始数据 (I&S / VG 两个独立域，不合并)
   │
   ├─ 文本 ──→ Qwen3-Embedding-0.6B ──→ (N, 1024)  已 L2 归一化
   └─ 图像 ──→ SigLIP-base-patch16-224 ─→ (N, 768)   已 L2 归一化
                    │
                    ↓  融合（四选一，统一输出 1024 维）
        ┌───────────┼───────────┬───────────┐
       text      concat       mlp         gate      ← 共现 InfoNCE 监督
   (单模态基线) (PCA线性)  (MLP非线性)  (门控, 当前最优)
                    │
                    ↓  RQ-VAE（端到端，encoder + 残差量化 + decoder）
              SID codes (N, 3)，码本 [256,256,256]
                    │
                    ↓  导出两版（解耦）
            sid_raw (纯 argmin)【默认交付 = 语义桶】/ sid_sk (末层 Sinkhorn 消解，存档可切回)
                    │
                    ↓  SFT / RL（MiniOneRec 路线）
              生成式召回
```

**为什么统一输出 1024 维**：保证四模式的 SID 消融可比。若维度不同，RQ-VAE 的
`in_dim` 与码本容量都不同，指标就没有对照意义。

---

## 1. 数据集与预处理

| | **I&S**（Industrial_and_Scientific） | **VG**（Video_Games） |
|---|---|---|
| items | 25,847 | 25,611 |
| users | 46,341 | — |
| train / valid / test | 251,576 / 16,074 / 13,232 | — |
| history_len 上限 | 20 | 20 |
| 图像 URL 覆盖 | 100% | 100% |
| 图像编码成功 | 25,824（**99.91%**） | 25,570（**99.84%**） |
| 文本 token 均/中位 | 173 / 175 | 127 / 114 |

- **域独立**：两域各跑各的，**不合并、不共享码本**。
- 目录：`data/Amazon23/raw/`（下载物，不入 git）、`data/Amazon23/<域>/`（处理产物）。
- `emb/` 下的正式产物：`emb_image_siglip.npy`(768d)、`emb_image_mask.npy`、
  `emb_text_title-features-category.npy`(1024d)、`emb_fused_<mode>.npy`(1024d)。
  长程产物在 `emb/long/`（带 `_e60` 后缀 = 60 轮）。

### 缺失图像的处理
缺主图的商品（I&S 23 个、VG 41 个）图像向量是**零向量**，由 `emb_image_mask.npy` 标记。
校验结论：零向量行与 `mask=False` **完全一一对应**，无错位。
融合时 `gate` 模式会自动给这些样本降图像权重（门控的设计初衷之一）。

---

## 2. 两个编码器

### 2.1 SigLIP-base-patch16-224（图像）

| 项 | 值 |
|---|---|
| 出处 | Google DeepMind, ICCV 2023, arXiv:2303.15343 |
| 骨干 | ViT-B/16，12 层，768 维，patch 16，224px → 196 patches |
| 输出 | **768 维**，已 L2 归一化 |
| 训练数据 | WebLI 英文图文对 |
| 评测 | ImageNet 零样本 73.4%；COCO 图文检索 R@1 49.7 / 67.5 |
| 本机 | 占 0.42GB，**全程冻结不微调** |

**核心贡献不是结构，是损失函数**：把 CLIP 的"整批 softmax 归一化"改成
"每对图文独立做一次二分类（sigmoid）"。

```
CLIP  : 需要 all-gather 全局负样本，显存 O(B²)，B 受限于硬件
SigLIP: 每对独立判正负，显存 O(b²)，batch 可自由切分
```

⚠️ **这个卖点我们用不到** —— 我们是纯推理、冻结权重、不训练。
真正选它的理由是另外三条：开箱即用的强图文对齐、只占 0.42GB、768 维比 CLIP-L 的 1024 维省下游开销。

⚠️ **未验证的假设**：SigLIP 的对齐学自 WebLI 通用网络图文，而商品主图大量是
白底 + 包装盒 + 文字特写，分布有偏。图像塔"冻结 vs 轻量域适应"这组对照**还没做**。

### 2.2 Qwen3-Embedding-0.6B（文本）

| 项 | 值 |
|---|---|
| 出处 | 阿里 Qwen，2025-06，Apache 2.0 |
| 骨干 | Qwen3-0.6B-Base + LoRA，28 层，hidden 1024 |
| 输出 | **1024 维**（MRL 支持截断到 32~1024） |
| 评测 | MTEB(Eng) **70.70**，MTEB-Code 75.41 |
| 本机 | 占 1.19GB，token 均 173(I&S)/127(VG) |

**三个必须知道的细节**（前两个是读本机 `config.json` 才发现的）：

1. **`head_dim` 与 `hidden/heads` 解耦**：16 头 × 128 = 2048 ≠ hidden 1024。
   Qwen3 刻意把 head_dim 钉在 128。按老公式 `head_dim = hidden/num_heads` 估 FLOPs 会**错一倍**。
2. **它是因果 LM，不是双塔编码器**：`architectures: Qwen3ForCausalLM`。
   embedding 是 decoder 最后一层抠出来的隐状态，所以必须 **last-token pooling**
   —— 因果注意力下只有末位 token 看得见全句。
3. **指令感知**：query 侧要加 `Instruct: {任务}\nQuery: {query}`，官方称不加掉 1%~5%。
   我们现在只编物品侧、不加指令（合理）；**将来编用户 query 必须走 `get_detailed_instruct`**。

⚠️ **停在 0.6B 不是最优，是 4GB 卡上唯一能跑的档位**。70.70 低于最佳竞品 73.30。
想验证"编码器规模是不是瓶颈"，只能上云 3090 换 4B 重编码。

---

## 3. 多模态融合

### 3.1 四模式（统一 1024 维输出）

| 模式 | 结构 | 参数量 | 是否有监督 |
|---|---|---|---|
| `text` | 直接用文本向量（单模态基线） | 0 | — |
| `concat` | `[e_t; e_i]` → **PCA** 线性降到 1024 | 0 | ❌ 无监督 |
| `mlp` | `[e_t; e_i]` → Linear(1792→512) → LN → ReLU → Drop → Linear(512→1024) | 1,444,352 | ✅ |
| `gate` | `g = sigmoid(MLP([e_t; e_i]))`，`e = g ⊙ W_t e_t + (1-g) ⊙ W_i e_i` | **3,278,336** | ✅ |

**门控不是标量，是小型 MLP**：`Linear(1792→512) → GELU → Linear(512→1024) → sigmoid`，
输出 **1024 个 0~1 权重**，逐输出维度决定"这一维信文本还是信图像"。
`gate` 分支单独 1,443,328 参数 ≈ 整个 `mlp` 模型。

### 3.2 训练损失：对称 InfoNCE（批内负样本）

```python
def info_nce(anchor, pos, temperature=0.07):
    logits = anchor @ pos.t() / temperature      # (B, B)
    labels = torch.arange(anchor.shape[0])       # 正样本 = 对角线
    return F.cross_entropy(logits, labels)

# 一批 512 个共现对
ea = model(et[a], ei[a]);  eb = model(et[b], ei[b])
loss = info_nce(ea, eb) + info_nce(eb, ea)       # 双向对称
```

- 一批 512 对 → 512×512 相似度矩阵 → 对角线是正样本，其余 **511 个批内负样本**。
- 超参：`bs=512`、`lr=1e-3`、AdamW `wd=1e-4`、`temperature=0.07`、60 轮。
- **正样本 = 共现物品**：同一用户行为序列内共现（`target` 与最近 5 个历史配对 + 历史内部随机配对）。
- **自检**：随机猜的基线是 `ln(512)=6.24`；实测单向 ≈ 5.11（总 10.215）< 6.24 → 确实在学。

### 3.3 实验结果（I&S，同一留出集 80,079 对）

| 模式 | R@10 | R@50 | R@100 | vs text |
|---|---|---|---|---|
| text | 0.0830 | 0.1783 | 0.2280 | — |
| concat | 0.0883 | 0.1790 | 0.2347 | +6% |
| mlp_e60 | 0.1013 | 0.2693 | 0.3623 | +22% |
| **gate_e60** | **0.1100** | **0.2787** | **0.3820** | **+33% / +56% / +68%** |

随机基线 R@10 = 0.003。**结论：监督融合才是多模态增益的来源，线性拼接几乎白给。**

### 3.4 轮次行为（R@10 与 R@50 不同步）

| epoch | 10 | 15 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|
| R@10 | 0.0960 | 0.0995 | 0.0990 | **0.1080** | 0.0955 | 0.0985 |
| R@50 | 0.2355 | 0.2330 | 0.2395 | 0.2445 | 0.2495 | **0.2515** |

- **R@10 在 e10 后即饱和**（e40 峰值后 e45 立刻掉回，是噪声）
- **R@50/R@100 到 e60 仍单调上升**（+17.8% / +16.1%），无拐点
- loss 从 11.29 单调降到 10.22，**完全无法判断早停**

→ 面向 top-10 取 10~15 轮；面向长尾/码本语义结构取 40~60 轮。

### 3.5 ⚠️ 两处已知不干净的地方

1. **gate vs mlp 参数量不对等**：3.28M vs 1.44M（**2.27×**），且 mlp 有 LayerNorm+Dropout
   而 gate 分支没有。所以现有证据只支持"gate_e60 是目前最好的可用配置"，
   **不支持"门控机制优于 MLP"这个机制性结论**。
2. **temperature 0.07 硬编码**在 `info_nce()` 默认参数里，没有命令行参数，**从未调过**。
   另：训练时只见到 511 个负样本，评估时是全局 25,846 个 —— 难度不匹配。
   模型才 3.3M、logits 2048² 才 16MB，**加大 batch 到 2048 是收益最便宜的一档**。

---

## 4. 指标口径：融合阶段的 R@k

**这是一个检索代理指标，不是最终推荐指标。** 定义（`recall_at_k`, `fuse_embeddings.py:285`）：

1. 抽 3000 个 query 物品
2. 融合向量对全库 25,847 件做余弦排序，**排除自身**
3. 看 top-k 里**是否至少有一个**正样本（= 留出共现伙伴）
4. 对 3000 个 query 取平均

`gate R@100 = 0.382` 的准确含义：**对 38.2% 的物品，它的某个共现伙伴出现在融合向量空间的 top-100 近邻里**。
随机基线 = `avg_pos × k ÷ N = 7.88 × 100 ÷ 25847 = 0.030`。

⚠️ **不是下游 HR@k**。它衡量"融合向量空间有没有把行为相关的物品聚到一起"——
而这正是 SID 需要的性质（共现物品该共享前缀）。真正好坏要等 SFT 出 HR@k。

⚠️ **`recall@k` 绝对值随 `avg_pos` 浮动**，不同留出集的数字不可直接横比
（早期 avg_pos=3.12 时 text R@10=0.0403；现在 8.15 时口径已变）。**只能在同一留出集内比**。

---

## 5. RQ-VAE（SID tokenizer）

### 5.1 结构（`rq/models/rqvae.py`）

```
x (1024) → encoder: 1024→2048→1024→512→256→128→64→32   [MLPLayers, xavier 初始化]
        → ResidualVectorQuantizer: 3 层 × 256 码 × 32 维
        → decoder: 32→64→128→256→512→1024→2048→1024
        → 重建 out，损失 = 重建 MSE + quant_loss
```

参数量 **9,819,040**（in_dim=1024 时）。码本容量 `256³ = 16,777,216` ≫ N=25,847。

### 5.2 残差量化（`rq/models/rq.py`）

```
r_0 = encoder(x)
for k in [0,1,2]:
    i_k = argmin || r_k - c ||²        # 最近码
    q_k = codebook_k[i_k]
    r_{k+1} = r_k - q_k                # 残差传下一层
z_q = q_0 + q_1 + q_2
```

### 5.3 损失（`rq/models/vq.py:90`）

```python
commitment_loss = MSE(x_q.detach(), x)      # 拉 encoder 输出向码本
codebook_loss   = MSE(x_q, x.detach())      # 拉码本向 encoder 输出
loss_layer      = codebook_loss + beta * commitment_loss   # beta = 0.25
# 总损失 = 重建 MSE + quant_loss_weight * mean(各层 loss)
# 梯度直通：x_q = x + (x_q - x).detach()
```

**straight-through estimator**：量化这一步不可导，用 `x + (x_q - x).detach()`
让梯度从 decoder 直接穿到 encoder。

### 5.4 训练配置（对齐开源，不自己拍）

| 超参 | 值 | 出处 |
|---|---|---|
| epochs | 5000（默认）/ 10000 | 上游 `rq/rqvae.py` / `rq/rqvae.sh` |
| batch_size | 2048（默认）/ 20480 | 同上 |
| lr | 1e-3 | 同上 |
| optimizer | AdamW，wd 0.0 | 同上 |
| scheduler | constant + warmup（50 epoch） | 上游 `rq/trainer.py` |
| grad_clip | 1.0 | 同上 |
| `num_emb_list` | [256, 256, 256] | 同上 |
| `e_dim` | 32 | 同上 |
| `layers` | [2048,1024,512,256,128,64] | 同上 |
| `beta` | 0.25 | 同上 |
| `kmeans_init` | True（100 iters） | 同上 |

**步数换算**：`rq/trainer.py:28` `max_steps = epochs × len(data_loader)`；
25,847 / bs 2048 = **13 batch/epoch** → 5000 轮 = **65,000 步**。
上游 `rqvae.sh`（bs 20480 → 2 batch/epoch）× 10000 轮 = 20,000 步。

### 5.5 训练集 / 验证集怎么取的

⚠️ **没有切分。训练集 = 验证集 = 全部 25,847 条物品向量。**

- `DataLoader(data, batch_size=2048, shuffle=True, drop_last=False)` → 训练
- `eval_collision` 用 `DataLoader(data, batch_size=4096, shuffle=False)` → **同一批数据、全量**

这不是 bug，是上游 `rq/trainer.py` 的原始行为（`fit()` 里 `_valid_epoch(data)` 传的就是训练 loader）。

**合理性**：RQ-VAE 是 **transductive（直推式）** 任务——码本就是要覆盖这批物品，
不追求泛化到新物品；且是自编码重建，无标签，留出的意义不大。
**对比**：融合阶段（`fuse_embeddings.py`）有真正的留出对切分，因为那里是度量学习、需要泛化。

### 5.6 ⚠️ 实测现象：k-means 初始化 vs 码本塌缩

**Q：为什么第 1 轮碰撞率最低？是接近随机初始化的原因吗？**

**A：恰恰相反 —— 首轮低是 k-means 初始化的功劳，不是随机。** 决定性对照：

| 初始化方式 | ep1 碰撞率 | 唯一码数 |
|---|---|---|
| k-means（`kmeans_init=True`，默认） | **0.044** | 24,708 |
| 真随机 `uniform(-1/256, 1/256)`（`--no_kmeans_init`） | **0.718** | 7,301 |

随机初始化下 256 个码几乎重合在原点附近，碰撞率高达 0.72。
k-means 用首批 2048 个样本的 **256 个真实质心**做初始化 → 分配天然均匀 → 碰撞率 0.044。

**完整轨迹（bs 2048，无 Sinkhorn，500 轮探针）**：

| epoch | 1 | 10 | 20 | 25 | 50 | 100 | 200 | 300 | 425 |
|---|---|---|---|---|---|---|---|---|---|
| 碰撞率 | 0.044 | 0.870 | 0.509 | 0.329 | 0.264 | 0.240 | 0.226 | 0.220 | 0.213 |
| recon loss | 0.00097 | — | — | 0.00046 | 0.00034 | 0.00027 | 0.00022 | 0.00020 | 0.00018 |

**"先飙到 0.87 再回落"的机制**：
1. ep1：k-means 质心贴合数据分布 → 均匀 → 低碰撞
2. ep2~10：encoder 被重建损失 + commitment 拉扯，输出分布快速漂移；
   码本只被 `codebook_loss` 缓慢更新且**只更新被选中的码**，追不上 → 样本挤到少数码上
3. ep10 之后：码本逐渐追上，碰撞率缓慢回落，最终稳定在 ~0.21

**分层诊断（更狠的发现）**：

| | 第0层用码数 | 最大簇 | 第1/2层 | R² | ICR |
|---|---|---|---|---|---|
| ep1（kmeans init） | **255**/256 | 935 | 250 / 254 | 0.0104 | **0.956** |
| best_loss（训练后） | **59**/256 ⚠️ | 1391 | 256 / 256 | **0.728** | **0.760** |

**训练让重建变好（R² 0.01 → 0.73），代价是第 0 层码本塌缩（255 → 59 个码）。**
这是经典的 **codebook collapse**：未被选中的码永远收不到梯度（dead codes），
被选中的码越来越强 → 富者愈富的正反馈。

### 5.6.1 ⚠️ 一次作废的对照实验（教训留档，数字已删）

早前有一组"三配方对照"（bs 2048/8192 × 训练期 Sinkhorn 开/关，1500 轮、单 seed），
曾得出"训练期开最后一层 Sinkhorn 最优"。**该结论已作废并从本文删除**，两个原因：

1. **自变量根本没生效**：当时 `train_rqvae.py` 硬编码 `use_sk=False` → `sk_epsilons` 从未进入前向。
   观测到的"第 0 层用码 59 vs 85"差异**全部来自 k-means 初始化的随机性**（当时 `KMeans` 无
   `random_state`，见 §5.11 Q6b）⟹ 副产品结论：**第 0 层塌缩程度在 seed 间波动可达 ±26 个码**。
2. **规模与口径已被取代**：1500 轮、单 seed 的数字均被 5000 轮固定 seed 实验覆盖
   （见 `docs/SID_PIPELINE.md` §2.4 / 本文 §5.9b）。

🔴 **可复用的教训**：写完消融先确认**自变量真的进入了计算图**（本次是 grep 训练循环才发现的）；
涉及 k-means / 数据 shuffle 的实验必须先固定 seed，否则读到的差异可能全是噪声。

真正的"训练期 Sinkhorn"消融已在 5000 轮实验里补上（`--train_use_sk`，见 §5.9b ③）。

**MiniOneRec vs MQL4GRec 的真实差异（已核实源码，结论仍然成立）**：
- MiniOneRec：`rq/rqvae.py:35` 默认 `--sk_epsilons [0,0,0]` → 训练期全 argmin；
  只在导出阶段 `rq/generate_indices.py` 把最后一层设 0.003、while ≤20 轮**只对碰撞组**定向重编码。
- MQL4GRec：`index/scripts/run.sh` 传 `[0,0,0,0.003]`，且 `index/trainer.py` 训练时调
  `self.model(data)`（→ `forward(use_sk=True)`）→ **训练期末层真的走 Sinkhorn**。
- 判定条件只有一个：`use_sk=True` **且** `sk_epsilon>0` 同时成立才走 Sinkhorn（`vq.py:74`）。
  注意 `forward(use_sk=True)` 与 `get_indices(use_sk=False)` 的**默认值是反的**。
- 本项目的 `rq/train_rqvae.py` 现用 `use_sk=args.train_use_sk`（默认 False = MiniOneRec 口径）。

详见 `docs/SID_PIPELINE.md` §1.3。

### 5.6.2 ⚠️ 选 ckpt 必须按碰撞率（+ burn-in），**不能按 loss**

| ckpt | 重建 R² | ICR | 第0层用码 |
|---|---|---|---|
| `best_loss` @ep120 | 0.7470 | 0.7947 | 85/256 |
| **`last` @ep500** | **0.8254** | **0.8154** | 87/256 |

**总 loss 在 ep120 触底（0.00033）后一路回升到 0.00039，但重建 R² 和碰撞率都在持续变好。**
原因是总 loss = 重建 MSE + quant loss，回升的是 **commitment 项**（encoder 输出逐渐远离码本），
而重建质量与唯一性都在改善。**按 loss 选会挑到一个既没训满、唯一性又差的模型。**

同时，**按碰撞率选又必须跳过 burn-in**（默认前 10% 轮次），否则会选到
ep1 的 k-means 初始化模型 —— 碰撞率 0.044（全场最低）但 R²=0.01，等于没训。

→ `train_rqvae.py --select_ckpt collision --burn_in_frac 0.1`（默认），
产物统一落到 `ckpt/selected_model.pth` 供下游引用。

### 5.7 Sinkhorn 均衡分配

`rq/models/layers.py:86`：把"每个样本选一个最近码"松弛成"样本-码的软分配矩阵 Q"，
用 Sinkhorn-Knopp 迭代使 **每行和 = 1/B、每列和 = 1/K** → 强制每个码分到同样多的样本。

```python
Q = exp(-distances / epsilon)
for _ in range(sk_iters):
    Q /= Q.sum(dim=1, keepdim=True); Q /= B      # 行归一化
    Q /= Q.sum(dim=0, keepdim=True); Q /= K      # 列归一化
Q *= B
indices = argmax(Q, dim=-1)
```

`epsilon → 0` 退化为硬分配（等价于 argmin）；`epsilon` 越大越接近均匀但越偏离最近码。

⚠️ **代价：唯一性↑ 保真度↓。** Sinkhorn 会把物品推给"次近但没那么挤"的码。
实测（20 轮冒烟模型）raw ICR 0.9559 → sk ICR 0.9983，但 **6.3% 的物品被改码**。

**为什么只开最后一层**：esci-ai-search 实测逐层 Sinkhorn 能把 ICR 拉到 100%，
但 **LCP（前缀语义保持）反而变差** —— 前缀被打散，层次聚类结构被破坏。
最后一层 Sinkhorn 是 ICR / LCP 的甜点区。（`build_sid_dual.py --sk_layers last|all`）

---

### 5.8 🔴 Sinkhorn 的 batch 敏感性——导出 batch 是最被低估的超参

> 2026-09-12 实测。这是一个**我们自己的实现 bug**，曾让 ICR 长期卡在 0.935，
> 并误导我们以为"轮次不够"。结论：`build_sid_dual.py --batch` 必须 **≤ 最后一层码本大小 K**。

#### 现象

I&S / text 模式跑完 1500 轮后，Sinkhorn 迭代 20 轮只把 ICR 从 0.8295 提到 **0.9351**，
剩 2,854 个物品仍碰撞。增量曲线边际递减（首轮 +0.059，末轮 +0.0006），看起来"没收敛"。

#### 根因：Sinkhorn 是 **batch 内** 的联合分配，`B >> K` 时结构性碰撞不可避免

`vq.py:63-83` 的 Sinkhorn 作用于 `[B, K]` 距离矩阵（B = 当前 batch 物品数，K = 码本大小）。
双随机约束下 **每列和 = B/K**，即每个码期望分到 `B/K` 个物品：

| 导出 batch | B/K | 每码期望物品数 | 能否无碰撞 |
|---|---|---|---|
| **2048**（我们原来的默认值） | 8.0 | 8.0 | ❌ 数学上不可能 |
| 256 | 1.0 | 1.0 | 边界 |
| **64**（上游 MiniOneRec 口径） | 0.25 | 0.25 | ✅ 可近似置换 |

**`B=2048` 时每码期望 8 个物品 → 单轮内必然碰撞 87.5%。**
迭代之所以还能缓慢爬升，只是因为每轮 `collided` 集合变小、**分批边界随之变化 = 重新洗牌**，
本质是随机爬山，不是收敛。

#### 实测对照（同一 ckpt，I&S/text，`best_collision_model.pth`）

| 配置 | 轮数 | ICR | 剩余碰撞物品 | LCP ratio | ΔMSE | 改码比例 | 耗时 |
|---|---|---|---|---|---|---|---|
| raw（纯 argmin） | — | 0.8295 | 7,290 | 105.76 | — | — | — |
| batch 2048（原默认） | 20 | 0.9351 | 2,854 | 97.77 | 4.32e-6 | 15.7% | 4s |
| batch 2048 | **100** | 0.9534 | 2,019 | — | — | — | 12s |
| batch 256 | 20 | 0.9981 | 94 | 93.42 | 7.62e-6 | 21.7% | 5s |
| **batch 64（现默认，= 上游）** | **17**（早停） | **0.9997** | **15** | 93.35 | 1.13e-5 | 23.9% | 9s |

**关键读数**：
- **轮次是次要因素**：batch 2048 从 20 轮加到 100 轮（×5）只换 +1.8 个点（0.9351→0.9534），外推也到不了 0.99。
- **batch 是主因**：同样 20 轮，batch 2048→64 换来 **+6.5 个点**（0.9351→0.9997），且第 17 轮就早停（更快）。
- 代价可接受：LCP 97.77→93.35（−4.5%，仍是 random 的 93 倍）、ΔMSE 4.3e-6→1.1e-5（R² 0.8526→0.8411）。
  生成式召回里**碰撞 = 模型无法区分物品 = 必然错召回**，ICR 优先于这点保真度损失。

#### 上游佐证

`MiniOneRec/rq/generate_indices.py:83` 用的是 `batch_size=64`（< K=256），
我们早年写 `build_sid_dual.py` 时误把**训练 batch 2048** 当成了导出 batch。
**这是我们自己引入的偏差，不是上游做法。**

#### 修正："重复 embedding 是硬下界"这个说法不成立

最初统计到 82 个物品、40 组 embedding 逐位相同，以为 ICR 上限是 0.9984。
实测 batch=64 下只剩 **15** 个碰撞物品 —— **40 个重复组里 33 组被成功分开**。

原因：Sinkhorn 是 batch 内联合分配，**逐位相同的物品只要落在不同的 batch，
就会因 batch 上下文不同而分到不同的码**。例：`item 107 / 1803` embedding 完全相同，
最终 codes `[190,42,179]` vs `[190,42,224]`（前两层相同、末层不同）。

→ 所以重复 embedding **不是硬下界**，真正的残余只有那些反复落在同一 batch 内的少数物品。
→ 也说明 ICR 的理论上限接近 1.0，而非 0.9984。

#### 修复

`rq/build_sid_dual.py` 的 `--batch` 默认 `2048 → 64`，并加注释锁死这条推理链。
诊断脚本：`scripts/multimodal/diag_collision.py`（拆解硬碰撞/软碰撞 + 打印收敛曲线）。

⚠️ **记住：这个 batch 不是训练 batch**，它只影响 Sinkhorn 的作用范围，
对 ICR 的影响（+6.5 个点）远大于训练轮次（+1.8 个点）。

#### 副作用检查：改这个 batch 会不会污染 raw 基线？**不会**

`--batch` 在代码里是**同一个参数**，两处共用：

```
build_sid_dual.py:169   codes_raw = quantize(..., args.batch, use_sk=False)   # raw 导出
build_sid_dual.py:206   new       = quantize(..., args.batch, use_sk=True)    # sk 迭代
```

但**只有 sk 会变**。原因：

- `use_sk=False` 走 `argmin`，是**逐样本独立**计算，batch 仅仅是并行度，不改变结果；
- `use_sk=True` 走 Sinkhorn，是 **batch 内联合分配**，batch 决定分配作用范围，会改变结果。

实测（同一 ckpt，I&S/text）三份 `sid_raw` 的 md5 完全相同 —— **bit-level 一致**：

| 导出 batch | sid_raw md5 | sid_raw ICR | sid_sk ICR |
|---|---|---|---|
| 64 | `4e4c79a170ab2518` | 0.8295 | **0.9997** |
| 256 | `4e4c79a170ab2518` | 0.8295 | 0.9981 |
| 2048 | `4e4c79a170ab2518` | 0.8295 | 0.9351 |

→ **raw 基线干净，不受导出 batch 影响**，改这个超参不会污染 raw 对照组，
可以放心只拿 sid_sk 做消融。（验证代码见 `scripts/multimodal/diag_collision.py` 的思路，
直接 `np.array_equal` 比三份 npy 即可。）

#### 📌 实验范围决策（2026-09-12）

**决定不做「训练期 Sinkhorn」这组消融**（即 `--train_use_sk --sk_epsilons 0 0 0.003`，
MQL4GRec 口径）。理由：

1. 跑它需要 3 个融合模式各自**重训 1500 轮**（≈60 min），成本高；
2. 训练期开 Sinkhorn 会削弱「训练期零 Sinkhorn」这个干净基线，是与 MiniOneRec 对标的主要口径；
3. 它要解决的原始问题是**码本塌缩**（第 0 层死码 64.8%），而这个问题在
   **导出期用 batch=64 的 Sinkhorn 已经顺带解决**（第 0 层用码数不变，但整体 ICR 已达 0.9997）。

→ 最终实验矩阵 = **3 融合模式 × 2 版 SID（raw / sk）= 6 组**，不是 9 组。
`--train_use_sk` 开关保留在 `train_rqvae.py` 里（已实现、已自校验），需要时随时可补跑。

---

### 5.9 三模式 SID 结果（I&S，1500 轮，导出 batch=64）

> ⚠️ **口径提示**：本表是**三模式唯一一次同口径横评**（1500 轮、导出 batch=64）。
> text 与 gate 已被 5000 轮固定 seed 实验取代（见 §5.9b / `docs/SID_PIPELINE.md` §2.4，
> 数值以那边为准）；**mlp 只跑过 1500 轮**，想让三模式在 5000 轮口径下重评需补跑 mlp。

完整表见 `results/sid/IandS/compare.md`（`scripts/multimodal/compare_sid_modes.py` 生成）。

| 指标 | text raw→sk | mlp raw→sk | gate raw→sk |
|---|---|---|---|
| **ICR** | 0.8295 → 0.9997 | 0.9901 → **1.0000** | 0.9639 → 0.9997 |
| 第 0 层死码率 | **0.6484** | 0.4102 | 0.4141 |
| **LCP ratio** | 105.76 → 93.35 | 90.53 → 89.20 | 109.55 → **105.50** |
| prefix-1 内聚比 | 1.5473 | 1.5021 | **2.0183** |
| 重建 R²（3 层） | 0.8526 → **0.8411** | 0.7140 → 0.7138 | 0.4746 → 0.4738 |
| 改码比例 / ΔMSE | 23.94% / 1.13e-5 | **1.62%** / **2.2e-7** | 5.56% / 7.9e-7 |
| best_collision 轮次 | **1500**（未收敛） | 150 | 150 |

#### 三个结论

**① Sinkhorn 之后，唯一性不再是区分点。** 三模式 sk 的 ICR 都 ≥0.9997，
碰撞已不是选型依据 —— 选型要看**结构指标（LCP / cohesion）和代价（改码比例、R²）**。

**② 存在明确的 trade-off：语义结构 ↔ 重建保真度，二者负相关。**

| | 语义结构 | 重建 R² |
|---|---|---|
| gate | **最好**（LCP 105.50、内聚 2.02） | 最差（0.474） |
| text | 中等（93.35） | **最好**（0.841） |
| mlp | 最差（89.20） | 中等（0.714） |

方向上与融合阶段的下游指标（R@10: text 0.083 < mlp 0.101 < **gate 0.110**）**同向**：
融合越强 → 语义结构越好，但向量分布越"聚拢" → 量化重建越难。
这不是 bug，是 **InfoNCE 把相关物品拉近后，分布更聚类化**的必然结果。

**③ 图像模态的增益有直接证据（raw 口径）。**

| | text（纯文本） | + 图像融合（mlp / gate） |
|---|---|---|
| 第 0 层死码率 | **0.648**（只用 90/256 码） | 0.410（用 151/256 码） |
| raw ICR | **0.8295** | 0.9901 / 0.9639 |
| 需要的改码比例 | 23.94% | 1.62% / 5.56% |

→ **加入图像让 embedding 分布显著更易量化**：死码率降 37%，raw 碰撞率降一个数量级。
这是"多模态 SID 优于纯文本 SID"在 tokenization 阶段的直接证据（不只是下游 R@k 的提升）。

#### 选型建议

- 若下游是 **Trie / beam search 生成式召回** → 优先 **gate**（LCP 105.50、内聚 2.02 最高，
  前缀结构对束搜索最关键），但要接受 R² 0.474 的信息损失。
- 若需要 **SID 本身携带更多信息**（如用 SID embedding 做粗排）→ 优先 **text**（R² 0.841）。
- **mlp 是最省事的选择**：raw 就几乎无碰撞（0.9901）、改码 1.62%、代价近零，
  是"什么都不用调"的安全默认。

⚠️ 注意：这三种模式的融合阶段本身**参数量不公平**（gate 3.28M vs mlp 1.44M，见 §3.5），
所以本表不能作为"门控机制优于 MLP"的证据，只能说明"gate_e60 这个配置产出的 SID 结构最好"。

---

### 5.9b 5000 轮五组实验：init_samples 消融 + 训练期 Sinkhorn 消融（2026-09-12）

**矩阵**：`text` 基线 + gate 上三档 init（0=首batch / 8192 / full=25,847）+ 训练期 Sinkhorn
（`gate__trainsk`，MQL4GRec 配方 [0,0,0.003]，**这次自变量真的在变**）。
全部固定 seed=2024、kmeans random_state=0（P-9 部分达成），结果在 `results/sid_e5000/IandS/`，
横向对比 `results/sid_e5000/IandS/compare.md`。

**① 5000 轮 vs 1500 轮（raw 口径，注意旧 run 未固定 seed，方向可信、数值不必精比）**

| | R² | LCP ratio | L0 死码率 |
|---|---|---|---|
| text 1500 → 5000 | 0.853 → 0.870（+0.017，边际） | 105.8 → 117.5 | 64.8% → 65.5%（不动） |
| gate 1500 → 5000 | **0.475 → 0.640（+0.165，巨大）** | 109.6 → 151.4 | 41.4% → 32.6% |

→ **R² 收益集中在 gate**：gate 融合分布更难量化，1500 轮远未到容量极限，5000 轮才吃到大半；
text 本来就接近 3×256 的表示上限，加轮只剩边际收益。`best_collision` 落点
text ep4400 / gate__init8192 ep4950 —— 5000 轮仍未完全收敛，但 ICR 已被导出期 Sinkhorn 兜住，
继续加轮只对 R² 有边际意义。

**② init_samples 消融（gate，看 raw）——8192 是甜点，全量反而略输**

| | L0 死码率 | L0 用码轨迹(50/500/1000/5000) | LCP ratio | R² | raw ICR | 改码比例 |
|---|---|---|---|---|---|---|
| init0（首batch） | 32.6% | 95→149→160→172 | 151.5 | 0.640 | 0.9505 | 7.58% |
| **init8192** | **0%（256 满用）** | **122→255→256→256** | **222.1** | **0.653** | 0.9582 | 6.54% |
| initfull | 8.6% | 122→232→232→234 | 212.7 | 0.647 | 0.9567 | 6.78% |

→ **init8192 在 ep500 就把 L0 拉满 256/256 并保持 4500 轮**——塌缩问题在这个配置下**不存在**，
不只是"缓解"。全量(25,847)反而比 8192 略差（死码 22 个、LCP 低 10），且 init 成本 28.5s vs ~9s。
单 seed 不能断言 8192>full 是本质规律，但 **8192 ≥ full ≫ 首batch** 这个排序是稳的。
（§5.11 Q6c 的 inertia/基尼探针只测了初始化时刻的质心质量，**预测不了训练 5000 轮后的结局**
——质心质量与训练后利用率并非单调对应，这是"必须真训"的教训。）

**③ 训练期 Sinkhorn 消融（gate__trainsk vs gate__init0，MQL4GRec 配方）**

| | L0 死码率 | LCP ratio | R² | 训练耗时 |
|---|---|---|---|---|
| trainsk | 4.3%（轨迹 118→170→197→231→255，缓慢爬满） | 205.5 | 0.643 | **4996s（+44%）** |
| init0 | 32.6% | 151.5 | 0.640 | 3473s |

→ 训练期 Sinkhorn **确实有效**（L0 死码 32.6%→4.3%、LCP +54），机理：前向时强制 batch 内
均衡分配 ⟹ 每个码周期性拿到梯度。但 **它被 init8192 全面支配**：init8192 零额外训练成本、
L0 直接满用、LCP 更高。**结论：MQL4GRec 配方在本设置下不采用；真正该做的是 init pass。**
未测组合：init8192 + trainsk（理论上可能叠出更稳的结构，留给后续）。

**④ 定版默认配方**：`gate` 融合 + `--init_samples 8192` + 5000 轮 + **直接导出 `sid_raw`（语义桶，
不做 Sinkhorn 消解）**。`sid_sk` 仍一并产出存档，随时可切回唯一化口径。
对应编排：`RESULTS_ROOT=results/sid_e5000 INIT_SAMPLES=8192 bash scripts/multimodal/run_sid_exp.sh IandS 5000 "gate"`。
完整数字与决策理由见 `docs/SID_PIPELINE.md`；碰撞口径的定版论证见 §5.13。

---

### 5.10 FAQ：R²、第 0 层死码、loss 回升、1500 轮够不够

> 以下全部基于**本仓已有实测结果 + 公开论文/源码**，不依赖新增实验。

#### Q1. R² 是什么指标

```
R² = 1 − recon_MSE / Var(x)
```

即"离散 SID 重建出的向量，解释掉了原始 embedding 多少比例的能量"。取值 (−∞, 1]，越高越好。

**为什么不直接看 MSE**：MSE 有量纲，`Var(x)` 因 embedding 不同而不同，跨模式不可比。
R² 是尺度无关的归一化版本。

**本项目的特殊性（让 R² 可比）**：三份融合 embedding 都是 **L2 归一化**的 1024 维向量，
实测 `input_var = 0.0009765 ≈ 1/1024` —— 三个模式**完全相同**（见 `eval_*.json` 的 `fidelity.input_var`）。
所以本仓的 R² 是可以直接横向比较的，不必担心"分母不同"。

物理含义：`||x||² = 1`，所以 **R² = 0.841 意味着重建后还剩 15.9% 的能量是量化噪声**。

| 模式 | R² (raw) | 解读 |
|---|---|---|
| text | 0.8526 | 量化噪声 14.7% |
| mlp | 0.7140 | 量化噪声 28.6% |
| gate | 0.4746 | 量化噪声 52.5% |

gate 的 R² 显著更低，**不是因为 embedding 收敛差，而是因为融合后的表示更难被 3×256 的离散码穷尽**
（详见 §5.9 的"语义结构 ↔ 重建保真度负相关"）。

#### Q2. 为什么第 0 层有死码，而第 1/2 层几乎满血

实测（`eval_raw.json` 的 `uniqueness.per_layer`）：

| 模式 | 层 | n_used | 死码率 | max_usage | usage_p50 |
|---|---|---|---|---|---|
| text | L0 | **90/256** | **64.8%** | **1385** | 262 |
| text | L1 | 256/256 | 0% | 319 | 93 |
| mlp | L0 | 151/256 | 41.0% | 450 | 171 |
| mlp | L1 | 256/256 | 0% | 311 | 98 |
| gate | L0 | 150/256 | 41.4% | 402 | 170 |
| gate | L1 | 256/256 | 0% | 223 | 94 |

**基准线：25,847 / 256 ≈ 101 个物品/码（均匀期望）**。
L1 的 p50 = 93~98 ≈ 101，**几乎是完美均匀**；而 L0 的 p50 = 170~262，**是均匀期望的 1.7~2.6 倍**，
text L0 更极端：最热的那个码独吞 1385 个物品（**均匀期望的 13.7 倍**）。

三个叠加成因，把握度从高到低：

1. **稀疏梯度 + 正反馈（VQ-VAE 通病，教科书级）**
   `vq.py` 里 `indices = argmin(d)`，然后 `x_q = self.embedding(indices)` ——
   **码本的梯度只流向被 argmin 选中的那一个码**，其余 255 个码本轮梯度恒为零。
   一个码只要偶然占据了密集区，它就被持续更新、越来越贴近该区，于是更容易继续赢 —— **富者愈富**。
   掉队的码永远收不到梯度，成为死码。这是最根本的机制。

2. **第 0 层的残差尺度最大 → 该项主导 loss → 码本漂移最快（为什么偏偏是第 0 层）**
   第 0 层输入是原始 embedding `r0 = x`（模长 ~1）；
   第 1 层输入是残差 `r1 = r0 − e_{c0}`（已被削掉一大截）；第 2 层更小。
   `rq.py:53` 是 `mean_losses = stack(all_losses).mean()` —— 三层**等权**，
   但第 0 层的 commitment/codebook loss 数值本身最大 ⟹ 它贡献了最大的梯度 ⟹
   第 0 层码本在同样的训练步数下"漂"得最多、最久（从 epoch 1 就开始积累），因此塌缩也最狠。

3. **输入分布的各向异性（推断，但有一致的旁证）**
   sentence embedding 存在 **表示各向异性 / 表示退化**（向量挤在一个窄锥里，有效秩远低于 1024）。
   k-means 在这种拉长的椭球上只能沿少数主方向切分，落到低密度区的质心拿不到样本。
   残差层面对的是**已经减去簇中心的簇内偏移**，主方差方向被层层剥离后更接近各向同性，
   因此每个质心都能分到样本。
   **旁证**：三个模式用**完全相同的 RQ-VAE 架构与超参**，只有 embedding 不同，
   L0 死码率就从 text 的 64.8% 降到融合版的 41.0% —— 说明它确实是**输入分布的函数**，
   而不只是训练动力学。这与 §5.9 的"融合质量决定量化难度"是同一件事。

#### Q3. MiniOneRec 有这个问题吗 —— 有，而且它没有修复机制

**代码级核实**（`D:\Codings\cs\Rec\MiniOneRec_oneGPU_preject\rq\` 全目录 grep
`restart|dead|usage|perplexity|utiliz|collaps|reinit|replace`）：
**零命中**。它只有两件武器：

- `VectorQuantizer.__init__` 的 `kmeans_init=True`（`vq.py:40 init_emb`），用第一批数据跑 k-means 作初始化；
- `generate_indices.py` 导出期的末层 Sinkhorn + 碰撞组迭代。

也就是说 MiniOneRec 把 TIGER/RQ-VAE 论文的"标准开局"直接搬了过来，**没有任何 dead-code 复活机制**，
因此必然共享同样的塌缩动力学。用户直觉正确。

**公开论述**：TIGER (Rajput et al., NeurIPS'23) §3.1 明确写道
"为了防止 RQ-VAE 发生 codebook 坍塌……我们使用基于 k-means 聚类的初始 codebook 初始化，
k-means 算法应用于**第一个训练批次**" —— 只治**初始化**，不治**训练过程**。

**更系统的对策清单**（arXiv 2602.07774 "Generative Reasoning Re-ranker" §2.3，
标题即为 *Techniques for Balanced Codebook Utilization*，明确列出五种互补技术）：

| # | 技术 | 我们有吗 |
|---|---|---|
| 1 | RQ-K-means 初始化（对残差逐层跑 k-means++） | ✅ `kmeans_init=True` |
| 2 | **EMA 码本更新**（γ∈[0.95,0.99]，替代梯度更新） | ❌ |
| 3 | **Diversity loss**：`λ·C·Σ_j p_j²`，最小化即在推向均匀分布 | ❌ |
| 4 | **Dead code reset**：连续 τ 个 batch 未使用 → 用当前最难重建的样本重置 | ❌ |
| 5 | **Random last levels**：最后 M 层用 Uniform 随机赋值保唯一性 | ❌（我们用 Sinkhorn 达到同样目的）|

第 2/3/4 条正是我们缺的。**这也是本项目明确的改进抓手**（见 §9 已知问题）。

#### Q4. 为什么总损失后面上升，而重建损失继续下降

不是 bug，是 VQ 类模型的**固有张力**，三模式曲线完全同构（`train.log` 实测，单位 ×10⁻⁶）：

| epoch | total | recon | rq = total − recon |
|---|---|---|---|
| 1 | 980 | 970 | 10 |
| 50 | 380 | 340 | 40 |
| **96** | **340（最低）** | 270 | 70 |
| 300 | 400 | 200 | 200 |
| 800 | 470 | 160 | 310 |
| 1500 | 510 | **150** | 360 |

mlp 同构（total 最低在 ep45，末期 rq 涨到 890，约 **89 倍**）；gate 同构（ep77，约 43 倍）。

**机理解释**：loss 拆三项（`vq.py:90-92`）

```
codebook_loss   = MSE(x_q, x.detach())     码本去追 encoder 输出
commitment_loss = MSE(x_q.detach(), x)     encoder 输出去追码本
总 loss = recon + mean_层( codebook + β·commitment )，β=0.25
```

训练后期 decoder 变强，要继续压 recon，**必须让 `z_q = Σ e_{c_i}` 携带更多区分信息**。
但 `z_q` 只能取码本里的**离散组合**（容量固定 256³ = 24 bit），唯一的出路是
**encoder 把 latent 空间拉得更开**——这样不同的 x 才能映射到差异更大的码组合。
latent 一"张开"，每个码到它负责样本的均方距离必然上升（codebook_loss ↑），
encoder 输出本身也离量化点更远（commitment_loss ↑）。

瓶颈容量被榨干后，继续压 distortion 只能以 latent spread 扩张为代价，
而**这笔账被记到了 rq 头上，不是 recon 头上** —— 两者只是在争夺同一份"预算"。

**结论：total loss 上升 ⟹ 该不该停？不该。** 我们真正要的是 collision / ICR / LCP，
它们在 rq 上升的同时**继续变好**。这正是 §5.6.2「按碰撞率选 ckpt，不按 loss」的由来 ——
有了这条机理，那句经验法则不再是 hack，而是必然推论。

#### Q5. 1500 轮够吗（别人 2000/5000/上万）

**先纠正一个常见误区：batch_size 相同时，跨数据集可直接比的不是 steps，是 epoch。**
`BatchLoader` 每 epoch 遍历一遍全量数据，所以 **epoch 数 = 每个 item 参与训练的次数**，与数据量无关。

MiniOneRec 源码 `rq/rqvae.py`：`--epochs default=5000`、`--batch_size default=2048`、
`--lr 1e-3`、`--warmup_epochs 50` —— **batch_size 与我们一模一样（2048）**。
所以对照极其干净：

| 工作 | epochs | 每 item 被训次数 | 相对我们 |
|---|---|---|---|
| **MiniOneRec 默认** | **5000** | 5000 | **3.33×** |
| 本项目现状 | 1500 | 1500 | 1× |
| MQL4GRec（`index/scripts/run.sh`） | 500 | 500 | 0.33× |

（注：坊间流传的 TIGER "20k epoch" 只见于二手博客，未在原论文核实，**不计入判断**。）

**未收敛的三个直接证据**：

1. 三模式的 `best_collision` **全部落在 ep1500（最后一轮）** —— 指标还在创新低。
2. recon 的最低点分别在 **ep1480 / ep1312 / ep1489**，末期仍在缓慢下降。
3. 早前发现第 0 层用码数在 seed 间波动可达 59~85（±26），说明 1500 轮的 ckpt 本身还没稳定。

**但必须分目标回答，结论相反**：

| 目标 | 1500 轮够吗 | 加轮有用吗 |
|---|---|---|
| 唯一性 ICR | ✅ 够 | ❌ 无用。瓶颈不在轮数：1500 轮 raw 就已有 0.9639（gate），5000 轮 0.9582 —— 且按 §5.13 的定版，**唯一性本来就不是要追的指标**（Snap：~70% 以上即平台期） |
| 重建保真 R² | ❌ 不够 | ✅ 有用，但边际递减（ep1200→1500 只动了 0.00001；5000 轮 gate +0.165） |
| 第 0 层码本利用率 | ❌ 不够 | 🔴 **加轮无效；要靠 init pass**：5000 轮实验里 init0 仍有 32.6% 死码，而 `--init_samples 8192` 直接 0%（§5.9b ②）。通用机制仍是 Q3 的 EMA / diversity loss / dead code reset |

**为什么加轮救不了塌缩**：塌缩是**单调的正反馈**（Q2 成因 1）——梯度只流向被选中的码，
赢者恒赢，训练越久 `max_usage` 只会继续涨。加步数改变的是"收敛程度"，不是"梯度富集"这个机制本身。
5000 轮实测 gate 死码 41.4%→32.6% 只是缓慢缓解，**真正归零靠的是把初始化质心铺满（init8192）**。

因此优先级建议：**先加机制（diversity loss + dead code reset），再谈加轮**。
若要加轮，直接对齐 MiniOneRec 的 5000（本地 ~70 min/模式，三模式 3.5 h），并同时打开
per-layer usage 日志确认没有加速塌缩。

---

### 5.11 k-means 初始化的真实规模 + 四种抗塌缩方法落地

#### Q6. 三层都有 k-means 初始化吗 —— TIGER 怎么说、我们是什么样

**TIGER 论文原文（§3.1，逐字）**：

> "As proposed in [40], to prevent RQ-VAE from a codebook collapse, where most of the input gets
> mapped to only a few codebook vectors, we use k-means clustering-based initialization for the
> codebook."
>
> "Specifically, we apply the k-means algorithm on the first training batch and use the centroids
> as initialization."

注意原文用的是**单数 "the codebook"**，且**没有一处写明"逐层"**。同节前面确实说过
"we chose to use a separate codebook of size K for each of the m levels"（每层有独立码本），
所以从上下文看初始化必然覆盖全部层 —— 但**严格说论文没有在层级粒度上明确限定**。
TIGER 官方未放出 RQ-VAE 实现，无法从源码核对。

**我们（以及 MiniOneRec / MQL4GRec 这条同源 fork）：三层都有，且是逐层残差 k-means。**
实测证据（`scripts/multimodal/_archive/probe_kmeans_init.py`，I&S/text，seed=0）：

```text
[before] 每层 initted = [False, False, False]
kmeans 被调用 3 次：
  L0: input=(2048, 32)  md5=e4b66af0cae1  ‖·‖=1.8775  mean|x|=0.005812
  L1: input=(2048, 32)  md5=8f83223d0a7e  ‖·‖=0.4776  mean|x|=0.001442
  L2: input=(2048, 32)  md5=8ba4f9563267  ‖·‖=0.3338  mean|x|=0.000998
三次输入互不相同 → 【逐层在残差上聚类】
[after ] 每层 initted = [True, True, True]
```

**代码链**（`rq.py:24-30` 每层都收到 `kmeans_init` → `vq.py:26` 每层独立 `initted=False`
→ `rq.py:45-47` 循环中 `residual = residual - x_res` 后由 `vq.py:67-68` 触发 init）。

⚠️ **更正上一版文档的说法**：之前写"三层看到的都是那 2048 个样本"，这个表述不准确。
**样本是同一批，但每层的输入张量完全不同** —— L1/L2 看到的是前层量化后的**残差**：

| 层 | k-means 输入 | 实测 Frobenius 范数 | 相对上层 |
|---|---|---|---|
| L0 | 原始 latent z | 1.8775 | — |
| L1 | z − q₀(z) | 0.4776 | **25.4%** |
| L2 | r₁ − q₁(r₁) | 0.3338 | 69.9% |

这条实测数字同时给了 Q2 成因 2 的量化支撑：**L0 吃掉绝大部分能量**，
它承担的重建损失最大 ⟹ 梯度最大 ⟹ 码本漂移最快 ⟹ 塌缩最狠（第 0 层死码 41%~65%，
而 L1/L2 全用满 256/256）。

#### Q6b. 🔴 k-means 初始化在当前实现下完全不可复现

`rq/models/layers.py:77` 是

```python
KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)   # 无 random_state
```

sklearn 1.9.1 下 `n_init='auto'`、`random_state=None`。实测（I&S/text `emb[:2048]`，float64）：

| 场景 | 两次质心的 max&#124;diff&#124; | 判定 |
|---|---|---|
| 同一输入跑两次（`random_state=None`，当前实现） | **1.605e-01** | 🔴 实质不同 |
| 不同 seed 之间（`random_state=0 vs 1`） | 1.692e-01 | （参照尺度） |
| 同一输入跑两次（`random_state=0`） | **6.939e-18** | ✅ 浮点精度内可复现 |
| 码本元素典型量级 | 1.661e-01 | — |

**当前两次初始化的差异是元素本身量级的 96.6%，等同于把骰子彻底重摇一遍。**
再叠加 `train_rqvae.py` 的 `DataLoader(shuffle=True)` —— **"第一个 batch"本身每次也不同**，
构成"数据源随机 + 算法随机"的**双重随机**。

**这是第 0 层用码数在 seed 间波动 59~85 的直接根因**（此前只能观察到现象，无法归因）。
它意味着：**此前任何单次运行的码本利用率 / ICR / R² 结论都不具有可复现性**，
包括 §5.9 三模式对比表 —— 那是三个单次 run 的横向读法。

**修复（零成本，一行）**：`layers.py:77` 改为
`KMeans(n_clusters=..., max_iter=..., random_state=0, n_init=10)`。
已落盘的 ckpt 不受代码改动影响，但**重训结果不再与历史 ckpt 对齐** —— 这是有意为之的取舍。

**修复后复测**（同代码同 seed 跑两次，比对各层 k-means 输入张量）：

| 层 | max&#124;diff&#124; | 相对误差 | 非零差异元素 |
|---|---|---|---|
| L0 | **0.000e+00** | 0 | 0 / 65,536 |
| L1 | 1.863e-09 | 1.8e-07 | 6,400 / 65,536 |
| L2 | 1.979e-09 | 3.1e-07 | 29,564 / 65,536 |

L0 **bit-level 完全一致**。L1/L2 的残留是 **CPU float32 矩阵乘的多线程归约噪声**
（相对误差 ~2e-7，远低于 sklearn 那 96.6% 的算法随机），对 argmin 判定几乎不可能翻转。
若需彻底 bit-level 复现，再设 `OMP_NUM_THREADS=1` 即可（会牺牲速度，一般不值得）。

**给后续实验的硬性要求**：从此**任何单次 run 的码本利用率 / ICR / R² 都必须标注 seed**，
跨结论的对比至少要在同一 seed 下做。特别是 §5.9 的三模式对比是三个单次 run，
修好 seed 之后应当在固定 seed 下重跑一轮以确认结论稳健。

（附带一条实测参考：首个 batch 2048 条 vs 全量 25,847 条，两者都聚出 256/256 非空簇，
但前者 cluster 大小中位数只有 7、后者 93 ⟹ 首 batch 质心是"过拟合到 2048 条"的，
对全量分布的代表性存疑。全量初始化仍是值得跑的消融。）

#### Q6c. 初始化样本量加大到多少有用 —— 结论（探针细节已归档）

**实现**：**不要**用调大训练 `--batch_size` 的方式（会改变训练动力学、显存占用，且破坏与
MiniOneRec bs=2048 的对齐）。正确做法是训练前一个**独立 init pass**：随机抽 N 条过 encoder
→ 逐层 k-means → 直接写入 `embedding.weight` 并把 `initted` 置 True，训练循环保持 bs=2048 不变。
成本约 9 秒（CPU），几乎免费。已实现为 `--init_samples N`（见 Q6d）。

**初始化时刻的质心质量**（探针 `scripts/multimodal/_archive/probe_kmeans_batch_size.py`，
seed=0 随机 encoder 的 latent 上测）：样本量 2048 → 8192 → 25,847 时，
inertia/N 与簇大小基尼系数**单调改善且边际递减**（基尼 0.344 → 0.176 → 0.128），
8192 拿走大部分收益。

⚠️ **但这个探针预测不了训练后的结局**：它只测初始化时刻的质心质量，而塌缩是训练期的正反馈
过程（Q2 成因 1）。真实裁决来自 5000 轮训练（§5.9b ②）：**init8192 最优（L0 死码 0%、LCP 222），
全量反而略输** —— 与探针"full 最好"的排序不同。**教训：初始化质量必须由真实训练验证。**

#### Q6d. init pass 已实现（`--init_samples`）+ 冒烟与 A/B 实测

**实现**（`rq/train_rqvae.py`）：新增 `--init_samples N`（默认 0 = 上游行为不变）。
`preinit_codebooks()`：RandomState(seed) 抽 N 条 → 过 encoder → 逐层
`init_emb(residual)` → argmin 量化 → 取残差（与训练期口径完全一致，`use_sk=False`）。
完成后各层 `initted=True`，训练首步不再触发 init_emb；init 信息写入
`train_metrics.json` 的 `kmeans_preinit`。编排层 `run_sid_exp.sh` 透传
`INIT_SAMPLES`（默认 0）。**训练 batch_size / 动力学完全不动。**

（实现期的冒烟与 10 轮 A/B 已删除：数字被 5000 轮定版实验覆盖，见 `docs/SID_PIPELINE.md` §2.4。
其中唯一保留下来的观察是「**塌缩谷**」：ep1 的低碰撞是 k-means init 假象，ep2~3 会崩到 ~1.0
再缓慢恢复，属 warmup 早期的固有过渡态，与 init 方式无关 —— 这条给 §5.6.2 的
`--burn_in_frac 0.1` 再补一层依据。）


#### Q7. 四种平衡码本利用率的方法（原理 + 我们的落地方式）

来源：arXiv 2602.07774 "Generative Reasoning Re-ranker" §2.3
*Techniques for Balanced Codebook Utilization*。

##### ① EMA 码本更新

```text
ẽ⁽ᵏ⁾ ← Optimizer(e⁽ᵏ⁾, ∇_{e⁽ᵏ⁾} L)        # 先算候选更新值
e⁽ᵏ⁾ ← γ·e⁽ᵏ⁾ + (1−γ)·ẽ⁽ᵏ⁾                 # γ ∈ [0.95, 0.99]
```

**为什么有用**：梯度更新容易被单个批次里的离群点一把推很远，而"赢者猛冲掉队去"正是靠这种大步位移发生的。
EMA 把每步压成小步并带上惯性，码本漂移变慢，掉队的码不容易被彻底甩开。
出处：VQ-VAE 原论文 van den Oord et al. 2017 附录里就用 EMA 更新字典。

**落地**：`rq/models/vq.py` 的 `self.embedding`（nn.Embedding）。
需要把它从 AdamW 的参数组里摘出来，`torch.no_grad()` 手工维护。
**代价**：几乎为零；风险是 γ 太大跟不上优化（γ=0.99 ⟹ 时间常数 100 步）。

##### ② Diversity loss

对每个码簿，令 `p_j = n_j / Σ_m n_m`（第 j 个码在当前 batch 的软分配占比，
原文用负距离 softmax 计算）：

```text
L_div = λ_div · Σ_k C_k · Σ_j p_{k,j}²
```

该式在 `p_j = 1/C` 时取最小值 ⟺ **把使用分布推向均匀**。

**为什么和 Sinkhorn 不同**：Sinkhorn 是**硬约束**（双随机矩阵，行列和强制 = 1/K、1/B），
Diversity loss 是**软约束**（有 λ 连续可调，允许保留一定的非均匀性以尊重数据本身的簇结构）。
理论上后者对语义聚类结构更友好，不会把不该分开的物品硬拆。

**落地**：`rq/models/rq.py:53` 把每层的 div loss 加进 `mean_losses`。
**风险**：λ 是敏感超参，**太大会破坏语义聚类 → LCP 掉**。建议从很小的值起扫
（λ ~ 0.01~0.1），并且**必须同时盯 LCP ratio 与 ICR**，只看 ICR 会被误导。

##### ③ Dead code reset

维护 `unused_count[k]`（某个码连续多少个 batch 未被选中），超过阈值 τ 就用当前 batch 里
**最难重建**的那个样本把它顶掉：

```text
i* = argmax_i ‖r_i − e_{z_i}‖²
e_k ← r_{i*}
```

**和前两种的关系**：EMA 和 diversity loss 是**预防**（让码不那么容易掉队）；
dead code reset 是**复活**（已经有码死了，把它救回来）。
它是三者里唯一直接针对"已死"状态的方法。

**落地**：训练循环每 N 步检查一次 usage，直接改 `embedding.weight.data`。
**风险**：重置引入跳变，建议配合小幅 warm-up 或在该步降低 lr；τ 太小会导致反复抖动。
同类经典：Jukebox (Dhariwal et al. 2020) 的 random restart。

##### ④ Random last levels（导出期）

```text
k_l = argmin_k ‖r⁽ˡ⁾ − e_k⁽ˡ⁾‖²    若 l ≤ L−M
k_l = Uniform(1, K)                  若 l > L−M
```

用最后 M 层的语义性换取唯一性。直觉：前几层承载粗粒度语义（商品品类），最后几层只做细粒度区分。

⚠️ **它和我们已在用的 Sinkhorn 是同一位置的竞争方案** —— 都是"牺牲末尾层的语义性换唯一性"。
差别在于 Sinkhorn 是 batch 内的**确定性最优分配**，仍贴合数据分布，所以保真度代价小得多：
实测我们 Sinkhorn(batch=64) 的 ΔMSE 只有 1.1e-5 ~ 2.2e-7，而 Uniform 随机赋值大概率明显更差。
**结论：不换，把 Sinkhorn 作为该方法更强的替代即可**（写论文时可作为对照引用）。

#### 落地优先级建议

| 顺序 | 方法 | 理由 |
|---|---|---|
| 0 | **先给训练日志加 per-layer usage 打印**（P-7） | 没有监控，任何抗塌缩消融都只能事后盲评 |
| 0' | **固定 seed 并重跑一次三模式**（P-9，已修 ++ random_state ++，重跑待做） | 不固定 seed 时利用率波动 ±26 个码，其余所有消融的结论都是噪声上的结论 |
| 1 | Dead code reset | 最对症（我们有 41%~65% 死码），改动局部、风险可控 |
| 2 | Diversity loss | 软约束，理论上对语义结构最友好；但要扫 λ |
| 3 | EMA 更新 | 几乎零成本，可与前两者叠加 |
| 4 | k-means 全量初始化 | 零成本改变，但需重新验证（可能与训练动力学交互） |
| — | Random last levels | ❌ 不采用，Sinkhorn 已是更优解 |

### 5.12 RQ-KMeans（MiniOneRec 原版代码）+ 同款 Sinkhorn 对照（2026-09-12）

为了回答「上游 MiniOneRec 的 FAISS-RQ 路线在我们数据上到底差在哪」，用**原版复刻代码**
（`rq/rqkmeans_faiss.py`：FAISS ResidualQuantizer，train_type=Train_default、max_beam_size=1、
3 层 × 256 码，直接量化 1024 维输入空间）+ 与 RQ-VAE **完全相同**的导出期 Sinkhorn
（碰撞物品 / 仅末层 / eps=0.003 / batch=64 / sk_iters=50 / 20 轮 patience=3）生成 SID。
实现：`rq/build_sid_rqkmeans.py`，对比表：`scripts/multimodal/compare_rqkmeans.py`
→ `results/sid_e5000/IandS/compare_rqkmeans.md`。同一 emb（gate_e60）、同一 eval_sid.py。

| 指标 | RQ-VAE(gate+init8192) raw→sk | RQ-KMeans(MiniOneRec) raw→sk |
|---|---|---|
| ICR | 0.9582 → 0.9997 | 0.8395 → 0.9996 |
| 最大冲突组 | 5 → 2 | 14 → 3 |
| 死码率 L0/L1/L2 | 0 / 0 / 0 | 0 / 0 / 0 |
| LCP(nn) | 0.9950 → 0.9383 | 0.8893 → 0.8240 |
| LCP ratio | 222.1 → 209.4 | 175.8 → 163.3 |
| prefix-1 内聚比 | 2.086 | 2.129 |
| 保真度 R² | 0.6530 → 0.6504（decoder 重建） | 0.4460 → 0.3797（直接量化） |
| Sinkhorn 改码比例 | 6.54% | **27.06%** |
| 训练/导出耗时 | 3,644 s | **6 s** |

结论（三条）：

1. **raw ICR 差距巨大（0.84 vs 0.96）**：beam=1 贪心残差量化在 1024 维高维空间里碰撞率高得多，
   这就是 TIGER/MQL4GRec 系都改用「低维 latent + 可学习码本 + 迭代训练」的动机——
   RQ-VAE 的 encoder 先把空间压到 32 维再量化，码本覆盖效率高一个量级。
2. **Sinkhorn 之后唯一性打平（0.9996 vs 0.9997），但语义结构 RQ-VAE 更好**：
   rqkmeans 要改写 27% 物品的码才能消完碰撞（RQ-VAE 只有 6.5%），LCP ratio 因此
   从 175.8 掉到 163.3；RQ-VAE 是 222.1→209.4。prefix-1 内聚比 rqkmeans 略胜
   （2.129 vs 2.086，k-means 质心天然均衡），但 LCP 是更全面的层次语义指标。
3. **耗时差 600 倍**：6 s vs 3,644 s。如果只看 ICR，rqkmeans 是"免费午餐"；
   但叠加 LCP 与保真度（R² 0.38 vs 0.65，口径差异见对比表注），RQ-VAE 的优势
   是系统性的。**维持决策：SID 默认 RQ-VAE(gate+init8192+5000 轮，导出 sid_raw)，
   rqkmeans 存档作上游锚点对照**；最终裁判留给下游 SFT（若 SFT 端到端指标
   rqkmeans 不落下风，则值得重估——毕竟训练成本可忽略）。

⚠️ 口径提醒：rqkmeans 的保真度是 emb 空间直接量化误差（无 decoder），RQ-VAE 是
decoder 重建，二者 R² 只能作参考性对照（都回答"SID 剩多少语义"），不可当同口径指标。

#### 为什么 Sinkhorn 消不完最后 0.08%？（实测诊断，`rq/diag_sinkhorn_residual.py`）

rqkmeans 早停在 ICR=0.9996（21 物品 / 10 组），RQ-VAE 侧 ~7 物品。对卡住组逐一解剖：

1. **根因 = 输入本身重复，不是 Sinkhorn 的锅**：全库 25,847 个物品里有 **30 个 embedding
   逐位相同**（重复商品/重复文本过了融合层）。卡住的 10 组里 **9 组（cos=1.0，d1 逐位相同）
   就是这些重复 embedding**——任何确定性分配规则（argmin、argmax-of-Sinkhorn 都一样）
   对相同输入必给相同码，数学上不可能分开。要分开只能靠打破对称性的机制：全局匹配
   （匈牙利/贪心占用）、随机 tie-break，或上游数据去重（正解）。
2. **次要机制 = softmax 饱和**：唯一一组残差不同的 (104,152,23)（组内 cos 0.72~0.82），
   次近码归一化距离差 0.42 ≫ eps=0.003 → exp(-d/eps) 下次近码质量 ≈ e^-140 ≈ 0，
   行是一根 one-hot，行列归一化只能等比缩放、搬不动质量 → argmax 恒定。**实测 iters
   50→200 即消掉这组**（行内的微小质量差异被反复迭代放大）。
3. **"调 eps / 加迭代"都救不了（实测扫描）**：eps ∈ {0.003..1.0} 最多消 1 组，且 MSE 代价
   +0.11~0.13（比正常消解的 ΔMSE 1e-5~1e-7 大 4~5 个量级）；eps≥0.3 反而 Q 趋均匀、
   argmax 退化、碰撞回升。iters 50→20000 同样只消 1 组。
4. **argmax(软计划) 本身不保证单射**：Sinkhorn 只产出近似双随机矩阵（Birkhoff 保证存在
   完美匹配，但那是"存在性"），对每行取 argmax 是贪心不是匹配——两行可以 argmax 同一列。
   大头（7304→21）能消掉是因为大 batch 里共享码被充分争抢、列归一化有压力差；
   剩下的硬骨头形成**确定性不动点**（同 batch 同结果）→ patience 早停。

工程含义：21/25847=0.08%（最大组仅 3）对 SFT 影响≈0；真要修，正解是**上游 embedding
去重**（30 个重复物品），或对残留组做一次"改判次近码"后处理（贪心占用，O(组大小)）。

##### 溯源：这 30 组重复 embedding 是什么（实测）

对 60 个逐位重复物品回查 `IandS.item.json` metadata：

* **是同样的商品**：30/30 组的 title+features+categories **逐字相同**、主图 URL 相同、
  但 **ASIN 不同**（9/30 组连价格都不同，如 idx1229 无价 vs idx3055 $78.99）。
  典型如 `VELCRO Brand Tape, Black 15ft` 两个 ASIN 同价同图——Amazon 目录里的
  **同品多 listing**（变体/重复上架），不是编码 bug。
* **不是"转码"产生**：文本层（title+features+categories → Qwen3）本身就有 40 组 82 个
  逐位重复（0.32%）——输入字符串逐字相同，编码器只是忠实输出。
* **融合层反而消掉了一部分**：文本层 40 组里有 **10 组被图像分支区分开**（同文不同图，
  mask=1 且 SigLIP 向量不同）；剩下 30 组**连图都一样**（主图 URL 相同 → 图像向量相同），
  融合 gate 对逐位相同的两路输入无能为力。
* 净效果：25,847 物品中 60 个（0.23%）是"文本+图像完全相同的重复商品"，其中 18 个
  正是 SID 碰撞消解消不掉的硬骨头（9 组）。

##### 孪生组为什么"该全碰撞却只剩几个"？——batch 上下文抽签（实测）

追问：30 组逐位相同 embedding 在确定性规则下应**全部**碰撞到底，为何 RQ-VAE sk 只剩
7 组、rqkmeans 剩 9 组？实测账本（对 30 组孪生逐一追踪）：

* **sid_raw 里确实是 30/30 全碰撞**（两套量化器都验证）——直觉正确。
* **Sinkhorn 循环能拆开孪生组，靠的是 batch 上下文不对称**：每轮把碰撞物品按 64 个
  重分桶。同一桶内，行分布相同的孪生物品经行列归一化后必然 argmax 同码；但落进
  **不同桶**时 [B,K] 上下文（同桶的其他碰撞物品）不同，列归一化压力不同，
  其中一个可能被推到次近码 → 拆开。拆开后即退出碰撞名单，结果固化。
  29/30 的孪生组索引差 ≥64，天然可能落进不同桶——这是一场**确定性但任意的抽签**。
* 抽签结果：RQ-VAE 拆开 23/30，rqkmeans 拆开 21/30。残留名单几乎不重叠
  （仅 (954,1284)、(2900,6489) 两组两边都卡）——证明是抽签不是规律。
* **残留 = 每轮都和孪生兄弟同桶的倒霉蛋**：碰撞名单缩到 <64 后变单桶，
  同桶孪生永远同码（不动点）→ patience 早停冻结。
* 精确账本：RQ-VAE sk 残留 14 物品/7 组**全部是孪生组**（非孪生 0 个）；
  rqkmeans 残留 21 物品/10 组 = 9 孪生 + 1 饱和组（(4035,4037,4040)，gap 0.42>>eps）。
  ⟹ "**残留碰撞 ≡ 重复 embedding**"严格成立（rqkmeans 差一个饱和特例）。
* RQ-VAE 拆得更多（23 vs 21）的原因：raw 碰撞基数小（1,067 vs 7,304 个物品）、
  32 维 latent 空间的 Sinkhorn 计划没那么饱和，跨桶翻转更容易。
* **语义 ID 的哲学边界**：重复商品在语义上"本来就该"同 SID，但生成式推荐训练协议要求
  item→SID 一一对应。两条路：(a) 数据侧按 title+features+categories 去重/合并 listing
  （会改 25,847 口径，且上游 MiniOneRec 也有同样问题，去重反而是对上游的改进）；
  (b) SID 侧对残留组强制改判次近码（打破对称，保真度代价 O(组大小) 极小）。

### 5.13 碰撞要不要消解：「语义桶」假说实测 + 定版决策（2026-09-13）

用户提议：RQ-VAE raw 最大冲突组才 5 个物品，生成式召回可以**不消解碰撞**——同码物品
本就是语义相似品，整桶召回后交给排序。实测检验（`gate__init8192` sid_raw），**结论：假说成立，
已定版为默认**（详见本节末尾的决策段，整理版见 `docs/SID_PIPELINE.md` §2.5）：

**账本**：965 组 / 2,046 物品（7.9%）参与碰撞，组大小 {2:872, 3:73, 4:17, 5:3}，
桶均 2.1 物品，最大 5。

**实拍 3 个最大组**（全部是"同品牌同系列、只差规格"）：

| 组（5 物品） | 内容 | 组内两两 cos | 组外最近邻 cos | 随机对基线 |
|---|---|---|---|---|
| (172,166,152) | Mr O-Ring 硅胶 O 圈 ×5（同 70A 硬度，只差内径外径） | 0.9456 | 0.9690 | 0.1814 |
| (179,149,164) | Fastenere 自攻螺丝 ×5（同盘头/驱动，只差 #4~#10、长度） | 0.9196 | 0.6158 | 0.1814 |
| (25,27,170) | CLUTCH 拉紧带 ×5（只差 16"/20"、2~4 pack） | 0.9359 | 0.9385 | 0.1814 |

→ **假说成立**：碰撞组 = 同品牌同系列的规格变体，推荐场景整桶返回完全合理。

**两个 nuance（诚实边界）**：

1. 碰撞组 ≠ "最相似集合"：碰撞物品的组外最近邻 100% 共享 prefix-1、但只有 31% 共享
   prefix-2；大组所在的 prefix-2 家族桶有 5~12 个成员（如 O 圈组 5 个散在 12 个的家族里）。
   **真实的语义家族边界在前缀层，比碰撞组大一圈**——同家族成员本来就散布在多个末码上。
2. Sinkhorn 拆开后前缀完全不变（如 (172,166,152) → (172,166,{13,30,127,152,209})），
   即**消解只影响末层的"家族内区分"，两种方案的家族结构完全一样**。差别仅是：
   raw 的末码 = argmin 顺序（谁先占谁），sk 的末码 = 均衡分配，都是任意的。

**方案对比**：

| | sk 消解（旧默认） | **raw 不消解（定版）** |
|---|---|---|
| 唯一性 | ICR 0.9997 | 0.9582（7.9% 物品在桶里，桶均 2.1） |
| LCP ratio | 209.4 | **222.1** |
| 保真度 R² | 0.6504 | **0.6530** |
| SFT 训练目标 | 每物品唯一 | 同一 SID 出现在 1~5 个物品的样本（= 学"SID↔商品族"，无标签冲突） |
| 推断侧 | SID→1 物品 | SID→整桶（≤5），beam/Trie 不受影响（同码同路径） |
| 先例 | TIGER/MQL4GRec/ETEGRec 均唯一化 | YouTube PLUM：code→桶→ranker |

**决策（2026-09-13 定版）：默认交付 `sid_raw`，不做碰撞消解。**

三条理由：

1. **碰撞组是语义簇不是噪声**（上表实拍）：整桶返回后交给排序正是工业界形态。
2. **消解不彻底且代价实打实**：sk 只能到 0.9997，残留 7 组**全部是逐位相同的重复 embedding**
   （确定性规则数学上不可分，见 §5.12）；代价是改写 6.54% 物品的码、LCP −12.7、R² −0.0026。
3. **唯一性本身不是硬指标（工业界证据）**：Snap《Semantic IDs for Recommender Systems at
   Snapchat》（SIGIR'26 Industry Track，arXiv 2604.03949；官方代码 = 本仓 `refs/GRID` 镜像的
   snap-research/GRID）—— ① Table 5：Amazon Beauty 上唯一性 92.95%（1024³）→ 70.58%（128³），
   GR Recall@10 仅 6.1 → 6.0，64³（65.40%）才掉到 5.8；原文 "once uniqueness surpasses a
   certain threshold (empirically 70%), this correlation plateaus"；② "Uniqueness should not
   be evaluated as a gold standard, but rather as a foundational sanity check against collapse"
   （§4.3 与 §5 两处）；③ Table 4 线上 A/B 用的就是**一码多品**：Top 10 SIDs 每码映射 100 个物品、
   relevance-guided 消歧 → view +0.57% / send +2.54% / share +4.39% / re-post +3.55%，
   随机映射只有 +0.13%。我们 raw 的 95.82% 远高于 70% 平台线，且 Snap 的
   relevance-guided 消歧正是我们下游排序阶段的角色。

🔴 **诚实边界**：本项目**不做** raw vs sk 的 SFT 端到端消融（时间与成本有限，且消解本身不彻底）。
上述是"离线结构指标 + 工业界先例"支撑的设计选择，**不是端到端验证过的结论**。
`sid_sk.npy` 仍随每次导出产出，随时可切回；若将来补做消融，评估口径需先定死
（一个桶占候选集几个名额，建议整桶展开与按桶计 Hit 两种都报）。

---

## 6. SID 导出：两版解耦

`rq/build_sid_dual.py` —— **同一个 ckpt 导出两版，互不覆盖**：

| 版本 | 量化方式 | 用途 |
|---|---|---|
| **`sid_raw`（默认交付）** | `get_indices(use_sk=False)` 纯 argmin | **保真度与语义结构最好**（LCP 222.1 / R² 0.6530）；碰撞 = 语义桶，交给下游排序消歧 |
| `sid_sk`（存档对照） | 对**冲突物品**施加最后一层 Sinkhorn，迭代至无碰撞 | 唯一性最高（ICR 0.9997）；改码 6.54%，LCP/R² 略降 |

产物（每个模式一个文件夹）：`sid_raw.npy/.json/.stats.json`、
`sid_sk.npy/.json/.stats.json`、`sid_stats.json`（两版对照）。

**为什么解耦**：之前 `rq/build_sid.py` 只导出消解后的最终版，原始量化结果被覆盖，
Sinkhorn 的代价无法在同一 ckpt 上量化。拆开后**同一权重、同一数据，唯一变量是"要不要 Sinkhorn"**。

训练期 `sk_epsilons` 强制全 0（见 `rq/train_rqvae.py`），保证导出的 `sid_raw`
就是训练时看到的 codes。

---

## 7. SID 质量评估：三件套（`rq/eval_sid.py`）

### ① uniqueness（唯一性）
- **ICR** = 唯一码数 / N；碰撞率 = 1 − ICR
- **dead_code_rate**（每层死码比例）、**per-layer entropy / ppl**（码本利用率）
- 最大冲突组大小

### ② fidelity（量化保真度）
用 ckpt 的 decoder 从 codes 重建输入，报告**两组**：
- **A) 实际交付**：保存的 codes（真正进 SFT 的那份）的逐层累积重建
- **B) 理论上限**：纯 argmin codes 的逐层累积重建
- **A − B = 碰撞消解的保真度代价**（`fidelity_cost_of_collision_resolution`）

报告字段：`cumulative[].recon_mse / r2 / cosine`（逐层）、
`final` / `final_argmin`、`codes_changed_by_collision_resolution`。

### ③ retrieval structure（检索结构）
- **LCP**（Longest Common Prefix）：语义近邻对 vs 随机对的 SID 前缀长度，
  输出 `nn_mean` / `random_mean` / `lcp_ratio` —— 衡量"相似物品是否共享前缀"
- **prefix cohesion**：同一前缀组内物品的向量相似度 vs 随机基线
- ⚠️ 深层前缀若组大小≈1（如本例 prefix2 均组大小 2.2、prefix3 均为 1.0），
  会被标记 **"样本不足/退化，不可信"** —— 别拿这个数字下结论

---

## 8. 开源配方对照（供 A/B）

| 项目 | 数据集 | tokenizer | 生成器 |
|---|---|---|---|
| **MiniOneRec**（本项目基座） | Amazon23 I&S | RQ-VAE 3×256，5000 轮(仓库默认) / 10000 轮(`rqvae.sh`)，bs 20480 | SFT ≤10ep（ES patience 1，lr 3e-4）+ GRPO 2ep |
| **ETEGRec**（SIGIR'25） | Amazon23 sci/inst/game | RQ-VAE 3×256 e_dim128，**10000 轮** bs1024 | `--cycle=2` 交替优化，`lr_id=1e-4` vs `lr_rec=5e-3`（**小 50 倍**） |
| **MQL4GRec**（ICLR'25） | Amazon18 | RQ-VAE **4×256 ×2**（文本/图像各一个），**`sk_epsilons 0 0 0 0.003`**，500ep bs2048 | pretrain 30ep + finetune 200ep |
| **GRID**（Snap, CIKM'25） | Amazon P5 | RQ-VAE **3000 步**，**Adagrad** lr1e-3 warmup1000，latent **64**，encoder `[768,256,128]→64`，beta 0.25，BN→L2 normalize | TIGER 320,000 步 |
| **MMGRec** | 多模态 | concat(视/听/文)→线性→拼 CF→**GCN**→RQ-VAE | R@10 0.1269 |

**两条路线的训练量差 300 倍**（这条只针对 tokenizer）：
- RQ-KMeans / 残差聚类（无梯度）：GRID 只跑 **30 步/层**
- RQ-VAE 端到端（有梯度）：**3,000 步 ~ 10,000 轮**

**本项目 = 第二行**（`rq/models/rqvae.py` 有完整 encoder/decoder + 梯度反传）。

**融合阶段学术界没有开源对照**：MQL4GRec 干脆不融合（双 SID）；
MMGRec 融合但塞进 RQ 一起训。我们"先融合成单向量、独立训好后交给 RQ"
是 **YouTube PLUM 路线，无先例** —— 这是空白点，也是潜在的创新点。

---

## 9. 当前已知问题（未修）

| # | 问题 | 状态 | 影响 / 修法 |
|---|---|---|---|
| P-1 | **第 0 层码本塌缩**：L0 只用 90~151/256 码（L1/L2 满 256），text 最惨（单码独吞 1385 个物品 = 均匀期望的 13.7×） | 🟡 未根治 | ⚠️ 早期"训练期 Sinkhorn 抗塌缩"结论**已作废**（自变量未生效，见 §5.6.1）。真正的修法是补机制：**EMA 码本更新(γ≈0.95) / diversity loss `λ·C·Σp²` / dead code reset(τ batch 未用则用最难重建样本重置)** —— arXiv 2602.07774 §2.3 五技术中的三条，我们一条都没有。**加轮数无效且可能更糟**（单调正反馈，见 §5.10 Q5） |
| P-2 | **选 ckpt 准则错误** | ✅ 已修 | 按碰撞率选 + `--burn_in_frac 0.1` 跳过 ep1 假象；统一产物 `ckpt/selected_model.pth`。**不要按 loss 选**（loss 是假信号，机理见 §5.6.2 与 §5.10 Q4） |
| P-3 | gate vs mlp 参数量不对等（2.27×） | 🔴 未修 | 机制性结论不可靠；mlp 加宽 hidden 对齐参数量，gate 加 LN+Dropout |
| P-4 | `temperature=0.07` 硬编码；训练只见到 511 个负样本 | 对比学习最敏感超参未调 | 加 `--temperature`；batch 提到 2048（显存几乎免费） |
| P-5 | 图像塔未做域适应 | SigLIP 在商品主图上分布有偏 | "冻结 vs 轻量域适应"对照 |
| P-6 | 编码器停在 0.6B（4GB 卡限制） | MTEB 70.70 vs 最佳 73.30 | 上云 3090 换 4B 重编码 |
| P-7 | ~~训练日志完全没有 per-layer usage 监控~~ | ✅ 已修 | `eval_collision` 顺带返回各层用码数（get_indices 本身返回逐层索引，零额外前向），eval_step 打印 `usage=[n0,n1,n2]` 并写入 `train_metrics.json` 的 `curve[*].layer_usage`（2026-09-12） |
| P-8 | **轮数 1500 vs MiniOneRec 5000** | ✅ 已跑（2026-09-12 五组实验） | R² 收益集中在 gate（0.475→0.640）；text 边际（+0.017）。**定版配方 = gate + init_samples 8192 + 5000 轮 + sid_raw**（L0 满用 256、LCP 222），训练期 Sinkhorn 被 init8192 支配、不采用。详见 §5.9b 与 `docs/SID_PIPELINE.md` |
| P-9 | **k-means 初始化不可复现**（`layers.py:77` 无 `random_state` + `shuffle=True`） | ✅ **已修** | 实测同一输入两次聚类质心 max&#124;diff&#124;=1.605e-01，而元素量级仅 1.661e-01（**96.6%**）⟹ 这是第 0 层用码数在 seed 间波动 59~85 的根因。已加 `random_state=0, n_init=10`，复测 L0 **bit-level 一致**（0 差异），L1/L2 残留 1.9e-9 系 CPU float32 多线程归约噪声（相对 2e-7，可忽略）。**残留待办：固定 seed 下把 mlp 也跑一遍 5000 轮**（text/gate 已在 seed=2024 下重跑） |
| P-10 | **重复 embedding 未在上游去重**：30 组 60 个物品（0.23%）文本+图像逐位相同（Amazon 同品多 listing） | 🔴 未修（有意保留） | 它们是 SID 残留碰撞的全部来源，也是评估口径的噪声源。去重会改 25,847 基准口径 → 与上游不可比，故暂不动；若决定去重，需同步重跑编码/融合/SID 全链路 |
| P-11 | **raw vs sk 无 SFT 端到端消融** | ⚠️ 已决定不做 | 语义桶为默认靠"离线结构指标 + Snap 论文先例"，**未经端到端验证**。将来若补做，先定死"一个桶占候选集几个名额"的评估口径。见 §5.13 |

---

## 10. 踩坑速查

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | 沙箱限速 ~110kB/s | 大文件下载慢 51 倍 | **>100MB 的下载必须在沙箱外执行** |
| 2 | HF 缓存符号链接失效 | snapshot 下出现 0 字节文件 | `scripts/tools/hf_repair_cache.py` 从 blobs 重建 |
| 3 | 批内 token 量过大 | 文本编码 2.6 item/s | 长度分桶 + token 预算装箱（`--token_budget 4096`）→ 101 item/s |
| 4 | 显存溢出到系统内存 | 速度衰减 10 倍 | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` + 周期 `empty_cache()` |
| 5 | 进度速率列假衰减 | 254 → 51 item/s | `rate = i/elapsed` 里的 `i` 含断点前历史。**做相邻行差分核对** |
| 6 | `os.remove` 被 safe-delete 钩子拦截 | 收尾崩溃导致 meta 没写、下游没跑 | **先写 meta，清理放最后并 `try/except` 包住** |
| 7 | 跨模态直接点积 | 1024 vs 768 维广播报错 | 跨空间点积**无定义**，先岭回归学线性映射 |
| 8 | CPU 单线程 BLAS 算协方差 | concat 的 PCA 220s | 走 GPU 特征分解 → **0.8s（275×）** |
| 9 | 训练/评估用同一份共现对 | 召回率随轮数单调虚高 | **切互斥留出对**；实测水分 **13.6 倍**（0.610 vs 0.045） |
| 10 | `rm -rf` 走 trash 很慢 | 删 100MB 要 10s | 用 `--out_dir` 换目录，别删 |
| 11 | 临时脚本放 `/tmp` | Git Bash 映射到不存在的 `D:\tmp` | 放 `C:/Users/z/AppData/Local/Temp/` |
| 12 | 🔴 **把别的项目实现串到本项目** | 断言"本项目用 FAISS k-means"是错的 | **凡写"本项目用什么"，落笔前必须 grep 本仓 import 核对** |

---

## 11. 文件索引

| 文件 | 作用 |
|---|---|
| `rq/train_rqvae.py` | RQ-VAE 训练（`--init_samples` init pass、`--train_use_sk` 开关、按碰撞率选 ckpt） |
| `rq/build_sid_dual.py` | 双版 SID 导出（**raw = 默认交付** / sk = 存档对照，导出 batch=64） |
| `rq/eval_sid.py` | SID 三件套评估（uniqueness / fidelity / retrieval structure） |
| `rq/build_sid_rqkmeans.py` | RQ-KMeans（MiniOneRec 原版 FAISS-RQ）+ 同款 Sinkhorn 的对照导出 |
| `rq/diag_sinkhorn_residual.py` | 残留碰撞诊断（eps / iters 双扫描 + 组内残差相似度） |
| `rq/models/rqvae.py` `rq.py` `vq.py` `layers.py` | RQ-VAE / 残差量化 / VQ / k-means+Sinkhorn |
| `scripts/multimodal/encode_text.py` `encode_image.py` | 两模态编码（断点续跑 + token 预算） |
| `scripts/multimodal/fuse_embeddings.py` | 四模式融合 + InfoNCE 训练 + R@k 评估 |
| `scripts/multimodal/run_sid_exp.sh` | SID 实验编排（模式 × 轮数 → 结果文件夹，透传 `INIT_SAMPLES`） |
| `scripts/multimodal/compare_sid_modes.py` `compare_rqkmeans.py` | 横向对比表生成（三模式 / RQ-VAE vs RQ-KMeans） |
| `scripts/multimodal/_archive/` | 已归档的一次性探针（k-means init / init 样本量），结论已并入本文 |
| `docs/SID_PIPELINE.md` | **SID 唯一入口**：参考方法综述 + 本项目实现与知识点 + 探索过程全记录 + 定版配方 |
| `docs/EXPERIMENT_LOG.md` | 实验流水账（含失败实验与踩坑） |
| `refs/GRID/` | Snap GRID 官方仓库镜像（27MB，已 gitignore）—— Snap SID 论文配套代码 |
