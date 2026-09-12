# 生成式推荐系统 — 完整技术文档

## 📋 项目概述

### 这个项目在做什么？

传统推荐系统（如协同过滤）用连续向量表示商品，然后计算用户向量和商品向量的相似度来推荐。本项目尝试一种全新思路：**把推荐当作"生成任务"**。

具体来说，给 LLM 一个用户的历史购买记录，让它"写出"用户下一个可能购买的商品。但 LLM 不能直接输出商品（商品数量太大），所以我们给每个商品分配一个简短的"编码"（称为 **语义 ID, SID**），让 LLM 学会生成这个编码。

**整个流程分三步：**
1. **给商品编码**（RQ-VAE 把商品变成 3 个 token 的编码）
2. **教 LLM 学编码**（SFT 监督微调）
3. **用奖励信号优化**（GRPO 强化学习）

### 为什么这么做？

| 传统推荐 | 生成式推荐（本项目） |
|----------|---------------------|
| 需要维护 embedding 矩阵 | LLM 自带世界知识 |
| 冷启动困难 | 可利用商品文本描述 |
| 泛化能力弱 | LLM 的语言理解能力强 |
| 推理需要额外模型 | 端到端生成即可 |

### 项目链接

- **论文**: [arXiv:2510.24431](https://arxiv.org/abs/2510.24431)
- **原始项目**: [GitHub - AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec)
- **本项目**: [GitHub - TangPeng7477/MiniOneRec_oneGPU_preject](https://github.com/TangPeng7477/MiniOneRec_oneGPU_preject)
- **官方权重**: [HuggingFace - kkknight/MiniOneRec](https://huggingface.co/kkknight/MiniOneRec)

---

## 🏗️ 技术架构

### 整体Pipeline

```
用户历史: [买过手机壳, 买过充电线, 买过贴膜]
                    │
                    ▼
        ┌─────────────────────┐
        │  商品语义 ID 编码    │  ← 离线阶段：RQ-VAE
        │  手机壳 → <a_5><b_12><c_88>
        │  充电线 → <a_5><b_12><c_33>
        │  贴膜   → <a_5><b_7><c_101>
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │  SFT 监督微调       │  ← 在线训练阶段 1
        │  输入: 用户历史 SID  │
        │  输出: 下一个商品 SID │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │  GRPO 强化学习      │  ← 在线训练阶段 2
        │  生成多个候选 → 奖励 │
        │  用奖励信号优化策略  │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │  约束解码推理       │  ← 推理阶段
        │  Beam Search 保证   │
        │  只生成合法 SID     │
        └─────────────────────┘
                    │
                    ▼
推荐结果: <a_5><b_12><c_33> → 对应商品"手机充电线"
```

### 数据流动全过程

```
原始数据 (CSV)                    语义 ID 文件              训练数据
┌──────────────┐    RQ-VAE    ┌──────────────┐    转换    ┌──────────────┐
│ user_id      │  ────────→   │ item_id      │  ────────→ │ prompt       │
│ item_id      │              │ semantic_id  │            │ "用户买了A,B │
│ item_title   │              │ <a_x><b_y>   │            │  下一个可能是?"│
│ timestamp    │              │ <c_z>        │            │ 答案: <SID>  │
└──────────────┘              └──────────────┘            └──────────────┘
```

---

## 1️⃣ 数据处理

### 1.1 原始数据格式

项目使用 Amazon 商品评论数据集（Amazon Reviews 2018）。经过预处理后，CSV 文件包含以下列：

```csv
user_id,history_item_title,item_title,history_item_id,item_id,history_item_sid,item_sid
A22,"['Gorilla Original Gorilla Glue, ...']",Grizzly G9849 Magnetic Base/Dial Indicator Combo,[117],118,['<a_165><b_107><c_44>'],<a_104><b_118><c_176>
```

| 列名 | 说明 | 示例 |
|------|------|------|
| `user_id` | 用户 ID | `A22` |
| `history_item_title` | 用户历史购买商品标题列表 | `['Gorilla Glue', 'Duct Tape']` |
| `item_title` | 目标商品标题 | `Magnetic Base/Dial Indicator` |
| `history_item_id` | 用户历史商品 ID 列表 | `[117, 130]` |
| `item_id` | 目标商品 ID | `118` |
| `history_item_sid` | 用户历史商品 SID 列表 | `['<a_165><b_107><c_44>']` |
| `item_sid` | 目标商品 SID | `<a_104><b_118><c_176>` |

每个样本是一条"用户历史 → 下一个商品"的训练对。train/valid/test 按时间切分：取用户最后 1 条交互作为 test，倒数第 2 条作为 valid，其余作为 train。

**数据预处理流程** (`data/amazon18_data_process.sh`):
1. **过滤**：只保留交互次数 ≥ 5 的用户和商品（去掉活跃度过低的数据）
2. **排序**：按时间排序每个用户的历史记录
3. **划分**：按时间切分 train/valid/test（最后 1 条作为 test）
4. **格式化**：转换为模型需要的 CSV 格式

### 1.2 预处理后的数据结构

```
data/Amazon/
├── train/
│   ├── Industrial_and_Scientific_5_2016-10-2018-11.csv    # 训练集
│   └── Office_Products_5_2016-10-2018-11.csv
├── valid/
│   ├── Industrial_and_Scientific_5_2016-10-2018-11.csv    # 验证集
│   └── Office_Products_5_2016-10-2018-11.csv
├── test/
│   ├── Industrial_and_Scientific_5_2016-10-2018-11.csv    # 测试集
│   └── Office_Products_5_2016-10-2018-11.csv
├── index/
│   ├── Industrial_and_Scientific.index.json  # SID 索引
│   ├── Industrial_and_Scientific.item.json   # 商品元信息
│   ├── Industrial_and_Scientific.emb-qwen-td.npy  # 商品 embedding
│   ├── Office_Products.index.json
│   ├── Office_Products.item.json
│   └── Office_Products.emb-qwen-td.npy
└── info/
    ├── Industrial_and_Scientific_5_2016-10-2018-11.txt    # 商品信息表
    └── Office_Products_5_2016-10-2018-11.txt
```

#### Industrial_and_Scientific 数据量

| 文件 | 行数/条数 | 说明 |
|------|----------|------|
| train CSV | 36,259 条（含 1 行表头，共 36,260 行） | 训练集样本 |
| valid CSV | 4,532 条 | 验证集样本 |
| test CSV | 4,533 条 | 测试集样本 |
| info TXT | 3,686 条 | 商品信息（去重后的商品数） |
| index.json | 3,686 个商品 | 商品 ID → SID token 映射 |
| item.json | 3,686 个商品 | 商品元信息（title, description） |

#### Office_Products 数据量

| 文件 | 行数/条数 | 说明 |
|------|----------|------|
| train CSV | 38,924 条 | 训练集样本 |
| valid CSV | 4,866 条 | 验证集样本 |
| test CSV | 4,866 条 | 测试集样本 |
| info TXT | 3,459 条 | 商品信息 |
| index.json | 3,459 个商品 | 商品 ID → SID token 映射 |
| item.json | 3,459 个商品 | 商品元信息 |

> **数据文件格式**: 文件名中的 `_5` 表示过滤阈值（用户/商品交互次数 ≥ 5），`2016-10-2018-11` 表示数据时间范围（2016年10月 ~ 2018年11月）。

**info 文件格式** (`semantic_id \t item_title \t item_id`)，每行一个商品：
```
<a_236><b_231><c_226>	SUPCO SPP6 Relay/Capacitor Hard Start Kit with 500% Increase Starting Torque	0
<a_42><b_80><c_160>	Stanley TRA708T Sharpshooter 1/2-Inch Leg Length Staples, Steel (1000 Count)	1
<a_42><b_194><c_177>	Stanley TRA708SST 1/2-Inch HD Stainless Steel Narrow Crown Staple	2
```

**index 文件格式** (`index.json`)，商品 ID → SID token 列表的映射：
```json
{
    "0": ["<a_236>", "<b_231>", "<c_226>"],
    "1": ["<a_42>", "<b_80>", "<c_160>"],
    "2": ["<a_42>", "<b_194>", "<c_177>"]
}
```

#### SID 的完整形状演变（从文本到 Tensor）

一个 SID 在项目不同阶段呈现不同的"形状"：

```
商品: "Wireless Mouse"

阶段 1: 原始 RQ-VAE 输出 (numpy array)
  形状: (3,) 
  值:   [236, 231, 226]
  类型: int32, 范围 [0, 255]

阶段 2: index.json (磁盘存储)
  形状: List[str] of length 3
  值:   ["<a_236>", "<b_231>", "<c_226>"]
  
阶段 3: info.txt (人类可读)
  形状: 单行文本，3个 token 拼接
  值:   <a_236><b_231><c_226>    Wireless Mouse    0

阶段 4: Tokenizer 编码后 (token IDs)
  添加 768 个 SID 特殊 token 后，每个 <a_N> 变成 1 个 token ID
  形状: (3,)  ← 3 个连续的 token ID
  值:   [152000, 152256, 152512]  (假设偏移量)
  
阶段 5: Embedding 矩阵查询后 (模型内部)
  形状: (3, 896)  ← 3 个 token, Qwen2.5-0.5B hidden_dim=896
  值:   [[-0.023, 0.142, ...], [0.051, -0.077, ...], ...]
  
阶段 6: Completion 文本 (推理输出)
  形状: 单行字符串
  值:   "<a_236><b_231><c_226>\n"
  如需 beam search decode: [batch, num_beams, seq_len] → 文本列表
```

**关键数值总结**：

| 属性 | 值 |
|------|-----|
| SID 层数 | 3 (L=3) |
| 每层 codebook 大小 | 256 (K=256) |
| 总商品容量 | 256³ = 16,777,216 |
| SID token 数（新增） | 3×256 = 768 |
| 每 SID 占用 token 数 | 3（不含 EOS）|
| 每 SID 占用 token 数 | 4（含 EOS）|
| max_completion_length | 16（padding 长度）|
| Embedding 维度 | 896（Qwen2.5-0.5B hidden_size）|
| 扩展后 vocab_size | 151,936 + 768 = 152,704 |

**在 LLM 中的实际形状变化**（以 batch=4 为例）：

```
训练/推理阶段:
  input_ids:     [4, seq_len]              ← token ID 序列
  attention_mask:[4, seq_len]              ← 1=有效, 0=padding
  labels:        [4, seq_len]              ← -100=忽略, else=target
  logits:        [4, seq_len, 152704]      ← 模型输出的概率分布
  loss:         scalar                     ← 聚合后的标量

SID 生成阶段 (RQ-VAE):
  embedding:      [3686, 32] 或 [batch, 32]   ← RQ-VAE 编码器输出
  indices:        [3686, 3]                   ← 每层选中的原型索引

Beam Search 推理:
  input_ids:     [batch, prompt_len]
  completions:   [batch, num_beams, completion_len]  ← beam 候选
  decoded:       List[List[str]]                      ← 解码后的文本
  candidates:    [batch, num_beams] 字符串           ← 用于奖励计算
```

#### Industrial_and_Scientific 完整统计（全部实验基于此数据集）

| 统计项 | 训练集 | 验证集 | 测试集 | 总计 |
|--------|--------|--------|--------|------|
| 用户数 | 7,034 | 393 | 267 | **7,694** |
| 商品数（作为目标） | 2,407 | 287 | 182 | **2,518** |
| 交互数（样本数） | 36,259 | 4,532 | 4,533 | **45,324** |
| 历史中出现商品数 | 2,427 | — | — | — |
| 总交互商品数 | — | — | — | **3,101** |
| info 文件商品数 | — | — | — | **3,686** |
| 平均历史长度 | 3.35 | — | — | — |
| 最大历史长度 | 14 | — | — | — |
| 数据稀疏度 | 0.21% | — | — | — |

**关键观察**：

1. **用户重叠度低**：7,694 总用户中，只有 524 人同时出现在 train/valid/test 三个集合。4,409 人仅在训练集中出现，说明 valid/test 大量包含未见过的用户（冷启动场景）。

2. **商品覆盖不全**：
   - 3,101 个商品在交互中出现（作为目标或历史）
   - 3,686 个商品在 info 文件中（全量商品库）
   - **585 个商品从未出现在任何交互中**——这些商品是纯粹的冷启动商品，即使模型训练完也无法推荐（除非利用商品文本描述做零样本推理）

3. **历史长度较短**：平均每个用户只有 3.35 条历史记录，最大 14 条。这是因为 5-core 过滤只保证每个用户至少有 5 条交互，其中 1 条作为 test、1 条作为 valid，留给训练的平均就只剩 3-4 条。

#### Office_Products 统计（辅助数据集）

| 统计项 | 训练集 |
|--------|--------|
| 用户数 | 7,510 |
| 商品数（作为目标） | 2,324 |
| 交互数 | 38,924 |
| info 文件商品数 | 3,459 |
| 数据稀疏度 | 0.22% |

#### 数据稀疏度计算

```
sparsity = interactions / (users × items)

Industrial_and_Scientific:  
  36,259 / (7,034 × 2,407) = 0.21%

Office_Products:  
  38,924 / (7,510 × 2,324) = 0.22%
```

#### 数据集大小讨论

**Q: Industrial_and_Scientific 只有 7,694 个用户、3,686 个商品、45,324 条交互，会不会太少？**

**A：在学术基准中属于中等规模，在工业场景中偏小。**

对比常用的推荐系统数据集：

| 数据集 | 用户数 | 商品数 | 交互数 | 稀疏度 |
|--------|--------|--------|--------|--------|
| **Industrial_and_Scientific（本项目）** | 7,694 | 3,101 | 45,324 | 0.21% |
| MovieLens-1M | 6,040 | 3,706 | 1M | 4.5% |
| MovieLens-20M | 138,000 | 27,000 | 20M | 0.53% |
| Amazon Beauty (5-core) | 22,363 | 12,101 | 198,502 | 0.07% |
| Amazon Sports (5-core) | 35,598 | 18,357 | 296,337 | 0.05% |
| Amazon Industrial (5-core) | 7,694 | 3,101 | 45,324 | 0.21% |
| Yelp (5-core) | 30,887 | 18,995 | 1.56M | 0.27% |
| Gowalla | 29,858 | 40,981 | 1.02M | 0.08% |

> Industrial_and_Scientific 是 Amazon 商品评论中的一个细分品类（工业与科学用品），天然比 Beauty、Sports 等大类数据量小。

**影响分析：**

| 方面 | 影响 |
|------|------|
| **模型容量限制** | 0.5B 的 LLM 对小数据集可能过拟合，需要在 SFT 阶段控制 epoch 数（3 epoch）和早停策略 |
| **统计显著性** | 测试集 4,533 条样本，HR/NDCG 指标的置信区间较宽；不同 run 之间的指标波动可能 > 1% |
| **冷启动挑战** | 验证/测试集中大量用户未出现在训练集中，模型必须学会泛化到未见用户 |
| **长序列建模** | 平均历史长度仅 3.35，无法充分发挥 LLM 的长上下文能力 |
| **指标易饱和** | 数据量小，HR@10 到 15% 以上后增长会越来越困难 |

**Q：这么小的数据集上做 RL（GRPO）有意义吗？**

有意义。实验结果已经验证：2 轮 RL 后 HR@10 从 9.3% → 10.9%（+17%），NDCG@10 从 6.0% → 7.4%（+23%）。即便在小数据集上，RL 仍然能显著提升推荐质量。但需要注意：

- GRPO 的 num_generations=4 在 3,686 个商品中搜索空间相对有限
- 如果数据集太小，RL 可能快速收敛到局部最优
- 本实验 2 轮 RL 后指标仍在上升（未饱和），说明 RL 还有潜力

**Q：迁移到别的数据集，泛用性如何？**

项目框架与具体数据集解耦，理论上可以迁移到任意推荐场景：

**需要重新做的**：
1. **RQ-VAE 重训**：新数据集的商品需要重新训练 RQ-VAE 生成 SID（如果商品文本特征差异大）
2. **Tokenizer 扩展**：新 SID 的 token 需要添加到 tokenizer
3. **Hash 字典重建**：约束解码依赖商品 → SID 映射，必须重新构建
4. **模型微调**：SFT + RL 需要在新数据上重新训练

**不需要改的**：
- 模型架构（Qwen2.5）、训练流程、损失函数、评估脚本均与数据集无关
- ConstrainedLogitsProcessor 等核心组件无需修改

**预期在不同规模数据集上的表现：**

| 数据集规模 | 预期效果 | 注意事项 |
|-----------|---------|---------|
| 类似（~5K items） | HR@10 ~8-15% | 与本项目接近，过拟合风险可控 |
| 中等（~50K items） | HR@10 ~1-5% | 搜索空间增大，需要更大模型或更多生成数 |
| 大规模（~1M items） | HR@10 <1% | beam search 很难在大空间中命中，需改进检索策略 |

**工业场景的实际限制**：

| 限制 | 说明 |
|------|------|
| **商品数上限** | RQ-VAE 三层 256 codebook 的组合空间 16M，理论上支持千万级商品 |
| **推理效率** | 生成式推荐需要 beam search，比 embedding 检索慢几个数量级 |
| **Hash 字典内存** | 百万级商品的 hash_dict 约几百 MB，内存可接受 |
| **增量更新** | 新商品需要增量编码 SID 并更新 hash_dict，不适合频繁更新的场景 |
| **延迟要求** | LLM 生成延迟远高于传统推荐，工业部署需大量 GPU |

### 1.4 数据集类详解

**源码位置**: `data.py`

项目实现了 10+ 种数据集类，分为 SFT 和 RL 两大类：

#### SFT 数据集

| 数据集类 | 输入 | 输出 | 用途 |
|---------|------|------|------|
| `SidSFTDataset` | 用户 SID 历史序列 | 下一个商品 SID | 核心序列推荐 |
| `SidItemFeatDataset` | SID | 商品标题 | SID-特征对齐 |
| `FusionSeqRecDataset` | SID 历史 + 商品标题 | 下一个 SID | 融合推荐 |
| `TitleHistory2SidSFTDataset` | 标题格式的历史 | 下一个 SID | 自然语言对齐 |
| `UserPreference2sidSFTDataset` | 用户偏好描述 | 推荐 SID | Thinking 模拟 |

**SidSFTDataset 示例**:
```python
{
    "prompt": "The user has interacted with items <a_5><b_12><c_88>, <a_5><b_12><c_33> in chronological order.\nCan you predict the next possible item that the user may expect?",
    "completion": "<a_5><b_7><c_101>"
}
```

#### RL 数据集

| 数据集类 | 输入 | 输出 | 用途 |
|---------|------|------|------|
| `SidDataset` | 用户 SID 历史序列 | 目标 SID | GRPO 训练 |
| `RLTitle2SidDataset` | 商品标题 | 目标 SID | 标题→SID 映射 |
| `RLSeqTitle2SidDataset` | 标题格式历史序列 | 目标 SID | 序列标题→SID |

RL 数据集额外维护两个字典：
- `prompt2history`: prompt → 用户历史 key
- `history2target`: 历史 key → 目标商品 SID

这两个字典用于奖励函数计算——模型生成候选后，通过 prompt 查找历史，再查找目标，判断是否命中。

---

## 2️⃣ 语义 ID 构建 (SID Construction)

### 2.1 为什么需要语义 ID？

商品数量可能有百万级，让 LLM 直接从百万个商品中选一个不现实。我们需要一种编码方式：
- **离散的**（LLM 擅长生成离散 token）
- **短的**（每个商品只需几个 token）
- **有语义结构的**（相似商品编码相似）

**解决方案**：用 RQ-VAE（残差量化变分自编码器）把商品的文本描述压缩成 3 个 token 的编码。

### 2.2 RQ-VAE 原理详解

**源码位置**: `rq/rqvae.py`, `rq/models/rqvae.py`

#### 直觉理解

想象你在给商品"画像"：

1. **第一层**：粗分类。所有商品先分成 256 个大类（如"电子产品"、"办公用品"、"运动器材"）
2. **第二层**：细分类。每个大类内再分 256 个小类（如"电子产品"下的"手机配件"、"电脑配件"）
3. **第三层**：更细分类。每个小类内再分 256 个具体品类

最终一个商品用 `<a_类别1><b_类别2><c_类别3>` 三个 token 表示。

#### 数学过程

令 $e \in \mathbb{R}^d$ 为商品文本通过 Qwen 编码器得到的 embedding 向量。RQ-VAE 将 $e$ 逐层残差量化到三个 codebook 中：

$$
\begin{aligned}
\text{Layer 1:}\quad & c_1 = \arg\min_{k \in [K]} \|e - \mathbf{e}^{(1)}_k\|_2, \quad q_1 = \mathbf{e}^{(1)}_{c_1}, \quad r_1 = e - q_1 \\[4pt]
\text{Layer 2:}\quad & c_2 = \arg\min_{k \in [K]} \|r_1 - \mathbf{e}^{(2)}_k\|_2, \quad q_2 = \mathbf{e}^{(2)}_{c_2}, \quad r_2 = r_1 - q_2 \\[4pt]
\text{Layer 3:}\quad & c_3 = \arg\min_{k \in [K]} \|r_2 - \mathbf{e}^{(3)}_k\|_2, \quad q_3 = \mathbf{e}^{(3)}_{c_3}
\end{aligned}
$$

其中 $\mathbf{e}^{(l)}_k \in \mathbb{R}^d$ 是第 $l$ 层 codebook 中第 $k$ 个原型向量，$K=256$。最终 SID 为三层索引的组合：$\text{SID} = [c_1, c_2, c_3]$。

**残差机制的核心**：每一层都在前一层**没有编码好的部分**（残差）上做量化，因此 $q_1 + q_2 + q_3$ 逐层逼近原始 $e$，保证信息损失最小化。

#### RQ-VAE 损失函数

RQ-VAE 的优化目标包含三项损失：

$$
\mathcal{L}_{\text{RQ-VAE}} = \underbrace{\|D(q_1 + q_2 + q_3) - e\|^2}_{\text{重建损失}} + \beta \cdot \big( \underbrace{\|e - \text{sg}[q_1]\|^2}_{\text{承诺损失}} + \underbrace{\|\text{sg}[e] - q_1\|^2}_{\text{量化损失}} \big)
$$

其中 $\text{sg}[\cdot]$ 表示 stop-gradient 操作（`detach()`），$\beta=0.25$ 为承诺损失权重。

| 损失项 | 作用 | 影响对象 |
|--------|------|----------|
| 重建损失 $\mathcal{L}_{recon}$ | 确保从离散编码重建后的 embedding 接近原始 | 编码器 + 解码器 + codebook |
| 承诺损失 $\mathcal{L}_{commit}$ | 强制 encoder 输出靠近选中的原型 | 编码器参数 |
| 量化损失 $\mathcal{L}_{quant}$ | 拉近原型向量到 encoder 输出 | codebook 原型向量 |

#### 核心参数

```python
num_emb_list=[256, 256, 256]    # 三层量化，每层 256 个 codebook
e_dim=32                        # codebook embedding 维度
encoder_dims=[2048, 1024, 512]  # 编码器逐层降维
decoder_dims=[256, 128, 64]     # 解码器逐层升维
beta=0.25                       # 承诺损失权重
lr=1e-3                         # 学习率
epochs=10000                    # 训练轮数
batch_size=20480                # 批次大小
kmeans_init=True                # 用 k-means 初始化 codebook（防止 collapse）
kmeans_iters=100                # k-means 初始化的迭代步数
loss_type="mse"                 # 重建损失类型（均方误差）
quant_loss_weight=1.0           # 量化损失权重
```

> **k-means 初始化（Warm-start Trick）**：RQ-VAE 训练前，先用 k-means 聚类第一个 batch 的 embedding，用聚类质心初始化 codebook。这防止了随机初始化导致的 codebook collapse（大量原型从未被使用），是 RQ-VAE 训练稳定性的关键技巧。

#### RQ-VAE 代码实现（四层嵌套结构）

四个文件由外到内层层调用，形成一个完整的 RQ-VAE 训练管线：

**文件层级：**
```
rq/rqvae.py          ← 入口：数据加载 + trainer 调用
  └─ rq/models/rqvae.py  ← RQVAE 类：encoder + RVQ + decoder
      └─ rq/models/rq.py ← ResidualVectorQuantizer：三层残差量化循环
          └─ rq/models/vq.py  ← VectorQuantizer：单层 VQ + codebook + Sinkhorn
              └─ rq/models/layers.py  ← MLP + Sinkhorn 算法实现
```

**① RQVAE — 顶层封装（`rq/models/rqvae.py`）：**

```python
class RQVAE(nn.Module):
    def __init__(self, in_dim=768, num_emb_list=[256,256,256], e_dim=32,
                 layers=[2048,1024,512,256,128,64], ...):
        # 编码器: 768 → 2048 → ... → 64 → 32（逐层降维）
        self.encoder = MLPLayers([in_dim] + layers + [e_dim])
        # 残差量化器: 3层codebook，每层256个32维原型
        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, ...)
        # 解码器: 32 → 64 → 128 → ... → 768（编码器的逆结构）
        self.decoder = MLPLayers(self.encode_layer_dims[::-1])

    def forward(self, x):
        x = self.encoder(x)                  # [B, 768] → [B, 32]
        x_q, rq_loss, indices = self.rq(x)    # RVQ 量化 → [B, 3] 索引
        out = self.decoder(x_q)               # [B, 32] → [B, 768] 重建
        return out, rq_loss, indices

    def compute_loss(self, out, quant_loss, xs):
        loss_recon = F.mse_loss(out, xs)      # 重建损失
        return loss_recon + self.quant_loss_weight * quant_loss
```

**② ResidualVectorQuantizer — 残差量化核心（`rq/models/rq.py`）：**

```python
class ResidualVectorQuantizer(nn.Module):
    """逐层残差量化: 当前层量化剩余残差，下一层量化新的残差"""
    def __init__(self, n_e_list, e_dim, sk_epsilons, ...):
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(256, 32, sk_epsilon=sk_epsilons[i], ...)
            for i in range(3)
        ])

    def forward(self, x):
        x_q = 0
        residual = x
        for quantizer in self.vq_layers:           # 逐层量化
            x_res, loss, indices = quantizer(residual)  # 当前层选取最近原型
            residual = residual - x_res             # 残差 = 输入 - 已量化部分
            x_q = x_q + x_res                       # 累加重建
        return x_q, mean_losses, all_indices        # all_indices: [B, 3]
```

**逐层量化过程可视化：**
```
原始向量 e = [0.5, -0.3, 0.8, 0.1, ...]  (768d)

第1层: 量化 → 找到最近原型 q₁  ─→ 残差 r₁ = e - q₁ = [0.1, -0.1, ...]
第2层: 量化残差 r₁ → 找到 q₂    ─→ 残差 r₂ = r₁ - q₂
第3层: 量化残差 r₂ → 找到 q₃

重建 = q₁ + q₂ + q₃ ≈ e         ← 逐层逼近原始向量
索引 = [<a_5>, <b_12>, <c_88>]  ← 最终 SID
```

**③ VectorQuantizer — 单层 VQ（`rq/models/vq.py`）：**

```python
class VectorQuantizer(nn.Module):
    def __init__(self, n_e=256, e_dim=32, ...):
        self.embedding = nn.Embedding(256, 32)   # codebook: [256, 32]

    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)

        # ① 如果尚未初始化，用 k-means 初始化 codebook
        if not self.initted and self.training:
            self.init_emb(latent)

        # ② 计算 L2 距离矩阵
        d = ||latent||² + ||codebook||² - 2×latent@codebookᵀ  # [B, 256]

        # ③ 选择原型：argmin（最近邻）或 Sinkhorn（均匀约束）
        if use_sk and self.sk_epsilon > 0:
            d = self.center_distance_for_constraint(d)    # 距离归一化
            Q = sinkhorn_algorithm(d, epsilon, iters)     # [B, 256] 概率矩阵
            indices = Q.argmax(dim=-1)                     # 概率最大的原型
        else:
            indices = d.argmin(dim=-1)                     # 普通最近邻

        x_q = self.embedding(indices).view(x.shape)       # 查表重建

        # ④ 损失计算
        commitment_loss = F.mse_loss(x_q.detach(), x)      # β×||x_q - sg(x)||²
        codebook_loss = F.mse_loss(x_q, x.detach())        # ||sg(x_q) - x||²
        loss = codebook_loss + self.beta * commitment_loss

        x_q = x + (x_q - x).detach()                       # 直通估计器(STE)
        return x_q, loss, indices
```

> **直通估计器（Straight-Through Estimator, STE）**：`x + (x_q - x).detach()` 在前向时 = x_q，反向时梯度 = 1（等价于跳过 argmin 不可微操作），使重建损失的梯度能流入 encoder。

**④ Sinkhorn 算法实现（`rq/models/layers.py`）：**

```python
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    """
    epsilon → 0:  退化为 argmin（纯语义，碰撞不控制）
    epsilon → ∞:  退化为纯均匀分配（无碰撞，但语义被破坏）
    """
    Q = torch.exp(-distances / epsilon)     # 距离 → 相似度（熵正则化）
    Q /= Q.sum()                            # 归一化到总和为1

    for _ in range(sinkhorn_iterations):
        # 行归一化：每个商品分配到各原型的概率和 = 1
        Q /= Q.sum(dim=1, keepdim=True);  Q /= Q.shape[0]
        # 列归一化：每个原型被分配到的商品总权重 = 1/K
        Q /= Q.sum(dim=0, keepdim=True);  Q /= Q.shape[1]

    Q *= Q.shape[0]                         # 恢复尺度使每行和=1
    return Q                                 # [B, 256] 分配矩阵
```

**训练入口（`rq/rqvae.py`）：**

```python
# ① 加载 embedding
data = EmbDataset(args.data_path)          # [3686, 768] numpy array

# ② 构建 RQVAE 模型
model = RQVAE(in_dim=data.dim,
              num_emb_list=[256,256,256],  # 三层 codebook
              e_dim=32,                    # codebook 向量维度
              layers=[2048,1024,512,256,128,64],  # 编码器 MLP 结构
              kmeans_init=True,            # k-means 初始化
              sk_epsilons=[0.0,0.0,0.0])   # Sinkhorn 强度（默认不开）

# ③ 训练
data_loader = DataLoader(data, batch_size=2048, shuffle=True)
trainer = Trainer(args, model, len(data_loader))
trainer.fit(data_loader)
```

**数据管线总结：**
```
item.json (title+description)
  → amazon_text2emb.py (Qwen编码+mean pooling)
  → .emb-qwen-td.npy [3686, 768]
  → EmbDataset → DataLoader
  → RQVAE.forward → [B, 3] indices
  → generate_indices.py → 商品→SID 映射表
```

### 2.3 码本坍塌（Codebook Collapse）与碰撞（Collision）

这两个问题是向量量化中最核心的挑战，项目中有一整套工程方案来解决它们。

#### 码本坍塌（Codebook Collapse）

**问题定义**：训练过程中，部分 codebook 向量被所有商品"冷落"，从未被选中作为最近邻。这些"死掉的"原型浪费了表示容量。

```
健康 codebook: [c₀, c₁, c₂, ..., c₂₅₅]  ← 256 个原型全部被使用
坍塌 codebook: [c₀, c₁, -, ..., c₂₅₅]    ← c₂ 从未被选中（死掉了）
```

**为什么 VQ-VAE 的 codebook collapse 更严重？**
- 单层 VQ-VAE 的 codebook 通常较大（如 4096），嵌入维度也更高
- 随机初始化时，大量原型落在数据分布的稀疏区域，永远无人问津
- 训练早期一旦某个原型被"冷落"，后续梯度无法传给它，形成死亡螺旋

**本项目 RQ-VAE 的应对策略**：

| 策略 | 实现位置 | 原理 |
|------|---------|------|
| **k-means 初始化** | `rqvae.py` → `kmeans_init=True` | 用 k-means 聚类第一个 batch 的数据，用聚类质心初始化 codebook。保证每个原型初始就在数据密集区，不会变成"孤岛" |
| **承诺损失（Commitment Loss）** | `models/rqvae.py` | `loss = ||e - q1.detach()||²`，强制 encoder 的输出靠近选中的原型。如果 encoder 的输出偏离原型太远，梯度会把它拉回来 |
| **量化损失（Quant Loss）** | `models/rqvae.py` | `loss = ||e.detach() - q1||²`，同时让原型靠近 encoder 输出。双向奔赴，防止原型脱离数据分布 |
| **每层 256 小 codebook** | 架构设计 | 每层只学 256 个原型，远小于商品数（3,686），确保每个原型都有足够的分配机会 |

**效果**：三项组合拳配合工作，实际训练中 codebook collapse 几乎不会发生。

#### 碰撞（Collision）

**问题定义**：两个不同的商品被分配到完全相同的 SID（三层的索引都一样）。碰撞导致 hash_dict 冲突——一个 SID 对应多个商品，模型不知道该推荐哪一个。

```
商品 A: "Wireless Mouse"   → SID: <a_5><b_12><c_88>
商品 B: "USB-C Cable"      → SID: <a_5><b_12><c_88>  ← 碰撞！两个商品同一个 SID
```

**碰撞来源**：
1. **语义上相似的商品**：商品 A 和商品 B 的 embedding 非常接近，残差量化后每层都选了同一个索引
2. **训练不充分**：RQ-VAE 尚未收敛时，codebook 分布不合理，部分区域过度聚集
3. **数据中的重复商品**：info 文件中的不同 ID 可能对应同一款商品的不同变体（如不同颜色）

**解决方案分训练期和生成期两个阶段**：

---

##### 阶段 1：训练期 — 监控碰撞率

训练过程中每个 eval_step 都会计算当前模型的碰撞率（`trainer.py`）：

```python
@torch.no_grad()
def _valid_epoch(self, valid_data):
    self.model.eval()
    indices_set = set()
    num_sample = 0
    for data in valid_data_loader:
        indices = self.model.get_indices(data)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = "-".join([str(int(_)) for _ in index])
            indices_set.add(code)
    
    # 碰撞率 = (总商品数 - 唯一 SID 数) / 总商品数
    collision_rate = (num_sample - len(indices_set)) / num_sample
    return collision_rate
```

**碰撞率公式**：

$$
\text{Collision Rate} = 1 - \frac{|\text{Unique SIDs}|}{N_{\text{items}}} = \frac{N_{\text{items}} - |\bigcup_i \text{SID}(i)|}{N_{\text{items}}}
$$

其中 $N_{\text{items}}$ 为商品总数，$|\bigcup_i \text{SID}(i)|$ 为所有商品分配到的不同 SID 数量。碰撞率为 0 表示每个商品都有唯一的 SID。

训练器同时保存两个最佳 checkpoint：
- **`best_loss_model.pth`**：重建损失最小的模型
- **`best_collision_model.pth`**：碰撞率最低的模型

生成 SID 时优先使用 `best_collision_model.pth`。

##### 阶段 2：生成期 — 迭代消除碰撞

SID 生成阶段（`generate_indices.py`）采用**迭代重编码**策略彻底消除碰撞：

```python
# 1. 先用最优模型为所有商品生成初始 SID
indices = model.get_indices(all_data, use_sk=False)

# 2. 检测碰撞：找出所有 SID 相同的商品组
collision_groups = get_collision_item(all_indices_str)

# 3. 迭代消除：对碰撞商品启用 Sinkhorn 约束重编码
while not collision_free and tt < 20:
    for collision_group in collision_groups:
        # 对这些冲突的商品启用 Sinkhorn 统一映射
        new_indices = model.get_indices(data, use_sk=True)
        # 更新 SID
        update_indices(collision_group, new_indices)
    
    # 重新检测碰撞
    collision_free = check_collision(all_indices_str)
    tt += 1
```

**Sinkhorn 约束**的核心作用：`sk_epsilon > 0` 时，RQ-VAE 的码本分配会倾向于均匀分布，强制冲突商品分配到不同的原型。

```python
# 不同层使用不同的 Sinkhorn 强度
for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon = 0.0                    # 前两层不用约束（保留语义）
model.rq.vq_layers[-1].sk_epsilon = 0.003 # 第三层用弱约束（微调分配）
```

**为什么第三层就够了？** 碰撞通常发生在最细粒度的分类层，前两层控制大类分配，第三层做最终消歧。保留前两层的自由分配可以维持语义结构。

##### 完整消除流程

```
训练 RQ-VAE → 选 best_collision_model.pth → 生成初始 SID
                                                │
                                                ▼
                                        检测碰撞
                                        /      \
                                    有碰撞    无碰撞 → 输出 index.json
                                      │
                                      ▼
                              对碰撞组启用 Sinkhorn 重编码
                                      │
                                      ▼
                                    重新检测
                                      │
                                   循环 ≤ 20 次
```

**效果**：
- 本项目的 3,686 个商品经过迭代消除后，**碰撞率降至 0%**
- 即使有剩余碰撞（如商品变体），hash_dict 也会在推理阶段通过 ConstrainedLogitsProcessor 做最后一层保障

#### 两个问题的关系

```
码本坍塌 + 碰撞 → 都源于 codebook 质量差
     ↓                    ↓
原型未被使用          多个商品抢同一个原型
     ↓                    ↓
k-means init        Sinkhorn 重编码 + 监控
commitment loss     迭代消除
     ↓                    ↓
   ✅ 解决             ✅ 解决
```

**核心经验**：codebook collapse 通过训练阶段的初始化 + 损失函数预防；collision 通过生成阶段的迭代重编码事后消除。两者互补，共同保证 SID 的质量。

### 2.4 Sinkhorn 算法原理

Sinkhorn 算法是解决碰撞问题的核心工具。它本质上是一种**最优传输（Optimal Transport, OT）**算法，用于在满足"均匀分配"约束的前提下，找到商品到 codebook 原型的最优匹配。

#### 直觉理解

把 Sinkhorn 想象成一个"交通调度问题"：

```
问题: 有 3686 个商品（人）要分配到 256 个原型（公交车）上
约束: 每辆车载 N/256 ≈ 14 人（均匀分配）
目标: 让每个人坐离自己最近的车（最小化总距离）
方案: Sinkhorn 算法调度
```

普通最近邻分配（`argmin`）只关心"每个商品离哪个原型最近"，不考虑整体均衡。结果可能是某些原型分配了几百个商品，某些原型一个都没有——这就是碰撞的根源。Sinkhorn 在"距离最近"和"均匀分布"两个目标之间做权衡。

#### 数学原理

Sinkhorn 求解带熵正则化的最优传输（Optimal Transport）问题：

给定商品嵌入 $\{x_i\}_{i=1}^N \subset \mathbb{R}^d$ 和原型向量 $\{c_j\}_{j=1}^K \subset \mathbb{R}^d$，代价矩阵 $D_{ij} = \|x_i - c_j\|^2$。

**标准最优传输**：寻找传输矩阵 $P \in \mathbb{R}^{N \times K}_+$ 最小化总运输代价：

$$
\min_{P} \sum_{i=1}^{N}\sum_{j=1}^{K} P_{ij} D_{ij}
\quad \text{s.t.} \quad
\sum_{j} P_{ij} = a_i,\; \sum_{i} P_{ij} = b_j
$$

其中 $a_i = 1/N$（每个商品等权），$b_j = 1/K$（每个原型均匀分配）。

**熵正则化（Sinkhorn 的核心创新）**：直接求解上述问题复杂度为 $O(N^3)$，Sinkhorn 加入熵正则项使问题变为严格凸且可高效迭代求解：

$$
\min_{P} \sum_{i,j} P_{ij} D_{ij} - \varepsilon H(P), \quad
H(P) = -\sum_{i,j} P_{ij} \log P_{ij}
$$

其中 $\varepsilon$（即 `sk_epsilon`）是正则化强度，$H(P)$ 是传输矩阵的熵。该问题的最优解具有形式 $P_{ij}^* = u_i v_j e^{-D_{ij}/\varepsilon}$，其中 $u, v$ 通过交替归一化求解。

**熵正则化（ε）的作用**：

| ε 取值 | 效果 | 场景 |
|--------|------|------|
| ε → 0 | 退化为普通最近邻分配（argmin），允许极端不平衡 | 无约束量化 |
| ε 适中 | 在距离和均匀性之间平衡 | 碰撞消除 |
| ε → ∞ | 完全忽略距离，纯均匀分配 | 过强约束，破坏语义 |

`sk_epsilon` 就是这里的 ε。项目中设为 0.003——一个很小的值，表示在最近邻的基础上做轻微的均匀性调整。

#### 迭代过程

Sinkhorn 通过简单的**交替归一化**迭代求解，这也是它被选中的原因——实现简单、收敛快、完全可微：

**Sinkhorn 迭代**：

$$
\begin{aligned}
P^{(0)} &= e^{-D / \varepsilon} & &\text{用距离和 } \varepsilon \text{ 初始化传输矩阵} \\
P^{(2k+1)} &= \frac{P^{(2k)}}{\sum_j P^{(2k)}_{ij}} \cdot a_i & &\text{行归一化（满足商品约束）} \\
P^{(2k+2)} &= \frac{P^{(2k+1)}}{\sum_i P^{(2k+1)}_{ij}} \cdot b_j & &\text{列归一化（满足均匀约束）}
\end{aligned}
$$

迭代约 50 步后收敛。

**为什么这个迭代有效？**
- **行归一化**确保每个商品的权重总和为 $1/N$
- **列归一化**确保每个原型分配的权重恰好为 $1/K$
- 交替进行，逐渐收敛到同时满足两个约束的解（固定点迭代）
- 这等价于在 **KL 散度**下投影到两个凸集交替进行（因此也叫 **Sinkhorn-Knopp 算法**）

**收敛性质**：收敛速度为线性（与迭代次数 $t$ 成 $O(e^{-t})$ 关系），远快于通用优化算法。

#### 从传输矩阵到离散分配

Sinkhorn 的输出 P[i] 是商品 i 分配到各原型的概率分布。需要转为确定的索引：

```python
# 源码 rqkmeans_faiss.py:148-171
P = ot.sinkhorn(a, b, D_full, tau)  # Sinkhorn 迭代得到传输矩阵

# 概率→离散分配（带容量约束）
remaining = capacities.copy()       # 每个原型的剩余容量
for i in order:                     # 逐个商品分配
    probs = P[i]                    # 商品 i 分配到各原型的概率
    cand = argsort(-probs)          # 按概率降序排列
    for c in cand:                  # 从最可能到最不可能尝试
        if remaining[c] > 0:       # 如果原型 c 还有容量
            assign[i] = c          # 分配
            remaining[c] -= 1
            break
```

**用容量约束替代纯概率采样**的原因：概率采样可能导致某些原型超载。显式跟踪剩余容量保证每个原型正好分配到 N/K 个商品。

#### 在项目中的实际应用

项目中 Sinkhorn 在两个地方使用：

**1. RQ-VAE 生成 SID 阶段** (`generate_indices.py`)

```python
# 前两层 sk_epsilon = 0.0 (不约束，保留语义层级)
model.rq.vq_layers[0].sk_epsilon = 0.0  
model.rq.vq_layers[1].sk_epsilon = 0.0

# 第三层 sk_epsilon = 0.003 (微调，解决最后一层碰撞)
model.rq.vq_layers[-1].sk_epsilon = 0.003
```

前两层不加约束是因为第一层划分大类、第二层划分中类，这两层的语义结构比均匀更重要。只在最细粒度的第三层做均匀性调整。

**2. RQ-Kmeans 后处理** (`rqkmeans_faiss.py`)

```
普通 RQ-Kmeans 输出 → Sinkhorn 均匀映射 → 均衡后的 SID
```

对 FAISS RQ-Kmeans 的三层分配都做 Sinkhorn 再平衡：

```python
for l in range(M):  # 对每一层
    residuals = compute_residuals_upto_level(data, codes, upto_level=l)
    new_ids = sinkhorn_balance_level(residuals, codebooks[l])
    codes[:, l] = new_ids
```

#### Sinkhorn vs 其他均衡方法

| 方法 | 原理 | 效果 | 复杂度 |
|------|------|------|--------|
| **argmin（无约束）** | 每个商品选最近原型 | 不均匀，碰撞高 | O(1) |
| **Sinkhorn（本项目）** | 最优传输 + 熵正则化 | 均衡且保持语义 | O(iter×N×K) |
| **k-means-constrained** | 硬约束最小/最大簇大小 | 严格均衡，但可能破坏语义 | 高 |
| **随机打乱** | 碰撞后随机重分配 | 不保持语义 | O(1) |

Sinkhorn 的独特优势：通过 ε 控制"均匀性"和"语义保持"的trade-off，不像硬约束方法那样一刀切。

### 2.5 RQ-VAE vs VQ-VAE vs RQ-Kmeans

项目实际使用了 **RQ-VAE**（Residual Quantized VAE）作为 SID 构建方法。项目中还提供了 RQ-Kmeans 系列方法作为对比。下面详细对比这三种技术路线。

#### 核心差异速览

| 方面 | VQ-VAE | RQ-VAE（本项目选用） | RQ-Kmeans（FAISS） |
|------|--------|---------------------|-------------------|
| **量化层数** | 单层（1 个 codebook） | 多层残差（3 层） | 多层残差（3 层） |
| **表示空间** | K（如 256） | K^L（256³=16M） | K^L（256³=16M） |
| **训练方式** | 端到端梯度下降 | 端到端梯度下降 | 无梯度，FAISS 聚类 |
| **编码器** | 可学习神经网络 | 可学习神经网络 | 无编码器（直接聚类） |
| **碰撞率** | 高（空间小） | 低（空间大） | 中（依赖初始化） |
| **实现复杂度** | 低 | 高 | 低 |
| **训练速度** | 快 | 中 | 极快 |

#### VQ-VAE 详解

VQ-VAE（Vector Quantized VAE）是 RQ-VAE 的前身和基础。其核心思想是：

```python
# VQ-VAE 单层量化
输入: 商品 embedding e (512维)
步骤:
  1. codebook C = [c_0, c_1, ..., c_255]  (1 个 codebook，256 个原型)
  2. 在 codebook 中找到最近邻: q = argmin ||e - c_i||
  3. 用 q 重建原始 embedding

# 表示能力
每个商品只能被分配到 256 个类别之一
最多表示 256 种商品 → 远远不够
```

**VQ-VAE 的核心问题**：
1. **表示空间太小**：单个 256 的 codebook 只能区分 256 个商品，而推荐系统有数万到数百万商品
2. **扩展方式有限**：增大 codebook 到 65536 会导致训练不稳定、codebook collapse（大量原型未被使用）
3. **无法表达层次语义**：所有商品在一个平面内分类，无法表达"大类→小类→具体商品"的层次关系

**扩展 VQ-VAE 的常见做法**：使用多个独立的 VQ-VAE（如多个 codebook 并联），但不同 codebook 之间缺乏交互，无法形成统一的层次化编码。

#### RQ-VAE 详解（为什么它是更好的选择）

**RQ-VAE 的改进：用"残差"代替"平面"**

```python
# RQ-VAE 三层残差量化
输入: 商品 embedding e (512维)

Layer 1: 粗粒度分类
  q1 = argmin ||e - c1_i||           # 从 256 个大类中选一个
  r1 = e - q1                         # 计算残差

Layer 2: 中粒度分类（在残差上做）
  q2 = argmin ||r1 - c2_j||          # 在 r1 基础上细分
  r2 = r1 - q2                        # 计算残差

Layer 3: 细粒度分类
  q3 = argmin ||r2 - c3_k||          # 最细粒度

最终 SID = [q1_idx, q2_idx, q3_idx]  # 如 <a_5><b_12><c_88>
```

**RQ-VAE 相对于 VQ-VAE 的优势**：

| 优势 | 说明 | 数值对比 |
|------|------|----------|
| **指数级更大的表示空间** | 每层 256 个原型，3 层组合 | VQ: 256 | RQ: 256³=16M |
| **层次化语义结构** | 第一层粗分类，逐层细化 | 同类商品共享前缀，如 `<a_42><b_*>...` |
| **更低的碰撞率** | 超大空间天然避免碰撞 | 实际碰撞率可 < 1% |
| **更好的重建质量** | 残差逐层补偿，重建误差更低 | 每层都有机会修正前层的误差 |
| **训练稳定性** | 每层只需学习局部映射，比大 codebook 更稳定 | 256 原型/层 vs 65536 原型单层 |

#### RQ-Kmeans 系列详解

**源码位置**: `rq/rqkmeans_faiss.py`, `rq/rqkmeans_constrained.py`, `rq/rqkmeans_plus.py`

RQ-Kmeans 系列是 RQ-VAE 的非学习替代方案，使用 FAISS 的 ResidualQuantizer 直接对 embedding 做聚类。相比 RQ-VAE，它们各有权衡：

**1. RQ-Kmeans (FAISS)**

```python
# rqkmeans_faiss.py 核心逻辑
rq = faiss.ResidualQuantizer(d, num_levels=3, nbits=8)  # 3层, 256/层
rq.train(data)   # 直接在 embedding 上跑 K-means
codes = rq.compute_codes(data)  # 得到 SID
```

| 优点 | 缺点 |
|------|------|
| 训练极快（几分钟 vs RQ-VAE 几小时） | 没有编码器，无法对未见过的商品编码 |
| 无需 GPU | 碰撞率通常高于 RQ-VAE |
| FAISS 高度优化，内存效率高 | 无法端到端优化，重建质量受限于聚类目标 |

**2. Constrained RQ-Kmeans**

在 RQ-Kmeans 基础上增加平衡约束：

```python
# 主要改进：Sinkhorn 均匀映射
def sinkhorn_balance_level(residuals, centroids):
    """用 Sinkhorn 算法让每个 prototype 分配到的商品数尽量均匀"""
    # 目标: 每个 centroid 分配 N/K 个商品
    # 方法: 最优传输 (Optimal Transport)
    P = ot.sinkhorn(a, b, cost_matrix, tau)  # Sinkhorn 迭代
    return optimal_assignment(P)
```

| 优点 | 缺点 |
|------|------|
| SID 分布更均匀，避免某些 SID 过度集中 | 强约束可能导致次优的语义划分 |
| 降低碰撞率 | 实现更复杂，需要调 Sinkhorn 参数 |
| 提高 codebook 利用率 | 计算量比普通 RQ-Kmeans 大 |

**3. RQ-Kmeans+**

两阶段策略：先粗聚类再在每个簇内细聚类，最后对冲突项增加额外层去重。

| 优点 | 缺点 |
|------|------|
| 碰撞率进一步降低 | 多阶段流程，工程复杂度高 |
| 层次结构更清晰 | 仍然没有编码器，无法泛化到新商品 |
| 灵活性高于普通 RQ-Kmeans | 去重过程可能破坏语义结构 |

#### 为什么选择 RQ-VAE？

综合权衡后选择 RQ-VAE，主要考虑：

| 考虑因素 | RQ-VAE | RQ-Kmeans |
|---------|--------|-----------|
| **端到端学习** | ✅ 编码器解码器联合优化，重建质量更好 | ❌ 只有聚类，缺乏对下游任务的自适应 |
| **新商品编码** | ✅ 编码器可直接推理新商品 SID | ❌ 新商品需要重新跑聚类，或训练映射网络 |
| **碰撞率** | ✅ 通常更低（端到端优化避免碰撞） | ⚠️ 依赖初始化，中等 |
| **与 LLM 的兼容性** | ✅ 经过 SFT 后，模型更容易学习 SID 规律 | ⚠️ 聚类边界可能不符合 LLM 的学习偏好 |
| **训练开销** | ⚠️ 需要 GPU 训练数小时 | ✅ FAISS 几分钟完成 |
| **成熟度** | 已被 TIGER、OneRec 等 generative recommendation 验证 | 项目探索性方案 |

**总结决策逻辑**：

```
为什么选 RQ-VAE 而不是 VQ-VAE？
→ VQ-VAE 单层 256 空间太小，无法区分 3686 个商品（碰撞率极高）
→ 增大 codebook 到 4096 会导致训练不稳定和 codebook collapse
→ RQ-VAE 用 3 层残差获得 16M 空间，同时保持每层 256 的稳定性

为什么选 RQ-VAE 而不是 RQ-Kmeans？
→ RQ-Kmeans 没有编码器，无法编码新商品（需要重新聚类）
→ RQ-VAE 端到端训练，编码器能理解"什么是好的 SID"
→ LLM 微调阶段，RQ-VAE 的编码规律更容易被 LLM 捕捉
→ 虽然训练更慢，但这是离线阶段，不影响在线推理效率

为什么不两者都用、选效果更好的？
→ 项目实际提供了两种方案的完整代码，可在同一数据集上对比
→ 本项目选取 RQ-VAE 作为主方案，RQ-Kmeans 作为备选参考
```

### 2.4 Hash 约束机制

**源码位置**: `evaluate.py`, `minionerec_trainer.py`

**问题**：LLM 可能生成不存在的 SID（如 `<a_999><b_0><c_0>`），导致推荐失败。

**解决方案**：构建前缀→后继的映射字典，约束每步只能生成合法 token。

```python
# 以 SID <a_5><b_12><c_88> 为例，构建约束字典：
hash_dict = {
    "prompt_prefix":   [<a_0>, <a_1>, ..., <a_255>],  # 第一层可选
    "<a_5>":           [<b_0>, <b_1>, ..., <b_255>],  # 以<a_5>开头的商品的第二层
    "<a_5><b_12>":     [<c_33>, <c_88>, ...],         # 以<a_5><b_12>开头的商品的第三层
    "<a_5><b_12><c_88>": [<eos>],                     # 完整 SID 后只能接结束符
}
```

**ConstrainedLogitsProcessor** 在 beam search 的每一步：
1. 查找当前已生成 token 序列对应的后缀
2. 在 hash_dict 中查找允许的下一个 token
3. 将不允许的 token 的 logit 设为 `-inf`
4. 这样 beam search 只会在合法 token 中搜索

**效果**：解码有效率从无约束的 ~70% 提升至 100%。

#### 一定会生成完整的 3 层 SID 吗？

**是的，hash 约束强制模型必须生成完整的 3 个 SID token + EOS，无法提前终止。**

hash_dict 的构造方式决定了 EOS token 只在完整 SID 之后才被允许：

```
以 SID <a_5><b_12><c_88> 为例，token 化后为 [a5_id, b12_id, c88_id, eos_id]：

hash("")                 → {a5_id}          ← 第一层，只能选 <a_N>
hash([a5_id])            → {b12_id}         ← 第二层，只能选 <b_N>
hash([a5_id, b12_id])    → {c88_id}         ← 第三层，只能选 <c_N>
hash([a5_id, b12_id, c88_id]) → {eos_id}    ← 完整后，只能选 EOS
```

**ConstrainedLogitsProcessor** 在每步解码时，将不允许的 token 的 logit 设为 `-inf`。这导致：

- **第 1 步**：EOS 不在合法集合中 → 被 mask → 模型必须输出 `<a_N>`
- **第 2 步**：EOS 仍不在合法集合中 → 被 mask → 模型必须输出 `<b_N>`
- **第 3 步**：EOS 仍不在合法集合中 → 被 mask → 模型必须输出 `<c_N>`
- **第 4 步**：只有 EOS 在合法集合中 → 模型必须输出 EOS

因此，**任何通过 hash 约束生成的序列都是完整的 3-token SID**，不存在只生成前 1 或 2 个 token 就停止的情况。

---

## 3️⃣ 监督微调 (SFT)

SFT（Supervised Fine-Tuning）是让 LLM 学会推荐的第一步。核心目标不是训练模型"生成语言"，而是训练模型**在推荐场景下生成正确的 SID 序列**。

### 3.1 为什么需要 SFT？

经过 tokenizer 扩展后，LLM 的词汇表新增了 SID token（`<a_0>`~`<a_255>` 等），但模型完全不知道这些 token 的含义。SFT 要解决三个问题：

1. **认识 SID**：让模型知道 `<a_5><b_12><c_88>` 代表一个具体的商品
2. **理解序列**：让模型学会从用户历史 SID 序列中推断下一个 SID
3. **多能力融合**：让模型同时具备序列推荐、SID-标题映射、自然语言理解的能力

这三个目标对应三个不同的训练任务，通过 **ConcatDataset** 混合在一起联合训练。

### 3.2 三个训练任务详解

#### 任务 1: 序列推荐 — SidSFTDataset

**源码位置**: `data.py:397`

**目标**：给定用户历史 SID 序列，预测下一个商品 SID。

**Prompt 模板**：
```text
Below is an instruction that describes a task, paired with an input that provides
further context. Write a response that appropriately completes the request.

### Instruction:
Can you predict the next possible item that the user may expect?

### Input:
The user has interacted with items <a_5><b_12><c_88>, <a_5><b_12><c_33>
in chronological order. Can you predict the next possible item that the
user may expect?

### Response:
<a_5><b_7><c_101>
```

**数据来源**：直接从训练 CSV 中取 `history_item_sid` 作为输入，`item_sid` 作为输出。

**Label 构造**（关键）：
```python
# data.py:453-457  SidSFTDataset.pre()
tokens = instruction_tokens + prompt_tokens      # 完整 prompt 部分
golden_tokens = tokenizer.encode(target_item)    # 目标 SID

input_prompt_len = len(tokens)
tokens = tokens + golden_tokens                  # 拼接 prompt + target

# mask 掉 prompt 部分，只对 target 计算 loss
labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
```

> `-100` 是 PyTorch CrossEntropyLoss 的 `ignore_index` 默认值——标签为 `-100` 的位置不参与梯度计算。

---

#### 任务 2: SID-标题双向映射 — SidItemFeatDataset

**源码位置**: `data.py:678`

**目标**：让模型理解 SID 和商品标题之间的对应关系。

为什么要这个任务？因为任务 1 只让模型看到了 SID 序列，模型可能只是在"死记硬背"SID 的共现模式，而不知道每个 SID 背后对应什么商品。任务 2 提供了语义锚点。

**双向训练**：

```python
# 构造两类样本，数量各半
# 类型 A: SID → 标题（sid2title）
{
    'task': 'sid2title',
    'input': '<a_5><b_12><c_88>',
    'output': 'Wireless Mouse'
}

# 类型 B: 标题 → SID（title2sid）
{
    'task': 'title2sid',
    'input': 'Wireless Mouse',
    'output': '<a_5><b_12><c_88>'
}
```

**Prompt 模板**：
```
# sid2title 任务
### User Input:
What is the title of item "<a_5><b_12><c_88>"?

### Response:
Wireless Mouse

# title2sid 任务
### User Input:
Which item has the title: Wireless Mouse?

### Response:
<a_5><b_12><c_88>
```

**这个任务的意义**：像"翻译"任务一样，强制模型建立 SID token 序列和自然语言词序列之间的跨模态对应关系。没有这个任务，模型可能只会机械地复制 SID 模式。

---

#### 任务 3: 融合推荐 — FusionSeqRecDataset

**源码位置**: `data.py:1126`

**目标**：输入 SID 历史，输出下一个商品的**标题**而非 SID。

```python
# 输入用 SID（离散 token），输出用自然语言（标题文本）
### User Input:
The user has sequentially interacted with items <a_5><b_12><c_88>,
<a_5><b_12><c_33>. Can you recommend the next item for him?
Tell me the title of the item

### Response:
Screen Protector for iPhone 15
```

**为什么输出标题而不是 SID？**
- 强制模型将 SID 序列"解码"为自然语言
- 激活 LLM 预训练阶段积累的语言知识
- 让 LLM 的"世界知识"（知道 Screen Protector 是一种手机配件）参与推荐决策

**数据清洗**：对 description 字段做了多层后备处理（`_process_description()`）：
1. 如果 description 非空，取最长的 description
2. 如果为空，用 title 替代
3. 如果 description 是列表，选最长的元素

---

#### 三种任务的对比总结

| 维度 | 任务1: 序列推荐 | 任务2: SID对齐 | 任务3: 融合推荐 |
|------|----------------|----------------|----------------|
| **输入** | SID 历史 | SID 或标题 | SID 历史 |
| **输出** | 下一个 SID | 标题 或 SID | 商品标题 |
| **核心能力** | 序列模式识别 | 跨模态对齐 | 语言理解 + 推荐 |
| **是否依赖 item.json** | ❌ | ✅ | ✅ |
| **数据量** | ~36K (全部样本) | ~7.4K (SID-标题对×2) | ~36K (全部样本) |

### 3.3 损失函数详解

三个任务共享同一个损失函数：**带标签掩码的因果语言模型交叉熵损失**。

#### 数学定义


给定输入序列 $\mathbf{x} = [x_1, x_2, \ldots, x_N]$，其中 $x_1, \ldots, x_P$ 为 prompt 部分，$x_{P+1}, \ldots, x_N$ 为 completion 部分。

**带标签掩码的因果语言模型损失**：

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{N-P}\sum_{t=P}^{N-1} \log P_\theta(x_{t+1} \mid x_1, \ldots, x_t)
$$

其中 $\theta$ 为模型参数，$P_\theta(x_{t+1} \mid x_{<t+1})$ 是模型在位置 $t$ 预测下一个 token $x_{t+1}$ 的概率。prompt 位置 $t < P$ 不参与损失计算（等价于 mask 为 0）。

**等价形式**（用 mask 表示）：

$$
\mathcal{L}_{\text{SFT}} = -\frac{\sum_{t=1}^{N-1} \mathbb{1}[t \geq P] \cdot \log P_\theta(x_{t+1} \mid x_1, \ldots, x_t)}{\sum_{t=1}^{N-1} \mathbb{1}[t \geq P]}
$$

其中 $\mathbb{1}[t \geq P]$ 是指示函数——当 $t \geq P$（即在 completion 区域）时为 1，否则为 0。



#### 代码中的精确实现

**Step 1: 构造 labels（每个数据集类的 pre() 方法）**

以 SidSFTDataset 为例：

```python
# data.py:440-457
def pre(self, idx):
    # 1. 编码 instruction + prompt → tokens
    instruction = """Below is an instruction that describes a task, ...
### Instruction:
Can you predict the next possible item that the user may expect?
"""
    tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
    
    history = self.get_history(self.data.iloc[idx])
    prompt = self.generate_prompt(history)
    tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
    
    # 2. 编码目标输出（completion）
    target_item = history['output']
    golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
    
    # 3. 拼接 prompt + completion
    input_prompt_len = len(tokens)
    tokens = tokens + golden_tokens
    
    # 4. 构造 labels: prompt 部分设 -100，completion 部分保留真实 token id
    labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
    
    return {
        "input_ids": tokens,
        "attention_mask": [1] * len(tokens),
        "labels": labels,           # ← 关键：prompt 位置被 mask
    }
```

**可视化一个实际样本**：

```
位置:   0   1   2  ...  20   21   22   23   24   25   26
tokens: [BOS] [inst] ... [prompt] [<a] [_5] [>] [<b] ... [EOS]
labels: -100 -100  ...  -100      [<a] [_5] [>] [<b] ... [EOS]
                            ↕ 只这些位置参与 loss 计算
```

**Step 2: HuggingFace Trainer 内部计算**

```python
# transformers 内部逻辑（非显式调用，Trainer 自动处理）
# 输入: input_ids=[batch, seq_len], labels=[batch, seq_len]

# 前向传播: 得到每个位置的 logits
outputs = model(input_ids=input_ids)
logits = outputs.logits  # [batch, seq_len, vocab_size]

# Shift: 每个位置预测下一个 token
shift_logits = logits[..., :-1, :].contiguous()    # [batch, seq_len-1, vocab]
shift_labels = labels[..., 1:].contiguous()         # [batch, seq_len-1]

# 展平后计算交叉熵（-100 的位置自动忽略）
loss = CrossEntropyLoss(shift_logits.view(-1, vocab_size), 
                         shift_labels.view(-1))
```

#### 为什么只对 completion 部分计算 loss？

这是 SFT 最关键的设计决策，原因有三：

**1. 因果语言模型的训练方式**

LLM 是因果（causal）模型，每个位置只能看到它之前的 token。对 prompt 部分计算 loss 等价于让模型"预测用户输入"，这没有意义——用户输入是给定的，不是模型生成的。

**2. 避免模型学会"偷懒"**

如果不 mask prompt，模型可能学会：**"不管输入是什么，我只需要输出固定模式"**。因为 prompt 部分已经占用了大量 loss budget，模型只要把 prompt 的 loss 降到很低，整体 loss 就很好看了——但对 completion 的生成能力可能完全没有提升。

**3. 训练-推理一致性**

推理时，模型接收 prompt，只生成 completion。训练时也只对 completion 计算 loss，保证了训练目标和推理目标一致。

#### 三个任务的 loss 权重

三个任务通过 `ConcatDataset` 简单地合并：

```python
# sft.py:204-214
train_data1 = SidSFTDataset(...)       # ~36,259 条
train_data2 = SidItemFeatDataset(...)  # ~7,400 条  (sid2title + title2sid)
train_data3 = FusionSeqRecDataset(...) # ~36,259 条
train_data = ConcatDataset(train_datasets)
```

**每个 batch 从三个数据集中均匀采样**。因此：
- 任务 1 和任务 3 各占约 45%（数据量大，采样概率高）
- 任务 2 占约 10%（数据量小，采样概率低）

这不是加权策略——数据量天然决定了任务 1 和 3 是训练主体，任务 2 是辅助。

### 3.4 为什么这样设计？

#### 三个任务的设计逻辑链

```
核心问题: LLM 如何做推荐？
    │
    ▼
需要让 LLM 输出 SID → 但 LLM 不认识 SID
    │
    ▼
任务1: 序列推荐 ─── 让模型学会"给定 SID 历史 → 输出 SID"
    │                   但模型可能只是死记硬背 SID 共现模式
    ▼
任务2: SID对齐 ──── 让模型真正理解每个 SID 对应什么商品
    │                   但模型仍然是在"SID 的世界"里运作
    ▼
任务3: 融合推荐 ─── 让模型把 SID 序列映射到自然语言
                        激活 LLM 的预训练知识参与推荐
```

**如果只做任务 1**：
- 模型本质上在学习一个**序列复制/模式匹配**任务
- 对未见过的 SID 组合泛化能力弱
- 无法利用 LLM 强大的语言理解能力

**加上任务 2 后**：
- 模型建立了 SID ↔ 自然语言的桥梁
- 能回答"SID 对应什么商品"和"这个商品的 SID 是什么"
- 相当于给 SID 加上了语义标签

**加上任务 3 后**：
- 模型必须将 SID 序列"解码"为有意义的文本
- 这迫使模型调用预训练阶段积累的知识
- 例如：看到 `<a_5><b_7><c_101>` 的历史，结合预训练知识"手机需要贴膜"，做出更好的推荐

#### 为什么用同一个损失函数处理三个差异很大的任务？

统一的因果 LM 损失 + 标签掩码机制天然支持多任务学习：

| 任务 | prompt（被 mask） | completion（被训练） | 学习信号 |
|------|------------------|---------------------|---------|
| 1 | Instruction + 用户历史 | `<a_5><b_7><c_101>` | SID 序列模式 |
| 2 | Instruction + SID 或标题 | 标题文本或 SID token | SID-语义映射 |
| 3 | Instruction + SID 历史 | "Screen Protector..." | 自然语言生成 |

三个任务共享底层的 transformer 层，只在输出 token 上不同。多任务训练迫使模型学习到更通用的 SID 表示。

### 3.5 Tokenizer 扩展

LLM 原生 tokenizer 不认识 SID token（如 `<a_5>`），需要扩展：

```python
# sft.py:160-170
# 1. 从 index.json 加载所有 SID token
token_extender = TokenExtender(data_path, dataset)
new_tokens = token_extender.get_new_tokens()
# new_tokens = ["<a_0>", "<a_1>", ..., "<a_255>",
#               "<b_0>", "<b_1>", ..., "<b_255>",
#               "<c_0>", "<c_1>", ..., "<c_255>"]

# 2. 添加到 tokenizer
tokenizer.add_tokens(new_tokens)

# 3. 调整模型 embedding 矩阵大小
model.resize_token_embeddings(len(tokenizer))
```

**可选：冻结 LLM，只训练新 token embedding**：

```python
# sft.py:173-200
if freeze_LLM:
    for param in model.parameters():
        param.requires_grad = False
    
    # 只对新加的 SID token embedding 反冻结
    embedding_layer = model.get_input_embeddings()
    embedding_layer.weight.requires_grad = True
    
    # 梯度掩码：冻结原 vocab 的梯度，只允许新 token 的梯度
    def mask_grad(grad):
        grad[:original_vocab_size].zero_()  # 原词表梯度归零
        return grad
    embedding_layer.weight.register_hook(mask_grad)
```

这个选项在 SFT 阶段通常不使用（默认 `freeze_LLM=False`），但在 cold-start 或数据极少的场景下有用。

此外还添加了 9 个特殊 token，用于更丰富的输入表示：

| Token | 含义 |
|-------|------|
| `[USER_HIGH_RATING]` | 用户高评分标记 |
| `[USER_MID_RATING]` | 用户中等评分标记 |
| `[USER_LOW_RATING]` | 用户低评分标记 |
| `[USER_UNKNOWN]` | 未知评分标记 |
| `[CTX_BROWSE]` | 浏览上下文 |
| `[CTX_SEARCH]` | 搜索上下文 |
| `[CTX_HOMEPAGE]` | 首页上下文 |
| `[O_TOKEN]` | 输出 token 标记 |
| `[I_TOKEN]` | 输入 token 标记 |

### 3.6 SFT 训练配置

```python
# sft_3090.sh 配置
base_model = "Qwen/Qwen2.5-0.5B-Instruct"
batch_size = 64              # 全局 batch
micro_batch_size = 4         # 每设备 batch
gradient_accumulation = 16   # 梯度累积 (64/4=16)
num_epochs = 3
learning_rate = 5e-4
cutoff_len = 64              # 最大序列长度
bf16 = True                  # bfloat16 混合精度
warmup_steps = 20
torch_compile = True         # PyTorch 编译加速
attn_implementation = "flash_attention_2"

# 数据组合
train_data = SidSFTDataset + SidItemFeatDataset + FusionSeqRecDataset

# 早停策略
EarlyStoppingCallback(early_stopping_patience=3)
```

**训练曲线解读**：
- 由于 `ConcatDataset` 混合三个任务，loss 曲线会有周期性波动（不同任务的难度不同）
- 任务 2（SID-标题映射）loss 最低，因为模式固定
- 任务 3（输出标题）loss 最高，因为生成长文本更难
- 关注 validation loss（仅任务 1）判断是否过拟合

---

## 4️⃣ 强化学习优化 (RL)

### 4.1 为什么需要 RL？

SFT 教会了模型基本的"生成 SID"能力，但存在根本性的局限：

| SFT 的局限 | 表现 | 原因 |
|-----------|------|------|
| **只学正例** | 模型只会输出训练集中出现的 SID | 交叉熵损失只最大化正确答案的似然，不看错误答案 |
| **优化目标不对齐** | 指标 HR@10=9.3%，但 loss 已经很低 | SFT 优化 token 级似然，推荐关心的是排名 |
| **缺乏探索** | 生成结果倾向于高频 SID | 没有奖励信号引导模型探索更好的推荐 |
| **无对比学习** | 不知道"A 比 B 好" | 每个样本独立计算 loss，没有组内比较 |

**RL 的核心作用**：不是让模型"生成更准确的 SID"，而是让模型学会 **"在多个候选中把最好的排到最前面"**。

### 4.2 RL 任务设计

RL 阶段使用 3 个任务（与 SFT 结构类似，但输出统一为 SID）：

```python
# rl.py:91-108
train_data1 = SidDataset(...)                # 任务1: 序列推荐（SID 历史 → SID）
train_data2 = RLTitle2SidDataset(...)         # 任务2: 标题 → SID 映射
train_data3 = RLSeqTitle2SidDataset(...)      # 任务3: 标题历史序列 → SID
train_data = ConcatDataset(train_datasets)
```

与 SFT 的关键区别：

| 维度 | SFT | RL |
|------|-----|-----|
| **输出格式** | 任务 3 输出标题文本 | **所有任务统一输出 SID** |
| **任务 2 方向** | 双向（SID↔标题） | **单向（标题→SID 和 描述→SID）** |
| **数据格式** | `prompt` + `completion`（固定对） | `prompt`（模型自生成，无固定 completion） |
| **任务 2 数据量** | ~7.4K | 全量（~7K title2sid + ~7K description2sid） |

**为什么 RL 所有任务都输出 SID？** 因为奖励函数需要比较模型生成的 SID 和真实目标 SID。如果输出文本，语义奖励很难设计。

### 4.3 GRPO 损失函数 — 完整代码级详解

**源码位置**: `minionerec_trainer.py:1033-1071`

GRPO 的损失函数是 RL 阶段的核心。下面从 `compute_loss` 的逐行代码入手分析。

#### 输入数据流

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # inputs 来自 _prepare_inputs() 方法
    # 包含:
    #   prompt_ids, prompt_mask       — 用户历史 SID (batch, prompt_len)
    #   completion_ids, completion_mask — 模型生成的候选 SID (batch, completion_len)
    #   ref_per_token_logps            — 参考模型对生成结果的 log 概率 (冻结的 SFT 模型)
    #   advantages                     — 组内归一化后的优势值 (batch,)
```

#### 第一步：获取策略模型的 log 概率

```python
    # minionerec_trainer.py:1038-1044
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)  # 只对 completion 部分计算

    # 策略模型的前向传播，只计算 completion 部分的 logits
    per_token_logps = self._get_per_token_logps(
        model, input_ids, attention_mask, logits_to_keep
    )
    # per_token_logps: [batch, completion_len]
    # 每个位置的值 = log π_θ(o_t | prompt + o_<t)
```

`_get_per_token_logps` 的实现（`minionerec_trainer.py:625-634`）：

```python
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, 
                       logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # 排除最后一个 logit (无需预测下一个 token)
        logits = logits[:, -logits_to_keep:]  # 只保留 completion 部分
        return selective_log_softmax(logits, input_ids)
        # 只返回 completion 中每个 token 对应的 log 概率，不返回全 vocab
```

> **为什么 logits_to_keep 要 +1 再 -1？** 因为 causal LM 的 logits[i] 预测的是 token[i+1]，我们需要最后一个位置的 logits 来预测 completion 之后的 token，但这里只对 completion 内部的 token 算 loss，所以去掉多余的。

#### 第二步：计算逐 token KL 散度

```python
    # minionerec_trainer.py:1046-1047
    ref_per_token_logps = inputs["ref_per_token_logps"]  # 参考模型的 log 概率
  
    # KL 散度的无偏近似估计（非对称形式）
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) \
                   - (ref_per_token_logps - per_token_logps) - 1
```

**这个 KL 公式的推导**：

标准 KL 散度：`KL(π_θ || π_ref) = Σ π_θ × log(π_θ / π_ref)`

但 GRPO 中使用的是**无偏近似**（来自 [Schulman et al. 2020]）：
```
KL ≈ exp(Δ) - Δ - 1,  其中 Δ = log π_ref - log π_θ

当 Δ=0（策略完全相同）时，KL = exp(0) - 0 - 1 = 0 ✓
当 π_θ < π_ref（策略倾向于参考模型）时，Δ > 0，KL 为正 ✓
当 π_θ > π_ref（策略比参考更自信）时，Δ < 0，KL 仍为正 ✓
```

这个近似的好处：
- 不需要显式计算完整分布，只需要两个模型的 log 概率
- 数值稳定，不会出现 log(0)
- 对正负偏差对称惩罚

#### 第三步：策略梯度损失

```python
    # minionerec_trainer.py:1049-1052
    advantages = inputs["advantages"]  # [batch]，组内归一化后的优势值

    # 策略梯度: exp(log π_θ - log π_θ.detach()) × advantage
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)

    # 组合: 策略梯度损失 + KL 惩罚
    per_token_loss = -(per_token_loss - self.beta * per_token_kl)
```

#### GRPO 损失的数学完整形式

结合前三步，标准 GRPO 损失的完整数学表达式为：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} 
\Bigg[ \underbrace{\frac{\pi_\theta(o_{i,t} | o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | o_{i,<t})}}_{\text{重要性采样权重}} \cdot \underbrace{\hat{A}_i}_{\text{优势函数}} - \beta \cdot \underbrace{\text{KL}[\pi_\theta \| \pi_{\text{ref}}](o_{i,t})}_{\text{KL 散度}} \Bigg]
$$

其中：
- $G$：每条 prompt 生成的候选数（= `num_generations`）
- $|o_i|$：第 $i$ 个候选的长度（= `completion_length`）
- $\pi_\theta$：当前策略模型，$\pi_{\text{ref}}$：参考模型（冻结的 SFT 模型）
- $\hat{A}_i$：组内归一化优势值，$\hat{A}_i = (r_i - \mu_g) / \sigma_g$
- $\beta$：KL 惩罚系数

**逐项解释**：

| 项 | 数学表达 | 代码实现 | 作用 |
|----|---------|---------|------|
| 重要性采样权重 | $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ | `exp(per_token_logps - per_token_logps.detach())` | 让损失函数在 $\theta$ 更新后仍有效 |
| 优势函数 | $\hat{A}_i = \frac{r_i - \mu_g}{\sigma_g}$ | `(rewards - mean_grouped) / (std_grouped + 1e-4)` | 衡量每个候选的相对好坏 |
| KL 散度 | $\exp(\Delta) - \Delta - 1$ | `exp(ref_logps - logps) - (ref_logps - logps) - 1` | 约束策略更新幅度 |

#### 优势函数归一化

$$
\hat{A}_{i}^{(g)} = \frac{r_i^{(g)} - \mu_g}{\sigma_g + \epsilon}, \quad 
\mu_g = \frac{1}{G}\sum_{j=1}^{G} r_j^{(g)}, \quad 
\sigma_g^2 = \frac{1}{G}\sum_{j=1}^{G} (r_j^{(g)} - \mu_g)^2
$$

其中 $r_i^{(g)}$ 是第 $g$ 组第 $i$ 个候选的奖励，$\epsilon=10^{-4}$ 防止除零。

#### KL 散度近似公式

标准 KL 散度 $\text{KL}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{x \sim \pi_\theta}[\log \pi_\theta(x) - \log \pi_{\text{ref}}(x)]$ 需要采样估计。GRPO 使用无偏近似：

$$
\hat{d}_{\text{KL}}(\Delta) = e^{\Delta} - \Delta - 1, \quad \Delta = \log \pi_{\text{ref}}(o_t) - \log \pi_\theta(o_t)
$$

**性质**：
- $\Delta = 0$ 时，$\hat{d}_{\text{KL}} = e^0 - 0 - 1 = 0$（策略一致时无惩罚）
- $\Delta \to \infty$ 时，$\hat{d}_{\text{KL}} \to \infty$（策略过度自信时强惩罚）
- $\Delta \to -\infty$ 时，$\hat{d}_{\text{KL}} \to -\infty + \infty - 1 \to \infty$（策略过度保守时也惩罚）

**为什么需要 `per_token_logps.detach()`？**

这是策略梯度定理的要求。梯度计算公式：

$$
\nabla_\theta \mathcal{L} = \mathbb{E}_{o \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(o) \cdot A \right]
$$
需要将 log 概率的**值**（用于加权）和**梯度**（用于更新）分离。`detach()` 确保 `per_token_logps` 作为数值参与计算梯度权重，但梯度只流过 `per_token_logps` 的第一个出现位置。

如果没有 `detach()`，梯度会计算为 `∇log π_θ × A + log π_θ × ∇A`，第二项不对。

#### 第四步：聚合损失

```python
    # minionerec_trainer.py:1054-1062
    if self.dapo:                      # DAPO 变体
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    elif self.gspo:                    # GSPO 变体
        per_token_ratio = per_token_logps - per_token_logps.detach()
        s_score = torch.exp((per_token_ratio * completion_mask).sum(dim=1) / completion_mask.sum(dim=1))
        sequence_kl = (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        loss = -(s_score * advantages - self.beta * sequence_kl).mean()
    else:                              # 标准 GRPO（本项目使用）
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
```

**标准 GRPO 聚合**：每个 completion 的 loss 取 token 维度的平均，再对 batch 取平均。

#### 完整损失函数图示

```
输入: prompt "user bought <a_5><b_12>, ..."

模型生成 G=4 个候选:
  candidate_1: <a_5><b_7><c_101>  ✅ 命中 → 奖励高
  candidate_2: <a_5><b_8><c_33>   ❌ 未命中 → 奖励低
  candidate_3: <a_7><b_12><c_88>  ❌ 未命中 → 奖励低
  candidate_4: <a_5><b_7><c_55>   ❌ 未命中 → 奖励低

normalize → advantage = (reward - mean) / std
  candidate_1: advantage 高 ✓ → 提升其概率
  candidate_2: advantage 低 ✗ → 降低其概率
  ... 

KL 散度约束: 防止策略更新过激，偏离 SFT 模型太远
```

#### GRPO vs PPO 的显存对比

| 组件 | PPO | GRPO |
|------|-----|------|
| 策略模型 | ✅ | ✅ |
| 参考模型 | ✅ | ✅ |
| Value 模型 | ✅ (额外 0.5B 参数) | ❌ |
| 优势计算 | Value 网络前向传播 | 组内均值归一化 O(G) |
| 总显存 (0.5B) | ~5.5 GB (3 个模型) | ~3.5 GB (2 个模型) |

### 4.4 奖励函数设计

**源码位置**: `rl.py:162-251`

项目设计了四种奖励函数，并通过 `reward_type` 参数切换：

#### 规则奖励（Rule Reward）

```python
# rl.py:192-203
def rule_reward(prompts, completions):
    """二进制奖励：命中目标=1.0，否则=0.0"""
    rewards = []
    for i, completion in enumerate(completions):
        if completion.strip("\n\" ") == targets[i].strip("\n\" "):
            rewards.append(1.0)   # 精确命中
        else:
            rewards.append(0.0)   # 未命中
    return rewards
```

**为什么用 0/1 离散奖励而不是连续值？**
- 推荐任务的"正确"是确定的——目标商品只有一个
- 连续奖励（如语义相似度）可能引入噪声
- 模型只需要知道"对了还是错了"，不需要知道"对了多少"

#### NDCG 排序奖励（Ranking Reward）

```python
# rl.py:162-190
# 预计算排名权重：排名越靠前，权重越高
ndcg_rewards = [-1.0 / math.log2(i+2) for i in range(num_generations)]  
# 例如 num_generations=4: [1.0, 0.63, 0.50, 0.41]

# 归一化: 使 sum(ndcg_rewards) = 1
ndcg_rewards = [-elm / sum(ndcg_rewards) for elm in ndcg_rewards]

def ndcg_rule_reward(prompts, completions):
    # 组内比较：如果组内有命中的
    #   → 命中的按排名给正奖励，未命中的给负奖励
    # 如果组内全部未命中
    #   → 所有人都给 0（不鼓励，也不惩罚）
    
    for i in range(0, len(completions), num_generations):
        group = completions[i:i+num_generations]
        if any(hit == target for hit in group):
            # 有命中的 → 命中的给正分，没命中的给负分
            for j, comp in enumerate(group):
                rewards.append(ndcg_rewards[j] if comp == target else ndcg_rewards[j])
        else:
            # 全没命中 → 都给 0
            rewards.extend([0.0] * num_generations)
```

**NDCG 奖励的设计逻辑**：

```
组内 4 个候选:
  位置 0: 命中了 → reward = +0.41  ← 高排名命中，高分
  位置 1: 未命中 → reward = -0.26  ← 被命中的挤到后面，负分
  位置 2: 未命中 → reward = -0.20
  位置 3: 未命中 → reward = -0.16

效果: 模型学会把正确答案放到 beam 的最前面
```

**为什么 NDCG 奖励比规则奖励更有效？**

规则奖励只告诉模型"这个对了/错了"，所有未命中的候选得到同样的 0 分。NDCG 奖励进一步区分了"排在前面但错了"和"排在后面但错了"——前者更糟糕，因为浪费了宝贵的 beam 位置。

#### 四种奖励模式

```python
# rl.py:255-264
if reward_type == "rule":
    reward_fun = rule_reward                    # 仅二进制
elif reward_type == "ranking":
    reward_fun = [rule_reward, ndcg_rule_reward] # 二进制 + NDCG（推荐）
elif reward_type == "ranking_only":
    reward_fun = ndcg_rule_reward                # 仅 NDCG
elif reward_type == "semantic":
    reward_fun = semantic_reward                 # 语义余弦相似度
elif reward_type == "sasrec":
    reward_fun = cf_reward                       # 协同过滤模型打分
```

本项目使用 `reward_type="ranking"`（`rule_reward` + `ndcg_rule_reward` 组合）。TRL 的 GRPOTrainer 将多个奖励函数的结果相加。

#### 如果只命中 SID 前两个 token 怎么给分？

**这个情况在 hash 约束下不会发生。** 如 2.4 节所述，hash 约束强制模型生成**完整的 3-token SID + EOS**——EOS 只有在完整 SID 之后才被添加到合法 token 集合中，模型无法提前终止。

所以 `rule_reward` 比较的永远是两个完整的 3-token SID：

```
生成: <a_5><b_12><c_88>  vs  目标: <a_5><b_12><c_88>  → 命中 ✅  reward=1.0
生成: <a_5><b_12><c_99>  vs  目标: <a_5><b_12><c_88>  → 未命中 ❌ reward=0.0
生成: <a_0><b_0><c_0>    vs  目标: <a_5><b_12><c_88>  → 未命中 ❌ reward=0.0
```

**前两个 token 匹配但第三个不匹配（`<a_5><b_12><c_99>` vs `<a_5><b_12><c_88>`）是否应该给部分奖励？**

从语义角度看应该给——前两层相同意味着属于同一大类，但**当前项目的奖励设计就是严格的 0/1 完全匹配**。原因如下：

| 考虑 | 分析 |
|------|------|
| **粒度要求** | 推荐任务要求精确推荐到具体商品，不是推荐到"大类"。如果只匹配前两层就给奖励，模型会倾向于只输出大类级别的 SID |
| **搜索空间** | 同一 `⟨a_5⟩⟨b_12⟩` 前缀下的商品数相对较少（~14 个），精确匹配的目标是可以达成的 |
| **RL 效率** | 0/1 奖励信号简单直接，梯度方差较小 |
| **语义奖励替代方案** | 项目提供了 `semantic_reward` 选项（余弦相似度），如果需要连续奖励可以切换 |

**如果要实现层级部分奖励**，可以这样设计：

```python
def hierarchical_reward(prompts, completions):
    """层级奖励：匹配第一层+0.2，匹配第二层+0.3，匹配第三层+0.5"""
    rewards = []
    for comp, target in zip(completions, targets):
        # 按 SID 层级拆分
        comp_tokens = comp.strip().split("><")
        target_tokens = target.strip().split("><")
        reward = 0.0
        if comp_tokens[0] == target_tokens[0]:  # 第一层匹配
            reward += 0.2
            if comp_tokens[1] == target_tokens[1]:  # 第二层匹配
                reward += 0.3
                if comp_tokens[2] == target_tokens[2]:  # 第三层匹配
                    reward += 0.5  # 精确命中
        rewards.append(reward)
    return rewards
```

不过本项目未使用这种方案，因为当前指标（HR@K）已经有效评估了"推荐列表中有没有目标商品"，RL 阶段只需要推动模型把正确答案排到前面即可——0/1 奖励+组内比较已经足够这个目的。

#### 如何从 prompt 获取 target？

RL 的奖励函数依赖两个映射字典：

```python
# rl.py:123-137
prompt2history: {"prompt文本": "用户历史SID(::分隔)"}
history2target: {"用户历史SID": "目标SID"}

# 在奖励函数中通过 prompt 间接查找 target
def rule_reward(prompts, completions):
    history = [prompt2history[prompt] for prompt in prompts]     # prompt → 历史
    targets = [history2target[elm] for elm in history]            # 历史 → 目标
    # 现在比较 completion 和 target
```

**为什么要用两层映射？**
- 同一个用户历史可能对应多个不同的 prompt 变体
- 分离 prompt 和 target 的逻辑，奖励函数不需要关心 prompt 的格式
- 不同的 Dataset 类可以复用同一套映射

### 4.5 RL 训练配置

```python
# rl_3090.sh 配置
train_batch_size = 4
eval_batch_size = 4
gradient_accumulation_steps = 8   # effective batch = 4 * 8 = 32
num_train_epochs = 2
learning_rate = 1e-5              # SFT 的 1/50，RL 更新必须小心
beta = 1e-3                       # KL 惩罚系数
num_generations = 4               # 每条 prompt 生成 4 个候选
temperature = 1.0                 # 采样温度
beam_search = True                # 使用 beam search 生成
max_completion_length = 16        # SID 仅 3 个 token，大幅缩减
max_grad_norm = 0.3               # 梯度裁剪
warmup_ratio = 0.03
test_during_training = False      # 训练中不评估，节省时间

# 优化器
optim = "paged_adamw_32bit"       # 分页 Adam，显存优化
lr_scheduler_type = "cosine"      # 余弦退火
torch_compile = True
attn_implementation = "flash_attention_2"
```

### 4.6 SFT vs RL 全面对比

| 维度 | SFT | RL |
|------|-----|-----|
| **核心目标** | 学习 SID 的生成模式 | 学习排序和探索 |
| **损失函数** | 交叉熵 `-log P(正确答案)` | 策略梯度 `-exp(logπ) × A + β×KL` |
| **梯度来源** | 正确答案的 token 位置 | 模型**自己生成**的候选 |
| **是否需要正确答案** | ✅ 每个样本都需要 | ❌ 只需要奖励信号 |
| **模型数量** | 1 个（策略模型） | 2 个（策略 + 参考） |
| **数据使用方式** | prompt + completion 固定对 | prompt → 模型生成 → 奖励 → 更新 |
| **探索机制** | ❌ 无 | ✅ 采样生成 + beam search |
| **优化指标** | token 级准确率 | 推荐级 HR/NDCG |
| **学习率** | 5e-4 | 1e-5（小 50 倍） |
| **训练模式** | 模仿学习（学正确答案） | 对比学习（学好 vs 差） |
| **过拟合风险** | 高（数据量少） | 低（模型自己生成数据） |

#### 核心区别的直观理解

```
SFT 就像: 学生抄标准答案
  - 给一个题目, 给正确答案
  - 学生学会输出和正确答案一样的 SID
  
RL 就像: 学生自己做题, 老师给分数
  - 给一个题目, 学生自己做 4 遍
  - 老师打分 (0/1), 学生对比哪次做得好
  - 做得好下次多这样, 做得差下次避免
```

#### 为什么 RL 学习率要小 50 倍？

SFT 阶段学习率 5e-4，RL 阶段 1e-5。原因：

1. **KL 约束的脆弱性**：RL 的目标是最大化奖励，如果学习率太大，策略模型会迅速偏离参考模型，KL 散度爆炸，loss 变成 NaN
2. **奖励信号的稀疏性**：大部分生成结果的奖励是 0，只有少数命中才有正奖励。大学习率会导致"奖励 hacking"——模型找到某个固定的"高奖励模式"并只输出那个
3. **策略梯度的高方差**：组内归一化只能减少部分方差，策略梯度的方差仍然远高于交叉熵，需要更小的更新步长

#### 训练动态的关键观察

| 现象 | 原因 | 应对 |
|------|------|------|
| **KL 逐渐增大** | 策略在偏离 SFT 初始点 | β=1e-3 提供持续约束 |
| **reward 波动大** | 生成候选的随机性 | 增加 num_generations 可降低方差 |
| **HR@10 ↑ 但 HR@1 ↓** | 多样性增加使正确答案分散 | 以 NDCG@10 为主要指标 |
| **训练初期 loss 升高** | 模型正在"忘掉"SFT 学到的东西 | 正常现象，KL 约束会控制幅度 |

**为什么 max_completion_length 从 128 降到 16？**

SID 只有 3 个 token（如 `<a_5><b_12><c_88>`），加上 EOS 也只有 4 个 token。原论文用 128 是因为 base model 可能生成更多内容，但我们实测 16 足够，且大幅降低 GRPO 训练的显存峰值。

---

## 5️⃣ 评估系统

### 5.1 评估流程

**源码位置**: `evaluate.py`, `calc.py`

```
测试数据 (prompt, target_SID)
        │
        ▼
┌───────────────────────────┐
│  加载模型 + 构建 hash_dict │  ← 确保约束解码
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  Constrained Beam Search  │  ← 每个 prompt 生成 10 个候选
│  beam_width = 10          │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  结果保存 (JSON)          │  ← {prompt, output, predict: [候选列表]}
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  指标计算                 │  ← HR@K, NDCG@K
└───────────────────────────┘
```

### 5.2 HR 和 NDCG 指标

**Hit Ratio@K (HR@K)**：在 top-K 个推荐中，命中目标的比例。

$$HR@K = \frac{\text{命中目标的用户数}}{\text{总用户数}}$$

**Normalized Discounted Cumulative Gain@K (NDCG@K)**：考虑排名位置的命中得分。

$$NDCG@K = \frac{DCG@K}{IDCG@K} = \frac{\sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}}{IDCG@K}$$

其中 $rel_i = 1$ 如果第 i 个候选命中目标，否则为 0。

**直觉理解**：
- HR@10 = 0.109：10 个推荐中，约 10.9% 的用户能在列表里找到目标商品
- NDCG@10 = 0.074：考虑排名位置后，标准化的推荐质量分数
- NDCG 比 HR 更严格——命中位置越靠前，NDCG 越高

### 5.3 评估配置

```bash
# evaluate_3090.sh
MODEL_PATH=./outputs/rl_Industrial_and_Scientific_3090/final_checkpoint
BATCH_SIZE=4
NUM_BEAMS=10
MAX_NEW_TOKENS=256
LENGTH_PENALTY=0.0
```

评估日志自动包含 checkpoint 名称：
- `./logs/evaluate_*_checkpoint-5280.log` — beam search 推理过程
- `./logs/calc_*_checkpoint-5280.log` — HR/NDCG 指标计算结果

---

## 📊 实验结果

### 实验配置

| 配置项 | 值 |
|--------|-----|
| 基础模型 | Qwen2.5-0.5B-Instruct (~1GB bf16) |
| 数据集 | Industrial_and_Scientific |
| 硬件 | 单卡 RTX 3090 24GB |
| 评估 beam 数 | 10 |

### 各阶段结果 (beam=10)

| 阶段 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|------|------|------|------|-------|--------|--------|---------|
| **SFT** | 0.036 | 0.057 | 0.068 | 0.093 | 0.036 | 0.052 | 0.060 |
| **RL-epoch1** | 0.045 | 0.066 | 0.079 | 0.103 | 0.045 | 0.062 | 0.070 |
| **RL-epoch2** | 0.047 | 0.070 | 0.083 | 0.109 | 0.047 | 0.066 | 0.074 |
| **官方 1.5B** | 0.085 | 0.113 | 0.133 | **0.154** | 0.085 | 0.109 | **0.116** |

### 结果分析

1. **GRPO 强化学习显著有效**：2 轮 RL 后 HR@10 相对 SFT 提升 +17%，NDCG@10 提升 +23%
2. **RL 持续提升**：epoch1→epoch2 仍在提升，说明 2 轮 RL 尚未饱和
3. **SFT 阶段已有竞争力**：多任务联合训练策略在 SFT 阶段已给出较强的 baseline
4. **Top-K 优化方向正确**：beam search 多样性策略下，HR@10 和 NDCG@10 的提升最为显著
5. **0.5B vs 1.5B 差距可接受**：0.5B RL-epoch2 (HR@10=10.9%) 约为官方 1.5B (15.4%) 的 71%，模型参数量仅为 1/3

---

## 🔧 项目结构

```
MiniOneRec/
├── 📄 训练脚本
│   ├── sft_3090.sh             # SFT 训练入口
│   ├── rl_3090.sh              # RL 训练入口 (支持断点续训)
│   ├── evaluate_3090.sh        # 评估脚本 (日志含 checkpoint 名)
│   └── scripts/download_models.sh  # 模型下载
│
├── 📝 核心代码
│   ├── sft.py                  # SFT 训练实现
│   ├── sft_gpr.py              # SFT + GPR 变体 (含 VAFT_Trainer)
│   ├── rl.py                   # RL/GRPO 训练入口
│   ├── rl_gpr.py               # RL + GPR 变体 (含 HEPO 奖励)
│   ├── evaluate.py             # 评估实现 (Constrained Beam Search)
│   ├── calc.py                 # HR/NDCG 指标计算
│   ├── minionerec_trainer.py   # ReReTrainer (GRPO 实现)
│   ├── data.py                 # 数据集定义 (10+ 种数据集类)
│   ├── LogitProcessor.py       # ConstrainedLogitsProcessor
│   └── sasrec.py               # SASRec 协同过滤
│
├── 📦 RQ-VAE 模块 (rq/)
│   ├── rqvae.py                # RQ-VAE 训练
│   ├── generate_indices.py     # SID 生成
│   ├── rqkmeans_faiss.py       # RQ-Kmeans (Faiss 加速)
│   ├── rqkmeans_constrained.py # Constrained RQ-Kmeans
│   ├── rqkmeans_plus.py        # RQ-Kmeans+
│   ├── generate_indices_plus.py
│   └── models/
│       ├── rqvae.py            # RQ-VAE 模型定义
│       └── layers.py           # 自定义层
│
├── 📊 数据目录 (data/Amazon/)
│   ├── train/, valid/, test/   # 训练/验证/测试集
│   ├── index/                  # SID 索引 (index.json + item.json)
│   └── info/                   # 商品信息
│
└── 📄 requirements.txt
```

---

## 🚀 快速开始

### 环境配置

```bash
# 云服务器（已有 CUDA/Python）
pip install -r requirements.txt
pip install trl flash-attn --no-build-isolation

# 或使用 conda
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
pip install -r requirements.txt
```

### 完整训练流程

```bash
# 1. 下载模型
bash scripts/download_models.sh

# 2. SFT 阶段 (~1-2h)
bash sft_3090.sh

# 3. SFT 评估
MODEL_PATH=./outputs/sft_Industrial_and_Scientific_3090/final_checkpoint bash evaluate_3090.sh

# 4. RL 阶段 (~3-5h)
MODEL_PATH=./outputs/sft_Industrial_and_Scientific_3090/final_checkpoint bash rl_3090.sh

# 5. RL 评估
MODEL_PATH=./outputs/rl_Industrial_and_Scientific_3090/final_checkpoint bash evaluate_3090.sh

# 6. 断点续训（如需）
RESUME=./outputs/rl_Industrial_and_Scientific_3090/checkpoint-5280 bash rl_3090.sh
```

---

## ❓ 面试常见问题

### Q1: 为什么用 RQ-VAE 而不是简单的 one-hot 编码？

One-hot 编码的问题：
- 商品数量百万级 → 维度爆炸
- 无法表达商品间的相似性
- 新商品需要重新扩展维度

RQ-VAE 的优势：
- 3 个 token 即可表示 256³=16M 种组合
- 层级结构天然表达商品相似性（同大类的商品第一层 token 相同）
- 新商品只需通过 RQ-VAE 编码，无需修改模型

### Q2: 为什么 SFT 之后还需要 RL？

SFT 的局限：
- 只学"正确答案"，没见过"错误答案"
- 优化的是 token 级似然，和推荐指标（HR、NDCG）不对齐
- 倾向于生成高频模式，缺乏多样性

RL 的作用：
- 通过奖励信号直接优化推荐指标
- 生成多个候选，通过比较学习好坏
- GRPO 的组内比较机制天然适合"从多个候选中选好的"

### Q3: Constrained Beam Search 的时间复杂度是多少？

- 每步查找 hash_dict 是 O(1) 操作
- 总复杂度和普通 beam search 相同：O(beam_width × vocab_size × seq_len)
- 实际上因为约束后有效 token 数远小于 vocab_size，速度更快

### Q4: GRPO 的 KL 惩罚有什么作用？

- 防止策略模型偏离参考模型太远（避免 mode collapse）
- β 太大 → 训练无进展（策略被束缚）
- β 太小 → 训练不稳定（策略可能崩塌）
- 实测 β=1e-3 在本任务上效果较好

### Q5: 如果商品数量增加到百万级，这个方案还能用吗？

- RQ-VAE 三层 256 的组合空间是 16M，理论上支持百万级
- 实际需要调整 RQ-VAE 的 codebook 数量（如每层 512 或 1024）
- hash_dict 会变大，但仍是 O(1) 查找
- beam search 的约束解码保证了效率不随商品数量线性增长

### Q6: max_completion_length 为什么设为 16 而不是 128？

- SID 固定为 3 个 token + 1 个 EOS = 4 个 token
- GRPO 训练时需要为每个候选存储 KV cache，长度越长显存越大
- 从 128 降到 16 显存峰值下降约 8 倍
- 对效果无影响（模型不会在 SID 后生成额外内容）

### Q7: 多任务训练的 3 个任务各有什么作用？

- **SidSFTDataset（序列推荐）**：核心任务，学用户历史→下一个商品的映射
- **SidItemFeatDataset（特征对齐）**：帮助模型理解 SID 的含义（SID ↔ 商品标题）
- **FusionSeqRecDataset（融合推荐）**：结合自然语言描述，让 LLM 的语言知识能迁移到推荐

如果只用任务 1，模型可能"死记硬背"SID 序列而不理解语义；加入任务 2 和 3 后，模型真正理解了每个 SID 对应什么商品。

### Q8: 为什么选 RQ-VAE 不选 VQ-VAE？RQ-VAE 和 RQ-Kmeans 比呢？

**RQ-VAE vs VQ-VAE**
- VQ-VAE 单层 256 codebook 只能表示 256 种商品，而本项目有 3,686 个商品，碰撞率极高
- 增大 VQ-VAE codebook 到 4096+ 会导致训练不稳定、codebook collapse（大量原型为空）
- RQ-VAE 用 3 层残差获得 256³=16M 的表示空间，同时保持每层 256 的训练稳定性
- 残差结构天然生成层次语义：第一层粗分类（如"电子产品"），第二层细分类（"手机配件"），第三层具体商品

**RQ-VAE vs RQ-Kmeans**
- RQ-VAE 有可学习的编码器，新商品可通过编码器直接获得 SID；RQ-Kmeans 没有编码器，新商品需重新聚类
- RQ-VAE 端到端训练，编码器能学习"什么是好的离散编码"；RQ-Kmeans 只优化聚类紧密度
- RQ-Kmeans 训练快（FAISS 几分钟），但碰撞率通常更高
- 项目提供了两种方案的完整代码，可对比实验。本项目选用 RQ-VAE 作为主方案

**决策树总结**：
```
需要离散编码商品？ 
├─ 商品数 < 256 → VQ-VAE 足够
├─ 商品数 256~16M → 
│  ├─ 需要编码新商品 + 追求最低碰撞 → RQ-VAE（本项目选择）
│  └─ 快速实验 + 无需泛化到新商品 → RQ-Kmeans（FAISS）
└─ 商品数 > 16M → 增大 codebook 或增加层数
```

### Q9: 码本坍塌和碰撞是如何解决的？

**码本坍塌（Codebook Collapse）** — 部分 codebook 原型从未被选中：

| 解决方法 | 原理 |
|---------|------|
| **k-means 初始化** | 用第一个 batch 数据的 k-means 质心初始化 codebook，避免随机初始化的"孤岛" |
| **承诺损失（Commitment Loss）** | `beta * ||e - q1.detach()||²`，强制 encoder 输出靠近选中的原型 |
| **量化损失** | 同时拉近原型和 encoder 输出，双向奔赴 |

**碰撞（Collision）** — 两个商品相同 SID：

| 解决方法 | 原理 |
|---------|------|
| **训练期监控碰撞率** | 每隔 eval_step 计算碰撞率，保存 `best_collision_model.pth` |
| **迭代重编码** | `generate_indices.py` 中用 Sinkhorn 约束对冲突商品重编码，循环最多 20 次 |
| **分层 Sinkhorn** | 前两层不用约束（保留语义），第三层 sk_epsilon=0.003 微调分配 |
| **最终碰撞率** | 迭代消除后碰撞率降为 0% |

**一句话总结**：codebook collapse 通过初始化+损失函数预防，collision 通过生成阶段的迭代重编码事后消除。

### Q10: Sinkhorn 算法的原理是什么？

Sinkhorn 是最优传输（Optimal Transport）中的经典算法，用于在满足均匀约束的前提下找到最优匹配。

**核心思想**：把商品分配到 codebook 原型看作"运输问题"——有 N 个商品（供应方）要分配给 K 个原型（需求方），每个原型必须分到恰好 N/K 个商品，目标是让总运输距离最小。

**数学形式**：
```
最小化 ΣᵢΣⱼ P[i][j] × ||xᵢ - cⱼ||² - ε × H(P)
约束: 每行求和 = 1/N  (每个商品总权重一致)
     每列求和 = 1/K  (每个原型分配均匀)
```

其中 H(P) 是熵正则化项，ε（即 sk_epsilon）控制"均匀性"和"距离最近"的 trade-off。

**为什么选 Sinkhorn 而不是硬约束方法？**
- 硬约束（如 k-means-constrained）强制每个原型分配完全相等，可能破坏语义结构
- Sinkhorn 的熵正则化允许"软"约束——语义上非常近的商品仍然倾向于分到同一原型
- 通过调整 ε，可以在"均匀"和"语义"之间平滑调节
- 实现简单：只需交替做行归一化和列归一化，收敛快

**项目中使用**：只在 RQ-VAE 的第三层（最细粒度）加 sk_epsilon=0.003，前两层保留语义不做约束。详见 2.4 节。

### Q11: SFT 和 RL 的损失函数有什么区别？为什么 RL 这么设计？

**SFT 损失函数**（带掩码的交叉熵）：

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{|C|}\sum_{t \in C} \log P_\theta(x_{t+1} \mid x_{<t+1})
$$

其中 $C$ 为 completion 位置集合，$\theta$ 为模型参数。

- 监督信号来自"标准答案"
- 每个 token 位置独立计算
- 只告诉模型"应该输出什么"

**RL 损失函数**（GRPO 策略梯度）：

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\Bigg[ \underbrace{\frac{\pi_\theta(o_{i,t})}{\pi_{\theta_{\text{old}}}(o_{i,t})}}_{\text{重要性采样}} \cdot \underbrace{\hat{A}_i}_{\text{优势}} - \beta \cdot \underbrace{\hat{d}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})(o_{i,t})}_{\text{KL 惩罚}} \Bigg]
$$

- 监督信号来自"奖励函数的比较"
- 模型自己生成候选，互相比较
- 告诉模型"哪些做法更好，哪些更差"

**核心设计差异**：

| 差异 | 为什么 |
|------|--------|
| **RL 需要两个模型**（策略+参考） | 计算 KL 散度防止偏离 SFT 太远 |
| **RL 需要 detach()** | 策略梯度定理要求：`∇E[f] = E[f × ∇log π]` |
| **RL 学习率小 50 倍** | 策略梯度方差大 + KL 约束脆弱，大步长导致崩溃 |
| **SFT 所有 token 参与 loss** | 交叉熵计算整个序列的概率 |
| **RL 只有 completion 参与** | 只有生成的候选需要更新，prompt 不变 |
| **SFT 任务 3 输出文本** | SFT 阶段需要激活语言能力 |
| **RL 所有任务输出 SID** | 奖励函数需要比较模型输出和目标 SID，统一格式方便计算 |

**一句话总结**：SFT 教模型"什么是正确答案"，RL 教模型"什么做法更容易得到高分奖励"。

### Q12: 约束解码一定会生成完整的 3 层 SID 吗？如果只命中前两层怎么给奖励？

**一定会生成完整 3 层 SID。** hash 字典中 EOS token 只被添加到完整 SID 的合法后继集合中。ConstrainedLogitsProcessor 在每步解码时将非法 token 的 logit 置为 -inf，所以：

- 第 1 步：只能选 `<a_N>`（EOS 被 mask）
- 第 2 步：只能选 `<b_N>`（EOS 仍被 mask）  
- 第 3 步：只能选 `<c_N>`（EOS 仍被 mask）
- 第 4 步：只能选 EOS（只有 EOS 合法）

不存在只生成 1 个或 2 个 token 就停止的情况。

**如果只命中前两层**（如生成 `<a_5><b_12><c_99>` 目标为 `<a_5><b_12><c_88>`）：当前项目的奖励就是 0 分——0/1 严格匹配。原因是：推荐任务要求精确到具体商品，且同一前缀下商品数很少（~14 个），精确匹配是可达目标。如果确实需要层级奖励，可以设计 hierarchical reward 给前两层部分分数（如 `[0.2, 0.3, 0.5]`），但本项目未使用。

### Q13: 为什么 SID 是 3 层而不是 2 层或 4 层？为什么 codebook 是 256 不是 128 或 512？

**层数选择**：3 层是表示能力和训练难度的平衡点。2 层只有 256²=65K 组合，对 >3K 商品够用但缺乏细粒度区分；4 层有 256⁴=4B 组合但训练中残差传播链路过长，梯度容易消失。256³=16M 对 3,686 个商品绰绰有余，每层 ~14 个商品/前缀的分配也合理。

**codebook 大小选择**：256 是经验值。太小（128）则每层表示能力不足，碰撞率上升；太大（512+）则每层训练困难，容易出现 codebook collapse（大量原型从未被选中）。256 在工业界被广泛验证（VQ-VAE 原论文、TIGER、OneRec 等）。

### Q14: 为什么选择 Qwen2.5-0.5B-Instruct 而不是其他 0.5B 模型？

| 候选 | 优点 | 缺点 |
|------|------|------|
| **Qwen2.5-0.5B-Instruct ✅** | 中文生态好、Instruct 版本已对齐指令、推理速度快 | 英文推荐场景可能不如 Llama |
| Llama-3.2-1B | 英文预训练充分 | 参数量 2 倍，RTX 3090 显存压力大 |
| GPT-2 (124M) | 参数量小，训练快 | 语言能力弱，无 instruct 对齐，效果差 |
| TinyLlama-1.1B | 小参数量 | 仍然比 0.5B 大，且社区支持不如 Qwen |

选 Qwen2.5-0.5B-Instruct 的核心原因：**在 RTX 3090 24GB 上 training + inference 都能跑、Instruct 版本天然理解任务指令、0.5B 参数量约 1GB（bf16）留出足够显存给 GRPO 的参考模型和生成缓存。**

### Q15: 为什么 SFT 不直接用 LoRA 或 QLoRA？全参数微调不怕过拟合吗？

全参数微调（full-parameter fine-tuning）在本项目是刻意的选择：

| 方式 | 可训练参数 | 推理部署 | 效果 |
|------|-----------|---------|------|
| **全参数微调 ✅** | ~494M | 需保存完整模型 | ✅ 最佳，SID token embedding 可充分学习 |
| LoRA (rank=8) | ~4M | 需加载 LoRA 权重 | ⚠️ 新增的 SID token 不在 LoRA 作用范围内 |
| QLoRA | ~4M (4bit 量化) | 量化后可能有损失 | ⚠️ 量化+推荐精度损失风险 |

关键原因：SFT 需要为 768 个新 SID token（256×3）学习高质量的 embedding。LoRA 的低秩更新主要影响 attention 层，对新 token embedding 的学习帮助有限。且 0.5B 全参数在 24GB 显存中完全可以容纳，没有使用 PEFT 的必要。

**关于过拟合**：3 epoch + 早停（patience=3）+ cutoff_len=64（限制序列长度，正则化效果），实测 validation loss 在 epoch 2-3 之间不再下降，说明过拟合被有效控制。

### Q16: cutoff_len=64 会不会太短？长历史用户的信息不是被截断了吗？

cutoff_len=64 是够用的，分析如下：

- SID 3 token + 分隔符 ≈ 4-5 tokens/商品
- 历史平均长度 3.35，最大 14
- prompt 模板本身约 30 tokens
- 总长：30 + 14×5 = ~100 tokens（最坏情况）
- cutoff_len=64 覆盖了平均情况（30 + 3×5 = 45 tokens）

64 的 cutoff 实际上起到了自然正则化的作用——极少数超长历史用户的数据会被截断，但模型不会因此丢失主要学习信号。如果 cut off 到 256 甚至 512，只会增加训练显存和计算量，对效果提升有限。

### Q17: SFT 的 3 个任务有没有做过消融实验？单独用任务 1 效果如何？

虽然本项目没有做完整的消融（参考原论文），但可以从原理上分析：

- **只用任务 1**：模型学会 SID 序列模式，但不知道 SID 对应什么商品。相当于"背诵"训练集中的 SID 共现模式，泛化能力弱。对完全相同的用户历史可能正确推荐，但对略作变化的 prompt 可能输出乱码
- **加任务 2**：模型建立了 SID 和自然语言的桥梁。即使遇到未见过的 SID 组合，也能通过"这个 SID 代表什么商品"来推断
- **加任务 3**：最关键的一步——模型必须用自然语言回答。这强制调用了 LLM 的预训练知识。比如 SID 指向 "Screen Protector"，LLM 的预训练知识告诉它"Screen Protector 是手机配件，和手机壳、贴膜一起购买"

原论文的消融实验显示多任务训练带来约 1-2% 的 HR@10 提升。

### Q18: RL 阶段为什么 num_generations=4？增大会更好吗？

num_generations=G 是 GRPO 的超参数，控制每条 prompt 生成的候选数。

**增大 G（如 8 或 16）的好处**：
- 优势函数归一化更准确（更大的样本量估计均值和方差）
- 探索更充分，更有可能命中目标
- KL 散度估计更稳定

**增大 G 的代价**：
- 显存线性增长：G 个候选需要 G 倍的前向/后向计算
- 在 RTX 3090 24GB 上，G=4 是显存和效果的实际平衡点
- 同时 beam search 生成 G 个候选的计算量也是 O(G)

| G | 显存占用 (GRPO) | 候选多样性 | 训练速度 |
|---|----------------|-----------|---------|
| 2 | ~4 GB | 低 | 快 |
| **4** | **~5.5 GB** | **中等** | **适中** |
| 8 | ~8 GB | 高 | 慢（OOM 风险）|
| 16 | >12 GB | 很高 | 3090 上不可行 |

### Q19: beam_search=True 不影响 RL 的探索性吗？

这是一个很好的观察。beam search 是确定性搜索（取 top-K 高概率路径），不是随机采样，理论上会降低探索多样性。

**为什么本项目仍然用 beam search？**

1. **搜索空间约束**：hash_dict 已经将合法 SID 限制在 3,686 个以内，beam search 的 4 个候选在这个空间内已经足够覆盖
2. **Constrained beam search 的特殊性**：每步只有几十个合法 token，beam search 的候选之间差异度比标准 LLM 生成更大
3. **效率优先**：在 RL 训练中，采样生成（`do_sample=True`）需要多次前向传播，beam search 在相同候选数下更高效

**实际使用**：代码中设置 `do_sample=True, temperature=1.0` 同时启用了采样，所以 beam search + 采样的组合在确定性中引入了随机性。

### Q20: 为什么用 bf16 而不是 fp16？混合精度对推荐效果有影响吗？

**bf16 vs fp16**：

| 特性 | fp16 | bf16 |
|------|------|------|
| 指数位 | 5 bit | 8 bit |
| 尾数位 | 10 bit | 7 bit |
| 动态范围 | ±65K | ±3.4×10³⁸ |
| 精度 | 高 | 低 |

bf16 保留了 fp32 相同的动态范围（8 bit 指数），只是尾数精度降低。这对深度学习训练更安全——**不会出现 fp16 常见的溢出问题**（gradient underflow/overflow）。

**对推荐效果的影响**：实验表明 bf16 训练的推荐指标与 fp32 几乎没有差异（差异 < 0.1%）。主要原因是推荐任务的 loss landscape 相对平滑，对数值精度不敏感。同时 bf16 降低了一半显存占用，是 RTX 3090 上限的必要选择。

### Q21: 如何评估模型的泛化能力？测试集上的指标会不会是"死记硬背"？

测试集是**按时间切分**的（最后一条交互作为 test），所以测试集用户虽然可能出现在训练集中，但测试样本是模型从未见过的（未来的交互）。这模拟了真实推荐场景。

**更严格的泛化测试**：

| 场景 | 本项目的表现 | 说明 |
|------|------------|------|
| **见过的用户 + 未见过的交互** | ✅ HR@10=10.9% | 时间切分，测试是未来行为 |
| **完全未见过的用户** | ⚠️ 用户仅出现在 valid/test | 约 471 用户从未在训练集中出现 |
| **完全未见过的商品** | ❌ 无法推荐 | 585 商品没有出现在任何交互中，需要冷启动策略 |

**冷启动商品的潜在解决方案**：
1. 利用 RQ-VAE 的编码器为新商品生成 SID（编码器可以处理未见过的文本）
2. 在 hash_dict 中加入新 SID，模型即可生成对应的 token
3. 但模型没有见过该 SID 的"购买模式"，推荐准确率会下降
4. 可通过模型的语义理解能力做零样本推荐（如"这个商品和用户历史中的商品相似"）

### Q22: 如果给你更多 GPU（如 4×A100 或 8×A100），你会怎么改进这个项目？

| 改进方向 | 具体方案 | 预期提升 |
|---------|---------|---------|
| **增大模型** | Qwen2.5-1.5B → 3B → 7B | 参考官方 1.5B HR@10=15.4%，猜测 7B 可达 18-20% |
| **增大 num_generations** | G=4 → G=16，更充分的优势估计 | NDCG 可能再提升 2-3% |
| **延长训练** | 3 epoch SFT → 5 epoch + 更激进的正则化 | 约 1-2% |
| **多数据集训练** | Industrial + Office + Sports 联合训练 | 泛化能力增强 |
| **更大 beam size** | 10 → 50，评估时更充分 | HR@20 显著提升 |
| **更复杂的奖励** | 引入 SASRec 或语义奖励 | 可能减少 reward hacking |
| **多阶段 RL** | GRPO → PPO 两阶段 | 更稳定 |
| **模型集成** | 多个 checkpoint 投票 | 稳定提升 1-2% |

### Q23: 这个方案在工业部署中会遇到什么问题？

| 问题 | 严重程度 | 原因 | 可能的解决方案 |
|------|---------|------|--------------|
| **推理延迟高** | 🔴 严重 | LLM 生成需 beam search，比 embedding 检索慢 100-1000 倍 | 小模型 + 量化 + KV cache 优化；或只在精排阶段使用 |
| **增量更新难** | 🟡 中等 | 新商品需要编码 SID + 更新 hash_dict + 可能需重训 | 预留 SID 空间 + 定期 reindex |
| **冷启动商品** | 🟡 中等 | 无交互的商品，模型无法推荐 | 利用 RQ-VAE 编码器做零样本 + 文本特征 |
| **可解释性** | 🟢 较好 | LLM 可生成自然语言解释 | 可设计 prompt 要求模型同时输出推荐理由 |
| **A/B 测试成本** | 🟡 中等 | 切换推荐系统涉及大规模的在线评估 | 渐进式上线、shadow test |

**最大瓶颈是推理延迟**：单条 beam search（10 beams, 4 steps）= ~40 次前向传播 ≈ 200ms 在 RTX 3090 上。对比传统 embedding 检索的 <1ms，差了 200 倍。工业部署需要大量 GPU 或模型压缩。

### Q24: 如果有多个商品对应到同一个标题怎么办？

**源码位置**: `data.py:706`

这是一个 title2sid 映射中的二义性问题。当前实现用一个 Python dict 存储映射：

```python
self.title2sid[title] = combined_sid  # dict覆盖
```

如果多个商品标题相同（例如同一产品不同卖家、不同颜色型号共用一个标题），后遍历到的 SID 会覆盖前面的。这意味着 title2sid 子任务的训练数据不完整——模型只学到了标题→某一个 SID 的映射，但真实世界中可能有多个正确答案。

**三个处理思路：**

| 方案 | 做法 | 优缺点 |
|------|------|--------|
| **保留全部 SID** | 改为 `self.title2sid[title] = self.title2sid.get(title, []) + [sid]`，一个标题创建多条训练样本 | ✅ 数据不损失，模型学会标题可对应多个商品；❌ 训练样本膨胀 |
| **频次保留** | 根据商品在训练集的出现频次保留最常见的那一个 | ✅ 推荐场景下更实用；❌ 需要额外统计信息 |
| **跳过歧义标题** | 标题对应 SID > 1 个的跳过不加入训练 | ✅ 标签确定无歧义；❌ 数据有损失 |

但方案 1 有一个**冲突问题**：SFT 训练时会出现多条相同 input、不同 target 的样本。

```
标题 "Safety Goggles":
  商品A → <a_10><b_5><c_3>  样本1: prompt → target <a_10><b_5><c_3>
  商品B → <a_10><b_5><c_7>  样本2: prompt → target <a_10><b_5><c_7>
```

因此 SFT 交叉熵损失永远降不到 0，模型处于"左右为难"的状态。

| 方案 | SFT 能否有效收敛 | 推理覆盖度 | 适用场景 |
|------|:-:|:-:|------|
| **保留全部 SID** | 否，loss 因歧义降不到 0 | 高（beam search 可覆盖多个候选项） | RL 阶段或者需要多样性的场景 |
| **频次保留** | 能 | 低 | 有商品热度统计信息可用时 |
| **跳过歧义标题** | 能 | 中（不影响非歧义数据） | SFT 评估对标签确定性要求高时 |

这个"冲突"在 RL 阶段反而无害——RL 不依赖固定 target，模型生成多个候选中只要有一个匹配就给正奖励。模型学会标题可能对应多个商品，反而是有用的多样性来源。

所以选择取决于所处的训练阶段：
- **RL 阶段**：方案 1 最合适（冲突不影响 RL，相反鼓励探索）
- **SFT 阶段**：方案 3 更干净（保证每个训练样本的标签是确定的；Amazon 数据集标题含具体型号，实际影响很小）

### Q25: SFT 是什么？怎么训练的？

SFT（Supervised Fine-Tuning，监督微调）是在预训练 LLM 基础上，用标注数据做全参数微调，让模型学会特定任务的输入输出模式。

**在本项目中的 SFT 流程：**

1. 加载 Qwen2.5-0.5B-Instruct 预训练权重
2. 扩展词表，加入 768 个新 SID token（3 层 × 256 codebook）
3. 初始化这些新 token 的 embedding（用 codebook 向量投影初始化）
4. 用 3 个任务的混合数据训练 3 epoch
5. 保存 final_checkpoint

**三个任务同时训练**：序列推荐（SID→SID）、特征对齐（SID↔标题）、融合推荐（SID→标题）。每 batch 从三个数据集中各取一部分，pad 到相同长度后混合训练。三个任务的 loss 都是交叉熵，直接相加回传。

### Q26: VQ-VAE 和 RQ-VAE 有什么区别？

**VQ-VAE（Vector Quantized VAE）**：将连续 embedding 量化到最近的一个 codebook 原型。单层、固定粒度。

```
输入 embedding → 找最近原型索引 → 通过对应原型重建
                一个 256 的 codebook 只能表达 256 种商品
```

**RQ-VAE（Residual Quantized VAE）**：逐层残差量化。第一层量化粗略语义，减去第一层残差后量化第二层，再减残差量化第三层。

```
输入 embedding → 第一层量化 (256种粗类)
                └─ 减去第一层重建 → 第二层量化 (256种细类)
                                   └─ 减去第二层重建 → 第三层量化 (256种具体)
```

| 对比 | VQ-VAE | RQ-VAE |
|------|--------|--------|
| **表示空间** | 256 | 256³ = 16M |
| **训练难度** | codebook 增大后易坍塌 | 每层 256 保持稳定 |
| **语义结构** | 无层次 | 天然层次（粗→细） |
| **本项目是否可用** | ❌ 3K+ 商品碰撞率极高 | ✅ |

**RQ-KMeans** 是 RQ-VAE 的无编码器版本，直接用 FAISS k-means 聚类的质心替代 codebook。训练快（几分钟），但无法为新商品生成 SID（需要重新聚类）。

### Q27: RQ-VAE 训练时有没有划分验证集和测试集？怎么评估 SID 质量？

**有划分。** 具体做法：

```
商品 embedding 数据 → 按商品 ID hash 分割
                    ├─ 训练集 (80%) — 训练 RQ-VAE 的 encoder + codebook
                    ├─ 验证集 (10%) — 每 eval_step 计算 collision_rate
                    └─ 测试集 (10%) — 最终评估 SID 质量
```

**在 NDCG 之前，SID 质量通过以下指标评估：**

| 指标 | 计算方式 | 目标 |
|------|---------|------|
| **碰撞率 (Collision Rate)** | 相同 SID 的商品数 / 总商品数 | 0%（完全无碰撞） |
| **重建损失 (Reconstruction Loss)** | \|e - D(q₁+q₂+q₃)\|² | 越小越好 |
| **码本利用率 (Codebook Usage)** | 每层被至少使用一次的原型数 / 256 | 100%（越高越好） |
| **最近邻命中 (Top-5/10 Retrieval)** | 重建 embedding 在原始 embedding 中做 KNN，命中原始商品的比例 | 越高越好 |

**碰撞率是最关键的指标**，因为碰撞直接导致两个商品共用同一个 SID，推荐时无法区分。项目在 RQ-VAE 训练中每 eval_step 计算验证集碰撞率，保存 `best_collision_model.pth`。

### Q28: 新商品加入后 SID 冲突会不会提升？怎么解决？

**会提升。** 新商品的文本特征可能和现有商品相似，导致 RQ-VAE 编码器的输出落到已有 SID 的 Voronoi 区域内。但不一定立刻导致碰撞——

**SQ-VAE（Stochastic Quantization）** 实际上比 RQ-VAE 更受这个问题困扰（因为 SQ-VAE 的 gumbel-softmax 采样不确定性更大），而 RQ-VAE 的确定性残差量化相对可控。

**解决方案分层级：**

| 时间点 | 措施 | 效果 |
|--------|------|------|
| **在线推理** | 新商品通过 RQ-VAE 编码器获得 SID | 可能和新商品共享 SID |
| **定时重训** | 每 N 天用全量数据重训 RQ-VAE | 消除累积碰撞 |
| **预留空间** | 训练时 codebook 只使用 75%（~192/256），留出空间给新商品 | 新商品分配到空原型 |
| **Sinkhorn 重编码** | 对碰撞组执行 Sinkhorn 约束分配 | 硬性消除碰撞，但需离线执行 |

**本项目的当前设计**：RQ-VAE 为所有 3,686 个商品生成 SID 时保证 0 碰撞。新商品出现时可直接运行编码器获得 SID，但碰撞是可能的。项目中未实现增量碰撞消除，这是工业部署前需要完善的部分。

### Q29: SASRec 的网络结构是什么？Loss 怎么设计的？

**网络结构（`sasrec.py:214-257`）：**

```
输入: 用户历史 item_id 序列 [N, seq_len]
  │
  ▼
Item Embedding + Positional Embedding  ← 可学习的 embedding 层
  │
  ▼
LayerNorm → Multi-Head Self-Attention  ← 捕捉序列中的 item-item 关联
  │
  ▼
LayerNorm → Positionwise Feed-Forward   ← 非线性变换
  │
  ▼
LayerNorm
  │
  ▼
取最后一个时间步的 hidden state → Linear(emb_dim, item_num) → 输出 logits
```

核心公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right) V
$$

其中 $M$ 是因果掩码（只允许关注历史位置，不能看未来）。最后一个时间步的 hidden state 通过线性层映射到 `[item_num]` 的 logits，表示每个商品作为下一个交互的概率。

**Loss 设计（`sasrec.py:458-529`）：**

支持两种 loss，通过 `--loss_type` 切换：

| Loss | 做法 | 公式 |
|------|------|------|
| **BCE (默认)** | 正样本(目标商品) score vs 随机负样本 score | `BCE(scores, labels)` |
| **CE** | 全 vocab 上的 softmax 交叉熵 | `CrossEntropy(logits, target)` |

**BCE loss 的详细实现**（`sasrec.py:496-529`）：

```python
# 对每个样本采样一个随机负样本
pos_scores = gather(model_output, target)     # 正样本分数
neg_scores = gather(model_output, target_neg)  # 随机负样本分数
scores = concat(pos_scores, neg_scores)         # [batch*2]
labels = concat(ones, zeros)                   # [1,1,...0,0,...]
loss = BCEWithLogitsLoss(scores, labels)
```

**为什么用 BCE 而不是 CE？** 商品数太多（~3K），CE 需要在全 vocab 上算 softmax，计算量大。BCE 只比较 1 个正样本和 1 个负样本，训练速度快。但 BCE 的缺点是梯度信号弱（只比较了两个商品），CE 理论上更优。本项目中 `loss_type=ce` 在 SASRec 上效果略好。

### Q30: SASRec 相比 RNN/LSTM 有什么优势？为什么不用 SIM/DIEN？

**SASRec vs RNN/LSTM：**

| 维度 | RNN/LSTM | SASRec (Transformer) |
|------|----------|---------------------|
| **序列建模** | 顺序处理，逐步压缩信息 | 并行处理，每步直接 attention 到所有历史 |
| **长程依赖** | 随序列增长，早期信息被遗忘门压缩 | attention 直接连接任意两个位置，无距离衰减 |
| **训练速度** | 串行，无法并行 | 可并行（非因果部分），训练快 5-10x |
| **位置编码** | 天然顺序（按时间步输入） | 需要额外 positional embedding |
| **推荐任务** | 最后一个 hidden state 包含所有历史压缩 | 每个历史 item 独立参与 attention，不丢失 |

**一个具体的比较**：用户买了「手机壳→充电线→贴膜→手机支架→???」。LSTM 在第 5 步时，"手机壳"的信息可能已被压缩到 1% 以下。SASRec 的 attention 直接计算"???"和"手机壳"的相关性，权重仍然可以很高。

**为什么不用 SIM/DIEN？**

| 模型 | 核心思想 | 本项目为什么不选 |
|------|---------|----------------|
| **SIM** | 搜索式长序列建模，两阶段检索 | 本项目的序列很短（平均 3.35），SIM 的超长序列优化用不上 |
| **DIEN** | 兴趣演化，用 GRU+兴趣门控建模兴趣变化 | 推荐场景是 Sequential Recommendation 而非 CTR 预估；本项目目标是验证 SID 的有效性，越简单的 baseline 对比越公平 |
| **SASRec** | 纯 attention，简单干净 | 作为 baseline 足够了，且原论文和其他 GR 工作（TIGER、OneRec）都使用 SASRec 作为序列推荐 backbone |

**核心原则**：Baseline 越简单，对比越有说服力。SASRec 在所有 GR 论文中都是标配 baseline，用它做对比是领域标准做法。

### Q31: SASRec 和 GR 模型的输入输出有什么区别？输入文件分别是什么？

**SASRec 的数据流：**

```
输入文件:
  data/Amazon/train/Industrial_and_Scientific_5_train.csv
  → 读取 history_item_id (list), item_id (target)
  → 转化为 item_id 序列 + 下一个 item_id

SASRec 输出:
  model_output = Linear(hidden_state)  # [batch, item_num]
  → 每个商品作为下一个交互的 logit 分数

评估: 对 model_output 做 argsort, 取 top-K 看是否命中 target
```

**GR 模型（本项目）的数据流：**

```
输入文件（3个）:
  ① train CSV:  用户交互序列 (history_item_id, item_id, history_item_sid, item_sid)
  ② item JSON:  商品特征 (item_id, title, description)
  ③ index JSON: 商品 SID 映射 (item_id → [<a_N>, <b_N>, <c_N>])

GR 模型输出:
  LLM生成的token序列 → decode 为 SID (如 <a_5><b_12><c_88>)
  
评估: beam search 生成 top-10 SID, 和 target SID 比较
```

**关键区别：**

| | SASRec | GR (本项目) |
|--|--------|------------|
| **输入形式** | 原始 item_id 序列 | 语义 SID token 序列 + 文本 |
| **输出形式** | item_num 维 logits | token 序列（需 decode） |
| **是否用文本** | ❌ | ✅ (SID 来源于文本特征) |
| **泛化能力** | 只能推荐训练集见过的 item_id | 新商品可通过 RQ-VAE 编码获得 SID |
| **模型** | 4 层 Transformer (~3M 参数) | 0.5B LLM |

### Q32: AUC 的定义和意义？手撕 AUC？同分样本怎么处理？

**AUC = Area Under the ROC Curve**，等价于「随机选一个正样本、一个负样本，模型给正样本的分数 > 负样本的概率」。

$$AUC = P(\text{score}_{\text{pos}} > \text{score}_{\text{neg}})$$

**意义**：AUC 衡量模型的排序能力，不依赖具体阈值。0.5 = 随机，1.0 = 完美。推荐场景中 AUC > 0.8 说明模型有好的区分度。

**手撕 AUC（O(n log n) 实现）：**

```python
def auc(y_true, y_score):
    """
    y_true: [1, 0, 1, 0, ...]  标签
    y_score: [0.9, 0.3, 0.8, ...]  预测分数
    """
    # 按分数排序（降序）
    data = list(zip(y_score, y_true))
    data.sort(key=lambda x: x[0], reverse=True)
    
    pos = sum(y_true)                     # 正样本总数
    neg = len(y_true) - pos               # 负样本总数
    if pos == 0 or neg == 0:
        return 0.5
    
    # 统计所有正样本排在负样本前面的次数
    rank_sum = 0
    pos_seen = 0
    for score, label in data:
        if label == 1:
            pos_seen += 1
        else:
            rank_sum += pos_seen          # 每个负样本，前面有多少正样本
    
    auc = rank_sum / (pos * neg)
    return auc
```

**另一种等价形式**（基于秩和）：

```python
def auc_rank(y_true, y_score):
    # 给所有样本按分数排序，取秩
    rank = np.argsort(np.argsort(-y_score)) + 1  # 从 1 开始
    pos_rank_sum = np.sum(rank[y_true == 1])
    pos = np.sum(y_true)
    neg = len(y_true) - pos
    return (pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg)
```

**同分样本处理：**

| 方法 | 做法 | 影响 |
|------|------|------|
| **法 1: 平均秩** | 同分的样本取平均秩 | AUC 不变，推荐做法 |
| **法 2: 顺序无关** | 认为同分时正负随机 | 等价于对每个同分组期望化为随机排序 |
| **法 3: 忽略同分** | 只考虑严格大于/小于的对 | 同分不多时影响小 |

**标准差公式**：AUC 的方差可以近似估计：

$$\sigma^2 = \frac{1}{P \cdot N} \left( \frac{Q_0}{N} + \frac{Q_1}{P} - 4 \cdot (AUC - 0.5)^2 \right)$$

其中 $Q_0 = \sum_{i: y_i=0} (r_i - i)^2$，$Q_1 = \sum_{i: y_i=1} (r_i - i)^2$，$r_i$ 是正样本中第 $i$ 个正样本的秩。

### Q33: Preference Accuracy 和 log-prob margin 是什么？

这两个指标来自 RL 阶段的模型行为分析。

**Preference Accuracy（偏好准确率）**：

模型在 GRPO 生成的 G 个候选中，正确答案的排名位置。反映模型已经学会的区分能力：

```
Preference Accuracy = 1 - (正确答案的排名 / G)

例如 G=4, 正确候选排在第 2:
  Preference Accuracy = 1 - 2/4 = 0.5
```

如果模型总是把正确答案排在 beam 最前面，Preference Accuracy = 1 - 1/G ≈ 0.75（G=4 时）。这是 NDCG 的轻量级代理指标，计算成本低，不需要完整的 beam search。

**log-prob margin（对数概率差）**：

模型对正确答案和错误答案的对数概率之差。衡量模型对选择的置信度：

```
log-prob margin = log P(正确答案 | prompt) - log P(错误答案 | prompt)
```

其中错误答案是模型最可能输出的错误候选（highest-scoring wrong answer）。这个值越大，说明模型的偏好越明确。

**两者在训练中的监控意义：**

| | Preference Accuracy | log-prob margin |
|--|-------------------|-----------------|
| **监测什么** | 正确候选中排第几 | 模型多确信这个选择 |
| **理想曲线** | 训练中稳步上升 | 先升后降（过度自信时 KL 惩罚会抑制） |
| **警报信号** | 下降说明训练不稳定 | 过大（>5）→ 策略与参考模型严重偏离 |
| **关联指标** | 与 NDCG@G 正相关 | 与 KL 散度正相关 |

两者共同刻画模型的偏好强度（margin）和偏好方向（accuracy），比单一 loss 更直观。

### Q34: 如果前两层一样、最后一层随机分配，会不会不准确？

**看你怎么定义"准确"。**

如果前两层一样（如 `<a_5><b_12>`），表示两个商品属于 RQ-VAE 的同一大类+同一子类。最后一层从 {0..255} 中随机分配：

| 情况 | 影响 |
|------|------|
| **推荐指标 (HR/NDCG)** | 如果目标商品和推荐商品的 <a><b> 相同但 <c> 不同 → 判为未命中 → 指标下降 |
| **语义合理性** | 同一 `<a_5><b_12>` 前缀下 ~14 个商品，在语义上确实相似（如不同颜色的同一型号） |
| **用户感知** | 推荐"同一款式的蓝色款"vs"红色款"，用户可能不一定在意 |

**问题在于**：RQ-VAE 的层次语义假设前两层捕捉粗粒度语义，第三层捕捉细粒度区分。如果第三层随机分配，就浪费了第三层的区分能力。

**实际落地的方案**：第三层不是随机分配的，而是：
1. 通过 RQ-VAE 编码器实际编码获得（最小化重建损失）
2. 对碰撞组执行 Sinkhorn 重编码（均匀分配 + 距离最优）
3. 初始化时 k-means warm-start 让 codebook 分布合理

所以实际项目中不存在真正"随机分配"的场景——即使在同等下，第三层也是通过优化重建损失 + Sinkhorn 约束得到的确定分配。

**更直接的回答**：如果第三层真是随机的，那 3 层 RQ-VAE 的表示能力退化为 2 层（65K 组合），碰撞率会从 0% 升到每个前缀区间 14 选 1 碰撞 ≈ 碰撞率大幅上升，模型无法区分同前缀商品。所以必须保证第三层也是可区分的高质量分配。

### Q35: 码本利用率、坍缩、Sinkhorn 怎么答？Sinkhorn 怎么落地的？

**面试回答结构：三步递进。**

**第一步：问题描述（30 秒）**

> 码本坍缩（Codebook Collapse）指大量 codebook 原型从未被选中，利用率很低。原因：随机初始化导致部分原型处于 embedding 空间的"无人区"，训练中 encoder 的输出始终落在附近几个原型，其他原型永远得不到梯度更新。

**第二步：解决方案（1 分钟）**

| 问题 | 解决方案 |
|------|----------|
| 码本坍缩 | k-means 初始化（用第一个 batch 的聚类质心初始化 codebook）+ commitment loss |
| 碰撞 | 迭代消除：循环 20 次 → 检测碰撞 → 对碰撞组执行 Sinkhorn 重编码 |
| 码本利用不均 | Sinkhorn 约束：均匀化分配，同时保持语义距离最优 |

**第三步：Sinkhorn 落地细节（2 分钟，展现工程能力）**

Sinkhorn 在本项目中的具体实现（`rq/generate_indices.py`）：

```python
# 伪代码流程
def sinkhorn_assign(embeddings, codebook, sk_epsilon=0.003, max_iter=50):
    # ① 计算距离矩阵 [N, K]
    distances = cdist(embeddings, codebook)
    
    # ② 初始化分配矩阵 P = exp(-distances / sk_epsilon)
    P = exp(-distances / sk_epsilon)
    
    # ③ 交替归一化（Sinkhorn-Knopp 迭代）
    for _ in range(max_iter):
        P = P / P.sum(dim=1, keepdim=True)    # 行归一化: 每个商品分配和=1
        P = P / P.sum(dim=0, keepdim=True)    # 列归一化: 每个原型总权重=1
        P = P * N / K                          # 调整尺度
    
    # ④ argmax 转为离散分配
    indices = P.argmax(dim=1)
    return indices
```

**关键参数 sk_epsilon 的物理意义**：

$$
P[i][j] = \frac{\exp(-\|x_i - c_j\|^2 / \varepsilon)}{\sum_k \exp(-\|x_i - c_k\|^2 / \varepsilon)}
$$

| ε | 行为 | 效果 |
|---|------|------|
| **ε → 0** | 退化为 argmin（最近邻） | 纯语义，碰撞不控制 |
| **ε → ∞** | 纯均匀分配 | 无碰撞，但语义被破坏 |
| **ε = 0.003** | 软约束 | 兼顾语义 + 均匀性 |

**为什么选 Sinkhorn 而不是其他方法？** 因为 Sinkhorn 可微（如果需要在训练中集成梯度）、收敛快（几十步就稳定）、不需要调参的梯度裁剪。

**和 structured embedding（结构化哈希）的关系**：structured embedding 是另一种防碰撞方案——用 LSH/locality-sensitive hashing 把相似 embedding 分到不同桶，但 LSH 不可微且难以保证均匀性。Sinkhorn 在这两方面都更好。

**项目的实际代码调用**（`rq/generate_indices_plus.py`）：

```python
# 第一二层: sk_epsilon=0 (纯语义, 不开均匀约束)
# 第三层: sk_epsilon=0.003 (语义优先 + 均匀调整)

sk_epsilon_list = [0, 0, 0.003]  # 三层分别配置

# 迭代重编码循环
for iteration in range(20):
    collision_groups = detect_collisions(indices)
    if not collision_groups:
        break
    for group in collision_groups:
        indices[group] = sinkhorn_assign(embeddings[group], codebook, sk_epsilon=0.003)
```

### Q36: GRPO 中一个 group 内所有 reward 都为 0 会怎样？使用 NDCG 排序奖励后还会出现全零吗？

**即使使用了 NDCG 排序奖励（`ndcg_rule_reward`），全零 reward 仍然会发生。**

看 `ndcg_rule_reward` 的核心逻辑（`rl.py:166-190`）：

```python
for i, completion in enumerate(completions):
    if completion.strip("\n\"") == targets[i].strip("\n\""):
        flag = True                    # 标记组内有命中
        lis.append(0.0)                # 命中的反而得 0.0
    else:
        lis.append(ndcg_rewards[i%num_generations])  # 未命中的得负值

    if (i+1)%num_generations == 0:     # 一个 group 结束
        if flag:
            rewards.extend(lis)        # 有命中 → 保留（0.0 + 负值混合）
        else:
            rewards.extend([0.0] * repeat)  # 全未命中 → 全给 0.0
```

**两种情况**：

| 情况 | 组内 reward 分布 | advantage |
|------|-----------------|-----------|
| **至少 1 个命中** | 命中=0.0，未命中=负值 | 有正有负，正常更新 ✅ |
| **全部未命中** | 全部=0.0 | 全为 0，该 group 不贡献梯度 ⚠️ |

**全零时发生了什么**（`minionerec_trainer.py:963-970`）：

```python
mean_grouped_rewards = 0.0
std_grouped_rewards = 0.0
advantages = (0 - 0) / (0 + 1e-4) = 0.0   # epsilon 防止除零
```

策略梯度项 = `exp(...) × 0 = 0`，只剩 KL 惩罚项 `β × KL`。

**为什么不致命：**
1. KL 惩罚仍在，模型不会偏离参考模型太远
2. batch 中其他 group 只要有命中，就能正常贡献梯度
3. 随着训练进行，命中率上升，全零 group 的比例会逐渐减少

**潜在改进**：代码中有 `mask_all_zero` 参数但未实际使用。`hepo_reward`（`rl_gpr.py:94-121`）是分级奖励（0/0.2/0.5/1.0），即使全部未精确命中，只要 SID 前缀匹配就有非零奖励，能从根本上避免全零问题。

---

## 📚 核心算法公式

### 一、GRPO 奖励函数

令第 $g$ 组（group）包含 $G$ 个候选 $\{o_1^{(g)}, \ldots, o_G^{(g)}\}$，对应同一个 prompt $p^{(g)}$，目标商品 SID 为 $y^{(g)}$。

#### 1. 规则奖励（Rule Reward）

**源码**：`rl.py:192-203`

$$
R_{\text{rule}}(o_i, y) = \mathbb{1}[o_i = y] =
\begin{cases}
1.0, & \text{若 } o_i \text{ 精确匹配 } y \\
0.0, & \text{否则}
\end{cases}
$$

比较前做 strip 处理：`completion.strip("\n\" ") == target.strip("\n\" ")`。

#### 2. NDCG 排序奖励（Ranking Reward）

**源码**：`rl.py:162-190`

首先预计算排名权重向量 $\mathbf{w} \in \mathbb{R}^G$：

$$
w_j = \frac{-1/\log_2(j+2)}{\sum_{k=0}^{G-1} -1/\log_2(k+2)}, \quad j = 0, 1, \ldots, G-1
$$

注意 $w_j < 0$（负值），且 $\sum_j w_j = -1$。以 $G=4$ 为例：$\mathbf{w} \approx [-0.41, -0.26, -0.20, -0.16]$。

组内奖励定义：

$$
R_{\text{ndcg}}^{(g)}(o_i) =
\begin{cases}
0.0, & \text{若 } o_i = y^{(g)} \text{（命中者得 0）} \\
w_{i \bmod G}, & \text{若组内存在命中（未命中者得负值）} \\
0.0, & \text{若组内全部未命中}
\end{cases}
$$

关键逻辑：`flag` 标记组内是否有命中。有命中时，命中候选得 0，未命中候选按生成顺序获得递减的负分；全未命中时所有人得 0。

#### 3. 语义奖励（Semantic Reward）

**源码**：`rl.py:205-219`

$$
R_{\text{semantic}}(o_i, y) = \cos\bigl(\mathbf{e}(y),\; \mathbf{e}(o_i)\bigr) = \frac{\mathbf{e}(y)^\top \mathbf{e}(o_i)}{\|\mathbf{e}(y)\| \cdot \|\mathbf{e}(o_i)\|}
$$

其中 $\mathbf{e}(\cdot) \in \mathbb{R}^d$ 是预计算的商品 Ada embedding 向量。

#### 4. 协同过滤奖励（CF Reward / SASRec Reward）

**源码**：`rl.py:221-251`

$$
R_{\text{cf}}(o_i, h) = f_{\text{SASRec}}\bigl(\text{seq}(h),\; \text{id}(o_i)\bigr)
$$

其中 $h$ 是用户历史（item ID 序列），$f_{\text{SASRec}}: \mathbb{R}^{\text{seq\_len} \times d} \times \mathbb{Z} \to \mathbb{R}$ 是 SASRec 模型的前向评估函数，输出 item ID 对应的 logit 分数。若 $o_i$ 不在合法商品集合中，随机采样一个商品 ID 替代。

#### 5. 层级奖励（HEPO Reward）

**源码**：`rl_gpr.py:94-121`

令 SID 的层级解析函数为 $\text{parse}(\cdot)$，将 SID 拆分为 $[l_1, l_2, l_3]$。定义匹配深度：

$$
\delta(o_i, y) = \max\{k \in \{0,1,2,3\} : l_j^{\text{comp}} = l_j^{\text{target}},\; \forall j \leq k\}
$$

层级奖励：

$$
R_{\text{hepo}}(o_i, y) =
\begin{cases}
0.0, & \delta = 0 \text{（第一层就不匹配）} \\
0.2, & \delta = 1 \text{（匹配第一层）} \\
0.5, & \delta = 2 \text{（匹配前两层）} \\
1.0, & \delta = 3 \text{（完全匹配）}
\end{cases}
$$

### 二、奖励聚合

当使用多个奖励函数时（如 `reward_type="ranking"` 同时使用 `rule_reward` 和 `ndcg_rule_reward`），TRL 的 GRPOTrainer 将各奖励加权求和：

$$
r_i = \sum_{m=1}^{M} \lambda_m \cdot R_m(o_i, y)
$$

其中 $M$ 是奖励函数数量，$\lambda_m$ 是权重（默认全为 1）。本项目中 `ranking` 模式 $M=2, \lambda_1=\lambda_2=1$，即：

$$
r_i = R_{\text{rule}}(o_i, y) + R_{\text{ndcg}}^{(g)}(o_i)
$$

**源码**：`minionerec_trainer.py:961`
```python
rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
```

### 三、优势函数（组内归一化）

**源码**：`minionerec_trainer.py:963-970`

将 $B \times G$ 个 reward 按组 reshape，计算每组的均值和标准差：

$$
\mu_g = \frac{1}{G} \sum_{j=1}^{G} r_j^{(g)}, \quad
\sigma_g = \sqrt{\frac{1}{G} \sum_{j=1}^{G} \bigl(r_j^{(g)} - \mu_g\bigr)^2}
$$

归一化优势值：

$$
\hat{A}_i^{(g)} = \frac{r_i^{(g)} - \mu_g}{\sigma_g + \epsilon}, \quad \epsilon = 10^{-4}
$$

其中 $\epsilon$ 防止 $\sigma_g = 0$（全零 reward 场景）时除零产生 NaN。

### 四、GRPO 损失函数（3 种变体）

#### 1. 标准 GRPO（默认使用）

**源码**：`minionerec_trainer.py:1033-1062`

令 $\pi_\theta$ 为当前策略模型，$\pi_{\text{ref}}$ 为参考模型（冻结的 SFT 模型），$o_{i,t}$ 为第 $i$ 个候选的第 $t$ 个 token，$m_{i,t} \in \{0,1\}$ 为 completion mask。

**Step 1：逐 token log 概率**

$$
\log \pi_\theta(o_{i,t} \mid o_{i,<t}), \quad \log \pi_{\text{ref}}(o_{i,t} \mid o_{i,<t})
$$

**Step 2：KL 散度近似（逐 token）**

$$
d_{\text{KL}}(o_{i,t}) = \exp\bigl(\underbrace{\log \pi_{\text{ref}}(o_{i,t}) - \log \pi_\theta(o_{i,t})}_{\Delta_{i,t}}\bigr) - \Delta_{i,t} - 1
$$

这是 KL 散度的无偏近似（Schulman et al., 2020），满足 $d_{\text{KL}}(\Delta=0) = 0$，且对正负 $\Delta$ 对称惩罚。

**Step 3：逐 token 损失**

$$
\ell_{i,t} = -\Biggl[ \underbrace{\exp\bigl(\log \pi_\theta(o_{i,t}) - \log \pi_\theta(o_{i,t})_{\text{detach}}\bigr)}_{=1 \text{（前向恒为 1，反向保留梯度）}} \cdot \hat{A}_i - \beta \cdot d_{\text{KL}}(o_{i,t}) \Biggr]
$$

注意 `exp(log π - log π.detach())` 在前向传播时恒等于 1（因为两个值相同），但反向传播时梯度为 $\nabla_\theta \log \pi_\theta$，实现了策略梯度定理的要求。

**Step 4：聚合（标准 GRPO）**

$$
\mathcal{L}_{\text{GRPO}} = \frac{1}{B} \sum_{i=1}^{B} \frac{\sum_{t=1}^{|o_i|} \ell_{i,t} \cdot m_{i,t}}{\sum_{t=1}^{|o_i|} m_{i,t}}
$$

每个候选的 loss 在 token 维度取加权平均，再对 batch 取平均。

#### 2. DAPO 变体

**源码**：`minionerec_trainer.py:1054-1055`

$$
\mathcal{L}_{\text{DAPO}} = \frac{\sum_{i=1}^{B} \sum_{t=1}^{|o_i|} \ell_{i,t} \cdot m_{i,t}}{\sum_{i=1}^{B} \sum_{t=1}^{|o_i|} m_{i,t}}
$$

区别：先对所有 token 求和再除以总 token 数（全局平均），而非每个候选独立平均。这使得长候选对 loss 的贡献更大。

#### 3. GSPO 变体

**源码**：`minionerec_trainer.py:1056-1060`

定义序列级 importance ratio：

$$
s_i = \exp\Biggl(\frac{\sum_{t=1}^{|o_i|} (\log \pi_\theta(o_{i,t}) - \log \pi_\theta(o_{i,t})_{\text{detach}}) \cdot m_{i,t}}{\sum_{t=1}^{|o_i|} m_{i,t}}\Biggr)
$$

定义序列级 KL 散度：

$$
\bar{d}_i = \frac{\sum_{t=1}^{|o_i|} d_{\text{KL}}(o_{i,t}) \cdot m_{i,t}}{\sum_{t=1}^{|o_i|} m_{i,t}}
$$

GSPO 损失：

$$
\mathcal{L}_{\text{GSPO}} = -\frac{1}{B} \sum_{i=1}^{B} \bigl(s_i \cdot \hat{A}_i - \beta \cdot \bar{d}_i\bigr)
$$

GSPO 在**序列级别**计算 importance ratio 和 KL，而非 token 级别。这减少了 token 级别的方差，但梯度信号更粗粒度。

### 五、完整公式汇总

以本项目默认配置（`reward_type="ranking"`, 标准 GRPO）为例，完整损失函数为：

$$
\boxed{
\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\Biggl[
\hat{A}_i - \beta \cdot \bigl(e^{\Delta_{i,t}} - \Delta_{i,t} - 1\bigr)
\Biggr]
}
$$

其中：

$$
\hat{A}_i = \frac{r_i - \mu_g}{\sigma_g + 10^{-4}}, \quad
r_i = \mathbb{1}[o_i = y] + R_{\text{ndcg}}^{(g)}(o_i), \quad
\Delta_{i,t} = \log \pi_{\text{ref}}(o_{i,t}) - \log \pi_\theta(o_{i,t})
$$

$$
\mu_g = \frac{1}{G}\sum_{j=1}^G r_j, \quad
\sigma_g = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \mu_g)^2}
$$

**超参数**：$\beta = 10^{-3}$，$G = 4$，$\epsilon = 10^{-4}$。

### 六、SFT 损失函数

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{N-P}\sum_{t=P}^{N-1} \log P_\theta(x_{t+1} \mid x_1, \ldots, x_t)
$$

其中 $x_1, \ldots, x_P$ 为 prompt 部分（被 mask），$x_{P+1}, \ldots, x_N$ 为 completion 部分（参与 loss）。

### 七、评估指标

#### NDCG@K

$$NDCG@K = \frac{DCG@K}{IDCG@K} = \frac{\sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}}{IDCG@K}$$

**单物品场景推导**（本项目评估场景——每条样本只有唯一目标商品）：

设候选集中只有一个相关物品（$rel=1$），其余 $rel=0$。模型返回长度为 $K$ 的推荐列表。

**DCG@K**：若相关物品出现在第 $pos$ 位（$1 \leq pos \leq K$），只有该位置贡献非零值：

$$
DCG@K = \frac{1}{\log_2(pos + 1)}
$$

若相关物品不在前 $K$ 位（$pos > K$ 或未召回），则 $DCG@K = 0$。

**IDCG@K**：理想情况下相关物品排在第一位（$pos=1$）：

$$
IDCG@K = \frac{1}{\log_2(1 + 1)} = \frac{1}{1} = 1
$$

（$K \geq 1$ 时 IDCG 恒为 1。）

**NDCG@K** 简化为：

$$
NDCG@K = \frac{DCG@K}{1} = DCG@K =
\begin{cases}
\frac{1}{\log_2(pos + 1)}, & \text{若 } pos \leq K \\
0, & \text{否则}
\end{cases}
$$

即 NDCG@K 仅由目标商品在推荐列表中的排名位置决定——排名越靠前，NDCG 越高。

#### Hit Ratio@K

$$HR@K = \frac{|\{u : target_u \in \text{topK}(u)\}|}{|U|}$$

---
## ❓ 常见问题 (FAQ)

### Q1: 评估时模型是逐个生成 10 个候选吗？

**不是。** 评估时通过**束搜索（beam search）**一次性并行生成全部 K 个候选。

在 `evaluate.py` 中（第170-198行），配置 `num_beams=K` 和 `num_return_sequences=K`：

```python
generation_config = GenerationConfig(
    num_beams=num_beams,              # K 个并行束
    num_return_sequences=num_beams,   # 一次返回 K 个候选
    top_k=None,                       # 禁用随机采样
    top_p=None,
)

# 单次 generate() 调用 — 所有 K 个候选同时生成
generation_output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    generation_config=generation_config,
)

# 将输出重塑为每样本 K 个候选
real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
```

**工作原理**：束搜索维护 **K 个并行的束**，每一步保留概率最高的 K 个序列前缀。所有 K 个候选在单次 `model.generate()` 调用中**同时推进、同时剪枝**，而非逐个跑 K 次自回归解码。

其他场景的候选生成策略：

| 路径 | 策略 |
|------|------|
| `evaluate.py` 评估 | **束搜索** — K 个束并行解码 |
| RL 训练（束搜索模式） | **束搜索** — 同上 |
| RL 训练（采样模式） | **提示复制 K 份，批量独立采样** |
| SASRec 等基线模型 | **单次前向传播对所有物品打分**，取 top-K |

---

### Q2: NDCG 的排序依据是什么？是根据生成候选的 token 概率排序吗？

**间接依据 token 概率，但实际依据束搜索返回的候选顺序。**

NDCG 的计算代码（`calc.py:59-74`）只看候选在列表中的**位置索引**：

```python
# 遍历生成候选，找到目标物品第一次出现的位置
for i in range(len(sample)):
    if sample[i] == target_item:
        minID = i    # 这个位置就是 ranking
        break

# NDCG = 1 / log2(rank + 1)，其中 rank = minID + 1
ALLNDCG[index] = ALLNDCG[index] + (1 / math.log(minID + 2))
```

代码**没有读取任何 token 概率或模型分数**，完全依赖 `sample[i]` 的索引位置作为排序依据。

**但候选的顺序从哪来？** HuggingFace `model.generate()` 在束搜索模式下，会按**总分（累计 log 概率）从高到低**返回 K 个候选序列：

```
output[0] = 分数最高的序列（最佳候选，累计 log 概率最大）
output[1] = 分数第二高的序列
...
output[K-1] = 分数最低的序列
```

所以 NDCG 的排序**隐式地等价于按束搜索的累计 log 概率排序**：

```
NDCG 排序 ≈ 束搜索的累计 log 概率排序
           （通过候选在 output 列表中的位置隐式表达）
```

**流程总结**：

```
model.generate()
    │
    ▼
束搜索解码 K 个候选，每个候选有累计分数（log 概率和）
    │
    ▼
HuggingFace 按分数从高到低排列输出 [cand_0, cand_1, ..., cand_{K-1}]
    │
    ▼
calc.py 遍历列表找目标物品在位置 pos
    │
    ▼
NDCG@K = 1 / log2(pos + 1)  （若 pos < K）
```

**与 SASRec 等基线模型的区别**：

| 模型类型 | 候选排序方式 |
|----------|-------------|
| 生成式模型（LLM + Beam Search） | **隐式排序**：束搜索按累计 log 概率排好候选顺序，calc.py 直接读取位置 |
| SASRec / GRU / Caser | **显式排序**：对所有物品一次打分，按分数降序取 top-K，分数就是排序依据 |

**一句话总结**：NDCG 排序不直接读取 token 概率，而是依赖束搜索返回的**候选顺序**，该顺序由束搜索的**累计 log 概率**决定。这是一个"隐式排序"机制——位置即排名，排名即概率顺序。

---

## ❓ 常见问题 (Q&A)

### Q1: Sinkhorn 算法在项目中具体是怎么解决 SID 碰撞的？

#### 碰撞的本质是什么？

SID 碰撞指的是两个不同的商品被分配到完全相同的三码 SID（三层的索引都一样）。要理解碰撞为什么发生，先看 RQ-VAE 默认的分配方式——**argmin（最近邻）**。

argmin 分配的逻辑非常简单：每个商品独立地在 256 个原型中找到离自己最近的那个，然后选它作为自己的 SID 码。这个策略的问题是：**每个商品只看自己，不看整体**。

具体来说，当两个语义相似的商品的 embedding 在向量空间中比较接近时，它们很可能在每一层都选了同一个原型：

```
商品 A: "无线鼠标" → embedding → 第1层选<a_5>, 第2层选<b_12>, 第3层选<c_88>
商品 B: "蓝牙键盘" → embedding → 第1层选<a_5>, 第2层选<b_12>, 第3层选<c_88>
                                          ↓
                                碰撞！SID = <a_5><b_12><c_88>
```

这不是巧合导致的异常现象，而是 argmin 机制的必然结果。更根本地说，碰撞的根源是 **"局部最优决策导致的全局分配不平衡"**——某些原型因为处于密集区域被大量商品选为最近邻（可能分配了几百个商品），而另一些原型落在稀疏区域，几乎无人问津。这种不平衡到了一定程度，碰撞就是不可避免的。

#### Sinkhorn 解决问题的思路

Sinkhorn 算法改变了分配的逻辑：**不是每个商品独立决策，而是所有商品一起协调，在"每个商品尽量选近的原型"和"每个原型分配的商品数大致均匀"两个目标之间找到最优平衡**。

这个思想被形式化为一个**带约束的最优传输（Optimal Transport）问题**：

$$
\min_{P} \sum_{i=1}^{N}\sum_{j=1}^{K} P_{ij} C_{ij} - \varepsilon H(P)
\quad \text{s.t.} \quad \sum_j P_{ij} = a_i,\ \sum_i P_{ij} = b_j
$$

这个数学公式的每一项、每一个符号都对应到项目中的具体变量，理解它们才能真正理解 Sinkhorn 为什么能消除碰撞。

#### 公式中每个参数的实物对应

**商品端参数：**

- **\(N\)**（代码：`N = residuals.shape[0]`）：参与这一层分配的商品总数。在全量 SID 生成时是 3,686（项目的全部商品数）；在碰撞消除阶段，每次只对碰撞组内的商品重编码，N 就是该碰撞组的商品数量。

- **\(a_i\)**（代码：`a = np.ones(N) / N`）：每个商品的"质量"权重，含义是每个商品在分配中具有相等的地位。数学上它对应行约束 \(\sum_j P_{ij} = a_i\)，即每个商品分配到所有原型的概率之和等于 \(1/N\)，所有商品加起来恰好为 1。这是一个归一化的权重分配，确保没有商品被特殊对待。

**原型端参数：**

- **\(K\)**（代码：`K = centroids.shape[0]`）：这一层 codebook 的原型数。本项目 RQ-VAE 的三层 codebook 大小固定为 **256**，即每层有 256 个可供分配的原型向量。

- **\(b_j\)**（代码：`b = capacities / float(N)`）：每个原型的"容量"约束。代码里 `capacities` 是这样计算的：

```python
capacities = np.full(K, N // K, dtype=np.int64)   # 每个原型分 N//K 个
capacities[: (N % K)] += 1                         # 余数均摊到前几个原型
```

对于 N=3,686、K=256，N // K = 14，余数 3686 - 14×256 = 102，所以前 102 个原型各分 15 个商品，后 154 个原型各分 14 个商品。\(b_j\) 就是 `capacities[j] / N`，即每个原型应分配的**相对权重**。列约束 $\sum_i P_{ij} = b_j$ 的物理含义就是：**每个原型恰好分配到 N/K 个商品**。

**这是 Sinkhorn 消除碰撞的核心机制——argmin 不限制每个原型分配多少商品，所以某些原型可能挤了几百个商品导致碰撞；Sinkhorn 通过 \(b_j\) 施加硬约束，强制分配均衡，从根上消除了碰撞的土壤。**

**代价和正则化参数：**

- **\(C_{ij}\)**（代码：`D_full = pairwise_sq_dists_batch(residuals, centroids)`）：代价矩阵，表示第 \(i\) 个商品的当前残差向量到第 \(j\) 个原型向量的 **L2 距离平方**。这是一个形状为 `[N, 256]` 的矩阵，\(C_{ij}\) 的值越大，表示商品 i 和原型 j 在语义上越不匹配。Sinkhorn 优化的目标之一就是让总的 \(\sum P_{ij} C_{ij}\) 尽可能小，即**让商品尽量分配到语义相近的原型上去**。

  注意"残差"的含义：在 RQ-VAE 的第三层使用 Sinkhorn 时，输入不是原始 embedding，而是经过前两层量化后的**残差** \(r_2 = x - q_1 - q_2\)。因为前两层已经编码了"大类"和"中类"的信息，残差代表的是最细粒度的语义差异。Sinkhorn 在这个残差空间做均匀化，相当于在"保持大小类归属不变"的前提下调整第三层的分配。

- **\(\varepsilon\)**（代码：`sk_epsilon` 或 `tau`）：熵正则化强度，是整个 Sinkhorn 算法中**最关键的参数**。它控制两个目标的权重——最小化传输代价和满足均匀约束。它的物理含义通过极端取值更容易理解：

  - \(\varepsilon \to 0\)：熵项 \(\varepsilon H(P)\) 消失，目标退化为 \(\min \sum P_{ij} C_{ij}\)，最优解是每个商品独占地选择最近的原型，P 退化为 one-hot 矩阵——这等价于 argmin。结果：语义距离最优，但分配极度不平衡，碰撞率高。
  
  - \(\varepsilon \to \infty\)：熵项主导，最优解是 P 变成完全均匀矩阵，每个商品等概率分配到所有原型。结果：分配绝对均衡，零碰撞，但语义距离完全被破坏，商品和原型之间的对应关系丢失。
  
  - **\(\varepsilon = 0.003\)（本项目取值）**：P 是"接近 one-hot 的软分布"，90% 以上的概率权重仍集中在最近的原型上，但会匀出极小的一部分给第二近、第三近的原型。结果：**碰撞消除，语义几乎无损**。

- **\(P_{ij}\)**（代码：`P = ot.sinkhorn(a, b, D_full, tau)`）：传输矩阵，形状 `[N, 256]`，是算法最终输出的中间结果。\(P_{ij}\) 表示第 \(i\) 个商品分配到第 \(j\) 个原型的概率。这个矩阵同时满足行约束（\(\sum_j P_{ij} = a_i\)，每个商品权重和为 1/N）和列约束（\(\sum_i P_{ij} = b_j\)，每个原型被分配的权重和为 capacities[j]/N）。

  P 是概率矩阵，不能直接作为离散的 SID 分配使用。需要一步**概率到离散的转换**：

```python
remaining = capacities.copy()        # 每个原型的剩余容量
order = np.arange(N)
rng.shuffle(order)                   # 随机打乱顺序避免系统偏差
for i in order:
    probs = P[i]                     # 商品 i 分配到各原型的概率
    cand = np.argsort(-probs)        # 按概率降序遍历
    for c in cand:
        if remaining[c] > 0:         # 如果原型 c 还有容量
            assign[i] = c            # 分配
            remaining[c] -= 1
            break
    # 兜底：如果所有原型都满了，选剩余容量最多的原型
```

**\(H(P) = -\sum_{i,j} P_{ij} (\log P_{ij} - 1)\)**：信息熵项。它的作用是让传输矩阵 P"软化"——当 \(\varepsilon = 0\) 时 P 是硬 one-hot，不可求导且没有灵活性；加入熵项后 P 变成软分布，既保留了"最近原型概率最大"的结构，又给次优原型留了概率空间，使得交替归一化迭代可以收敛。

**完整的参数对照表：**

| 数学符号 | 代码中的变量 | 在本项目中的含义 | 具体值 |
|---------|-------------|----------------|--------|
| \(N\) | `N = residuals.shape[0]` | 待分配的商品数 | 全量 3,686；或碰撞组内商品数 |
| \(K\) | `K = centroids.shape[0]` | 本层 codebook 原型数 | 固定 **256**（每层相同） |
| \(a_i\) | `a = ones(N) / N` | 每个商品的分配权重，行约束 | `[1/N, ..., 1/N]`，和 = 1 |
| \(b_j\) | `b = capacities / N` | 每个原型的容量约束，列约束 | `capacities[j] ≈ N/256 ≈ 14` |
| \(C_{ij}\) | `pairwise_sq_dists_batch(residuals, centroids)` | 残差到原型的 L2 距离平方 | 形状 `[N, 256]`，浮点数矩阵 |
| \(\varepsilon\) | `sk_epsilon` / `tau` | 熵正则化强度 | **0.003**（第三层） |
| \(P_{ij}\) | `ot.sinkhorn(a, b, D_full, tau)` | 传输概率矩阵 | 形状 `[N, 256]`，每行和 ≈ 1/N |
| \(H(P)\) | —（隐含在 Sinkhorn 迭代中） | 熵项，软化传输矩阵 | \(-\sum P_{ij}(\log P_{ij} - 1)\) |

#### Sinkhorn 迭代的直观过程

Sinkhorn 求解上述问题的方法非常优雅——**交替行/列归一化**：

```
初始化: P[i][j] = exp(-C[i][j] / ε)    ← 将距离转化为相似度

迭代:
  第1步: 行归一化 — 每行除以行和，再乘以 a_i
          → 满足"每个商品权重和 = 1/N"
  第2步: 列归一化 — 每列除以列和，再乘以 b_j
          → 满足"每个原型分配权重 = capacities[j]/N"
  
重复行/列归一化直到收敛（通常 50 步以内）
```

为什么这个简单的交替操作能解决问题？它的数学本质是 **KL 散度下的交替投影**——行归一化把 P 投影到满足行约束的凸集上，列归一化把 P 投影到满足列约束的凸集上。两个约束同时满足的解就是最优传输矩阵。这个过程也叫 **Sinkhorn-Knopp 算法**，从 1960 年代就被证明是线性收敛的。

#### 项目中的三层递进策略

项目中 Sinkhorn 不是在所有层无差别使用的，而是采用**三层递进、最小必要干预**的策略：

```python
# 生成 SID 阶段 (generate_indices.py)
for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon = 0.0           # 前两层：不用 Sinkhorn，纯 argmin
model.rq.vq_layers[-1].sk_epsilon = 0.003  # 第三层：极轻的 Sinkhorn
```

为什么前两层不用？这是对 RQ-VAE 层次结构的深刻理解：

- **第一层**（`<a_N>`）划分**大类**——功能类似的商品归入同一大类。例如所有"办公用品"共享同一个第一层码。这一层的语义结构至关重要，用均匀约束反而会打乱大类归属。
- **第二层**（`<b_N>`）划分**中类**——在大类下进一步细分。例如"办公用品"下的"打印耗材"和"文具"。同样，这一层的细分应该由数据本身的聚类结构决定，不应强制均匀。
- **第三层**（`<c_N>`）划分**具体商品**——最细粒度的区分。碰撞的本质就是第三层上的冲突：两个商品前两层码相同（同属一个中类下），第三层本应区分它们但 argmin 没有做到，因为它们的残差向量太相似。这时候在第三层上施加轻量的 Sinkhorn 约束，等于说"既然你们被分到了同一个中类，第三层你们各自选不同的细类，把你们区分开"。

**这就是为什么 \(\varepsilon = 0.003\) 就够了**——不是在均匀分布和语义之间取舍，而是在"第三层保持最优距离"和"第三层避免重复"之间做一个极轻微的调整。因为前两层已经保证了语义的大框架，第三层的调整幅度很小，不需要大的 epsilon。

#### 迭代消除——从"有碰撞"到"零碰撞"

Sinkhorn 不是一次运行就搞定所有碰撞，而是配合一个**迭代检测-重编码-再检测**的闭环：

```python
tt = 0
while True:
    if tt >= 20 or check_collision(all_indices_str):
        break

    # 检测：找出当前所有碰撞组
    collision_groups = get_collision_item(all_indices_str)

    # 消除：对每个碰撞组，只提取这些碰撞商品启用 Sinkhorn 重编码
    for collision_items in collision_groups:
        d = data[collision_items].to(device)
        indices = model.get_indices(d, use_sk=True)  # Sinkhorn 分配
        # 更新这些商品的 SID
        all_indices[item] = code
        all_indices_str[item] = str(code)
    tt += 1
```

这个流程有几个关键设计点：

1. **只对有碰撞的商品重编码**：没有碰撞的商品保持不动。这保证了最小化对原有语义结构的扰动。
2. **每次重编码后重新检测碰撞**：因为重新分配可能解决原有碰撞，但也可能（理论上）产生新的碰撞。重新检测确保可以收敛。
3. **最多 20 轮上限**：防止死循环。实际中通常在 1-3 轮内就收敛到零碰撞。
4. **全量商品用 argmin，碰撞组用 Sinkhorn**：初始分配尽可能保持语义结构，只在必要的时候用均匀约束纠偏。

**效果**：本项目 3,686 个商品经过迭代消除后，碰撞率降至 **0%**。

#### RQ-Kmeans 路线中的另一种用法

在 `rqkmeans_faiss.py` 中还有一个不同的用法——使用 POT 库（Python Optimal Transport）的 `ot.sinkhorn()` 对 FAISS ResidualQuantizer 的输出做**全三层的后处理平衡**：

```python
for l in range(M):  # 对每一层
    residuals = compute_residuals_upto_level(data, codes, upto_level=l)
    new_ids = sinkhorn_balance_level(residuals, codebooks[l])
    codes[:, l] = new_ids
```

这里每一层都重新用 Sinkhorn 做均衡分配。区别在于：

| 方法 | 在哪个阶段用 | 覆盖范围 | epsilon 来源 |
|------|------------|---------|------------|
| RQ-VAE 路线 | 迭代重编码时 | 仅碰撞组 | 固定 0.003（第三层） |
| FAISS RQ-Kmeans 路线 | 全量后处理 | 所有商品、所有层 | `estimate_tau()` 自动估计 |

FAISS 路线中 `estimate_tau()` 用一个启发式方法估算合适的 epsilon：

```python
def estimate_tau(residuals, centroids, sample_size=4000, percentile=90):
    # 1. 随机采样 4000 个商品
    # 2. 计算每个商品到所有原型的距离矩阵
    # 3. 对每个商品，算"第二近的距离 - 最近的距离"
    #    （这个差值反映了距离分布的散开程度）
    # 4. 取第 90 百分位 × 0.1 作为 tau
    spread = np.percentile(D - D.min(axis=1, keepdim=True), 90, axis=1)
    tau = float(np.median(spread) * 0.1)
    return max(tau, min_tau)
```

这样估算的 tau 可以随着每层残差的分布特性自适应调整，不需要手调。

#### 为什么 Sinkhorn 比别的方法更适合？

| 方法 | 原理 | 碰撞控制 | 语义保持 | 适合场景 |
|------|------|---------|---------|---------|
| **argmin（无约束）** | 每个商品独立选最近原型 | ❌ 完全不控制 | ✅ 语义距离最优 | 无碰撞要求的纯量化 |
| **Sinkhorn（本项目）** | 最优传输 + 熵正则化 | ✅ 均匀约束，ε 可调 | ✅ ε 控制 trade-off | **碰撞消除 + 语义保持** |
| 硬约束均衡 | 强制每簇大小完全相等 | ✅ 严格均衡 | ❌ 可能强行分割语义相近的商品 | 纯均衡任务 |
| 随机重分配 | 碰撞后随机打乱 | ✅ 零碰撞 | ❌ 语义完全破坏 | 仅作对比基线 |

Sinkhorn 的独特优势用四个字总结就是——**连续可调**。\(\varepsilon\) 从 0 到 ∞ 连续变化，算法行为从"纯 argmin"平滑过渡到"纯均匀分布"，不存在"要么不管碰撞、要么破坏语义"的二元取舍。项目中选择 \(\varepsilon = 0.003\)，相当于在最优解附近做了一步微调——把最近邻分配中不平衡的那部分"推"向均衡，但推的幅度只有 0.3%，对语义距离的影响微乎其微。

#### 完整流程总结

```
① RQ-VAE 训练：监控碰撞率，保存 best_collision_model.pth
     │
     ▼
② 用 argmin 为所有商品生成初始 SID
     │
     ▼
③ 检测碰撞
     │
     ├── 无碰撞 → 输出 index.json (collision_rate = 0%)
     │
     └── 有碰撞 → 提取碰撞组
                     │
                     ▼
                  ④ 对碰撞组商品启用 Sinkhorn 重编码
                     （第三层, ε=0.003）
                     │
                     ▼
                  ⑤ 更新 SID → 回到步骤③重新检测
                     （最多循环 20 轮）
```

最终结果：**3,686 个商品，100% 唯一 SID，0% 碰撞率**。

---

### Q2: 为什么 tokenizer 只添加了 560 个 SID token（不是 768 个）？第一层 codebook 利用率为什么只有 48/256？

**实际数据**（通过分析 `index.json` 得出）：

| 层 | 实际使用的原型数 | codebook 总大小 | 利用率 |
|----|---------------|----------------|-------|
| 第一层 `<a_>` | 48 | 256 | **18.8%** |
| 第二层 `<b_>` | 256 | 256 | **100%** |
| 第三层 `<c_>` | 256 | 256 | **100%** |
| **总计** | **560** | **768** | **72.9%** |

560 = 48 + 256 + 256。所以不是所有层都"利用不足"，全部缺口来自第一层。

#### 为什么第一层只有 48 个原型被选中？

**关键：这不是 codebook collapse（死亡螺旋）。**

真正的 collapse 是全局性的——如果初始化差或训练不稳定，所有层都会崩。但这里第二层和第三层都是 100% 用满的，说明训练本身没有问题。

**根本原因：编码器在训练中不断更新，而 argmin 决定了哪些原型存活。**

k-means 初始化确实用 k=256 在第一个 batch（2,048 个商品）上铺了 256 个质心。但初始化之后的事情更关键：

```python
# 初始化阶段
first_batch = data[0:2048]           # 第一个 batch
kmeans = KMeans(n_clusters=256)       # k=256 铺满
centroids = kmeans.fit(first_batch)   # 256 个质心
codebook = centroids                  # 初始值

# 训练阶段—编码器在变
# encoder(x): 768d → MLP(可学习) → 32d
# 编码器权重在梯度更新 → 编码器输出分布不断漂移
# 只有分布在密集区的原型被选中 → 有梯度 → 继续存活
# 分布在稀疏区的原型始终不被选中 → 梯度为零 → 饿死
```

这是 argmin 机制的必然结果：

```
初始化时 (编码器输出分布 A):       收敛后 (编码器输出分布 B):
  o  o  o  o  o                      ●  ●  ●  ·  ·
  o  o  o  ·  ·              →       ●  ●  ●  ·  ·
  ·  ·  ·  ·  ·                      ·  ·  ·  ·  ·
  ^  ^  ^                            ∣
  ∣                                  256 个坑位只填了 48 个
  初始化均匀铺了 256 个质心          数据密度决定有多少"活下来"
```

#### 为什么 48 这个数字？不是 43、不是 52？

**48 没有特殊含义。** 它是数据、模型、初始化、训练动态共同作用下的结果。换一个随机种子重训，可能是 44 或 53，但会稳定在 **40~60 之间**，不会是 200。

真正的问题是：**为什么 256 个坑位只填了 40~60？**

回答这个问题需要看 RQ-VAE 第一层的角色和数据的本征结构：

1. **第一层的角色是粗粒度分类**。3,686 个 Industrial & Scientific 商品，在 32 维的编码空间里的自然大类数大约就是 40~60 个。不会有 256 个那么多——亚马逊自己在这个品类下的粗分类也远不到 256 个。

2. **32 维空间的本质维度受限**。编码器把 768 维的 Qwen embedding 压缩到 32 维，降维过程本身就在丢弃信息、保留最主要的语义轴。在 32 维空间中，3,686 个工业用品的点的**本征维度**就更低了——它们不会均匀散开，而是聚集在少数几个区域。

3. **argmin 的赢家通吃**。密集区的质心会持续获得分配和更新，边缘区的质心梯度永远为零，进入"不死不活"的状态——参数还在，但从不被选中，也不会更新。

#### 强制填满 256 个会怎样？

如果对第一层施加 Sinkhorn 均匀约束（`sk_epsilon > 0`）：

```
<a_5>: 测量仪器类 → 强行分成 <a_5> 和 <a_100>
<a_12>: 紧固件类   → 强行分成 <a_12> 和 <a_101>
```

这会**把一个语义上自然的粗类劈成两半**，破坏 RQ-VAE 精心设计的层次结构——第一层本应做粗分类，均匀约束会让语义相近的商品被分到不同"大类"下，打乱整个树状编码结构。

**这就是为什么项目代码中明确把前两层的 `sk_epsilon` 设为 0.0——宁可利用率低，也要保护语义结构。**

#### 那这到底是不是问题？

不是问题，而且有客观证据：

| 指标 | 本项目 | 如果是真 collapse 的表现 |
|------|-------|------------------------|
| 第一层利用率 18.8% | ⚠️ 看起来低 | ✅ 符合 |
| 第二层利用率 100% | ✅ 正常 | ❌ 也会很低 |
| 第三层利用率 100% | ✅ 正常 | ❌ 也会很低 |
| Sinkhorn 消除碰撞后 | ✅ 0% 碰撞 | ❌ 不可能 |
| 推荐 HR@10 | ✅ ~10%+ | ❌ 应该极差 |

**第二、三层的 100% 利用率是决定性的证据**：如果真的是 codebook collapse（初始化差、承诺损失不对、优化不稳定），不可能出现"所有层都 collapse 但只有第一层死了"的情况。唯一合理的解释是：**这个数据集在粗粒度上的自然聚类数就是 40~60 个，256 个坑位对第一层来说太多了。**

换个品类更多样化的数据集（如 Amazon Beauty，12,000+ 商品跨多个品类），第一层利用率自然会上升。

#### 一句话总结

> 只添加 560 个 token 是因为第一层 codebook 的 256 个坑位对于这个数据集的粗粒度分类来说过多了——3,686 个工业用品在 32 维编码空间只有约 40~60 个自然粗类。这不是 codebook collapse（证据是后两层 100% 利用），而是 **argmin 机制下数据本身聚类结构决定的正常现象**。项目在前两层刻意不用 Sinkhorn，正是为了保护这种自然聚类不被强行均匀约束破坏。

---

## 📖 引用

```bibtex
@misc{MiniOneRec,
    title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation},
    author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
    year={2025},
    eprint={2510.24431},
    archivePrefix={arXiv},
    primaryClass={cs.IR},
}
```
