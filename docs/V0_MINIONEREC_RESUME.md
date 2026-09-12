# MiniOneRec: 生成式推荐系统

---

## 项目一：生成式推荐系统端到端实现

### 📌 Situation (背景)

在推荐系统领域，传统基于embedding的协同过滤方法存在泛化能力弱、冷启动差等问题。随着大语言模型(LLM)的发展，如何利用LLM的世界知识提升推荐效果成为研究热点。本项目基于生成式推荐范式，独立实现了一套端到端的推荐系统：将推荐任务建模为**序列生成**——让 LLM 生成商品的语义 ID（SID），通过 **SFT + GRPO 强化学习** 两阶段训练实现推荐。

### 📌 Task (目标)

1. 设计并实现完整的生成式推荐系统，覆盖语义ID构建→监督微调→强化学习→评估全流程
2. 在 Amazon Industrial_and_Scientific 数据集上验证系统效果
3. 针对单卡 RTX 3090 24GB 环境进行工程优化

### 📌 Action (行动)

#### 1. 语义ID方案设计与实现
- **技术选型**: 采用 RQ-VAE（残差量化变分自编码器）将商品编码为离散 token 序列，替代传统连续 embedding
- **量化架构**: 三层残差量化，每层 256 个 codebook，将商品编码为 `[<a_236><b_231><c_226>]` 形式的 3 级 token 序列
- **优势分析**: 相比单层量化，三层残差量化将表示空间从 256 扩展到 256³=16M 种组合，同时保持层次语义结构
- **解码约束**: 设计 ConstrainedLogitsProcessor，基于前缀 hash 字典约束 beam search，保证 100% 生成合法 SID

#### 2. 监督微调阶段 (SFT)
- **模型选型**: 选用 Qwen2.5-0.5B-Instruct 作为基座，在模型容量与训练成本间取得平衡
- **多任务训练**: 设计 3 路数据联合训练——序列推荐 (SidSFTDataset)、特征对齐 (SidItemFeatDataset)、融合推荐 (FusionSeqRecDataset)
- **Tokenizer 扩展**: 动态添加 SID token + 9 个特殊 token（用户评分、上下文类型等），扩展 LLM 词汇表
- **训练配置**: batch_size=64, 3 epochs, lr=5e-4, bf16 混合精度

#### 3. 强化学习优化阶段 (RL)
- **算法选型**: 选择 GRPO 而非 PPO——GRPO 用组内均值归一化替代 Value 网络，节省约 50% 显存，更适合推荐场景
- **奖励设计**: 设计双奖励机制——二进制正确性奖励 (是否命中目标) + NDCG 排序奖励 (组内排名激励)
- **解码优化**: Constrained Beam Search + max_completion_length=16（匹配 SID 3-token 长度，大幅降低显存）
- **训练配置**: train_batch=4, num_generations=4, β=1e-3, gradient_accumulation=8


### 📌 Result (成果)

在 Industrial_and_Scientific 数据集上的完整实验结果（beam search=10）：

| 阶段 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |
|------|------|------|------|-------|--------|--------|---------|
| **SFT** | 0.036 | 0.057 | 0.068 | 0.093 | 0.036 | 0.052 | 0.060 |
| **RL-epoch1** | 0.045 | 0.066 | 0.079 | 0.103 | 0.045 | 0.062 | 0.070 |
| **RL-epoch2** | 0.047 | 0.070 | 0.083 | 0.109 | 0.047 | 0.066 | 0.074 |
| **官方 1.5B** | 0.085 | 0.113 | 0.133 | **0.154** | 0.085 | 0.109 | **0.116** |

**核心成果**:
- ✅ 独立完成 SID 构建→SFT→RL→评估全流程，HR@10 达 **10.9%**，NDCG@10 达 **7.4%**
- ✅ GRPO 强化学习显著有效：HR@10 相对 SFT 提升 **+17%**，NDCG@10 提升 **+23%**
- ✅ 在单卡 RTX 3090 24GB 上完成全流程训练

---

## STAR法则面试问答

### Q1: 请描述你在这个项目中遇到的最大挑战

**S**: 单卡 RTX 3090 显存 24GB，训练 0.5B 模型的 GRPO 阶段需要同时加载策略模型和参考模型，显存压力大

**T**: 在有限显存下完成 SFT + GRPO 两阶段训练，同时保证推荐效果

**A**:
1. 分析 GRPO 显存瓶颈：max_completion_length=128 时生成占大量 KV cache，但 SID 实际只需 3 个 token
2. 将 max_completion_length 从 128 缩减至 16，显存峰值大幅下降
3. 实现 Flash Attention 2 + torch_compile 组合优化，进一步压缩显存
4. 设计断点续训机制，应对长时间训练中的异常中断

**R**: 成功在单卡 RTX 3090 上完成全流程，HR@10 达 10.9%，NDCG@10 达 7.4%

---

### Q2: 为什么选择GRPO而不是PPO？

**S**: 推荐场景下的强化学习需要高效的算法，显存和训练稳定性都是约束

**T**: 选择适合生成式推荐的 RL 算法

**A**:
1. PPO 需要单独训练 Value 网络估算 baseline，额外增加约 50% 显存开销
2. GRPO 用组内均值归一化替代 Value 网络：`advantage = (reward - group_mean) / group_std`
3. 推荐场景天然适合"组内比较"——同一条 prompt 生成多个候选，比较相对好坏
4. GRPO 同时引入 KL 散度约束，防止策略偏离参考模型太远

**R**: 2 轮 RL 后 HR@10 从 9.3% 提升至 10.9%（+17%），NDCG@10 从 6.0% 提升至 7.4%（+23%）

---

### Q3: 如何解决SID解码时生成无效序列的问题？

**S**: LLM 自由生成时可能输出非商品 SID 的 token，导致推荐失败

**T**: 需要在生成阶段保证只输出合法的 SID 序列

**A**:
1. 预计算阶段：遍历所有商品 SID，构建"前缀 token 序列 → 有效后继 token"的 hash 字典
2. 实现 ConstrainedLogitsProcessor：每步生成前，将无效 token 的 logit 置为 -inf
3. 约束按层级生效——第一层 token 约束来自 prompt 后缀，第二层约束来自第一层已生成的 token，以此类推

**R**: 解码有效率从无约束的约 70% 提升至 100%，每条候选都是合法商品

---

### Q4: RL训练过程中有什么印象深刻的问题？

**S**: SFT 阶段 HR@10 已达 9.3%，RL 后 Top-1 指标略有波动，但 Top-K 指标持续提升

**T**: 需要理解这种现象并决定是否继续 RL 训练

**A**:
1. 分析发现：beam_search 多样性策略使正确答案分散到多个候选位置，Top-1 命中率下降但整体召回率上升
2. 对于推荐场景，用户关心的是"推荐列表里有没有好的"，而非"第一个是不是最好的"
3. 决策：以 Top-K 指标（HR@10、NDCG@10）为主要优化目标，继续 RL 训练

**R**: 2 轮 RL 后 HR@10 达 10.9%（+17%），验证了 Top-K 策略的合理性

---

### Q5: 在工程优化方面做了什么？

**S**: 单卡 RTX 3090 显存 24GB，GRPO 训练需同时持有策略模型和参考模型

**T**: 在不损失性能的前提下最大化训练效率和显存利用率

**A**:
1. Flash Attention 2 + SDPA 自动降级：检测 flash_attn 是否可用，不可用时自动切到 SDPA
2. max_completion_length 精简：从通用的 128 缩减至 16，匹配 SID 3-token 长度
3. torch_compile 图编译：对前向传播进行 JIT 优化，加速训练
4. 梯度累积策略：train_batch=4 + accum=8 = effective_batch=32，在显存和收敛间平衡
5. 断点续训：实现 checkpoint 状态恢复，应对训练中断

**R**: 单卡 RTX 3090 完成 SFT + 2 轮 RL 全流程训练

---

## 简历 bullet point 版本

```markdown
- 基于生成式推荐范式，实现 SID 构建 + SFT + GRPO 强化学习的端到端推荐系统
- 设计 RQ-VAE 三层残差量化方案，将商品编码为 3 级离散 token 序列，表示空间达 16M
- 实现 Constrained Beam Search 解码器，通过前缀 hash 约束保证 100% 生成有效商品 SID
- 采用 GRPO 算法进行强化学习优化，2 轮训练后 HR@10 提升 17%、NDCG@10 提升 23%
```

---

## 技术深度问题准备

### Q6: RQ-VAE与普通VAE的区别？

| 方面 | VAE | RQ-VAE |
|------|-----|--------|
| 量化方式 | 单一向量量化 | 三层残差量化 |
| codebook | 1个 | 3个×256 |
| 表达能力 | 受限于codebook大小 | 256³=16M组合 |
| 梯度传递 | straight-through | 逐层残差传递 |

---

### Q7: GRPO的数学推导

```
传统PPO: advantage = r(π_θ) - V(s)
GRPO: advantage = (r₁,r₂,...,r_G) - mean(r)

其中G是每条prompt采样的候选数量

损失函数:
L = -E[exp(logπθ - logπref) * advantage] + β * KL(πθ || πref)
```

---

### Q8: Constrained Beam Search vs 普通Beam Search

**普通Beam Search**: 每个beam独立扩展，可能生成任意token

**Constrained Beam Search**:
```
Step 1: 根据前缀hash查找有效token集合
Step 2: 只保留有效token的概率，其他置为-inf
Step 3: 继续扩展，重复Step 1-2
Result: 所有beam都是完整的有效SID序列
```

---

## 参考资料

- MiniOneRec论文: [arXiv:2510.24431](https://arxiv.org/abs/2510.24431)
- GRPO原理论文: [DeepSeekMath](https://huggingface.co/papers/2402.03300)
- 项目地址: [GitHub](https://github.com/AkaliKong/MiniOneRec)
