# 解码策略详解：束搜索 / 束采样 / 各类采样

> 2026-09-19 ｜ 面向 `GenRetrieval` 的生成式检索链路。
> 本文回答三个问题：① **采样束搜索**与**贪心/纯束搜索**差在哪；② 还有哪些采样方法；
> ③ 对本项目（SID 结构化约束解码）**什么才是合理选择**。
>
> 数字标注：`[实测]` = 本机跑出来的；`[源码]` = transformers 4.57.1 原文；`[原理]` = 教科书结论。

---

## 0. 一句话总览

| 解码方式 | 每步怎么选 | 随机性 | 输出条数 | 本项目适用性 |
|---|---|---|---|---|
| 贪心 greedy | 取 top-1 | 无 | 1 | ❌ 只有 1 条候选，`HR@K` 无从谈起 |
| 束搜索 beam search | 保留 top-B 条**路径** | 无 | B | ✅ 确定性、可复现 |
| 纯采样 sampling | 按概率**抽** 1 条 | 有 | n | ⚠️ SID 空间大，易抽到无关码字 |
| **束采样 beam sampling** | **每个 beam 独立抽样** | **有** | B | 🔴 **本项目当前实际走的**（继承自基座） |
| 分组束搜索 grouped | 分组 + 组内多样性惩罚 | 无 | B | ⚠️ 见 §4 |
| 对比搜索 contrastive | 惩罚退化重复 | 无 | 1 | ❌ 面向长文本生成 |
| 受限束搜索 constrained | 合法集内 beam | 无 | B | — 本项目已用 Trie，属另一维度 |

**核心区分只有两条轴**：① 每步是**取最大**还是**按概率抽**；② 是否保留**多条路径**。

---

## 1. 从贪心到束搜索：为什么要保留多条路径

### 1.1 贪心解码（greedy）

每步取概率最大的 token：

```
ŷ_t = argmax P(y_t | y_<t)
```

**问题**：局部最优 ≠ 全局最优。第 1 步选错的 token，后面无法回头。

**对本项目**：贪心只产出 **1 条** SID。而 `EVAL_PROTOCOL` 的口径里
`HR@K` 需要 K 条候选 —— **贪心直接让评估失去意义**。

### 1.2 束搜索（beam search）

每步保留累计对数概率最高的 **B 条完整前缀**：

```
score(y_1..y_t) = Σ log P(y_i | y_<i)
```

**关键点：`[源码]` 束搜索的"分数"是累计 log 概率，不是当前步的概率**。
所以一个在第 1 步概率略低的候选，可能因为后续步骤概率很高而最终胜出。

**对本项目**：B=50 ⟹ 产出 50 条 SID 候选，`HR@50` 才有定义。
这也是 MiniOneRec 原版 `evaluate.sh` 写死 `--num_beams 50` 的原因 ——
**不是随手设的，是因为 SID 有 3 层分支，窄 beam 覆盖不住**。

### 1.3 一张图看清三者差异

`[实测]` 我用一个 8 词表、手造 logits 的可控实验跑出来的（`torch.manual_seed` 固定）：

```
实验 1：do_sample=False（纯束搜索）
  seed=0: [[0,1,2,0], [0,1,2,1], [0,1,2,2]]
  seed=0: [[0,1,2,0], [0,1,2,1], [0,1,2,2]]   ← 重复跑
  seed=1: [[0,1,2,0], [0,1,2,1], [0,1,2,2]]   ← 换 seed 也一样
  ⟹ 完全确定性，seed 不影响结果

实验 2：do_sample=True（束采样）
  seed=0: [[0,2,2,7], [0,2,2,3], [0,2,3,0]]
  seed=0: [[0,2,2,7], [0,2,2,3], [0,2,3,0]]   ← 同 seed 可复现
  seed=1: [[0,3,2,7], [0,3,2,0], [0,0,3,5]]   ← 换 seed 结果全变
  ⟹ 同 seed 可复现；不同 seed 结果不同
```

**两个可直接用的结论**：

1. **纯束搜索与 seed 无关**（确定性）⟹ 换 seed 重跑不会改变结果。
2. **束采样下 seed 决定结果** ⟹ 如果你没固定 seed，"同一个模型跑两次结果不同"是正常的，
   **不是 bug**。这一点在跨版本比 HR 时是隐患（见 §5）。

---

## 2. 采样束搜索 vs 纯束搜索：本质区别

### 2.1 机制差异

| | 纯束搜索 | 束采样 |
|---|---|---|
| 每步候选怎么来 | 从 B 条路径各展开，取全局 top-B | **每条路径各自按概率抽**，再取全局 top-B |
| 是否看概率大小 | 是（严格排序） | 是（抽的时候按概率加权）+ 随机 |
| 同一步的 B 条路径 | 尽量分散在高分区 | **可能彼此高度相似**（都从同一高概率区抽） |
| 确定性 | 确定 | 随机 |

**`[原理]` 关键洞察**：束采样的随机性带来的是**候选间多样性**，代价是**牺牲最优性**。
在长文本生成（故事、对话）里这是优点 —— 用户要的是多样性。
**在 SID 检索里这是双刃剑**：多样性可能让你捞到纯 beam 捞不到的候选，
但也可能让本该排前面的正确 SID 被挤到后面（`calc.py` 用 `minID` 首次命中位排名，
**beam 内的顺序就是最终名次**）。

### 2.2 transformers 内部怎么判定 `[源码]`

`GenerationConfig.get_generation_mode()`（`configuration_utils.py:382` 起）原文：

```python
if self.constraints is not None or self.force_words_ids is not None:
    generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
elif self.num_beams == 1:
    if self.do_sample is False:
        ... GenerationMode.GREEDY_SEARCH
    else:
        generation_mode = GenerationMode.SAMPLE
else:                                    # num_beams > 1
    if self.num_beam_groups > 1:
        generation_mode = GenerationMode.GROUP_BEAM_SEARCH
    elif self.do_sample is True:
        generation_mode = GenerationMode.BEAM_SAMPLE     # ← 本项目落在这里
    else:
        generation_mode = GenerationMode.BEAM_SEARCH
```

⟹ **`num_beams > 1` 且 `do_sample=True` 时，HF 走的是 `BEAM_SAMPLE`，不是 `BEAM_SEARCH`。**

`[实测]` 各组合实测归类：

| 配置 | 实际模式 |
|---|---|
| `num_beams=1, do_sample=False` | greedy |
| `num_beams=1, do_sample=True` | sample |
| `num_beams=50, do_sample=False` | **beam search** |
| `num_beams=50, do_sample=True` | **beam sample** ← 本项目现状 |

### 2.3 本项目当前的真实处境

`evaluate.py:207-217` 构造 `GenerationConfig` 时**没有传 `do_sample`**：

```python
generation_config = GenerationConfig(
    num_beams=num_beams,              # 50
    length_penalty=length_penalty,
    num_return_sequences=num_beams,   # 50
    pad_token_id=...,
    eos_token_id=...,
    max_new_tokens=max_new_tokens,
    top_k=None,
    top_p=None,
    **kwargs
)
```

但 `generate()` 内部的 `_prepare_generation_config` 会**用基座
`generation_config.json` 的非默认值填充未显式设置的字段** ⟹
本项目 `models/Qwen3-0.6B/generation_config.json` 的
`do_sample: true` / `temperature: 0.6` **生效**。

⟹ **实际执行的是 `BEAM_SAMPLE`（束采样），不是纯束搜索。**
（完整溯源见 `SFT_PIPELINE §3.5.4`。）

---

## 3. 其他采样方法一览

### 3.1 纯采样（multinomial sampling）

`num_beams=1, do_sample=True`。每步**按 softmax 概率抽 1 个 token**。
用 `num_return_sequences=n` 拿到 n 条互不相同的输出。

**对本项目的问题**：SID 词表有 768 个 token（256×3），
`[实测]` 本项目第 1 层实际候选是 **256 个满基数**。
在全词表上无截断采样，会把大量概率质量分给**Trie 随后会 mask 掉**的 token，
纯属浪费算力；且抽出来的 SID 质量方差极大。

### 3.2 Top-K 采样

只在概率最高的 K 个 token 里抽。`top_k=None` 表示不启用。

**`[实测]` 本项目 `top_k=None` 是显式传入的**（`evaluate.py:214`），
且 `None != 全局默认值` ⟹ **没有被基座的 `top_k: 20` 覆盖**。
所以当前采样是在**全词表无截断**状态下做的。

### 3.3 Top-P（核）采样

按概率**从大到小累加**，取到累计概率 ≥ p 的最小集合，在该集合内采样。

与 top-K 的区别：**候选集大小自适应**。分布尖锐时候选少，分布平坦时候选多。

**`[实测]` 本项目同样 `top_p=None`**，未启用。

### 3.4 温度采样（temperature）

在 softmax 前除以 T：`P = softmax(logits / T)`。

- `T → 0` ⟹ 退化为 argmax（贪心）
- `T = 1` ⟹ 原始分布
- `T > 1` ⟹ 分布更平坦，更多样但更乱

**`[实测]` 本项目的 `temperature = 0.6`（< 1）来自基座文件**，
方向上是合理的 —— SID 是结构化输出，温度高会生成"合法但无关"的码字。

⚠️ **`--temperature` 在本项目是死参数**：`evaluate.py:7` import 了
`TemperatureLogitsWarper`、原版 `evaluate.sh` 也传了 `--temperature 1.0`，
但 `main()` 签名（`evaluate.py:39-57`）**根本没有这个参数** ⟹ fire 吃掉后丢弃。
**改命令行 `--temperature` 不会改变任何结果。**

### 3.5 重复惩罚 / 频率惩罚（repetition / frequency penalty）

对已生成 token 降分。**对 SID 无意义** —— 3 个 token 分属 3 个不同层，
不存在"重复"概念，反而可能误伤同层重码。

### 3.6 对比搜索（contrastive search）

`penalty_alpha > 0` 时启用，惩罚与上文高度相似的 token，专门对抗
"退化重复"。面向**长文本**，本项目不适用。

### 3.7 分组束搜索（group beam search / diverse beam）

`num_beam_groups > 1`。把 B 条 beam 分成若干组，**组内正常 beam，
组间加多样性惩罚**，迫使不同组探索不同区域。

`[原理]` 这其实是**最贴近"SID 结构化检索"需求**的一种 ——
它想解决的正是"beam 里的候选过于集中在同一分支"的问题。
比束采样更有控制力（多样性的强度是显式参数，不靠随机）。

### 3.8 受限/约束解码（constrained decoding）

严格说这**不是采样方法**，而是在 logits 上加 mask，把非法 token 置 `-inf`。

**本项目已用**（`LogitProcessor.py` 的 `ConstrainedLogitsProcessor`），
机制：每步查 Trie，只保留合法后继；查不到时**强制 EOS**（`LogitProcessor.py:58-66`）。

🔴 **重要**：约束解码与上述采样方法是**正交**的 ——
Trie 决定"能选什么"，采样/束搜索决定"怎么从能选的里挑"。
所以**约束不会消除采样的随机性**，只会把随机性限制在合法集合内。

`[实测]` `logits_processor` 每步调用一次，**与 beam 宽度无关**：
```
beam=3, 3 步 ⟹ logits_processor 调用 3 次（每步 1 次）
```

---

## 4. 本项目的实测结构：为什么"采样方式"不是主要矛盾

### 4.1 SID Trie 的真实形状 `[实测]`

IandS 域，来自 `data/Amazon23/IandS/sft/info/sid2items.json`：

| 层 | 分支数 | 说明 |
|---|---|---|
| 第 1 层 `a` | **256**（满基数） | 每个 `a` 都有商品 |
| 第 2 层 `b` | 均值 **72.9**、中位 70、范围 11~155 | 每个 `a` 下平均 73 个 `b` |
| 第 3 层 `c` | **76.1% 的前缀只有唯一候选**，均值 1.3、最大 10 | 给定 `(a,b)` 后基本没得选 |

其他量：唯一 SID **24,766** 个（空间 256³ = 16,777,216，稀疏度 **0.1476%**）；
碰撞桶 **100%**，平均桶 3.00（74,298 商品 → 24,766 桶）。

### 4.2 由此得到的三条判断

**① 第 3 层几乎没有采样空间。**
76% 的位置 Trie 只允许 1 个 token ⟹ `argmax` 与采样**结果完全相同**。
在那里调 `temperature` / `top_p`，七成以上的步数是空转。

**② 真正的约束在第 1、2 层的分支数。**
目标 SID 要求第 1 层选对 1/256、第 2 层选对 1/73。

⚠️ **但要注意口径**：`256 × 72.9 ≈ 18,662` 这个组合空间**是均匀随机假设**。
它只说明"如果模型完全没学到东西，靠宽度撞上的概率"，**不能用来论证 beam 无用**。

**③ 实测数据反驳了"beam 无用"。**
项目记忆记录 `sid_gr` 的 `beam_ceiling@20 = 0.0312`（beam=20，v2 LOO 口径），
而随机基准是 `20 / 18662 = 0.107%` ⟹ **实测是随机的 29 倍**。
说明**模型对 SID 的排序是有信息的**，不是靠撞。

⚠️ **引用注意**：`sid_gr` 已于 2026-09-13 **移出 `baseline/RESULTS.md` 主榜**
（归档在 `baseline/_dropped/2026-09-13_cut_models/`，见 `RESULTS.md` caveat 第 5 条）。
此处**只作为"排序包含信息量"的量级参考**，**不可写进任何正式对照表**，
也不可与 `RESULTS.md` 现行矩阵横比。SFT/RL 的达标线以 `docs/EVAL_PROTOCOL.md §5` 为准。

⟹ 所以"beam 够不够宽"**必须实测**，判据是：ceiling 随 beam 增长是
**快速饱和**（加宽无意义）还是**持续线性**（宽度是真瓶颈）。

---

## 5. 对本项目的建议

### 5.1 概念纠正：采样属于推理，不属于训练

🔴 **SFT 训练阶段根本没有解码**。teacher forcing 下整条 SID 已知，
一次前向算 cross-entropy ⟹ **没有 beam、没有采样**。

| | 训练阶段 | 推理/评估阶段 |
|---|---|---|
| 机制 | teacher forcing 一次前向 | 逐 token 解码 |
| 有 beam 吗 | **无** | 有（`num_beams=50`） |
| 有采样吗 | **无** | 有（`do_sample` 继承自基座） |
| 该调什么 | prompt / 任务配比 / 冻结策略 / LoRA 目标模块 | `num_beams` / `do_sample` / `top_p` |

⟹ 所以"**采样归推理，训练归损失与数据**"，两者不可混谈。

### 5.2 当前建议：先不动采样，但必须记账

| 参数 | 现值 | 来源 | 评价 |
|---|---|---|---|
| `num_beams` | 50 | `evaluate_run0.sh:54` | 与 MiniOneRec 原版一致，保留 |
| `do_sample` | `true` | **继承自基座** | 保留 —— Qwen2.5 与 Qwen3 基座**都为 true**，关掉会引入新旧不可比变量 |
| `temperature` | `0.6` | **继承自基座** | 方向正确（< 1，抑制乱码） |
| `top_k` / `top_p` | `None` | `evaluate.py:214-215` 显式 | 未截断；可考虑加 `top_p≈0.9` 收窄 |

**理由**：`do_sample=true` 是**两个基座的共同默认**，MiniOneRec 原版也在采样。
**关掉它 = 引入一个新旧不可比的变量**，而当前首要任务是跑通硬串行三阶段、
拿到一套内部一致的数。

### 5.3 待办清单（按性价比排序）

1. **补记 meta**（成本最低）：现有 meta 只记 `num_beams` / `prompt_format`，
   **没记 `do_sample` / `temperature`** ⟹ 跨版本比 HR 时无法确认口径。
2. **beam 扫描**（终结争论）：
   `MAX_SAMPLES=1000` 扫 `NUM_BEAMS ∈ {20, 50, 100, 256}`，**只报 `beam_ceiling`**。
   ⚠️ `EXP_ID` 要分开命名（如 `-b20`/`-b50`），否则结果互相覆盖。
   ⚠️ 显存按 `batch × beam` 算，beam=256 时 batch 需压到 1~2。
3. **采样开关消融**（不插在当前链路）：`do_sample=False` 两套都跑，
   作为独立消融，与"基座切换"这个变量**分开记账**。

### 5.4 判据纪律

🔴 **评估生成式方法一律用 `beam_ceiling`（目标是否出现在 beam 里），不要只看 HR。**
`baseline/generative/sid_gr.py:19` 的定义就是干这个的：
HR 混了"没生成出来"和"没排到前面"两件事，`beam_ceiling` 只看前者 ⟹
**只有它能区分瓶颈在解码宽度还是在排序质量**。

⚠️ `sid_gr` 本身已归档（见 §4.2 第 ③ 条的引用注意），
但 `beam_ceiling` 这个**指标定义**仍然有效、且在正式链路里保留使用。

---

## 6. 图示

### 6.1 变体分支结构

```
prompt ──▶ [第1层 a]  ──▶ [第2层 b]  ──▶ [第3层 c]
            256 分支        均值 73 分支      76% 唯一候选
               │                │
               │                └─ beam≤50 时此处 18662 选 beam
               └─ beam≤50 时此处 256 选 50
```

**采样在第 3 层无用（76% 唯一），在第 1、2 层才有意义。**

### 6.2 各方案代价/收益

```
方案            代价              收益                    建议
─────────────────────────────────────────────────────────────
① 关采样        改 1 行           可复现、KV 复用率回升    ⏸ 等 baseline 全跑完再做
② 加宽 beam     显存×5、耗时×5    ★ 唯一能抬 ceiling      ▶ 当前最该先测
③ 分层采样      改解码接口        覆盖分支数 50→256+       ⏸ 确认②是瓶颈后再动
```

### 6.3 采样参数在流程中的位置

```
SID 数据 ──▶ SFT 训练 ──▶ 训练产物 ──▶ 评估/推理 ──▶ HR/NDCG
              (teacher forcing)         (逐token解码)
              无 beam / 无采样           beam=50 + do_sample=true
                                          ↑ 采样只在这里起作用
```

---

## 7. 面试速答版

- **问：束搜索和束采样的区别？**
  束搜索每步取累计 log 概率 top-B（确定）；束采样每条 beam 各自按概率抽再取 top-B（随机）。
  HF 里 `num_beams>1 && do_sample=True` 走的是 `BEAM_SAMPLE`。

- **问：SID 结构化检索该用哪个？**
  先分清阶段 —— 训练是 teacher forcing，没有解码；采样只属于推理。
  推理侧：窄 beam + 低温度 + 窄 top-p，且 **Trie 约束与采样正交**
  （约束决定"能选什么"，采样决定"怎么挑"）。

- **问：你怎么知道 beam 宽度够不够？**
  用 `beam_ceiling`（目标是否在 beam 里）扫描 beam 宽度。
  本项目历史实测 `beam_ceiling@20 = 0.0312`（v2 LOO），是随机基准的 29 倍
  ⟹ 排序有信息。涨不涨、涨多少，靠实测曲线判断，不靠类比 LLM。

- **问：SID 三层结构对解码的特殊影响？**
  实测第 3 层 76% 前缀唯一 ⟹ 采样在那里无作用；
  瓶颈在第 1、2 层分支（256 与 73）⟹ 加宽 beam 或做分层采样才有意义。

---

## 附：本文档的核验记录

| 结论 | 判据 |
|---|---|
| HF 模式判定 | `transformers/generation/configuration_utils.py` `get_generation_mode()`（源码第 382 行起） |
| 可复现性差异 | 可控实验（8 词表 / 手造 logits / 固定 seed），纯 beam 换 seed 不变、束采样变 |
| `logits_processor` 调用次数 | 实测每步 1 次，与 beam 宽度无关 |
| SID 分支结构 | 读 `data/Amazon23/IandS/sft/info/sid2items.json` 统计 |
| `do_sample` 继承 | 实跑 `_prepare_generation_config()`，`False → True`，复现 warning；详见 `SFT_PIPELINE §3.5.4` |
| `temperature` 是死参数 | `grep temperature evaluate.py` → 仅 import 行；`main()` 签名无此参数 |
| `beam_ceiling@20 = 0.0312` | 项目记忆 `2026-09-13.md`（v2 LOO 口径）；⚠️ `sid_gr` 已移出主榜，仅作量级参考 |
