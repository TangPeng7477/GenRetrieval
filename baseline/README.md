# baseline —— 召回阶段的对照基线（本地可复现）

> **为什么在 SFT/RL 之前先做这个**：生成式召回的增益只有跟**同口径**的基线比才有意义。
> 这一步的产出是"标尺"——SFT 出来之后要回答的第一个问题是"它比 SASRec 强多少、比零训练的 SID 前缀检索强多少"，
> 没有这把尺子，后面所有数字都无法解释。
>
> 入口：**`SURVEY.md`（有哪些 baseline、公开数字多少）** → 本文（怎么复现）→ **`RESULTS.md`（实测结果表）**。

---

## 1. 目录结构

```
baseline/
├── README.md                     # 本文：口径 + 复现设置 + 命令 + 踩坑
├── SURVEY.md                     # 文献综述：经典/生成式 baseline 清单 + 公开数字（含核验等级）
├── RESULTS.md                    # 自动生成的实测结果表（python -m baseline.scripts.summarize --out baseline）
├── run.py                        # 统一入口（模型注册表在这里，改它等于改实验登记表）
├── common/
│   ├── data.py                   # .inter → 序列样本；完整训练序列重建；屏蔽集构造
│   ├── metrics.py                # HR@K / NDCG@K / MRR / coverage / gini ——**全项目唯一口径实现**
│   ├── nn.py                     # 训练循环 + 全库排序评估骨架（含 NaN 硬拦截）
│   └── base.py                   # Baseline 统一接口 fit / score / info
├── models/                       # A 组：经典非生成式
│   ├── heuristic.py              #   pop / itemknn（零训练）
│   ├── mf.py                     #   bprmf
│   └── seq.py                    #   gru4rec / sasrec（BERT4Rec 实现保留可回溯，已移出对比矩阵）
├── generative/                   # B 组检索式 + C 组生成式
│   ├── retrieval.py              #   content_ann（融合向量近邻）/ sid_prefix（SID 前缀检索），均零训练
│   └── sid_gr.py                 #   SID 自回归生成（TIGER 范式，小 Transformer + Trie 约束解码）
├── scripts/
│   ├── summarize.py              # 汇总所有 metrics.json → Markdown 表
│   ├── diagnose_coldstart.py     # test 冷/热目标拆分：判定 sid_prefix 的领先是否来自内容泛化
│   └── probe_sequence_reconstruction.py   # 验证"完整训练序列重建"口径的三条前提
└── results/<domain>/<model>/     # metrics.json（机器可读）+ run.log
```

---

## 2. 评估口径（**不能改，改了所有数字作废**）

> 📌 **完整定版协议 → `docs/EVAL_PROTOCOL.md`**（划分依据、指标定义、冷热分桶、报数模板、给 SFT/RL 的三条闸门）。
> 本节是执行层的落地检查表；**两者冲突时以 `EVAL_PROTOCOL.md` 为准**。

| 项 | 取值 | 说明 |
|---|---|---|
| 训练样本 | `train.inter` 的滑窗样本，history ≤ 20 | 与后续 SFT 完全同一批样本 |
| valid / test 输入 | 文件给出的 history（≤20） | test 的 history = **该用户前 K−1 个物品**（LOO 标准），见下 |
| 候选集 | **全库**（I&S 25,848 / VG 25,612），全量排序 | 不做负采样 |
| 屏蔽 | history（含 valid 目标）∪ **完整训练序列** | 见 §2.2 |
| 指标 | HR@1/5/10、NDCG@1/5/10、MRR、coverage@10、gini@10 | `common/metrics.py`；与 `utility.py:calculate_hit` 同定义 |
| rank 定义 | `rank = #{score > score_target} + 1`（同分取最优位次） | 单正样本 ⇒ NDCG 的 IDCG = 1 |
| 模型选择 | valid 的 NDCG@10 早停（`sid_gr` 用 valid 的 teacher-forcing CE） | test 只报数，不参与选择 |

> **这套口径与同类目 7/8 论文的关系**（对外报数前必读）：**协议完全对齐** —— 同一份 5core 数据、
> 同一份 LOO 切分（test = 每用户最后一条、valid = 倒数第二条）、同样的 hist 长度（20）、
> 同样的全库排序候选集。绝对数字可以做**量级对齐**，不再只是"目标线参考"。
> 三套口径的逐项量化对照与官方原文见 **`SURVEY.md §1.5`**。

### 2.1 LOO 的 test 输入口径（实测确认）

`prepare_amazon23_loo.py` 的 test 样本 history = **该用户前 K−1 个交互**（按 timestamp 升序去重后截断 20），
target = **最后一条**（K-th）。这是标准 LOO：CCFRec / MTGRec / TIGER / LC-Rec 全部如此。
实测（2026-09-13）：I&S K 最小 = 3 / 最大 = 705；VG K 最小 = 3 / 最大 = 1242。

> **诚实边界**：LOO 有数据泄漏（Ji TOIS'23 / Meng RecSys'20 论证），让模型见过目标之前的近邻交互；
> 同协议下所有 baseline **共享**这份偏差，组内排序可比，绝对值与同类目论文只能量级对齐。
> 冷启动命题改在 MiniOneRec-style sliding-window time-split 附录单独跑（§7 待办）。

### 2.2 为什么必须重建"完整训练序列"（这一步不做就是放水）

`train.inter` 只落盘滑窗样本，history 被截断到 20。若只用文件里的 history 做屏蔽，
用户第 21 个以前的训练物品会**留在候选集里**——模型只要学会"推荐用户很久以前看过的东西"就能白刷分。
实测这份额外屏蔽在 I&S 补掉 **≈10 万**个、VG 补掉 **≈20 万**个本应屏蔽的物品。

重建公式（脚本 `scripts/probe_sequence_reconstruction.py`）：

```
tr = [第一个样本的 history[0]] + [各样本的 target]（按文件顺序）
```

| 前提 | 实测 |
|---|---|
| ① 首样本 history 长度恒为 1 | I&S 50,985 / VG 94,762 个用户**全部**为 1 ✅ |
| ② 重建序列能还原 valid 输入 `train_seq[-20:]` | 重建可还原 ✅ |
| ③ 训练段只有 1 个物品的用户不在 train.inter 里 | LOO 要求 K≥3，5core 保证 K≥5，**无残余** ✅ |

补齐后 `train_seq` 覆盖用户数 = I&S **50,985** / VG **94,762**，与 `stats.json` 的 `n_users`
**逐位相同**（这是一个强一致性校验）。

**已知残余（诚实披露）**：LOO 下**残余为 0**（v1 timestamp 协议下 I&S 0.249% / VG 0.106% 的残余**结构性消失**）。

---

## 3. 本地复现设置（默认超参，4GB 卡）

| 组 | 模型 | 关键超参 | 训练成本（实测/估计） |
|---|---|---|---|
| A 经典 | `pop` | 训练集交互计数排序 | 零训练，~10 s（含评估） |
| A | `itemknn` | 余弦相似 + 每行截断 top-100 近邻 | 零训练，~8 s |
| A | `bprmf` | d=64, lr 3e-3, bs 4096, 25 轮, patience 3 | 分钟级 |
| A | `gru4rec` | hidden 64, 1 层, dropout 0.2, bs 512, lr 1e-3 | ~1 min/轮（I&S） |
| A | `sasrec` | hidden 64, 2 层 2 头, dropout 0.2, 因果掩码, bs 512 | ~1 min/轮（I&S） |

> `bert4rec`：I&S 已跑过（HR@10 = 0.0153，产物在 `baseline/_dropped/bert4rec_IandS/`），但单轮 **364~675 s**
> （`sasrec` 的约 **25 倍**），双域跑完约 2.5 h，且**未超过** `sasrec`(0.0157) / `gru4rec`(0.0144)
> → 2026-09-13 **移出对比矩阵**（决定记录见 `run.py` 的 registry 注释与 `docs/EVAL_PROTOCOL.md §6`）。
| B 检索 | `content_ann` | 融合向量 `emb_fused_gate_e60` 的历史均值，余弦 | **零训练**，~12 s |
| B | `sid_prefix` | SID 前缀逐层计数，权重 (1.0, 1.5, 2.0) | **零训练**，~18 s |
| C 生成 | `sid_gr` | d=128, 2+2 层 4 头, dropout 0.1, bs 512, **beam 20**, Trie 约束解码 | ~45 s/轮（I&S） |

**所有模型共用的口径**：maxlen=20、全库 softmax / 全量排序、同一套早停、同一份屏蔽集。

### 与文献默认值的**有意偏离**（必须写清楚，否则数字对不上）

1. **训练目标统一为全库 softmax**（SASRec 原版用 sampled softmax、GRU4Rec 原版用 pairwise BPR）。
   理由：生成式召回本来就是全词表 softmax，统一后"生成式 vs 非生成式"的差异不掺损失函数这一项。
2. **序列长度 20**（原论文常用 50/100）—— 服从本项目数据的 history 截断口径。
3. **BERT4Rec 做了 next-item 化改编**：尾部追加 [MASK] 作为查询位预测下一个物品，
   同时保留随机位置 mask 的 MLM 辅助任务。原版是纯 MLM，推理接口与本项目其它 baseline 不一致。
   （该模型**已移出对比矩阵**，此条改编说明留档备查。）
4. **输出层与 item embedding 共享权重**（SASRec 原版不共享），更贴近生成式模型"生成 token = 查同一个 embedding"的形态。
5. **BPR-MF 负样本均匀采样、不排已交互**（与 BPR 原文一致）；对绝对指标略偏乐观，但方向对所有对比对象一致。

---

## 4. 复现命令

```bash
# 0) 环境：项目内 .venv（torch 2.6.0+cu118, CUDA 可用）
#    所有命令在项目根目录执行，用 -m 方式（models 用相对导入）

# 1) 准备 LOO 数据（已一次性生成；若重做需先备份）
./.venv/Scripts/python.exe scripts/data/prepare_amazon23_loo.py --categories Industrial_and_Scientific --short IandS
./.venv/Scripts/python.exe scripts/data/prepare_amazon23_loo.py --categories Video_Games --short VG

# 2) 零训练组（双域，约 2 分钟，先拿标尺）
./.venv/Scripts/python.exe -m baseline.run --model pop,itemknn,content_ann,sid_prefix --domain all

# 3) 可训练组（双域，数小时；按重要度排序，先出生成式与 SASRec）
./.venv/Scripts/python.exe -m baseline.run --model sid_gr,sasrec,bprmf,gru4rec --domain all --beam 20

# 4) 冒烟测试（1 轮 + 2 万样本，用来改代码后快速自检；结果不会进对比表）
./.venv/Scripts/python.exe -m baseline.run --model sasrec,sid_gr --domain IandS --quick

# 5) 汇总成表
./.venv/Scripts/python.exe -m baseline.scripts.summarize --out baseline

# 6) 复核数据口径（LOO K≥3 余零）
./.venv/Scripts/python.exe -m baseline.scripts.probe_sequence_reconstruction
```

产物：`baseline/results/<domain>/<model>/metrics.json`（含 `git_commit` / 数据 meta / 超参 / valid+test 指标 / 用时）
与 `run.log`。**任何一个对外引用的数字都必须能指回这两个文件。**

---

## 5. 结果与发现

见 **`RESULTS.md`**（自动生成；v2 LOO 重跑进行中，跑完会重生成）。以下每一条都锚在 `results/<域>/<模型>/metrics.json` 上。

### 5.1 ⚠️ v1 时代（timestamp 协议）的红旗 + v2 LOO 的现状

**v1（timestamp）发现**：I&S 上第一批可训练模型跑完后，零训练 `sid_prefix` HR@10=0.0449
**比所有训练过的模型高约 3 倍**（sasrec 0.0157 / gru4rec 0.0144 / bprmf 0.0084 / sid_gr 0.0081）。

按两条假设做了可证伪的检查（脚本 `scripts/diagnose_coldstart.py`）：

- **H1（真机理 · 内容泛化）→ 已证实。** 本项目 test 有 **31%（I&S）/ 49%（VG）** 的目标商品
  在 train 中从未出现（样本级口径）。纯流行度 `pop` 在冷目标上的 HR@10
  **两域都恰好是 0.0000**；而 `sid_prefix` 的冷目标 HR@10 达到热目标的
  **80%（I&S）/ 94%（VG）** —— SID 来自内容量化，所以没见过的商品照样能被召回。
- **H0（口径 / 泄漏）→ 未发现问题。** 判据是"若 `pop` 在冷目标上也能命中，则评估有毛病"，
  实测它严格为 0。

**v2（LOO）现状**：冷启动命题从主榜消失（LOO + 5core 决定，冷目标 ≈0），
**`sid_prefix` 是否仍是 LOO 下最强非神经基线 待重跑验证**（任务 OAx33E 进行中）。
→ 冷启动命题改在 `MiniOneRec-style sliding-window time-split` 附录单独跑（§7 待办）。

### 5.2 v1 已经能确定的结论（timestamp 协议下，待 v2 重跑验证是否仍成立）

1. **`sid_prefix` 在 v1 是两个域上最强的基线**（I&S 0.0449 / VG 0.0418），且几乎免疫冷启动。
   v2 下数字会变（**测试集从 13,232/11,276 涨到 50,982/94,759**，候选数 +1），但仍应是最强非神经基线。
   → **这是本项目最重要的一条标尺**：SFT/RL 必须打败的不是 SASRec，
   而是"用现成 SID 做前缀检索"。
2. **`content_ann` 在 v1 两个域都优于 `pop` 与 `itemknn`**，说明融合向量本身携带可检索语义。
3. **`itemknn` 在 I&S 上弱于 `pop`**（0.0043 vs 0.0094）—— 用户交互中位数只有 6，
   共现信号太稀疏，协同过滤在这个数据密度下基本失效。
4. **`sid_gr` 的瓶颈在解码，不在排序**：`beam_ceiling@20 = 0.0166` 而 HR@10 = 0.0081，
   `e^(-valid_ce) = e^(-4.14) = 0.0159` 与 `beam_ceiling 0.0166` 几乎相等 —— 模型校准良好但不够准。
   对照 `sid_prefix` 用**同一份 SID** 拿到 0.0449，说明**信息在 SID 里，输在了生成器**。

---

## 6. 踩坑记录（都是本轮真实踩到的）

| # | 坑 | 症状 | 根因与解法 |
|---|---|---|---|
| B-01 | **左填充 + 因果掩码 ⇒ softmax 全屏蔽 ⇒ NaN** | SASRec 未训练就报 HR@10 = 0.9721，且 `NDCG@10 = HR@10 = MRR`（全是 rank 1） | 位置 0 的前缀全是 pad，该 query 的所有 key 被屏蔽 → softmax 得到 NaN，第二层沿注意力传染到末位。解法：掩码改为 `causal ∪ (key_pad ∧ ¬自注意力)`，每层后清零 pad 位。**并已在 `common/nn.py` 的评估里加 NaN 硬拦截**，防止类似问题再以"假性高分"的形式溜过去 |
| B-02 | **`np.isin`（布尔 OR）让 `sid_prefix` 退化成流行度排序** | coverage@10 = 0.0007（≈18 个物品），HR@10 仅 0.0097 | "与历史任一物品共享前缀"的物品全部同分，排序被流行度兜底项主导。解法：改为**按层计数**（与更多历史物品共享前缀的候选分更高），HR@10 → 0.0449、coverage → 0.878 |
| B-03 | **beam 展开后 memory 批次不匹配 / 爆显存** | `RuntimeError: shape '[60, 40960, 32]' is invalid` | beam search 把 memory 复制成 `(b*beam, S, d)`，batch 512×beam 20 = 10240 条序列直接失控。解法：内部分块 `max_seqs=2048`，第 0 步的 `dec_in` 也展开成 `b*beam`（各 beam 相同，取第 0 个算一次） |
| B-04 | **`nohup ... &` 起的训练进程会被会话回收** | 日志停在 epoch 1 不再增长，`ps` 里没有 python | 任务结束时进程组被清理。解法：把长命令本身交给后台任务机制，**不要再套一层 `&` + `sleep`** |
| B-05 | 每轮评估的 masking 逐用户 for 循环耗时 ~8 s | 1.3 万条测试样本 × 每轮 valid 都付一遍 | 摊平成 `(rows, cols)` 一维索引后批量置 `-inf`，缓存到 `EvalSet` |
| B-06 | 训练段只有 1 个物品的用户"消失" | valid/test 里 1,221 / 828 个用户不在 train.inter，最初被当成"无训练历史" | 滑窗要求 target 之前至少有 1 个物品，所以单物品训练段产生不出样本。解法：用 valid history 首元素补齐（前提③） |
| B-07 | 冒烟结果污染对比表 | `--quick` 的 1 轮结果与正式结果混在一张表里 | 结果 JSON 加 `quick` 字段，`summarize.py` 默认跳过 |

---

## 7. 与后续 SFT / RL 的衔接

- **口径复用**：SFT/RL 阶段的 HR/NDCG 必须用 `common/metrics.py` 的同一实现，
  以及同一份 `EvalSet`（同样的 history、同样的屏蔽集），否则"生成式比 baseline 强多少"这句话不成立。
- **交付给 SFT 的三条闸门**（建议写死）：
  1. HR@10 必须超过 `sid_prefix`（零训练，最强非神经基线）；
  2. HR@10 必须超过 `content_ann`（问"量化成 SID 再生成"是否比"直接用融合向量检索"更值）；
  3. `beam_ceiling` 必须显著高于 HR@10 —— 否则指标被 beam 宽度卡住，说明问题在解码而不是模型。
- **待办（按优先级；完成前不要对外写结论）**：
  1. 🔴 **孪生效应诊断**：统计 `sid_prefix` 的命中里，"候选与历史物品 **3 码全同（同桶）**"占多少，
     把"近似重复商品的召回"与"真正的跨物品语义召回"分开 —— 这决定 §5.1 那面红旗能不能结案；
  2. 🔴 **借官方参考实现做交叉验证**（依据见 `SURVEY.md §1.5`）：官方 `seq_rec_results/` 自带全套
     RecBole 配置 + 处理脚本，两条路线 —— (a) 拉 `0core_timestamp_w_his_{domain}` 复现**官方口径**
     参考数字；(b) 把本项目 `5core/timestamp` 的 `.inter` 灌进 RecBole 跑 SASRec，
     与自研 `sasrec` **逐格对表**。用于排除"自研 baseline 偏弱"这个最常见的方法论漏洞；
  3. `sid_gr` 的 beam 宽度消融（10 / 20 / 50）+ 继续训练 —— `valid_ce` 仍在缓慢下降，
     `e^(-ce)` 的追赶目标是 `sid_prefix` v2 LOO 数字（I&S **0.0545** / VG **0.0627**，`RESULTS.md §4.1`）；
  4. 与 MTGRec 的 I&S 0.0506 / VG 0.0956 做**同口径**对比：需先在原始 5core 数据上按 LOO 重切一轮。
