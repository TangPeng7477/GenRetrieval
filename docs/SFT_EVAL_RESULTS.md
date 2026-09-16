# SFT 评估结果表

> 自动生成（`./.venv/Scripts/python.exe scripts/sft/collect_eval_results.py`），共 **1** 条记录，生成于 2026-09-16T23:13:39。**请勿手改。**
>
> 数据源：`results/sft/<EXP_ID>/eval_*.{meta,metrics}.json`（该目录被 `.gitignore` 忽略，明细不入仓；**只有本表入仓**，供跨会话 / 跨环境对照）。

## 读表前必看（口径边界）

> 1. 🔴 **候选集 = 模型自己生成的 beam，不是全库。** `HR@K` / `NDCG@K` 都是 **beam 内排名**（`calc.py` 的 `minID < K`），**本质是 `beam_ceiling@K`** —— 与 [`baseline/RESULTS.md`](../baseline/RESULTS.md) 的「全库排序、无负采样」口径**不可直接横比**（`EVAL_PROTOCOL` §3.4 / §5）。
> 2. 🔴 **`HR@K` 的硬上限 = beam 宽度。** 读表必须带 `beam` 列：`beam=20` 的行天然被锁在 20 个候选内，**不能与 `beam=50` 的行比强弱**。
> 3. **不报告 MRR** —— 生成式下 `MRR ≈ 1/beam` 是结构常数，不携带排序质量信息。
> 4. **抽样行（`样本` 列 < 全量）与全量行不可混比**，样本量不同。
> 5. **推理时间** = `evaluate.py` 开始生成的那一刻（`started_at`）；**耗时** = 该步墙钟时长（不含指标计算），括号内是每样本值 —— 跨行比速度看括号，它已归一化掉 batch / beam / 样本数差异。

---

## IandS（n_items=25847）

| EXP_ID | 模型版本 | 推理时间 | 样本 | beam | HR@1 | HR@5 | HR@10 | HR@20 | NDCG@10 | 耗时 | commit | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| **IandS-untrained** | `models/Qwen3-0.6B` | 2026-09-16 20:41 | 300 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | `3ac43d5` | 现场注册(dry-run) / 非训练产物 / 抽样 |

---

## 参考锚点（非本表口径，仅供理解量级）

| 参照 | HR@10 | 口径 |
|---|---:|---|
| 未训练 Qwen3-0.6B | 0.0000 | 本表口径（beam 内），见上表 |
| `sid_gr`（零 LLM，训练了自己的解码器） | 0.0168 | beam 内，`EVAL_PROTOCOL` §3.4 |
| `sasrec` | 0.0395 | **全库排序**，`baseline/RESULTS.md` |
| `content_ann` | 0.0288 | **全库排序**，`baseline/RESULTS.md` |

🔴 `EVAL_PROTOCOL` §5 的 SFT 达标线是 **HR@10 > `sasrec`（I&S 0.0395）**，那是**全库口径**；本表的 beam 内数字要往上打赢它，需要 `beam` 足够宽（`beam_ceiling` 就是天花板）—— 比较前先确认两边的口径。

