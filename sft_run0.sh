#!/bin/bash
set -euo pipefail

# ============================================================
# GenRetrieval - SFT Run-0  (先 IandS 单域跑通，再上 VG)
# ------------------------------------------------------------
# 上游产物落点与参数映射：docs/SFT_PIPELINE.md §3.2
#
# Run-0 的定义 = 原样复刻 MiniOneRec 的单阶段 concat(T1+T2+T3)，
# **只改「基座」一个变量**（Qwen2.5-0.5B -> Qwen3-0.6B）。
# 因此刻意不动提示词口径（data.py 保持 verbatim 带引号版，
# 见 SFT_PIPELINE §6.4(5)），否则出了数字归因不干净。
#
# 根目录的 sft.sh / sft_3090.sh 是 MiniOneRec 原版，数据路径指向
# ./data/Amazon/... —— 本仓不存在，不要直接用。
# ============================================================

export NCCL_IB_DISABLE=1
export WANDB_MODE="${WANDB_MODE:-offline}"   # 不设 wandb 项目时训练端也不上报

# ---------------- 可覆盖配置 ----------------
DOMAIN="${DOMAIN:-IandS}"                            # IandS | VG
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"    # category_dict 的键（要全名，不是域代号）
BASE_MODEL="${BASE_MODEL:-models/Qwen3-0.6B}"        # post-trained，依据 UPGRADE_PLAN §5.1.1
SFT_DIR="${SFT_DIR:-data/Amazon23/${DOMAIN}/sft}"
TASKS="${TASKS:-T1,T2a,T2b,T3}"   # 哪几路进训练集。默认全开 = MiniOneRec ConcatDataset 锚点；
                                  # 单任务消融 e.g. TASKS=T1。T2a/T2b 由同一个类产出，拆不开
                                  # （sft.py resolve_tasks 会 WARN）。
RUN_TAG="${RUN_TAG:-run0}"        # 实验标签：run0 / S0 / S1 / base-cmp ...（任意字符串，只用于命名）

# ---------------- 实验 ID：一个 EXP_ID 串起训练产物与评估结果 ----------------
#   EXP_ID = <域>-<RUN_TAG>[-<任务集>]      例 IandS-run0 / IandS-run0-T1T3
#   训练产物  outputs/<EXP_ID>/      评估结果  results/sft/<EXP_ID>/      日志 logs/sft/<EXP_ID>/
#   （results/ 按阶段分层：SID 阶段在 results/sid*/（rq/*.py 写），SFT 阶段统一在 results/sft/ 下）
#   （evaluate_run0.sh 会从模型路径自动反推 EXP_ID，两边命名天然一致）
if [ "${TASKS}" = "T1,T2a,T2b,T3" ]; then
  TASK_SUFFIX=""                                        # 默认全开不加后缀，Run-0 名字保持干净
else
  TASK_SUFFIX="-$(printf '%s' "${TASKS}" | tr -d ',')"   # T1,T3 -> -T1T3
fi
EXP_ID="${EXP_ID:-${DOMAIN}-${RUN_TAG}${TASK_SUFFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_ID}}"

BATCH_SIZE="${BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
CUTOFF_LEN="${CUTOFF_LEN:-320}"    # Run-0 不含 T4 -> 320 够；带 T4 要 400（SFT_PIPELINE §3.1）
SEED="${SEED:-42}"
SAMPLE="${SAMPLE:--1}"             # -1 = 全量；想先冒烟可设 SAMPLE=5000
FREEZE_LLM="${FREEZE_LLM:-False}"  # Run-0 全参训练；S0 warmup 才置 True（SFT_PIPELINE §6.4(1)）

# ---------------- 上游产物路径（SFT_PIPELINE §3.2） ----------------
TRAIN_FILE="${SFT_DIR}/train/${DOMAIN}_5_train.csv"
EVAL_FILE="${SFT_DIR}/valid/${DOMAIN}_5_valid.csv"
SID_INDEX="${SFT_DIR}/index/${DOMAIN}.index.json"
ITEM_META="${SFT_DIR}/index/${DOMAIN}.item.json"
SID_VOCAB="${SFT_DIR}/info/sid_vocab.json"

# ---------------- 解释器：优先仓库自带 venv（Windows 本地），否则 PATH 里的 python（云端） ----------------
PY="python"
if [ -x "./.venv/Scripts/python.exe" ]; then
  PY="./.venv/Scripts/python.exe"
elif [ -x "./.venv/bin/python" ]; then
  PY="./.venv/bin/python"
fi

# ---------------- 前置检查：上游产物必须齐 ----------------
# 缺文件就立刻停，别等到 load_dataset 才炸、白占 GPU。
missing=0
for f in "${TRAIN_FILE}" "${EVAL_FILE}" "${SID_INDEX}" "${ITEM_META}" "${SID_VOCAB}"; do
  if [ ! -f "${f}" ]; then
    echo "  [MISSING] ${f}"
    missing=1
  fi
done
if [ ! -d "${BASE_MODEL}" ]; then
  echo "  [MISSING] base model dir: ${BASE_MODEL}"
  echo "            -> bash scripts/download_base_models.sh"
  missing=1
fi
if [ "${missing}" -ne 0 ]; then
  echo ""
  echo "上游产物不齐，先跑："
  echo "  ./.venv/Scripts/python.exe scripts/data/prepare_sft_data.py --domain ${DOMAIN}"
  echo "  bash scripts/download_base_models.sh"
  exit 1
fi

# 权重文件单独确认（tokenizer 目录存在但权重缺失是最容易踩的坑）
if [ ! -f "${BASE_MODEL}/model.safetensors" ] && [ ! -f "${BASE_MODEL}/model.safetensors.index.json" ]; then
  echo "  [MISSING] ${BASE_MODEL}/model.safetensors  (只有 tokenizer？)"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "./logs/sft/${EXP_ID}"

echo "=========================================="
echo " GenRetrieval SFT  (EXP_ID = ${EXP_ID})"
echo "=========================================="
echo " Domain      : ${DOMAIN}  (category=${CATEGORY})"
echo " Base model  : ${BASE_MODEL}"
echo " Output dir  : ${OUTPUT_DIR}"
echo "   -> 评估用：MODEL_PATH=${OUTPUT_DIR}/final_checkpoint bash evaluate_run0.sh"
echo " SFT dir     : ${SFT_DIR}"
echo " Train file  : ${TRAIN_FILE}"
echo " tasks       : ${TASKS}"
echo " cutoff_len  : ${CUTOFF_LEN}"
echo " Batch       : ${BATCH_SIZE} (micro ${MICRO_BATCH_SIZE}, accum $((BATCH_SIZE / MICRO_BATCH_SIZE)))"
echo " Epochs / LR : ${NUM_EPOCHS} / ${LEARNING_RATE}"
echo " sample      : ${SAMPLE}   (=-1 全量)"
echo " freeze_LLM  : ${FREEZE_LLM}"
echo "=========================================="

"${PY}" sft.py \
  --base_model "${BASE_MODEL}" \
  --batch_size "${BATCH_SIZE}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --cutoff_len "${CUTOFF_LEN}" \
  --sample "${SAMPLE}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --category "${CATEGORY}" \
  --train_from_scratch False \
  --seed "${SEED}" \
  --sid_index_path "${SID_INDEX}" \
  --item_meta_path "${ITEM_META}" \
  --sid_vocab_path "${SID_VOCAB}" \
  --tasks "${TASKS}" \
  --freeze_LLM "${FREEZE_LLM}" \
  2>&1 | tee "./logs/sft/${EXP_ID}/sft.log"

echo ""
echo "训练完成。产物：${OUTPUT_DIR}"
echo "  - sid_token_map.json   （token->id，供评估/M4 对齐）"
echo "  - final_checkpoint/    （含扩展后的 tokenizer，evaluate 的 --base_model 指这里）"
echo ""
echo "下一步（EXP_ID 自动反推，不用手动传）："
echo "  bash evaluate_run0.sh                 # DOMAIN 默认 IandS"
echo "  DOMAIN=VG bash evaluate_run0.sh"
