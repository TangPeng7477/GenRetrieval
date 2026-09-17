#!/bin/bash
set -euo pipefail

# ============================================================
# GenRetrieval - RL Run-0 (GRPO)  —— 先 IandS 单域跑通
# ------------------------------------------------------------
# 🔴 上游依赖：**RL 不需要生成任何新数据集**。三个在用的 Dataset 类
#    全部直接读 SFT 阶段已落盘的同名产物（详见 docs/RL_PIPELINE.md §1）：
#      SidDataset            <- train/*.csv                （与 SFT T1 同一个 CSV）
#      RLTitle2SidDataset    <- index/*.item.json + *.index.json（与 SFT T2b 同两个文件）
#      RLSeqTitle2SidDataset <- train/*.csv                （用 history_item_title 列）
#
# 🔴 --model_path 必须是 **SFT 训练产物目录**（自带扩展 tokenizer + resize 过的
#    embedding）。rl.py / ReReTrainer 都**不做** add_tokens / resize_token_embeddings，
#    指回原始基座会让 SID 碎裂、约束映射全废，而且不报错。rl.py 已加主动护栏。
#
# 根目录的 rl.sh / rl_3090.sh 是 MiniOneRec 原版，路径指向 ./data/Amazon/...
# —— 本仓不存在，不要直接用。
# ============================================================

export NCCL_IB_DISABLE=1
export WANDB_MODE="${WANDB_MODE:-offline}"

# ---------------- 可覆盖配置 ----------------
DOMAIN="${DOMAIN:-IandS}"                            # IandS | VG
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"    # category_dict 的键（全名，不是域代号）
RL_DIR="${RL_DIR:-data/Amazon23/${DOMAIN}/sft}"      # RL 复用 SFT 阶段的产物目录
SFT_EXP_ID="${SFT_EXP_ID:-${DOMAIN}-run0}"           # 从哪个 SFT 实验接着训
MODEL_PATH="${MODEL_PATH:-outputs/${SFT_EXP_ID}/final_checkpoint}"
RUN_TAG="${RUN_TAG:-rl0}"                            # rl0 = GRPO 锚点（rule 奖励 + ref 模型）

# ---------------- 实验 ID ----------------
#   EXP_ID = <域>-<RUN_TAG>      例 IandS-rl0
#   训练产物 outputs/<EXP_ID>/   日志 logs/rl/<EXP_ID>/   元数据 logs/rl/<EXP_ID>/run.meta.json
EXP_ID="${EXP_ID:-${DOMAIN}-${RUN_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_ID}}"

# ---------------- 训练超参（默认给 3090 24G，单卡）----------------
# GRPO 的有效规模 = per_device_train_batch_size(唯一 prompt 数) × num_generations
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"            # MiniOneRec: GRPO 2 ep
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BETA="${BETA:-1e-3}"                                 # KL-to-ref 系数（rl.sh 用 1e-3）
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-16}" # 3 SID + \n + EOS 只需 5，留头寸
REWARD_TYPE="${REWARD_TYPE:-rule}"                   # UPGRADE_PLAN §6 的 R0 锚点 = rule
BEAM_SEARCH="${BEAM_SEARCH:-True}"
TEST_DURING_TRAINING="${TEST_DURING_TRAINING:-True}" # 训练内 beam 评测（HR/NDCG 就靠它）
TEST_BEAM="${TEST_BEAM:-10}"
SYNC_REF_MODEL="${SYNC_REF_MODEL:-False}"
EVAL_STEP="${EVAL_STEP:-0.0999}"                     # <1 = 占训练总步数的比例
SEED="${SEED:-42}"

# checkpoint：全参 GRPO 单个 ckpt ≈ 6 GB（bf16 权重 + paged_adamw_32bit 的 fp32 状态）
SAVE_STEPS="${SAVE_STEPS:-0.1}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"            # 原版 20 -> ~120 GB，本仓下调
TORCH_COMPILE="${TORCH_COMPILE:-False}"              # [红线] 默认关
OPTIM="${OPTIM:-paged_adamw_32bit}"
# 计算精度：bf16（Ampere+ 默认）| fp16（V100 等 Volta 必须用这个）
PRECISION="${PRECISION:-bf16}"
RESUME="${RESUME:-}"

# ---------------- 上游产物路径 ----------------
TRAIN_FILE="${RL_DIR}/train/${DOMAIN}_5_train.csv"
EVAL_FILE="${RL_DIR}/valid/${DOMAIN}_5_valid.csv"
SID_INDEX="${RL_DIR}/index/${DOMAIN}.index.json"
ITEM_META="${RL_DIR}/index/${DOMAIN}.item.json"
INFO_FILE="${RL_DIR}/info/${DOMAIN}.item_info.txt"

# ---------------- 解释器：优先仓库自带 venv（Windows 本地），否则 PATH 里的 python（云端）----------------
PY="python"
if [ -x "./.venv/Scripts/python.exe" ]; then
  PY="./.venv/Scripts/python.exe"
elif [ -x "./.venv/bin/python" ]; then
  PY="./.venv/bin/python"
fi

# ---------------- 前置检查（缺文件立刻停，别白占 GPU）----------------
missing=0
for f in "${TRAIN_FILE}" "${EVAL_FILE}" "${SID_INDEX}" "${ITEM_META}" "${INFO_FILE}"; do
  if [ ! -f "${f}" ]; then echo "  [MISSING] ${f}"; missing=1; fi
done
if [ ! -f "${MODEL_PATH}/model.safetensors" ] && [ ! -f "${MODEL_PATH}/model.safetensors.index.json" ]; then
  echo "  [MISSING] ${MODEL_PATH}/model.safetensors"
  missing=1
fi
# tokenizer 里必须已注册 SID —— 比"目录存在"强得多，能拦住"指回原始基座"的静默失败
if [ ! -f "${MODEL_PATH}/tokenizer.json" ] || ! grep -q '<a_0>' "${MODEL_PATH}/tokenizer.json"; then
  echo "  [BAD] ${MODEL_PATH}/tokenizer.json 里找不到 SID token '<a_0>'"
  echo "        -> RL 必须从 SFT 产物接着训，先跑 bash sft_run0.sh"
  missing=1
fi
if [ "${missing}" -ne 0 ]; then
  echo ""
  echo "上游产物不齐。RL 复用 SFT 阶段的产物，先补齐："
  echo "  bash sft_run0.sh                       # 产出 outputs/${SFT_EXP_ID}/final_checkpoint"
  echo "  ./.venv/Scripts/python.exe scripts/data/prepare_sft_data.py --domain ${DOMAIN}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "./logs/rl/${EXP_ID}"

echo "=========================================="
echo " GenRetrieval RL / GRPO  (EXP_ID = ${EXP_ID})"
echo "=========================================="
echo " Domain      : ${DOMAIN}  (category=${CATEGORY})"
echo " SFT source  : ${SFT_EXP_ID}  ->  ${MODEL_PATH}"
echo " Output dir  : ${OUTPUT_DIR}"
echo " RL data dir : ${RL_DIR}   (复用 SFT 产物，无新增数据集)"
echo " reward_type : ${REWARD_TYPE}"
echo " Batch       : ${TRAIN_BATCH_SIZE} prompt x ${NUM_GENERATIONS} gen"
echo "               x grad_acc ${GRAD_ACC_STEPS} = effective $((TRAIN_BATCH_SIZE * GRAD_ACC_STEPS)) prompt / step"
echo " Epochs / LR : ${NUM_TRAIN_EPOCHS} / ${LEARNING_RATE}   beta=${BETA}"
echo " beam_search : ${BEAM_SEARCH}   test_during_training=${TEST_DURING_TRAINING} (beam=${TEST_BEAM})"
echo " save        : save_steps=${SAVE_STEPS}  save_total_limit=${SAVE_TOTAL_LIMIT}"
echo " optim       : ${OPTIM}   torch_compile=${TORCH_COMPILE}"
echo " precision   : ${PRECISION}   (V100/Volta 请用 fp16)"
echo "=========================================="

# 版本元数据（与评估侧同口径，便于回溯"这条 RL 是从哪个 SFT 接着训的"）
"${PY}" scripts/sft/eval_report.py meta --out "./logs/rl/${EXP_ID}/run.meta.json" \
  --set "exp_id=${EXP_ID}" \
  --set "stage=rl-grpo" \
  --set "domain=${DOMAIN}" \
  --set "sft_exp_id=${SFT_EXP_ID}" \
  --set "model_path=${MODEL_PATH}" \
  --set "reward_type=${REWARD_TYPE}" \
  --set "num_generations=${NUM_GENERATIONS}" \
  --set "train_batch_size=${TRAIN_BATCH_SIZE}" \
  --set "grad_accum=${GRAD_ACC_STEPS}" \
  --set "learning_rate=${LEARNING_RATE}" \
  --set "beta=${BETA}" \
  --set "num_train_epochs=${NUM_TRAIN_EPOCHS}" \
  --set "max_completion_length=${MAX_COMPLETION_LENGTH}" \
  --set "beam_search=${BEAM_SEARCH}" \
  --set "optim=${OPTIM}" \
  --set "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  > /dev/null

"${PY}" rl.py \
  --model_path "${MODEL_PATH}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --beta "${BETA}" \
  --num_generations "${NUM_GENERATIONS}" \
  --temperature "${TEMPERATURE}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --reward_type "${REWARD_TYPE}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --info_file "${INFO_FILE}" \
  --sid_index_path "${SID_INDEX}" \
  --item_meta_path "${ITEM_META}" \
  --category "${CATEGORY}" \
  --eval_step "${EVAL_STEP}" \
  --beam_search "${BEAM_SEARCH}" \
  --test_during_training "${TEST_DURING_TRAINING}" \
  --test_beam "${TEST_BEAM}" \
  --sync_ref_model "${SYNC_REF_MODEL}" \
  --add_gt False \
  --dynamic_sampling False \
  --sample_train False \
  --dapo False \
  --gspo False \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --optim "${OPTIM}" \
  --torch_compile "${TORCH_COMPILE}" \
  --precision "${PRECISION}" \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  ${RESUME:+--resume_from_checkpoint "${RESUME}"} \
  2>&1 | tee "./logs/rl/${EXP_ID}/rl.log"

echo ""
echo "训练完成。产物：${OUTPUT_DIR}"
echo "  - final_checkpoint/    （自包含 tokenizer，可直接做后续评估的 --base_model）"
echo "  - checkpoint-*/        （最近 ${SAVE_TOTAL_LIMIT} 个）"
echo ""
echo "训练内指标（HR/NDCG@3/5/10/20）在日志里，由 ReReTrainer.test_during_training 打印："
echo "  grep -E 'HR|NDCG' logs/rl/${EXP_ID}/rl.log | tail"
