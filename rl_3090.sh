#!/bin/bash
set -euo pipefail

export NCCL_IB_DISABLE=1
export WANDB_MODE="${WANDB_MODE:-online}"

# 3090 24G + Qwen2.5-0.5B RL (GRPO) entrypoint.
# Conservative defaults for 24GB VRAM.
MODEL_PATH="${MODEL_PATH:-./outputs/sft_Industrial_and_Scientific_3090/final_checkpoint}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/rl_${CATEGORY}_3090}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-rl_${CATEGORY}_3090}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
TEST_BEAM="${TEST_BEAM:-10}"
BETA="${BETA:-1e-3}"
SEED="${SEED:-42}"
SAMPLE_TRAIN="${SAMPLE_TRAIN:-True}"
BEAM_SEARCH="${BEAM_SEARCH:-True}"
SYNC_REF_MODEL="${SYNC_REF_MODEL:-False}"
TEST_DURING_TRAINING="${TEST_DURING_TRAINING:-False}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-16}"
RESUME="${RESUME:-}"

train_file=$(ls -f ./data/Amazon/train/${CATEGORY}*.csv | head -1)
eval_file=$(ls -f ./data/Amazon/valid/${CATEGORY}*11.csv | head -1)
info_file=$(ls -f ./data/Amazon/info/${CATEGORY}*.txt | head -1)
sid_index_path="./data/Amazon/index/${CATEGORY}.index.json"
item_meta_path="./data/Amazon/index/${CATEGORY}.item.json"

mkdir -p "${OUTPUT_DIR}" ./logs

echo "=========================================="
echo "MiniOneRec RL - RTX 3090 24G + Qwen2.5-0.5B"
echo "=========================================="
echo "Category: ${CATEGORY}"
echo "Model path: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Train batch: ${TRAIN_BATCH_SIZE}, Eval batch: ${EVAL_BATCH_SIZE}"
echo "Grad acc: ${GRAD_ACC_STEPS} (effective batch: $((TRAIN_BATCH_SIZE * GRAD_ACC_STEPS)))"
echo "Num generations: ${NUM_GENERATIONS}"
echo "Max completion length: ${MAX_COMPLETION_LENGTH}"
echo "=========================================="

python rl.py \
  --model_path "${MODEL_PATH}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --train_file "${train_file}" \
  --eval_file "${eval_file}" \
  --info_file "${info_file}" \
  --category "${CATEGORY}" \
  --sample_train "${SAMPLE_TRAIN}" \
  --eval_step 0.2 \
  --reward_type ranking \
  --num_generations "${NUM_GENERATIONS}" \
  --mask_all_zero False \
  --dynamic_sampling False \
  --sync_ref_model "${SYNC_REF_MODEL}" \
  --beam_search "${BEAM_SEARCH}" \
  --test_during_training "${TEST_DURING_TRAINING}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --temperature 1.0 \
  --learning_rate "${LEARNING_RATE}" \
  --add_gt False \
  --beta "${BETA}" \
  --dapo False \
  --test_beam "${TEST_BEAM}" \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --sid_index_path "${sid_index_path}" \
  --item_meta_path "${item_meta_path}" \
  --seed "${SEED}" \
  ${RESUME:+--resume_from_checkpoint "${RESUME}"} \
  2>&1 | tee "./logs/rl_${CATEGORY}_3090.log"
