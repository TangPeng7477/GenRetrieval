#!/bin/bash
set -euo pipefail

# ============================================================
# GenRetrieval - SFT Run-0 评估
# ------------------------------------------------------------
# 🔴 关键：--base_model 必须指向**训练输出目录**（自带扩展后的 tokenizer），
#    不是 models/Qwen3-0.6B。evaluate.py 里没有任何 add_tokens
#    （evaluate.py:72 只做 from_pretrained），指回原始基座会让 SID
#    被切成 6 个碎片、Trie 全挂。详见 docs/SFT_PIPELINE.md §3.2。
#
# 根目录的 evaluate.sh / evaluate_3090.sh 是 MiniOneRec 原版，
# 数据路径指向 ./data/Amazon/... —— 本仓不存在。
# ============================================================

DOMAIN="${DOMAIN:-IandS}"                                 # IandS | VG
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
SFT_DIR="${SFT_DIR:-data/Amazon23/${DOMAIN}/sft}"
MODEL_PATH="${MODEL_PATH:-outputs/sft_${DOMAIN}_run0/final_checkpoint}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_BEAMS="${NUM_BEAMS:-50}"          # 与 EVAL_PROTOCOL 的 beam 口径一致
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}" # 生成目标只有 3 SID + \n + EOS，16 足够
LENGTH_PENALTY="${LENGTH_PENALTY:-0.0}"

TEST_FILE="${SFT_DIR}/test/${DOMAIN}_5_test.csv"
INFO_FILE="${SFT_DIR}/info/${DOMAIN}.item_info.txt"

# ---------------- 解释器：优先仓库自带 venv（Windows 本地），否则 PATH 里的 python（云端） ----------------
PY="python"
if [ -x "./.venv/Scripts/python.exe" ]; then
  PY="./.venv/Scripts/python.exe"
elif [ -x "./.venv/bin/python" ]; then
  PY="./.venv/bin/python"
fi

# ---------------- 前置检查 ----------------
for f in "${TEST_FILE}" "${INFO_FILE}"; do
  [ -f "${f}" ] || { echo "  [MISSING] ${f}"; exit 1; }
done
if [ ! -d "${MODEL_PATH}" ]; then
  echo "  [MISSING] model dir: ${MODEL_PATH}"
  echo "            -> 先跑 bash sft_run0.sh"
  exit 1
fi
# 训练产物必须自带扩展后的 tokenizer（词表 152437），否则说明指错了目录
if [ ! -f "${MODEL_PATH}/tokenizer.json" ]; then
  echo "  [MISSING] ${MODEL_PATH}/tokenizer.json"
  echo "            --base_model 要指向训练输出目录，而不是原始基座"
  exit 1
fi

exp_name_clean=$(basename "${MODEL_PATH}")
output_dir="./results/${exp_name_clean}_${DOMAIN}"
mkdir -p "${output_dir}" ./logs
result_json="${output_dir}/final_result_${DOMAIN}.json"

# ---------------- 训练/评估口径一致性（回归检查） ----------------
# 2026-09-16 实测抓到上游遗留 bug：data.py:624 EvalSidDataset 的输入句式与三个训练类
# 不一致（共同前缀仅 49 token，长度差 4），模型能部分泛化所以不会崩到 0，但必然掉点。
# 已统一。此检查防复发。跳过用 SKIP_PROBE=1。
if [ "${SKIP_PROBE:-0}" != "1" ] && [ -d "${PROBE_MODEL_DIR:-models/Qwen3-0.6B}" ]; then
  echo "[probe] 训练/评估 prompt 一致性 + Trie 形状 ..."
  if ! "${PY}" scripts/sft/probe_constrained_decoding.py \
        --domain "${DOMAIN}" --n-rows 20 \
        --model-dir "${PROBE_MODEL_DIR:-models/Qwen3-0.6B}" \
        > "./logs/probe_${DOMAIN}.log" 2>&1; then
    echo "  [FAIL] 约束解码自检未通过 —— 训练/评估口径可能已漂移"
    echo "         详见 ./logs/probe_${DOMAIN}.log（或临时跳过：SKIP_PROBE=1）"
    exit 1
  fi
  echo "  [OK] 约束解码自检通过（prompt 逐 token 一致 / Trie 5 步）"
fi

echo "=========================================="
echo " GenRetrieval Evaluate - Run-0"
echo "=========================================="
echo " Domain     : ${DOMAIN}  (category=${CATEGORY})"
echo " Model      : ${MODEL_PATH}"
echo " Test file  : ${TEST_FILE}"
echo " Info file  : ${INFO_FILE}"
echo " Beams      : ${NUM_BEAMS}   Batch: ${BATCH_SIZE}"
echo " Output     : ${result_json}"
echo "=========================================="

"${PY}" ./evaluate.py \
  --base_model "${MODEL_PATH}" \
  --info_file "${INFO_FILE}" \
  --category "${CATEGORY}" \
  --test_data_path "${TEST_FILE}" \
  --result_json_data "${result_json}" \
  --batch_size "${BATCH_SIZE}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --length_penalty "${LENGTH_PENALTY}" \
  2>&1 | tee "./logs/evaluate_run0_${DOMAIN}.log"

"${PY}" ./calc.py \
  --path "${result_json}" \
  --item_path "${INFO_FILE}" \
  2>&1 | tee "./logs/calc_run0_${DOMAIN}.log"

echo ""
echo "评估完成：${result_json}"
echo "⚠️ 指标口径以 docs/EVAL_PROTOCOL.md 为准（HR@10 等），"
echo "   与 MiniOneRec calc.py 自带的 HR/NDCG 定义对照后再引用。"
