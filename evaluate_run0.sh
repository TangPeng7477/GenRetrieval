#!/bin/bash
set -euo pipefail

# ============================================================
# GenRetrieval - SFT 评估（通用入口：只改环境变量，不改脚本）
# ------------------------------------------------------------
# 换版本只需给两个变量：
#   MODEL_PATH   模型目录   （默认 outputs/<EXP_ID>/final_checkpoint）
#   EXP_ID       实验 ID    （不传则从 MODEL_PATH 自动反推）
#
# ---- 命名规范：一个 EXP_ID 串起训练与评估 ----
#   EXP_ID = <域>-<RUN_TAG>[-<任务集>]      例 IandS-run0 / IandS-untrained / IandS-run0-T1T3
#   results/ 按**阶段**分层：SID 阶段的产物在 results/sid*/（rq/*.py 写），
#   SFT 阶段统一收在 results/sft/ 下，互不混淆。
#   模型        outputs/<EXP_ID>/final_checkpoint/
#   评估结果    results/sft/<EXP_ID>/eval_<域>_beam<B>[_n<N>].json
#   版本元数据  results/sft/<EXP_ID>/eval_<域>_beam<B>[_n<N>].meta.json
#   指标        results/sft/<EXP_ID>/eval_<域>_beam<B>[_n<N>].metrics.json
#   日志        logs/sft/<EXP_ID>/
#   ⟹ 光看路径就知道是哪个阶段、哪个版本；换 beam / 样本数也不会互相覆盖。
#
# ---- 指标口径（只看 HR / NDCG，不看 MRR）----
#   HR@K / NDCG@K = **beam 内排名**（calc.py 的 minID < K）。
#   未生成物品得分 -inf ⟹ 全库排序下 top-K 等价 beam 内 top-K。
#   🔴 beam 宽度就是 HR@K 的硬上限（beam_ceiling），别调小。
#   ⚠️ MRR 不报告：候选集 = beam 内 SID，MRR ≈ 1/beam 是结构常数，
#      不携带排序质量信息（EVAL_PROTOCOL §3.4 已标 n/a）。
#
# 🔴 --base_model 必须指向**训练输出目录**（自带扩展后的 tokenizer）。
#    evaluate.py 默认不做 add_tokens，指回 models/Qwen3-0.6B 会让 SID 碎裂、Trie 全挂。
#    例外：显式给 SID_VOCAB_PATH 时评估端现场注册（dry-run 用，见 SFT_PIPELINE §3.5）。
#
# 根目录的 evaluate.sh 是 MiniOneRec 原版（路径指向本仓不存在的
# ./data/Amazon/...），不要用。
# ============================================================

DOMAIN="${DOMAIN:-IandS}"                                 # IandS | VG
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
SFT_DIR="${SFT_DIR:-data/Amazon23/${DOMAIN}/sft}"
RUN_TAG="${RUN_TAG:-run0}"                                # 与 sft_run0.sh 保持一致的标签

# ---------------- EXP_ID：优先级 显式 EXP_ID > 从 MODEL_PATH 反推 > 默认 ----------------
if [ -z "${EXP_ID:-}" ] && [ -n "${MODEL_PATH:-}" ]; then
  EXP_ID="$(basename "$(dirname "${MODEL_PATH}")")"   # outputs/IandS-run0/final_checkpoint -> IandS-run0
fi
EXP_ID="${EXP_ID:-${DOMAIN}-${RUN_TAG}}"
MODEL_PATH="${MODEL_PATH:-outputs/${EXP_ID}/final_checkpoint}"

# ⚠️ 显存由 `BATCH_SIZE × NUM_BEAMS`（beam 展开后的序列总数）决定，**不是 batch 单独决定**。
#    实测 4GB 卡（3050Ti）：batch4 × beam20 = 80 条序列 ✓ 跑得动；
#    batch8 × beam50 = 400 条 → CUDA OOM（KV cache 爆）。本地想跑就压 batch，**别压 beam**
#    （beam 宽度 = HR@K 的硬上限，压它等于自降天花板）。
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_BEAMS="${NUM_BEAMS:-50}"            # 生成式 HR@K 的硬上限 = beam 宽度，别调小
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"  # 目标只有 3 SID + \n + EOS，16 足够
LENGTH_PENALTY="${LENGTH_PENALTY:-0.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"         # 0=全部；>0 随机取 N 条（dry-run 提速）
SID_VOCAB_PATH="${SID_VOCAB_PATH:-}"    # 仅 dry-run：非空则评估端现场注册词表

TEST_FILE="${SFT_DIR}/test/${DOMAIN}_5_test.csv"
INFO_FILE="${SFT_DIR}/info/${DOMAIN}.item_info.txt"

# ---------------- 落点（命名规范见文件头） ----------------
SAMPLE_TAG=""
if [ "${MAX_SAMPLES}" != "0" ]; then SAMPLE_TAG="_n${MAX_SAMPLES}"; fi
EVAL_TAG="beam${NUM_BEAMS}${SAMPLE_TAG}"
OUT_DIR="results/sft/${EXP_ID}"
LOG_DIR="logs/sft/${EXP_ID}"
RESULT_JSON="${OUT_DIR}/eval_${DOMAIN}_${EVAL_TAG}.json"
META_JSON="${OUT_DIR}/eval_${DOMAIN}_${EVAL_TAG}.meta.json"
METRICS_JSON="${OUT_DIR}/eval_${DOMAIN}_${EVAL_TAG}.metrics.json"
EVAL_LOG="${LOG_DIR}/eval_${EVAL_TAG}.log"
CALC_LOG="${LOG_DIR}/calc_${EVAL_TAG}.log"

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
  echo "            -> 先训练：bash sft_run0.sh   （EXP_ID=${EXP_ID}）"
  exit 1
fi
if [ ! -f "${MODEL_PATH}/tokenizer.json" ]; then
  echo "  [MISSING] ${MODEL_PATH}/tokenizer.json"
  echo "            --base_model 要指向训练输出目录，而不是原始基座"
  exit 1
fi
# 比"目录存在"更强的判据：tokenizer 里必须真的有 SID token
if [ -z "${SID_VOCAB_PATH}" ] && ! grep -q '<a_0>' "${MODEL_PATH}/tokenizer.json" 2>/dev/null; then
  echo "  [FAIL] ${MODEL_PATH}/tokenizer.json 里没有 SID token（<a_0>）—— 这个目录不是训练产物"
  echo "         要么指对目录，要么按 dry-run 显式传："
  echo "           SID_VOCAB_PATH=${SFT_DIR}/info/sid_vocab.json bash evaluate_run0.sh"
  exit 1
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# ---------------- 训练/评估口径一致性（回归检查） ----------------
# 2026-09-16 实测抓到上游遗留 bug：EvalSidDataset 的输入句式与三个训练类不一致
# （共同前缀仅 49 token，长度差 4）。模型能部分泛化所以不会崩到 0，但必然掉点。
# 已统一，此检查防复发。跳过用 SKIP_PROBE=1。
if [ "${SKIP_PROBE:-0}" != "1" ] && [ -d "${PROBE_MODEL_DIR:-models/Qwen3-0.6B}" ]; then
  echo "[probe] 训练/评估 prompt 一致性 + Trie 形状 ..."
  if ! "${PY}" scripts/sft/probe_constrained_decoding.py \
        --domain "${DOMAIN}" --n-rows 20 \
        --model-dir "${PROBE_MODEL_DIR:-models/Qwen3-0.6B}" \
        > "${LOG_DIR}/probe.log" 2>&1; then
    echo "  [FAIL] 约束解码自检未通过 —— 训练/评估口径可能已漂移"
    echo "         详见 ${LOG_DIR}/probe.log（或临时跳过：SKIP_PROBE=1）"
    exit 1
  fi
  echo "  [OK] 约束解码自检通过（prompt 逐 token 一致 / Trie 5 步）"
fi

STARTED_AT="$(date -Iseconds 2>/dev/null || date)"

# ---------------- 口径可追溯：记录本次生效的提示词格式与模板真源 ----------------
# 🔴 骨架（chatml / alpaca）是**会改数字的变量**，不记进 meta 就会出现"同 EXP_ID 两行不可比"。
#    取自 prompt_templates 唯一真源；读不到就写 unknown（不猜）。
PROMPT_FMT="$("${PY}" -c "import prompt_templates as pt; print(pt.default_format())" 2>/dev/null || echo unknown)"
PROMPT_SRC="$("${PY}" -c "import prompt_templates as pt; print(pt.source('${INFO_FILE}'))" 2>/dev/null || echo unknown)"

echo "=========================================="
echo " GenRetrieval Evaluate   (EXP_ID = ${EXP_ID})"
echo "=========================================="
echo " Domain     : ${DOMAIN}  (category=${CATEGORY})"
echo " Model      : ${MODEL_PATH}"
echo " Test file  : ${TEST_FILE}"
echo " Info file  : ${INFO_FILE}"
echo " Beams      : ${NUM_BEAMS}   Batch: ${BATCH_SIZE}   max_new_tokens: ${MAX_NEW_TOKENS}"
echo " Samples    : ${MAX_SAMPLES}   (0 = 全部)"
echo " Result     : ${RESULT_JSON}"
echo " Meta / 指标: ${META_JSON}"
echo "              ${METRICS_JSON}"
echo "=========================================="

T_EVAL_0="$(date +%s)"
"${PY}" ./evaluate.py \
  --base_model "${MODEL_PATH}" \
  --info_file "${INFO_FILE}" \
  --category "${CATEGORY}" \
  --test_data_path "${TEST_FILE}" \
  --result_json_data "${RESULT_JSON}" \
  --batch_size "${BATCH_SIZE}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --length_penalty "${LENGTH_PENALTY}" \
  --sid_vocab_path "${SID_VOCAB_PATH}" \
  --max_samples "${MAX_SAMPLES}" \
  2>&1 | tee "${EVAL_LOG}"
T_EVAL_1="$(date +%s)"
EVAL_SECONDS=$((T_EVAL_1 - T_EVAL_0))

"${PY}" ./calc.py \
  --path "${RESULT_JSON}" \
  --item_path "${INFO_FILE}" \
  2>&1 | tee "${CALC_LOG}"
T_CALC_1="$(date +%s)"
CALC_SECONDS=$((T_CALC_1 - T_EVAL_1))

echo ""
echo "[timing] 推理(evaluate.py) ${EVAL_SECONDS}s   指标(calc.py) ${CALC_SECONDS}s"

# ---------------- 落盘版本元数据 + 指标（不动 calc.py 的口径，只在外层解析） ----------------
"${PY}" scripts/sft/eval_report.py meta --out "${META_JSON}" \
  --set "exp_id=${EXP_ID}" \
  --set "domain=${DOMAIN}" \
  --set "category=${CATEGORY}" \
  --set "prompt_format=${PROMPT_FMT}" \
  --set "prompt_templates_source=${PROMPT_SRC}" \
  --set "base_model=${MODEL_PATH}" \
  --set "base_model_has_sid_token_map=$([ -f "${MODEL_PATH}/sid_token_map.json" ] && echo true || echo false)" \
  --set "registered_at_eval=$([ -n "${SID_VOCAB_PATH}" ] && echo true || echo false)" \
  --set "test_file=${TEST_FILE}" \
  --set "info_file=${INFO_FILE}" \
  --set "n_items=$(wc -l < "${INFO_FILE}" | tr -d ' ')" \
  --set "num_beams=${NUM_BEAMS}" \
  --set "max_new_tokens=${MAX_NEW_TOKENS}" \
  --set "length_penalty=${LENGTH_PENALTY}" \
  --set "max_samples=${MAX_SAMPLES}" \
  --set "batch_size=${BATCH_SIZE}" \
  --set "result_json=${RESULT_JSON}" \
  --set "eval_seconds=${EVAL_SECONDS}" \
  --set "calc_seconds=${CALC_SECONDS}" \
  --set "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  --set "started_at=${STARTED_AT}" \
  --set "finished_at=$(date -Iseconds 2>/dev/null || date)"

set +e
"${PY}" scripts/sft/eval_report.py metrics --log "${CALC_LOG}" --out "${METRICS_JSON}" \
  --result-json "${RESULT_JSON}"
METRICS_RC=$?
set -e

echo ""
echo "=========================================="
echo " 评估完成   EXP_ID = ${EXP_ID}"
echo "   结果 json   : ${RESULT_JSON}"
echo "   版本元数据  : ${META_JSON}"
echo "   指标(HR/NDCG): ${METRICS_JSON}"
echo "   日志        : ${LOG_DIR}/"
echo "=========================================="
echo "口径：HR@K / NDCG@K 都是 **beam 内排名**（calc.py），beam 宽度即 HR@K 的硬上限。"
echo "      MRR 不报告 —— 生成式下 ≈1/beam 是结构常数，不携带排序质量信息。"
echo "      与 baseline 表（全库排序）比较前，先读 docs/EVAL_PROTOCOL.md §3.4 与 §5。"
if [ "${METRICS_RC}" -ne 0 ]; then
  echo "⚠️ 指标 json 解析失败（原始 log 已保留）：${CALC_LOG}"
fi

# ---------------- 汇总：重扫 results/sft/ 全部 EXP_ID，重写 docs/SFT_EVAL_RESULTS.md ----------------
# 全量重扫而非增量追加 ⟹ 幂等，永远反映最新全貌；硬串行三阶段跑完自动得到完整对照表。
# 🔴 与 evaluate 本身解耦：eval 失败也照写（把已有结果保住），只告警。
# ⚠️ 输出文件 `docs/SFT_EVAL_RESULTS.md` **入 git 且自动生成，勿手改**。
if [ "${COLLECT:-1}" != "0" ]; then
  echo ""
  echo "---------------- 汇总 --------------------"
  set +e
  "${PY}" scripts/sft/collect_eval_results.py
  COLLECT_RC=$?
  set -e
  echo "（汇总表已重写：docs/SFT_EVAL_RESULTS.md；单个 EXP_ID 的明细在 results/sft/${EXP_ID}/）"
  if [ "${COLLECT_RC}" -ne 0 ]; then
    echo "⚠️ 汇总失败（不影响本次评估产物）；可手动重跑："
    echo "     ${PY} scripts/sft/collect_eval_results.py"
    echo "   跳过汇总：COLLECT=0 bash evaluate_run0.sh"
  fi
fi
