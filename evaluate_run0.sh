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
#   评估结果    results/sft/<EXP_ID>/eval_<域>_beam<B>[_samp][_n<N>][_<TAG>].json
#   版本元数据  results/sft/<EXP_ID>/eval_<域>_beam<B>[_samp][_n<N>][_<TAG>].meta.json
#   指标        results/sft/<EXP_ID>/eval_<域>_beam<B>[_samp][_n<N>][_<TAG>].metrics.json
#   日志        logs/sft/<EXP_ID>/
#   ⟹ 光看路径就知道是哪个阶段、哪个版本；换 beam / 采样 / 样本数 / **评估集**都不会互相覆盖。
#   🔴 `_<TAG>` = **非默认 TEST_FILE** 时自动加，取文件名最后一段（`IandS_5_test.u5k.csv` -> `_u5k`）。
#      没有它的时候踩过一次真事故（[实测] 2026-09-25）：同一个 EXP_ID 先在**全量** test 上评、
#      再在**子集** test 上以 `MAX_SAMPLES=0` 评 ⟹ 两次写的是**同一个文件**
#      ⟹ 汇总表里 `n=50,982` 那行被 `n=5,000` 静默顶掉（全量锚点从表里消失）。
#      默认 TEST_FILE 不加后缀 ⟹ 旧路径与旧脚本行为逐位不变。
#
# ---- 汇总表（docs/SFT_EVAL_RESULTS.md）----
#   收尾自动重扫 results/sft/ 重写该表（全量重扫 ⟹ 幂等）。
#   COLLECT=auto（默认）：Linux/Darwin 写表，MINGW/MSYS/CYGWIN（本机 Windows）不写。
#   AUTO_COMMIT=0（默认）：置 1 时写完表**自动 git commit + push**（云端用），
#   彻底消除"表入 git 但工作区 dirty ⟹ 下次 git pull 被拒"的反复撞车。详见文末注释。
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
# [本项目 2026-09-19 提效] 16 -> 8。目标恒为 3 个 SID token + EOS（第 4 步），
#   实测 2000 条生成长度全落在 16~21 字符（= 3 个 SID），无一例外 ⟹ 8 有 2 倍余量。
#   ⚠️ 改 SID 层数 / 放开多候选生成时必须同步调大。
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"         # 0=全部；>0 随机取 N 条（dry-run 提速）
SID_VOCAB_PATH="${SID_VOCAB_PATH:-}"    # 仅 dry-run：非空则评估端现场注册词表
# [本项目 2026-09-19 提效] attention 后端。sdpa = PyTorch 原生，在 4090D（Ada, sm_89）上
#   自动走 FlashAttention-2 内核，**零额外依赖**（无需 flash_attn 包）。一般不用改。
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

# ---------------- 解码采样（口径开关，2026-09-19 新增） ----------------
# 🔴 背景：evaluate.py 原来**不传** do_sample ⟹ HF 会用基座 generation_config.json 的
#    `do_sample: true` 回填 ⟹ **一直在跑束采样（BEAM_SAMPLE）**，而本仓代码里看不见。
#    Qwen2.5-0.5B 与 Qwen3-0.6B 两个基座都写着 true ⟹ MiniOneRec 原版同样在采样。
#    溯源见 docs/DECODING_STRATEGIES.md §2.2/§2.3、docs/SFT_PIPELINE.md §3.5.4。
# ⟹ 现在改成**显式可控**：
#    DO_SAMPLE=False（默认）= 纯束搜索，确定性、可复现 —— 基准口径。
#    DO_SAMPLE=True          = 束采样，需同时给 TEMPERATURE / TOP_P 做消融。
# ⚠️ 与 hs1 那次已跑的评估（跑在"隐式 do_sample=True, temp=0.6"下）**不可直接横比**，
#    重跑后 meta 里的 do_sample 字段会如实区分。详见 §解码口径注意事项。
DO_SAMPLE="${DO_SAMPLE:-False}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
# 🔴 真值陷阱防护（与 sft_run0.sh 的 EVAL_BY_EPOCH 同款）：
#    fire 把 "True"/"False" 解析成 Python bool；但若传成 TRUE/1/yes 之类，
#    Python 里非空字符串恒为真 ⟹ 会静默走采样分支，与意图相反且不报错。当场拦掉。
case "${DO_SAMPLE}" in
  True|False) ;;
  *) echo "[ERROR] DO_SAMPLE 只接受 True / False，收到 '${DO_SAMPLE}'"; exit 1 ;;
esac

# 🔴 [本项目新增] 支持环境变量覆盖 —— 指向"同一批用户"的子集 test CSV，使**评估集与训练集同源**
#    （scripts/rl/make_user_subset.py 生成；不传时行为与覆盖前完全一致）。
TEST_FILE="${TEST_FILE:-${SFT_DIR}/test/${DOMAIN}_5_test.csv}"
INFO_FILE="${SFT_DIR}/info/${DOMAIN}.item_info.txt"

# ---------------- 落点（命名规范见文件头） ----------------
SAMPLE_TAG=""
if [ "${MAX_SAMPLES}" != "0" ]; then SAMPLE_TAG="_n${MAX_SAMPLES}"; fi
# 采样消融用后缀区分，否则会互相覆盖（do_sample=False 是默认，不加后缀保持旧路径可读）
if [ "${DO_SAMPLE}" = "True" ]; then
  SAMP_TAG="_samp"
else
  SAMP_TAG=""
fi
# 评估集后缀：非默认 TEST_FILE 必须带，否则会与全量评估**互相覆盖**（详见文件头「命名规范」）。
#   取文件名最后一段做标签：IandS_5_test.u5k.csv -> _u5k；无扩展段时退化为整名。
TEST_TAG=""
if [ "${TEST_FILE}" != "${SFT_DIR}/test/${DOMAIN}_5_test.csv" ]; then
  _tt="$(basename "${TEST_FILE}" .csv)"     # IandS_5_test.u5k
  _tt="${_tt##*.}"                          # u5k（无 '.' 时原样返回）
  TEST_TAG="_${_tt}"
fi
EVAL_TAG="beam${NUM_BEAMS}${SAMP_TAG}${SAMPLE_TAG}${TEST_TAG}"
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
echo " Attn       : ${ATTN_IMPL}   (batch×beam = $((BATCH_SIZE * NUM_BEAMS)) 条序列/批)"
echo " Samples    : ${MAX_SAMPLES}   (0 = 全部)"
# 解码模式回显：一眼看出这次跑的是纯束搜索还是束采样
if [ "${DO_SAMPLE}" = "True" ]; then
  echo " Decoding   : BEAM_SAMPLE（束采样）  do_sample=True  temperature=${TEMPERATURE}  top_p=${TOP_P}"
  echo "              ⚠️ 含随机性；与 do_sample=False 的结果不可直接横比"
else
  echo " Decoding   : BEAM_SEARCH（纯束搜索，确定性）  do_sample=False"
fi
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
  --do_sample "${DO_SAMPLE}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --attn_impl "${ATTN_IMPL}" \
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
# 🔴 `base_model_has_sid_token_map` 的判据（2026-09-25 修正）
#    `sft.py:372` 把 `sid_token_map.json` 写在 **run 根目录**（`output_dir`），**不在 `final_checkpoint/` 里**；
#    而 `MODEL_PATH` 指的就是 `<run 根>/final_checkpoint`（见本文件 :49 的 EXP_ID 反推）。
#    原写法判 `${MODEL_PATH}/sid_token_map.json` ⟹ **恒为 false** ⟹ 汇总表每一行都被打上「非训练产物」
#    （含 `IandS-all` 这种货真价实的训练产物）。
#    ⟹ 改为查 **run 根**；并补一条 tokenizer 判据，覆盖 RL 产物（`rl.py` 不写 `sid_token_map.json`）。
_SID_MAP=false
if [ -f "${MODEL_PATH}/sid_token_map.json" ] || [ -f "${MODEL_PATH%/*}/sid_token_map.json" ]; then
  _SID_MAP=true
elif [ -f "${MODEL_PATH}/tokenizer.json" ] && grep -q '<a_0>' "${MODEL_PATH}/tokenizer.json"; then
  _SID_MAP=true
fi

"${PY}" scripts/sft/eval_report.py meta --out "${META_JSON}" \
  --set "exp_id=${EXP_ID}" \
  --set "domain=${DOMAIN}" \
  --set "category=${CATEGORY}" \
  --set "prompt_format=${PROMPT_FMT}" \
  --set "prompt_templates_source=${PROMPT_SRC}" \
  --set "base_model=${MODEL_PATH}" \
  --set "base_model_has_sid_token_map=${_SID_MAP}" \
  --set "registered_at_eval=$([ -n "${SID_VOCAB_PATH}" ] && echo true || echo false)" \
  --set "test_file=${TEST_FILE}" \
  --set "info_file=${INFO_FILE}" \
  --set "n_items=$(wc -l < "${INFO_FILE}" | tr -d ' ')" \
  --set "num_beams=${NUM_BEAMS}" \
  --set "do_sample=${DO_SAMPLE}" \
  --set "temperature=${TEMPERATURE}" \
  --set "top_p=${TOP_P}" \
  --set "attn_impl=${ATTN_IMPL}" \
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
#
# 🔴 [2026-09-19] **本机（Windows）默认不写这张表**，云端（Linux）照写。
#    原因：`results/sft/` 被 .gitignore 忽略（明细不入仓），但本汇总表**入 git**。
#    两台机器各自的 results/ 是独立的 ⟹ 谁最后写，表里就只剩谁的内容，
#    互相挤掉对方的记录（实测：云端 push 后本机 3 条记录消失，反之亦然）。
#    约定：**表格以云端为准**（云端才有训练产物与全量结果）。
#    本机想看表就显式 `COLLECT=1 bash evaluate_run0.sh`（清楚自己在做什么时）。
#    判据用 `uname -s`：Windows 下 git-bash 返回 `MINGW*/MSYS*/CYGWIN*`，Linux 返回 `Linux`。
_uname_s="$(uname -s 2>/dev/null || echo unknown)"
if [ "${COLLECT:-auto}" = "auto" ]; then
  case "${_uname_s}" in
    Linux|Darwin) COLLECT=1 ;;
    *)            COLLECT=0 ;;   # MINGW / MSYS / CYGWIN = 本机 Windows
  esac
fi
# 真值陷阱防护（与 DO_SAMPLE / EVAL_BY_EPOCH 同款）：只接受 0 / 1。
case "${COLLECT}" in
  0|1) ;;
  *) echo "[ERROR] COLLECT 只接受 0 / 1（或不传=自动），收到 '${COLLECT}'"; exit 1 ;;
esac

if [ "${COLLECT}" != "0" ]; then
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
else
  echo ""
  echo "---------------- 汇总（已跳过）----------------"
  echo " 环境=${_uname_s} ⟹ 本机不写 docs/SFT_EVAL_RESULTS.md（该表以云端为准，详见 docs/SFT_PIPELINE.md §3.7）。"
  echo " 本次结果：results/sft/${EXP_ID}/   （明细已落盘，未汇总进表）"
  echo " 确实需要本机写表时：COLLECT=1 bash evaluate_run0.sh"
fi

# ---------------- 可选：自动提交汇总表（AUTO_COMMIT=1，默认关） ----------------
# 🔴 [2026-09-20] 起因：汇总表**入 git**，但 collect 脚本只改文件、不提交
#    ⟹ 云端工作区长期 dirty ⟹ 下次 `git pull` 因
#    "Your local changes to the following files would be overwritten" 被拒，
#    每轮评估后都要手动 `cp` 备份 + `git checkout -- docs/SFT_EVAL_RESULTS.md` + 重扫。
#    开 AUTO_COMMIT=1 后，由云端**单向**提交并 push ⟹ 落实本表"以云端为准"的约定，
#    撞车从根上消失（本机 COLLECT=0 本就不写表，不会反向竞争）。
#
# 设计要点：
#   - **只提交这一张表**（`git add -- <path>` + `git commit -- <path>` 双重 pathspec 限定）
#     ⟹ 不会顺手把其它未完成的改动一起带走。
#   - **push 失败不算错**：提交已在本地、工作区已干净 ⟹ "拉取被拒"这个核心问题已解决，
#     只告警并给出手动命令（云端可能未配 GitHub 凭据）。
#   - 只接受 0 / 1（与 DO_SAMPLE / EVAL_BY_EPOCH / COLLECT 同款真值陷阱防护）。
# 🔴 必须先**归一化赋值**再引用（`set -u` 下裸引用未定义的变量会直接崩）。
#    2026-09-25 云端实测踩坑：原先写成 `case "${AUTO_COMMIT:-0}"`，
#    下游却用裸 `${AUTO_COMMIT}` ⟹ 不传该变量时 `evaluate_run0.sh: line 339: AUTO_COMMIT: unbound variable`。
#    ⚠️ 教训：隔离测试若**总是显式设置**该变量，就永远测不到"不传"这条路径。
AUTO_COMMIT="${AUTO_COMMIT:-0}"
case "${AUTO_COMMIT}" in
  0|1) ;;
  *) echo "[ERROR] AUTO_COMMIT 只接受 0 / 1（不传=0），收到 '${AUTO_COMMIT}'"; exit 1 ;;
esac

if [ "${AUTO_COMMIT}" = "1" ]; then
  echo ""
  echo "---------------- 自动提交汇总表 ----------------"
  if [ "${COLLECT}" = "0" ]; then
    echo "[AUTO_COMMIT] 跳过：本机 COLLECT=0，未写表，无内容可提交。"
  elif ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[AUTO_COMMIT] 跳过：当前目录不在 git 工作区内。"
  elif git diff --cached --quiet -- docs/SFT_EVAL_RESULTS.md 2>/dev/null \
       && git diff --quiet -- docs/SFT_EVAL_RESULTS.md 2>/dev/null; then
    echo "[AUTO_COMMIT] 跳过：表相对 HEAD 无变化。"
  else
    set +e
    git add -- docs/SFT_EVAL_RESULTS.md
    if [ "$?" -ne 0 ]; then
      echo "⚠️ [AUTO_COMMIT] git add 失败。手动："
      echo "     git add docs/SFT_EVAL_RESULTS.md && git commit -m 'chore(eval): 汇总表' && git push"
    else
      AC_MSG="chore(eval): 自动汇总 ${EXP_ID}（${_uname_s}）"
      git commit -m "${AC_MSG}" -- docs/SFT_EVAL_RESULTS.md
      if [ "$?" -ne 0 ]; then
        echo "⚠️ [AUTO_COMMIT] git commit 失败（多半是云端 git 身份未配置）。手动："
        echo "     git config user.name  '<你的名字>'      # 仓库级，勿加 --global"
        echo "     git config user.email '<你的邮箱>'"
        echo "     git add docs/SFT_EVAL_RESULTS.md && git commit -m '${AC_MSG}' && git push"
      else
        echo "[AUTO_COMMIT] 已提交：${AC_MSG}"
        if git push; then
          echo "[AUTO_COMMIT] 已推送到远端。"
        else
          echo "⚠️ [AUTO_COMMIT] push 失败（提交已在本地、工作区已干净，拉取不会再撞车）。"
          echo "     手动重试： git push"
          echo "     排查凭据： ssh -T git@github.com   （应回 'Hi TangPeng7477!'）"
        fi
      fi
    fi
    set -e
  fi
fi

