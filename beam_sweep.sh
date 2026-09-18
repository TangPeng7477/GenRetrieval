#!/usr/bin/env bash
# ============================================================================
# beam_sweep.sh —— 生成式 beam 宽度扫描（只报 beam_ceiling，不发散结论）
# ============================================================================
# 目的：终结「加宽 beam 有没有用」的争论。判据不是 HR（HR 混了"没生成出来"
#       和"没排到前面"两件事），而是 beam_ceiling —— 目标是否**出现在生成出来的
#       beam 里**（不限前 10）。定义见 baseline/generative/sid_gr.py:19。
#
# 🔴 为什么必须单独一个脚本：每个 beam 宽度的结果必须落到**不同的 EXP_ID**，
#    否则 results/sft/<EXP_ID>/eval_*.json 互相覆盖，扫完只剩最后一个。
#
# 用法（云端 4090D）：
#   bash beam_sweep.sh                                    # 用默认设置
#   MAX_SAMPLES=1000 bash beam_sweep.sh                   # 先小样本探路
#   MODEL_PATH=outputs/IandS-hs1-T2aT2b/final_checkpoint \
#     EXP_ID_BASE=IandS-hs1-T2aT2b bash beam_sweep.sh     # 指定 hs1 产物
#   BEAMS="20 50 100 256" bash beam_sweep.sh
#
# 环境变量：
#   BEAMS          要扫的 beam 宽度列表（空格分隔），默认 "20 50 100 256"
#   MODEL_PATH     模型目录（默认 outputs/<EXP_ID_BASE>/final_checkpoint）
#   EXP_ID_BASE    基础 EXP_ID（默认取 MODEL_PATH 的父目录名）
#   DOMAIN         数据域，默认 IandS
#   MAX_SAMPLES    0=全部；>0 随机取 N 条（建议先 1000 探路）
#   BASE_BATCH     每档 beam 的 batch 由它推导（见下）
#   ATTN_IMPL      attention 后端，默认 sdpa
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DOMAIN="${DOMAIN:-IandS}"
BEAMS="${BEAMS:-20 50 100 256}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

# ---- 解析 python 解释器（evaluate_run0.sh 里定义的是脚本内变量，这里拿不到）----
# 口径与 evaluate_run0.sh:109-113 一致：优先 venv，本机/云端分 Windows 与 Linux。
if [ -z "${PY:-}" ]; then
  if [ -x "./.venv/Scripts/python.exe" ]; then
    PY="./.venv/Scripts/python.exe"     # Windows
  elif [ -x "./.venv/bin/python" ]; then
    PY="./.venv/bin/python"             # Linux（云端）
  else
    PY="python"
  fi
fi
export PY

# ---- 解析 EXP_ID_BASE / MODEL_PATH ----
if [ -z "${MODEL_PATH:-}" ] && [ -n "${EXP_ID_BASE:-}" ]; then
  MODEL_PATH="outputs/${EXP_ID_BASE}/final_checkpoint"
fi
if [ -z "${MODEL_PATH:-}" ]; then
  echo "[ERROR] 必须给 MODEL_PATH 或 EXP_ID_BASE 之一"
  echo "        例：MODEL_PATH=outputs/IandS-hs1-T2aT2b/final_checkpoint bash beam_sweep.sh"
  exit 1
fi
EXP_ID_BASE="${EXP_ID_BASE:-$(basename "$(dirname "${MODEL_PATH}")")}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[ERROR] 模型目录不存在：${MODEL_PATH}"
  exit 1
fi

# ---- batch 自适应 ----------------------------------------------------------
# 显存 ≈ batch × beam（beam 展开成 batch 倍序列）。
# 实测基线（3050Ti 4GB, 0.6B）：batch4 × beam20 = 80 条序列 ≈ 2.6 GB 峰值显存。
#   ⟹ 约 32 MB/序列（含 KV cache，max_new_tokens 已压到 8，KV 极小）。
# 4090D 24GB 留 4GB 给其他开销 ⟹ 可用约 20GB ⟹ 单批 **512 条序列**是安全的，
#   再往上边际收益很低（GPU 已被打满），且 OOM 风险上升。
# 🔴 只压 batch，绝不压 beam —— beam 宽度是 HR@K / beam_ceiling 的硬上限。
MAX_SEQ_PER_BATCH="${MAX_SEQ_PER_BATCH:-512}"
BASE_BATCH="${BASE_BATCH:-32}"

pick_batch() {
  local beam="$1"
  local b=$(( MAX_SEQ_PER_BATCH / beam ))
  [ "$b" -lt 1 ] && b=1
  [ "$b" -gt "$BASE_BATCH" ] && b="$BASE_BATCH"
  echo "$b"
}

echo "=========================================="
echo " Beam Sweep   (base EXP_ID = ${EXP_ID_BASE})"
echo "=========================================="
echo " Model       : ${MODEL_PATH}"
echo " Domain      : ${DOMAIN}"
echo " Beams       : ${BEAMS}"
echo " Max samples : ${MAX_SAMPLES}  (0 = 全部)"
echo " Attn        : ${ATTN_IMPL}"
echo "------------------------------------------"
echo " 计划（batch 按 batch×beam<=${MAX_SEQ_PER_BATCH} 自适应）："
for B in ${BEAMS}; do
  BB="$(pick_batch "$B")"
  printf "   beam=%-4s -> batch=%-3s  (%s 条序列/批)\n" "$B" "$BB" "$((B * BB))"
done
echo "=========================================="

# ---- 预检模式：只打印计划，不跑 --------------------------------------------
# 用途：上云前先确认 EXP_ID 不冲突、模型目录在、数据在。
#   DRY_RUN=1 bash beam_sweep.sh
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ""
  echo "[预检模式] 不执行评估。逐档确认："
  for B in ${BEAMS}; do
    BB="$(pick_batch "$B")"
    EID="${EXP_ID_BASE}-b${B}${MAX_SAMPLES:+_n${MAX_SAMPLES}}"
    R_DIR="results/sft/${EID}"
    L_DIR="logs/sft/${EID}"
    R_N="$(ls "${R_DIR}"/eval_*.json 2>/dev/null | grep -v meta | grep -v metrics | wc -l)"
    printf "  beam=%-4s batch=%-3s EXP_ID=%-28s 已有结果=%s 个\n" "$B" "$BB" "$EID" "$R_N"
    [ "$R_N" != "0" ] && echo "           ⚠️ 该 EXP_ID 已有结果，重跑会覆盖（若要保留请改 EXP_ID_BASE）"
  done
  echo ""
  echo "模型目录: ${MODEL_PATH} $([ -d "${MODEL_PATH}" ] && echo '✓' || echo '✗ 不存在')"
  echo "测试数据: data/Amazon23/${DOMAIN}/sft/test/${DOMAIN}_5_test.csv $([ -f "data/Amazon23/${DOMAIN}/sft/test/${DOMAIN}_5_test.csv" ] && echo '✓' || echo '✗ 不存在')"
  echo "（预检结束，去掉 DRY_RUN=1 即可正式跑）"
  exit 0
fi

SWEEP_T0="$(date +%s)"
SUMMARY_ROWS=()

for B in ${BEAMS}; do
  BB="$(pick_batch "$B")"
  # 🔴 每档一个独立 EXP_ID，结果才不会互相覆盖（EVAL_TAG 里已含 beam 宽度）
  EXP_ID="${EXP_ID_BASE}-b${B}${MAX_SAMPLES:+_n${MAX_SAMPLES}}"

  echo ""
  echo "############################################"
  echo "#  beam=${B}   batch=${BB}   EXP_ID=${EXP_ID}"
  echo "############################################"

  T0="$(date +%s)"
  if ! MODEL_PATH="${MODEL_PATH}" EXP_ID="${EXP_ID}" \
       NUM_BEAMS="${B}" BATCH_SIZE="${BB}" \
       MAX_SAMPLES="${MAX_SAMPLES}" ATTN_IMPL="${ATTN_IMPL}" \
       DOMAIN="${DOMAIN}" \
       bash -c 'bash evaluate_run0.sh' 2>&1 | tee "logs/sft/${EXP_ID}.sweep.log"; then
    echo "  [FAIL] beam=${B} 这一档失败，继续下一档（详见上方日志）"
    SWEEP_ROWS+=("${B}|${BB}|FAIL|—|—")
    continue
  fi
  T1="$(date +%s)"
  EL=$((T1 - T0))

  # ---- 提取 beam_ceiling ---------------------------------------------------
  # calc.py 的指标 json 里若已含 beam_ceiling 就直接用；否则现场算（只看目标
  # 是否落在 predict 列表里，与排名无关）。
  MJ="$(ls -t results/sft/${EXP_ID}/eval_*_beam${B}*.metrics.json 2>/dev/null | head -1)"
  RJ="$(ls -t results/sft/${EXP_ID}/eval_*_beam${B}*.json 2>/dev/null | grep -v meta | grep -v metrics | head -1)"

  if [ -z "$RJ" ]; then
    echo "  [WARN] 找不到结果 json，跳过统计"
    SWEEP_ROWS+=("${B}|${BB}|NO_JSON|—|${EL}s")
    continue
  fi

  CEIL="$("${PY:-./.venv/bin/python}" - "$RJ" "$MJ" <<'PYEOF' 2>/dev/null
import json, sys, glob, os
rj, mj = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
# 优先用 calc.py 已算好的
if mj and os.path.exists(mj):
    try:
        m = json.load(open(mj, encoding="utf-8"))
        for k in ("beam_ceiling", "beam_ceiling@10", "beam_ceiling_at_10"):
            if k in m:
                print(f"{float(m[k]):.4f}|calc.py"); sys.exit()
    except Exception:
        pass
# 现场算：目标出现在 predict 列表里即命中（不限名次）
d = json.load(open(rj, encoding="utf-8"))
n = hit = nonempty = 0
for row in d:
    pr = row.get("predict") or []
    preds = pr if isinstance(pr, list) else [pr]
    tgt = row.get("output") or row.get("target") or ""
    tgt = tgt.strip() if isinstance(tgt, str) else ""
    n += 1
    if any(str(p).strip() for p in preds):
        nonempty += 1
    if tgt and any(str(p).strip() == tgt for p in preds):
        hit += 1
print(f"{hit/max(n,1):.4f}|n={n},非空={nonempty}")
PYEOF
)"
  CEIL_V="${CEIL%%|*}"
  CEIL_N="${CEIL##*|}"
  echo "  [result] beam=${B}  beam_ceiling=${CEIL_V}  (${CEIL_N})  耗时=${EL}s"
  SWEEP_ROWS+=("${B}|${BB}|${CEIL_V}|${CEIL_N}|${EL}s")
done

SWEEP_T1="$(date +%s)"

# ---- 汇总 -----------------------------------------------------------------
echo ""
echo "=========================================="
echo " Beam Sweep 汇总   (总计 $((SWEEP_T1 - SWEEP_T0))s)"
echo "=========================================="
printf "%-6s %-7s %-14s %-24s %s\n" "beam" "batch" "beam_ceiling" "说明" "耗时"
echo "----------------------------------------------------------------------"
for R in "${SWEEP_ROWS[@]}"; do
  IFS='|' read -r b bb cv cn el <<< "$R"
  printf "%-6s %-7s %-14s %-24s %s\n" "$b" "$bb" "$cv" "$cn" "$el"
done
echo "----------------------------------------------------------------------"
cat <<'NOTE'

🔴 读法纪律（别弄错）：
  1. beam_ceiling = 目标**出现在 beam 里**的比例，与名次无关。
     HR@K 混了"没生成出来"和"没排到前面"⟹ 只有 ceiling 能定位瓶颈。
  2. 若 ceiling 随 beam 上升 ⟹ 瓶颈在**解码宽度**，该继续加宽。
     若 ceiling 早已饱和 ⟹ 瓶颈在**排序质量**，加宽无用，该改 prompt / 数据。
  3. 本扫描**不产出**可直接引用的成绩：换 beam 会改变 HR@K 的硬上限，
     与已有 baseline 数字**不可横比**（详见 docs/EVAL_PROTOCOL.md）。
  4. 采样默认已关（do_sample=False），本扫描是确定性结果，可复现。
NOTE
