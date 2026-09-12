#!/usr/bin/env bash
# =============================================================
#  SID 实验编排：RQ-VAE 训练 -> 双版 SID 导出 -> 三件套评估
# =============================================================
#  三个对比模式（**不做 concat**：PCA 线性拼接已实测几乎无增益，见 EXPERIMENT_LOG EXP-M1-11）
#    text = 纯文本单模态基线
#    mlp  = concat -> MLP 有监督降维（InfoNCE）
#    gate = 门控融合（当前最优）
#
#  每个模式一个结果文件夹，训练日志 / 指标 / ckpt / 两版 SID / 两份评估报告全在里面：
#    results/sid/<域>/<模式>/
#        train.log            RQ-VAE 训练日志
#        train_metrics.json   超参 + 逐 epoch 曲线 + best 指标
#        build_sid.log        两版 SID 导出日志
#        ckpt/                best_collision / best_loss / last
#        sid_raw.npy|.json|.stats.json    纯最近邻量化（无 Sinkhorn）
#        sid_sk.npy|.json|.stats.json     Sinkhorn 碰撞消解后
#        eval_raw.log|.json   三件套评估（raw）
#        eval_sk.log|.json    三件套评估（sinkhorn）
#        sid_stats.json       两版对照
#        summary.json         本模式总览
#
#  用法:
#    bash scripts/multimodal/run_sid_exp.sh IandS 5000 "text mlp gate"
#    bash scripts/multimodal/run_sid_exp.sh IandS 20    # 冒烟（epochs=20，模式默认三个）
# =============================================================
set -u
set -o pipefail   # 让 TRAIN_RC 捕到 python 的真实返回码（否则管道返回的是 tee 的 0）

SHORT="${1:-IandS}"
EPOCHS="${2:-1500}"
MODES="${3:-text mlp gate}"
RESULTS_ROOT="${RESULTS_ROOT:-results/sid}"
EVAL_STEP="${EVAL_STEP:-50}"
BATCH="${BATCH:-2048}"
LR="${LR:-1e-3}"
# 训练期 Sinkhorn：
# ⚠️ sk_epsilons 只有在 --train_use_sk 打开时才生效（vq.py:74 要求 use_sk=True 且 epsilon>0）。
#    默认 TRAIN_USE_SK 为空 = 训练期纯 argmin = **MiniOneRec 官方口径**（解耦最干净）。
#    想做 MQL4GRec 口径的消融：TRAIN_USE_SK="--train_use_sk" TRAIN_SK="0.0 0.0 0.003"
# 注：早年"训练期 Sinkhorn 抗塌缩"的对照已作废——当时 use_sk 被硬编码 False，自变量没变。
TRAIN_SK="${TRAIN_SK:-0.0 0.0 0.003}"
TRAIN_USE_SK="${TRAIN_USE_SK:-}"
# 训练前独立 k-means init pass 的样本量（0 = 上游行为：首 batch 2048 触发）。
# 实测 8192 拿走大部分收益（基尼 0.344→0.176），全量 25847 约 9s；不影响训练 batch/动力学。
INIT_SAMPLES="${INIT_SAMPLES:-0}"
# 实测：总 loss 在 ep120 触底后回升（commitment 项在涨），但重建 R² 一路升到 0.825、
# 碰撞率一路降到 0.185 —— **loss 是假信号**。按碰撞率选 + 跳过 burn-in 才对
# （burn-in 用于避开 ep1 k-means 初始化造成的"未训练但碰撞率最低"假象）
SELECT_CKPT="${SELECT_CKPT:-collision}"
SK_EPS="${SK_EPS:-0.003}"                 # 导出期碰撞消解的 epsilon
EMB_SUBDIR="${EMB_SUBDIR:-long}"          # I&S 的 60 轮长程产物在 emb/long/
DEVICE="${DEVICE:-cuda:0}"
CKPT_NAME="selected_model.pth"

PY=".venv/Scripts/python.exe"
ROOT="data/Amazon23/${SHORT}"

echo "=================================================================="
echo " SID 实验  short=${SHORT}  epochs=${EPOCHS}  modes=${MODES}"
echo " batch=${BATCH}  lr=${LR}  eval_step=${EVAL_STEP}  sk_eps=${SK_EPS}"
echo " results_root=${RESULTS_ROOT}"
echo "=================================================================="

for M in ${MODES}; do
  # --- 标签语法：`gate__init8192` / `gate__trainsk`。
  #     `__` 前面是融合模式（用于找融合向量），后面是消融标签，决定该 run 的覆盖项：
  #       init0 / init2048 / init8192 / initfull  -> 覆盖 init pass 样本量
  #         （initfull 传超大数，preinit_codebooks 内部 min(N) 截断为全量）
  #       trainsk                                 -> 训练期开 Sinkhorn（MQL4GRec 口径）
  #     无标签 = 全局默认（INIT_SAMPLES / TRAIN_USE_SK）
  BASE="${M%%__*}"
  TAG="${M#*__}"
  if [ "${TAG}" = "${M}" ]; then TAG=""; fi

  RUN_INIT="${INIT_SAMPLES}"
  RUN_USE_SK="${TRAIN_USE_SK}"
  case "${TAG}" in
    init0)    RUN_INIT=0 ;;
    init2048) RUN_INIT=2048 ;;
    init8192) RUN_INIT=8192 ;;
    initfull) RUN_INIT=999999999 ;;
    trainsk)  RUN_USE_SK="--train_use_sk" ;;
    "") ;;
    *) echo "[warn] 未知标签 '${TAG}'（run=${M}），按无标签处理" ;;
  esac

  # --- 融合向量文件名：按 BASE 查找；I&S 的长程产物带 _e60 后缀，VG 的不带 ---
  if [ -f "${ROOT}/emb/${EMB_SUBDIR}/emb_fused_${BASE}_e60.npy" ]; then
    EMB="${ROOT}/emb/${EMB_SUBDIR}/emb_fused_${BASE}_e60.npy"
  elif [ -f "${ROOT}/emb/${EMB_SUBDIR}/emb_fused_${BASE}.npy" ]; then
    EMB="${ROOT}/emb/${EMB_SUBDIR}/emb_fused_${BASE}.npy"
  elif [ -f "${ROOT}/emb/emb_fused_${BASE}.npy" ]; then
    EMB="${ROOT}/emb/emb_fused_${BASE}.npy"
  else
    echo "[skip] ${M}: 找不到融合向量"; continue
  fi

  OUT="${RESULTS_ROOT}/${SHORT}/${M}"
  mkdir -p "${OUT}/ckpt"

  echo
  echo "##################################################################"
  echo "# [${M}] $(date +%H:%M:%S)  emb=${EMB}"
  echo "#        out=${OUT}  init_samples=${RUN_INIT}  train_use_sk=${RUN_USE_SK:-off}"
  echo "##################################################################"

  # ---------- 1) 训练 RQ-VAE ----------
  # 训练期是否走 Sinkhorn 由本 run 的 RUN_USE_SK 决定（全局 TRAIN_USE_SK 只作默认，
  # 标签 trainsk 会覆盖它）；RUN_INIT 同理。默认 = 纯 argmin + 首 batch init（MiniOneRec 口径）。
  # 无论哪种，导出期都会另外产出 raw / sk 两版 SID（见第 2 步）。
  echo "[1/4] train  $(date +%H:%M:%S)"
  PYTHONPATH="rq:${PYTHONPATH:-}" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    "${PY}" rq/train_rqvae.py \
      --emb "${EMB}" --out_dir "${OUT}" \
      --epochs "${EPOCHS}" --batch_size "${BATCH}" --lr "${LR}" \
      --eval_step "${EVAL_STEP}" --device "${DEVICE}" \
      --init_samples "${RUN_INIT}" \
      --sk_epsilons ${TRAIN_SK} ${RUN_USE_SK} --select_ckpt "${SELECT_CKPT}" 2>&1 | tee "${OUT}/train.log"
  TRAIN_RC=$?
  if [ "${TRAIN_RC}" -ne 0 ] || [ ! -f "${OUT}/ckpt/${CKPT_NAME}" ]; then
    echo "[fail] ${M} 训练失败 rc=${TRAIN_RC}"; continue
  fi

  # ---------- 2) 导出两版 SID ----------
  echo "[2/4] build_sid  $(date +%H:%M:%S)"
  PYTHONPATH="rq:${PYTHONPATH:-}" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    "${PY}" rq/build_sid_dual.py \
      --ckpt "${OUT}/ckpt/${CKPT_NAME}" --emb "${EMB}" \
      --out_dir "${OUT}" --sk_epsilon "${SK_EPS}" --sk_layers last \
      --device "${DEVICE}" 2>&1 | tee "${OUT}/build_sid.log"

  # ---------- 3) 三件套评估（raw / sinkhorn 各一份）----------
  for V in raw sk; do
    echo "[3/4] eval_${V}  $(date +%H:%M:%S)"
    PYTHONPATH="rq:${PYTHONPATH:-}" \
      "${PY}" rq/eval_sid.py \
        --short "${SHORT}" \
        --codes "${OUT}/sid_${V}.npy" \
        --emb "${EMB}" \
        --ckpt "${OUT}/ckpt/${CKPT_NAME}" \
        --out "${OUT}/eval_${V}.json" 2>&1 | tee "${OUT}/eval_${V}.log"
  done

  # ---------- 4) 汇总 ----------
  echo "[4/4] summary  $(date +%H:%M:%S)"
  "${PY}" scripts/multimodal/make_sid_summary.py \
      --out_dir "${OUT}" --short "${SHORT}" --mode "${M}"

  echo "[done] ${M} -> ${OUT}/summary.json   $(date +%H:%M:%S)"
done

echo
echo "=================================================================="
echo " 全部完成 $(date +%H:%M:%S)"
echo " 结果根目录: ${RESULTS_ROOT}/${SHORT}/"
echo "=================================================================="
