#!/usr/bin/env bash
# =============================================================
#  VG 域 SID 构建：定版 RQ-VAE（gate+init8192+5000 轮）+ RQ-KMeans 对照
# =============================================================
#  与 I&S 定版严格同口径（否则两边的 SID 数字不可比）：
#    · 融合：全量共现对（--max_pairs 0）+ 60 轮 → emb/long/emb_fused_gate_e60.npy
#    · RQ-VAE：3 层 × 256 码 × 32 维，--init_samples 8192，5000 轮，按碰撞率选 ckpt
#    · RQ-KMeans：MiniOneRec 原版 FAISS ResidualQuantizer + 同款导出期 Sinkhorn
#    · 评估：rq/eval_sid.py 三件套（raw / sk 各一份）
#
#  产物位置（全部在 results/sid_e5000/VG/ 下）：
#    gate__init8192/{train.log, train_metrics.json, ckpt/selected_model.pth,
#                    sid_raw.* , sid_sk.*, eval_raw.json, eval_sk.json, summary.json}
#    rqkmeans/{sid_raw.*, sid_sk.*, eval_raw.json, eval_sk.json, summary.json}
#    compare_rqkmeans.md / .json       两量化器的对比表
#    data/Amazon23/VG/emb/long/        融合向量 emb_fused_gate_e60.npy + 曲线
#
#  用法：
#    bash scripts/multimodal/run_vg_sid.sh
# =============================================================
set -uo pipefail
# 本会话 bash 的 coreutils（mkdir/date/tee/tail/cp）不在默认 PATH 里，必须显式前置 usr/bin
export PATH="/c/Users/z/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:/c/Users/z/.workbuddy/binaries/PortableGit/versions/1.2.0/bin:$PATH"
cd "$(dirname "$0")/../.."

SHORT=VG
ROOT_DIR="results/sid_e5000/${SHORT}"
EMB="data/Amazon23/${SHORT}/emb/long/emb_fused_gate_e60.npy"
mkdir -p "${ROOT_DIR}"

echo "=================================================================="
echo " VG SID 构建启动 $(date +%H:%M:%S)   （总耗时预计 ~2h：融合 ~1h + RQ-VAE ~1h）"
echo " 日志：${ROOT_DIR}/_run.log"
echo "=================================================================="

# ---------- 0) 融合（与 I&S 定版同口径：全量对 + 60 轮） ----------
echo "[0/4] fuse_long 60 轮  $(date +%H:%M:%S)"
bash scripts/multimodal/fuse_long.sh "${SHORT}" 60 5
if [ ! -f "${EMB}" ]; then
  echo "[fail] 融合向量未生成：${EMB}"; exit 1
fi

# ---------- 1) RQ-VAE（定版） ----------
echo "[1/4] RQ-VAE gate__init8192 5000 轮  $(date +%H:%M:%S)"
RESULTS_ROOT=results/sid_e5000 INIT_SAMPLES=8192 \
  bash scripts/multimodal/run_sid_exp.sh "${SHORT}" 5000 "gate__init8192"

# ---------- 2) RQ-KMeans（MiniOneRec 原版对照） ----------
echo "[2/4] RQ-KMeans  $(date +%H:%M:%S)"
PYTHONPATH="rq:${PYTHONPATH:-}" .venv/Scripts/python.exe rq/build_sid_rqkmeans.py \
  --emb "${EMB}" --out_dir "${ROOT_DIR}/rqkmeans"

# ---------- 3) rqkmeans 的三件套评估（无 ckpt → fidelity 跳过，与 I&S 一致） ----------
echo "[3/4] eval rqkmeans  $(date +%H:%M:%S)"
for V in raw sk; do
  PYTHONPATH="rq:${PYTHONPATH:-}" .venv/Scripts/python.exe rq/eval_sid.py \
    --short "${SHORT}" \
    --codes "${ROOT_DIR}/rqkmeans/sid_${V}.npy" \
    --emb "${EMB}" \
    --out "${ROOT_DIR}/rqkmeans/eval_${V}.json" 2>&1 | tee "${ROOT_DIR}/rqkmeans/eval_${V}.log"
done

# ---------- 4) 对比表 ----------
echo "[4/4] compare  $(date +%H:%M:%S)"
.venv/Scripts/python.exe scripts/multimodal/compare_rqkmeans.py \
  "${ROOT_DIR}" gate__init8192 rqkmeans

echo
echo "=================================================================="
echo " VG 全部完成 $(date +%H:%M:%S)"
echo " 结果根目录: ${ROOT_DIR}/"
echo "=================================================================="
