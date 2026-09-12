#!/usr/bin/env bash
# M1-5 长程融合训练：全量共现对 + 多轮次 + 留出对曲线
# 目的：回答"轮次该给多少"——用留出对指标找拐点，而不是盯着训练 loss。
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/Scripts/python.exe"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

SHORT="${1:?用法: fuse_long.sh <IandS|VG> <epochs> [eval_every]}"
EPOCHS="${2:?需要 epochs}"
EVAL_EVERY="${3:-5}"
OUT="data/Amazon23/$SHORT/emb/long"
mkdir -p "$OUT"

# 把未训练的基线向量放进来，便于统一评估（它们不受共现对切分影响）
for B in text concat; do
  SRC="data/Amazon23/$SHORT/emb/emb_fused_$B.npy"
  [ -f "$SRC" ] && [ ! -f "$OUT/emb_fused_$B.npy" ] && cp "$SRC" "$OUT/" && echo "[cp] $B 基线 -> $OUT"
done

for M in mlp gate; do
  echo "=============================================================="
  echo "---- [$(date +%H:%M:%S)] $SHORT / $M / epochs=$EPOCHS / 全量对 ----"
  echo "=============================================================="
  $PY scripts/multimodal/fuse_embeddings.py --short "$SHORT" --mode "$M" \
      --max_pairs 0 --epochs "$EPOCHS" --eval_every "$EVAL_EVERY" \
      --out_dir "$OUT" --out_name "${M}_e${EPOCHS}"
done

echo "---- [$(date +%H:%M:%S)] $SHORT 统一评估 ----"
$PY scripts/multimodal/fuse_embeddings.py --short "$SHORT" --eval --max_pairs 0 \
    --out_dir "$OUT" \
    --modes text concat mlp_e${EPOCHS} gate_e${EPOCHS} 2>&1 | tail -60
echo "[$(date +%H:%M:%S)] $SHORT 全部完成"
