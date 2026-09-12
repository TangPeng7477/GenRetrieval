#!/usr/bin/env bash
# M1-5 融合消融：两域 × 四模式（text / concat / mlp / gate）+ 统一评估
# 用法: bash scripts/multimodal/fuse_all.sh [IandS VG]
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=".venv/Scripts/python.exe"
SHORTS="${*:-IandS VG}"

EPOCHS="${EPOCHS:-3}"
MAX_PAIRS="${MAX_PAIRS:-200000}"
TARGET_DIM="${TARGET_DIM:-1024}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

for S in $SHORTS; do
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] === $S 四模式融合 ==="
  echo "=============================================================="
  for M in text concat mlp gate; do
    echo "---- [$(date +%H:%M:%S)] $S / $M ----"
    $PY scripts/multimodal/fuse_embeddings.py \
        --short "$S" --mode "$M" \
        --target_dim "$TARGET_DIM" --epochs "$EPOCHS" --max_pairs "$MAX_PAIRS"
  done
  echo "---- [$(date +%H:%M:%S)] $S / 评估 ----"
  $PY scripts/multimodal/fuse_embeddings.py \
      --short "$S" --eval --modes text concat mlp gate --max_pairs "$MAX_PAIRS"
done

echo "[$(date +%H:%M:%S)] 全部完成"
