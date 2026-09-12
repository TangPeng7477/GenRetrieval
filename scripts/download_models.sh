#!/bin/bash
set -euo pipefail

# ============================================================
# Download models for MiniOneRec 3090 experiments
#
# Usage:
#   bash scripts/download_models.sh                 # default: download all
#   bash scripts/download_models.sh --base-only     # only Qwen2.5-0.5B base
#   bash scripts/download_models.sh --ckpt-only     # only official 1.5B ckpts
#   bash scripts/download_models.sh --mirror        # use HF mirror (China)
# ============================================================

MODEL_DIR="${MODEL_DIR:-./models}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
USE_MIRROR=false
DOWNLOAD_BASE=true
DOWNLOAD_CKPT=true

for arg in "$@"; do
  case $arg in
    --mirror)     USE_MIRROR=true ;;
    --base-only)  DOWNLOAD_CKPT=false ;;
    --ckpt-only)  DOWNLOAD_BASE=false ;;
    --model-dir=*) MODEL_DIR="${arg#*=}" ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

# HF endpoint
if $USE_MIRROR; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "[mirror] Using HF mirror: $HF_ENDPOINT"
else
  export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
fi

mkdir -p "${MODEL_DIR}"

# Ensure huggingface-cli is available
if ! command -v huggingface-cli &>/dev/null; then
  echo "Installing huggingface_hub CLI..."
  pip install -q huggingface_hub
fi

echo ""
echo "=========================================="
echo "MiniOneRec Model Download"
echo "Model dir: ${MODEL_DIR}"
echo "=========================================="

# --- 1. Download Qwen2.5-0.5B-Instruct (training base) ---
if $DOWNLOAD_BASE; then
  echo ""
  echo "[1/2] Downloading Qwen2.5-0.5B-Instruct base model..."
  echo "------------------------------------------"
  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir "${MODEL_DIR}/Qwen2.5-0.5B-Instruct"
  echo "Done: ${MODEL_DIR}/Qwen2.5-0.5B-Instruct"
fi

# --- 2. Download official 1.5B checkpoints from kkknight/MiniOneRec ---
if $DOWNLOAD_CKPT; then
  echo ""
  echo "[2/2] Downloading official 1.5B checkpoints from kkknight/MiniOneRec..."
  echo "------------------------------------------"

  # Industrial checkpoint
  echo "  -> Industrial_ckpt"
  mkdir -p "${MODEL_DIR}/MiniOneRec_1.5B/Industrial_ckpt"
  for f in model.safetensors config.json generation_config.json \
           tokenizer.json tokenizer_config.json special_tokens_map.json \
           added_tokens.json vocab.json merges.txt chat_template.jinja; do
    if [[ ! -f "${MODEL_DIR}/MiniOneRec_1.5B/Industrial_ckpt/${f}" ]]; then
      echo "     Downloading ${f}..."
      huggingface-cli download kkknight/MiniOneRec \
        "Industrial_ckpt/${f}" \
        --local-dir "${MODEL_DIR}/MiniOneRec_1.5B"
    fi
  done

  # Office checkpoint
  echo "  -> Office_ckpt"
  mkdir -p "${MODEL_DIR}/MiniOneRec_1.5B/Office_ckpt"
  for f in model.safetensors config.json generation_config.json \
           tokenizer.json tokenizer_config.json special_tokens_map.json \
           added_tokens.json vocab.json merges.txt chat_template.jinja; do
    if [[ ! -f "${MODEL_DIR}/MiniOneRec_1.5B/Office_ckpt/${f}" ]]; then
      echo "     Downloading ${f}..."
      huggingface-cli download kkknight/MiniOneRec \
        "Office_ckpt/${f}" \
        --local-dir "${MODEL_DIR}/MiniOneRec_1.5B"
    fi
  done

  echo "Done: ${MODEL_DIR}/MiniOneRec_1.5B/"
fi

echo ""
echo "=========================================="
echo "Download complete!"
echo ""
echo "Files:"
if $DOWNLOAD_BASE; then
  echo "  Base model:  ${MODEL_DIR}/Qwen2.5-0.5B-Instruct"
fi
if $DOWNLOAD_CKPT; then
  echo "  1.5B Industrial ckpt: ${MODEL_DIR}/MiniOneRec_1.5B/Industrial_ckpt"
  echo "  1.5B Office ckpt:     ${MODEL_DIR}/MiniOneRec_1.5B/Office_ckpt"
fi
echo "=========================================="
