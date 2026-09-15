#!/usr/bin/env bash
# ============================================================
# Download base models for GenRetrieval v2 (M3 SFT stage).
#
# Base model decision: docs/UPGRADE_PLAN.md 5.1.1
#   student : Qwen/Qwen3-0.6B  (post-trained)  ~1.50 GB  single-file safetensors
#   teacher : Qwen/Qwen3-1.7B  (post-trained)  ~4.06 GB  2 shards
#   Qwen3-0.6B-Base is a probe-only control and is NOT downloaded here.
#
# Usage:
#   bash scripts/download_base_models.sh                  # student only
#   bash scripts/download_base_models.sh --target all     # student + teacher
#   bash scripts/download_base_models.sh --direct         # use huggingface.co
#   bash scripts/download_base_models.sh --model-dir=/data/models
#
# NOTE: this is a >100MB download. Run it OUTSIDE the agent sandbox, which is
#       throttled to roughly 118 kB/s.
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${REPO_DIR}/models}"
TARGET="student"
ENDPOINT="https://hf-mirror.com"
DRY_RUN=false

# NOTE: must be a while/shift loop, not `for arg in "$@"` -- inside a for loop
# over "$@" the positional parameters are already expanded, so `shift` cannot
# consume the value of an option that takes a separate argument.
while [ $# -gt 0 ]; do
  case "$1" in
    --target=*)    TARGET="${1#*=}";  shift ;;
    --target)      TARGET="${2:-}";   shift 2 ;;
    --model-dir=*) MODEL_DIR="${1#*=}"; shift ;;
    --model-dir)   MODEL_DIR="${2:-}"; shift 2 ;;
    --mirror)      ENDPOINT="https://hf-mirror.com"; shift ;;
    --direct)      ENDPOINT="https://huggingface.co"; shift ;;
    --dry-run)     DRY_RUN=true; shift ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

case "$TARGET" in
  student|teacher|all) ;;
  *) echo "Invalid --target: $TARGET (expected student|teacher|all)"; exit 1 ;;
esac

# ---- locate the hf CLI (prefer the project venv) -------------
HF=""
for c in "${REPO_DIR}/.venv/Scripts/hf.exe" "${REPO_DIR}/.venv/bin/hf" \
         "${REPO_DIR}/.venv/Scripts/huggingface-cli.exe" "${REPO_DIR}/.venv/bin/huggingface-cli"; do
  if [ -x "$c" ] || [ -f "$c" ]; then HF="$c"; break; fi
done
if [ -z "$HF" ]; then
  if command -v hf >/dev/null 2>&1; then HF="$(command -v hf)"
  elif command -v huggingface-cli >/dev/null 2>&1; then HF="$(command -v huggingface-cli)"
  else
    echo "hf CLI not found. Activate the project venv or run: pip install -U huggingface_hub"
    exit 1
  fi
fi

export HF_ENDPOINT="$ENDPOINT"
mkdir -p "$MODEL_DIR"

echo ""
echo "=========================================="
echo "GenRetrieval v2 - base model download"
echo "  endpoint : $ENDPOINT"
echo "  model dir: $MODEL_DIR"
echo "  target   : $TARGET"
echo "  cli      : $HF"
echo "=========================================="

# repo|dir|size|weight files
PLAN=()
if [ "$TARGET" = "student" ] || [ "$TARGET" = "all" ]; then
  PLAN+=("Qwen/Qwen3-0.6B|Qwen3-0.6B|1.50 GB|model.safetensors")
fi
if [ "$TARGET" = "teacher" ] || [ "$TARGET" = "all" ]; then
  PLAN+=("Qwen/Qwen3-1.7B|Qwen3-1.7B|4.06 GB|model-00001-of-00002.safetensors model-00002-of-00002.safetensors")
fi

for entry in "${PLAN[@]}"; do
  IFS='|' read -r REPO DIR SIZE WEIGHTS <<< "$entry"
  DEST="${MODEL_DIR}/${DIR}"

  echo ""
  echo "[${REPO}]  ->  ${DEST}   (${SIZE})"
  echo "------------------------------------------------------------"

  mkdir -p "$DEST"
  if $DRY_RUN; then
    echo "  [dry-run] HF_ENDPOINT=$ENDPOINT"
    echo "  [dry-run] would run: $HF download $REPO --local-dir $DEST"
    continue
  fi
  "$HF" download "$REPO" --local-dir "$DEST"

  # verify weight files landed
  MISSING=""
  for w in $WEIGHTS; do
    if [ ! -f "${DEST}/${w}" ]; then MISSING="${MISSING} ${w}"; fi
  done
  if [ -n "$MISSING" ]; then
    echo "ERROR: weights missing after download:${MISSING}" >&2
    exit 1
  fi
  for w in $WEIGHTS; do
    MB=$(( $(wc -c < "${DEST}/${w}") / 1000000 ))
    echo "  OK  ${w}  (${MB} MB)"
  done
done

echo ""
echo "=========================================="
echo "Done."
for entry in "${PLAN[@]}"; do
  IFS='|' read -r REPO DIR SIZE WEIGHTS <<< "$entry"
  echo "  ${REPO}  ->  ${MODEL_DIR}/${DIR}"
done
echo "=========================================="
