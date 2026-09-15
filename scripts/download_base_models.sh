#!/usr/bin/env bash
# ============================================================
# Download base models for GenRetrieval v2 (M3 SFT stage).
#
# Base model decision: docs/UPGRADE_PLAN.md 5.1.1
#   student : Qwen/Qwen3-0.6B  (post-trained)  1,503,300,328 B  1.50 GB
#   teacher : Qwen/Qwen3-1.7B  (post-trained)  4,063,515,592 B  4.06 GB, 2 shards
#   Qwen3-0.6B-Base is a probe-only control and is NOT downloaded here.
#
# WHY THE DEFAULT PATH IS NOT huggingface_hub
# -------------------------------------------
# Measured 2026-09-16 from this network: hf-mirror.com is flaky at the TLS layer.
# /resolve/... and /api/... both intermittently die with
#     SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING]'))
# (4 of 5 attempts on the model.safetensors resolve URL), and a HEAD sometimes
# omits the `X-Repo-Commit` header, making huggingface_hub raise
#     FileMetadataError "Distant resource does not seem to be on huggingface.co"
#     -> LocalEntryNotFoundError
# (file_download.py:1568 / :1661). It re-raises a raw SSLError out of the worker
# thread (:1600-1602), so ONE blip on ONE worker aborts the whole snapshot.
#
# ModelScope serves the same official repos from cdn-lfs-cn-1.modelscope.cn
# (domestic CDN, Range/206, byte-identical). Cross-checked 2026-09-16, agreeing:
#   hf-mirror X-Linked-Etag  ==  ModelScope X-Linked-Etag  ==  LFS object path
# huggingface.co answered 502 from this network, so it could NOT be used as an
# independent third source.
#
# Usage:
#   bash scripts/download_base_models.sh                    # student, ModelScope
#   bash scripts/download_base_models.sh --target all
#   bash scripts/download_base_models.sh --source hf        # old huggingface_hub path
#   bash scripts/download_base_models.sh --dry-run
#
# NOTE: this is a >100MB download. Run it OUTSIDE the agent sandbox, which is
#       throttled to roughly 118 kB/s.
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${REPO_DIR}/models}"
TARGET="student"
SOURCE="modelscope"
ENDPOINT="https://hf-mirror.com"
RETRIES=6
MAX_WORKERS=2
SKIP_HASH=false
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
    --source=*)    SOURCE="${1#*=}";  shift ;;
    --source)      SOURCE="${2:-}";   shift 2 ;;
    --retries=*)   RETRIES="${1#*=}"; shift ;;
    --retries)     RETRIES="${2:-}";  shift 2 ;;
    --workers=*)   MAX_WORKERS="${1#*=}"; shift ;;
    --workers)     MAX_WORKERS="${2:-}";  shift 2 ;;
    --mirror)      ENDPOINT="https://hf-mirror.com"; shift ;;
    --direct)      ENDPOINT="https://huggingface.co"; SOURCE="hf"; shift ;;
    --skip-hash)   SKIP_HASH=true; shift ;;
    --dry-run)     DRY_RUN=true; shift ;;
    -h|--help)     sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

case "$TARGET" in
  student|teacher|all) ;;
  *) echo "Invalid --target: $TARGET (expected student|teacher|all)"; exit 1 ;;
esac
case "$SOURCE" in
  modelscope|hf) ;;
  *) echo "Invalid --source: $SOURCE (expected modelscope|hf)"; exit 1 ;;
esac

# ---- locate tools -------------------------------------------
find_tool() {
  local name="$1"; shift
  local c
  for c in "$@"; do
    if [ -x "$c" ] || [ -f "$c" ]; then echo "$c"; return 0; fi
  done
  if command -v "$name" >/dev/null 2>&1; then command -v "$name"; return 0; fi
  return 1
}

CURL="$(find_tool curl /usr/bin/curl /bin/curl)" || {
  echo "curl not found."; exit 1; }
HF="$(find_tool hf "${REPO_DIR}/.venv/Scripts/hf.exe" "${REPO_DIR}/.venv/bin/hf" \
        "${REPO_DIR}/.venv/Scripts/huggingface-cli.exe" "${REPO_DIR}/.venv/bin/huggingface-cli")" || HF=""
PY="$(find_tool python "${REPO_DIR}/.venv/Scripts/python.exe" "${REPO_DIR}/.venv/bin/python")" || PY=""

if [ "$SOURCE" = "hf" ] && [ -z "$HF" ]; then
  echo "hf CLI not found. Activate the project venv or run: pip install -U huggingface_hub"
  exit 1
fi

# sha256 helper (sha256sum on Linux/git-bash, shasum on macOS)
if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  sha256_of() { echo ""; }
fi

# curl may be a *native* Windows binary (C:\Windows\System32\curl.exe) while this
# script runs under git-bash. git-bash paths like /d/Codings/... are NOT understood
# by native binaries: handing $P straight to curl fails with
#     curl: (23) Failed to open the file /d/Codings/.../x: No such file or directory
# (measured 2026-09-16 -- it only ever worked because the test used a relative
# --model-dir). So run curl from inside the target directory with a bare filename,
# which is portable across MSYS curl, native Windows curl and Linux curl.
curl_get() {
  local url="$1" fname="$2" dir="$3"
  ( cd "$dir" && "$CURL" -L -C - --retry 5 --retry-delay 5 --retry-all-errors \
        --connect-timeout 30 --progress-bar -o "$fname" "$url" )
}

# same reasoning as curl_get: hash from inside the directory with a bare filename.
sha256_at() {  # $1 = dir, $2 = filename
  ( cd "$1" && sha256_of "$2" )
}

# ---- manifest: repo|dir|bytes|other files (space separated)|weight:sha256 ...
PLAN=()
if [ "$TARGET" = "student" ] || [ "$TARGET" = "all" ]; then
  PLAN+=("Qwen/Qwen3-0.6B|Qwen3-0.6B|1503300328|.gitattributes LICENSE README.md config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json|model.safetensors:f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b")
fi
if [ "$TARGET" = "teacher" ] || [ "$TARGET" = "all" ]; then
  PLAN+=("Qwen/Qwen3-1.7B|Qwen3-1.7B|4063515592|.gitattributes LICENSE README.md config.json generation_config.json merges.txt model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json|model-00001-of-00002.safetensors:169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5 model-00002-of-00002.safetensors:912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9")
fi

mkdir -p "$MODEL_DIR"

# env for the hf fallback; Xet off because hf-mirror redirects Xet blobs to
# cas-bridge.xethub.hf.co, which we do not want to depend on.
export HF_ENDPOINT="$ENDPOINT"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=60

echo ""
echo "=========================================="
echo "GenRetrieval v2 - base model download"
echo "  source   : $SOURCE"
if [ "$SOURCE" = "hf" ]; then echo "  endpoint : $ENDPOINT"; fi
echo "  model dir: $MODEL_DIR"
echo "  target   : $TARGET"
echo "  retries  : $RETRIES"
echo "=========================================="

for entry in "${PLAN[@]}"; do
  IFS='|' read -r REPO DIR BYTES OTHERS WEIGHTS <<< "$entry"
  DEST="${MODEL_DIR}/${DIR}"
  GB=$(awk -v b="$BYTES" 'BEGIN{printf "%.2f", b/1e9}')

  echo ""
  echo "[${REPO}]  ->  ${DEST}   (${GB} GB, ${BYTES} B)"
  echo "------------------------------------------------------------"
  mkdir -p "$DEST"

  if $DRY_RUN; then
    for f in $OTHERS; do
      echo "  [dry-run] $f  <-  https://modelscope.cn/models/${REPO}/resolve/master/$f"
    done
    for w in $WEIGHTS; do
      echo "  [dry-run] ${w%%:*}  <-  https://modelscope.cn/models/${REPO}/resolve/master/${w%%:*}"
    done
    continue
  fi

  if [ "$SOURCE" = "hf" ]; then
    # outer retry loop: without it a single TLS blip aborts the whole snapshot.
    # The .cache/huggingface/download/*.incomplete file makes each retry resume.
    OK=false
    a=1
    while [ "$a" -le "$RETRIES" ]; do
      echo "  attempt ${a}/${RETRIES} ..."
      if "$HF" download "$REPO" --local-dir "$DEST" --max-workers "$MAX_WORKERS"; then
        OK=true; break
      fi
      echo "  attempt ${a} failed; partial files kept, next attempt resumes" >&2
      a=$((a + 1))
      if [ "$a" -le "$RETRIES" ]; then sleep $((3 * a)); fi
    done
    if ! $OK; then
      echo "download failed: ${REPO} after ${RETRIES} attempts." >&2
      exit 1
    fi
  else
    for f in $OTHERS; do
      P="${DEST}/${f}"
      if [ -s "$P" ]; then echo "  skip  $f"; continue; fi
      echo "  get   $f"
      curl_get "https://modelscope.cn/models/${REPO}/resolve/master/$f" "$f" "$DEST"
    done
    for w in $WEIGHTS; do
      F="${w%%:*}"; WANT="${w##*:}"
      P="${DEST}/${F}"
      if [ -f "$P" ] && [ "$(sha256_at "$DEST" "$F")" = "$WANT" ]; then
        echo "  skip  $F  (sha256 ok)"; continue
      fi
      if [ -f "$P" ]; then echo "  redo  $F  (sha256 mismatch, resuming)"; fi
      echo "  get   $F"
      curl_get "https://modelscope.cn/models/${REPO}/resolve/master/$F" "$F" "$DEST"
    done
  fi

  # ---- verify ----
  echo ""
  for w in $WEIGHTS; do
    F="${w%%:*}"; WANT="${w##*:}"
    P="${DEST}/${F}"
    if [ ! -f "$P" ]; then echo "ERROR: weights missing after download: $F" >&2; exit 1; fi
    ACTUAL_B=$(wc -c < "$P")
    if $SKIP_HASH; then
      echo "  OK  ${F}  (${ACTUAL_B} B, hash not checked)"
      continue
    fi
    GOT="$(sha256_at "$DEST" "$F")"
    if [ "$GOT" != "$WANT" ]; then
      echo "ERROR: sha256 mismatch for $F" >&2
      echo "  expected $WANT" >&2
      echo "  actual   $GOT" >&2
      echo "  The bytes on disk are not the bytes we pinned. Do NOT train on it." >&2
      exit 1
    fi
    echo "  OK  ${F}  (${ACTUAL_B} B, sha256 ok)"
  done
done

# ---- functional smoke test ----------------------------------
echo ""
echo "=========================================="
echo "Done."
for entry in "${PLAN[@]}"; do
  IFS='|' read -r REPO DIR BYTES OTHERS WEIGHTS <<< "$entry"
  echo "  ${REPO}  ->  ${MODEL_DIR}/${DIR}"
done
echo "=========================================="

if ! $DRY_RUN && [ -n "$PY" ]; then
  echo ""
  echo "Smoke test (config + tokenizer + shard index, no weights in VRAM):"
  for entry in "${PLAN[@]}"; do
    IFS='|' read -r REPO DIR BYTES OTHERS WEIGHTS <<< "$entry"
    # NOTE: run from INSIDE the directory. $PY may be a native Windows python.exe,
    # which does not understand git-bash paths like /d/Codings/... -- it treats them
    # as a repo id and transformers raises
    #   HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
    # Same root cause as curl_get() above; see EXPERIMENT_LOG E-24.
    ( cd "${MODEL_DIR}/${DIR}" && "$PY" - <<'PYEOF'
import json, os
from transformers import AutoConfig, AutoTokenizer
p = os.getcwd()
cfg = AutoConfig.from_pretrained(p)
tok = AutoTokenizer.from_pretrained(p)
print('  ' + os.path.basename(p) + ': vocab=' + str(cfg.vocab_size)
      + ' hidden=' + str(cfg.hidden_size) + ' layers=' + str(cfg.num_hidden_layers))
print('    eos=' + str(tok.eos_token_id)
      + '  <|im_end|>=' + str(tok.convert_tokens_to_ids('<|im_end|>')))
idx = os.path.join(p, 'model.safetensors.index.json')
if os.path.exists(idx):
    wm = json.load(open(idx))['weight_map']
    shards = sorted(set(wm.values()))
    missing = [s for s in shards if not os.path.exists(os.path.join(p, s))]
    print('    index shards=' + str(shards) + ' missing=' + str(missing))
PYEOF
    )
  done
fi
echo ""
