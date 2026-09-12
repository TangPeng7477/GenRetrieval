#!/usr/bin/env bash
# ============================================================
# GenRetrieval — 环境一键安装（Linux / 云平台 / WSL）
# ------------------------------------------------------------
# 用法:
#   bash scripts/setup_env.sh              # 默认 CUDA 11.8
#   CUDA=cu121 bash scripts/setup_env.sh   # 指定 CUDA 版本
#   NO_VENV=1 bash scripts/setup_env.sh    # 用当前/系统 python，不建 venv
#   SKIP_TORCH=1 bash scripts/setup_env.sh # 跳过 torch（平台已预装）
#   PIP_MIRROR=<url> bash scripts/setup_env.sh
#
# 实测: RTX 3090 / Ubuntu 22.04 / Python 3.11 → 约 2-4 分钟
# ============================================================
set -euo pipefail

CUDA="${CUDA:-cu118}"
TORCH_VER="${TORCH_VER:-2.6.0}"
TV_VER="${TV_VER:-0.21.0}"
PIP_MIRROR="${PIP_MIRROR:-https://mirrors.cloud.tencent.com/pypi/simple/}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/${CUDA}}"

# Override any stale global pip config (a pip.ini pointing at an unreachable
# mirror is a common cause of "No matching distribution found for ...").
export PIP_INDEX_URL="$PIP_MIRROR"
export PIP_DISABLE_PIP_VERSION_CHECK=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "==> 项目根目录: $ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "!! 找不到 $PYTHON_BIN"; exit 1; }
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] >= (3,10), "需要 Python >= 3.10"' \
  || { echo "!! Python 版本过低"; exit 1; }
echo "==> Python: $("$PYTHON_BIN" -V)"

# ---------- 1. 虚拟环境 ----------
if [ "${NO_VENV:-0}" = "1" ]; then
  PIP="$PYTHON_BIN -m pip"
  echo "==> [1/4] 跳过 venv（NO_VENV=1）"
else
  if [ ! -d .venv ]; then
    echo "==> [1/4] 创建 .venv"
    "$PYTHON_BIN" -m venv .venv
  else
    echo "==> [1/4] 复用已存在的 .venv"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PIP="python -m pip"
  python -m pip install -U pip setuptools wheel -i "$PIP_MIRROR" -q
fi

# ---------- 2. PyTorch ----------
if [ "${SKIP_TORCH:-0}" = "1" ]; then
  echo "==> [2/4] 跳过 torch（SKIP_TORCH=1）"
else
  echo "==> [2/4] 安装 PyTorch ${TORCH_VER}+${CUDA}（约 2.5GB）"
  # torch/torchvision come from the pytorch index; their pure-python deps
  # (filelock, sympy, networkx, ...) are NOT hosted there, so the mirror
  # must be passed as an extra index or the install fails with
  # "No matching distribution found for filelock".
  # numpy is pinned here because torchvision declares it without a version
  # bound and pip would otherwise pick numpy 2.x, forcing a downgrade later.
  $PIP install "torch==${TORCH_VER}" "torchvision==${TV_VER}" "numpy==1.26.3" \
      --index-url "$TORCH_INDEX" \
      --extra-index-url "$PIP_MIRROR"
fi

# ---------- 3. 核心依赖 ----------
echo "==> [3/4] 安装核心依赖 (requirements-core.txt)"
$PIP install -r requirements-core.txt -i "$PIP_MIRROR"

# ---------- 4. 自检 ----------
echo "==> [4/4] 环境自检"
python - <<'PYEOF'
import importlib.metadata as md
import sys

fails = []
for name in ["torch", "torchvision", "numpy", "pandas", "scipy", "pyarrow",
             "transformers", "trl", "peft", "accelerate", "datasets",
             "faiss", "ot", "bitsandbytes", "einops", "sklearn"]:
    try:
        mod = __import__(name)
        print(f"  [OK]   {name:<16} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        fails.append(name)
        print(f"  [FAIL] {name:<16} {type(e).__name__}: {e}")

import torch
print(f"\n  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  Compute cap.   : {torch.cuda.get_device_capability(0)}")

if fails:
    print(f"\n!! 以下模块导入失败: {fails}")
    sys.exit(1)
print("\n==> 环境就绪 ✔")
PYEOF

echo ""
echo "============================================================"
echo " 完成。激活环境:  source .venv/bin/activate"
echo " 下一步: 见 docs/QUICKSTART.md"
echo "============================================================"
