# ============================================================
# GenRetrieval - Environment setup for Windows (local dev)
# ------------------------------------------------------------
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
#   -Recreate   delete the existing .venv first (clean rebuild)
#   -SkipTorch  keep the torch already installed in .venv
#   -Mirror     override the PyPI mirror
#
# Why this script looks the way it does:
#   1. torch is installed from the LOCAL wheel in .wheels\ (no 2.6GB download)
#   2. every pip call explicitly passes -i <mirror>, because a stale
#      pip.ini under %APPDATA%\pip may point at a mirror that is
#      unreachable (e.g. the Tsinghua mirror returns 403 on some networks)
#   3. every external command is checked via $LASTEXITCODE - PowerShell's
#      $ErrorActionPreference does NOT catch non-zero exit codes of
#      native executables, so without this a failed pip run would still
#      print a success banner
#
# Cloud/Linux users: use scripts/setup_env.sh instead (2-4 min).
# ============================================================
param(
    [switch]$Recreate,
    [switch]$SkipTorch,
    [string]$Mirror = "https://mirrors.cloud.tencent.com/pypi/simple/"
)

$ErrorActionPreference = "Stop"

$Root   = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Venv   = Join-Path $Root ".venv"
$Py     = Join-Path $Venv "Scripts\python.exe"
$Wheels = Join-Path $Root ".wheels"

# Override any stale global pip config (env vars outrank pip.ini).
$env:PIP_INDEX_URL = $Mirror
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

Write-Host "==> Project root : $Root"
Write-Host "==> PyPI mirror  : $Mirror"

function Invoke-Pip {
    param(
        [string[]]$Arguments,
        [string]  $Step
    )
    & $Py -m pip @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "!! pip FAILED (exit $LASTEXITCODE) during: $Step" -ForegroundColor Red
        Write-Host "   Nothing further was attempted." -ForegroundColor Red
        exit 1
    }
}

# ---------- 0. Recreate ----------
if ($Recreate -and (Test-Path $Venv)) {
    Write-Host "==> [0/4] Removing existing .venv (may take a while) ..."
    Remove-Item -Recurse -Force $Venv
}

# ---------- 1. Virtual environment ----------
if (-not (Test-Path $Py)) {
    Write-Host "==> [1/4] Creating .venv ..."
    $Base = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $Base) { throw "python not found on PATH" }
    & $Base -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Write-Host "!! venv creation failed" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "==> [1/4] Reusing existing .venv"
}
Write-Host "         python $(& $Py -V 2>&1)"

# Pin numpy while installing torch: torchvision only declares an unversioned
# "numpy" dependency, so pip would pull numpy 2.x, and the later
# requirements-core.txt step would then have to UNINSTALL and downgrade it.
# On Windows that uninstall step is exactly what we want to avoid.
$NumpyPin = "numpy==1.26.3"

# ---------- 2. PyTorch (local wheel + mirror for its deps) ----------
if ($SkipTorch) {
    Write-Host "==> [2/4] Skipping torch (-SkipTorch)"
} else {
    $TorchWhl = Get-ChildItem "$Wheels\torch-2.6.0+cu118-*.whl"       -ErrorAction SilentlyContinue | Select-Object -First 1
    $TvWhl    = Get-ChildItem "$Wheels\torchvision-0.21.0+cu118-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1

    if ((-not $TorchWhl) -or (-not $TvWhl)) {
        Write-Host "==> [2/4] Local wheels missing - downloading from pytorch.org (~2.5GB, slow in CN)"
        Invoke-Pip -Step "torch (remote)" -Arguments @(
            "install", "torch==2.6.0", "torchvision==0.21.0", $NumpyPin,
            "--index-url", "https://download.pytorch.org/whl/cu118",
            "--extra-index-url", $Mirror,
            "--no-warn-script-location"
        )
    } else {
        Write-Host "==> [2/4] Installing torch from local wheels"
        Write-Host "         $($TorchWhl.Name) / $($TvWhl.Name)"
        Invoke-Pip -Step "torch (local wheel)" -Arguments @(
            "install", $TorchWhl.FullName, $TvWhl.FullName, $NumpyPin,
            "-i", $Mirror,
            "--no-warn-script-location"
        )
    }
}

# ---------- 3. Core dependencies ----------
Write-Host "==> [3/4] Installing core deps (requirements-core.txt)"
Invoke-Pip -Step "requirements-core.txt" -Arguments @(
    "install", "-r", (Join-Path $Root "requirements-core.txt"),
    "-i", $Mirror,
    "--no-warn-script-location"
)

# ---------- 4. Self-check ----------
Write-Host "==> [4/4] Environment self-check"
$Check = @'
import sys
fails = []
for name in ["torch","torchvision","numpy","pandas","scipy","pyarrow",
             "transformers","trl","peft","accelerate","datasets",
             "faiss","ot","bitsandbytes","einops","sklearn"]:
    try:
        mod = __import__(name)
        print(f"  [OK]   {name:<16} {getattr(mod,'__version__','?')}")
    except Exception as e:
        fails.append(name)
        print(f"  [FAIL] {name:<16} {type(e).__name__}: {e}")
import torch
print(f"\n  torch          : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
if torch.__version__.endswith("+cpu"):
    fails.append("torch is the CPU build")
if fails:
    print(f"\n!! FAILED: {fails}")
    sys.exit(1)
print("\n==> Environment ready.")
'@
$Check | & $Py -
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "!! Environment self-check FAILED - see the list above." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Done. Activate:  .\.venv\Scripts\Activate.ps1"
Write-Host " Next: docs/QUICKSTART.md"
Write-Host "============================================================" -ForegroundColor Green
