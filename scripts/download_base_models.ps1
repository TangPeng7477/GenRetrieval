<#
.SYNOPSIS
    Download base models for GenRetrieval v2 (M3 SFT stage).

.DESCRIPTION
    Base model decision is recorded in docs/UPGRADE_PLAN.md 5.1.1:
      student : Qwen/Qwen3-0.6B   (post-trained)  ~1.50 GB  single-file safetensors
      teacher : Qwen/Qwen3-1.7B   (post-trained)  ~4.06 GB  2 shards
    Qwen3-0.6B-Base is kept as a probe-only control and is NOT downloaded here.

    IMPORTANT - run this OUTSIDE the agent sandbox. The sandbox is throttled to
    roughly 118 kB/s, which turns a 1.5 GB fetch into a 3+ hour job.

    The script is incremental: files already present under the target directory
    are verified and skipped, so re-running it is cheap.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\download_base_models.ps1
    powershell -ExecutionPolicy Bypass -File scripts\download_base_models.ps1 -Target all
    powershell -ExecutionPolicy Bypass -File scripts\download_base_models.ps1 -Direct
#>
[CmdletBinding()]
param(
    [ValidateSet("student", "teacher", "all")]
    [string]$Target = "student",

    # Defaults to <repo>/models
    [string]$ModelDir = "",

    # hf-mirror.com is reachable from CN networks; huggingface.co often is not.
    [string]$Endpoint = "https://hf-mirror.com",

    # Use the official endpoint instead of the mirror.
    [switch]$Direct,

    # Print the plan without downloading anything.
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrEmpty($ModelDir)) {
    $ModelDir = Join-Path $RepoRoot "models"
}
if ($Direct) {
    $Endpoint = "https://huggingface.co"
}

# ---- locate the hf CLI (prefer the project venv) -------------------------
$candidates = @(
    (Join-Path $RepoRoot ".venv\Scripts\hf.exe"),
    (Join-Path $RepoRoot ".venv\Scripts\huggingface-cli.exe")
)
$Hf = $null
foreach ($c in $candidates) {
    if (Test-Path $c) { $Hf = $c; break }
}
if ($null -eq $Hf) {
    $onPath = Get-Command hf -ErrorAction SilentlyContinue
    if ($null -eq $onPath) { $onPath = Get-Command huggingface-cli -ErrorAction SilentlyContinue }
    if ($null -ne $onPath) { $Hf = $onPath.Source }
}
if ($null -eq $Hf) {
    throw "hf CLI not found. Activate the project venv or run: pip install -U huggingface_hub"
}

# ---- planned downloads ---------------------------------------------------
# Sizes measured from the official ModelScope repo file listing (2026-09-16).
$models = @()
if ($Target -eq "student" -or $Target -eq "all") {
    $models += @{ Repo = "Qwen/Qwen3-0.6B"; Dir = "Qwen3-0.6B"; Size = "1.50 GB"; Weights = @("model.safetensors") }
}
if ($Target -eq "teacher" -or $Target -eq "all") {
    $models += @{ Repo = "Qwen/Qwen3-1.7B"; Dir = "Qwen3-1.7B"; Size = "4.06 GB";
                  Weights = @("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors") }
}

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
$env:HF_ENDPOINT = $Endpoint

Write-Host ""
Write-Host "==========================================" -ForegroundColor DarkGray
Write-Host "GenRetrieval v2 - base model download"
Write-Host "  endpoint : $Endpoint"
Write-Host "  model dir: $ModelDir"
Write-Host "  target   : $Target"
Write-Host "==========================================" -ForegroundColor DarkGray

foreach ($m in $models) {
    $dest = Join-Path $ModelDir $m.Dir

    Write-Host ""
    Write-Host "[$($m.Repo)]  ->  $dest   ($($m.Size))" -ForegroundColor Cyan
    Write-Host ("-" * 60) -ForegroundColor DarkGray

    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    if ($DryRun) {
        Write-Host "  [dry-run] HF_ENDPOINT=$Endpoint" -ForegroundColor DarkYellow
        Write-Host "  [dry-run] would run: `"$Hf`" download $($m.Repo) --local-dir `"$dest`"" -ForegroundColor DarkYellow
        continue
    }
    & $Hf download $m.Repo --local-dir $dest
    if ($LASTEXITCODE -ne 0) {
        throw "download failed: $($m.Repo) (exit code $LASTEXITCODE)"
    }

    # ---- verify the weight files actually landed ----
    $missing = @()
    foreach ($w in $m.Weights) {
        $p = Join-Path $dest $w
        if (-not (Test-Path $p)) { $missing += $w }
    }
    if ($missing.Count -gt 0) {
        throw "weights missing after download: $($missing -join ', ')"
    }
    foreach ($w in $m.Weights) {
        $p = Join-Path $dest $w
        $mb = [math]::Round((Get-Item $p).Length / 1MB, 1)
        Write-Host ("  OK  {0}  ({1} MB)" -f $w, $mb) -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor DarkGray
Write-Host "Done." -ForegroundColor Green
foreach ($m in $models) {
    Write-Host ("  {0}  ->  {1}" -f $m.Repo, (Join-Path $ModelDir $m.Dir))
}
Write-Host "==========================================" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Smoke check (loads config + tokenizer, no weights in VRAM):" -ForegroundColor Yellow
Write-Host "  .venv\Scripts\python.exe -c \`"from transformers import AutoTokenizer, AutoConfig; p=r'$(Join-Path $ModelDir 'Qwen3-0.6B')'; print(AutoConfig.from_pretrained(p).vocab_size); print(AutoTokenizer.from_pretrained(p).encode('hello'))\`""
