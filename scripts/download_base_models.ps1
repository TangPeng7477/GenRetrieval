<#
.SYNOPSIS
    Download base models for GenRetrieval v2 (M3 SFT stage).

.DESCRIPTION
    Base model decision is recorded in docs/UPGRADE_PLAN.md 5.1.1:
      student : Qwen/Qwen3-0.6B   (post-trained)  1,503,300,328 B  single-file safetensors
      teacher : Qwen/Qwen3-1.7B   (post-trained)  4,063,515,592 B  2 shards
    Qwen3-0.6B-Base is a probe-only control and is NOT downloaded here.

    WHY THE DEFAULT PATH IS NO LONGER huggingface_hub
    -------------------------------------------------
    Measured 2026-09-16 from this network, hf-mirror.com is flaky at the TLS
    layer. Both /resolve/... and /api/... intermittently die with:
        SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING]'))
    (observed 4 of 5 attempts on the model.safetensors resolve URL), and a HEAD
    sometimes comes back without the `X-Repo-Commit` header, which makes
    huggingface_hub raise

        FileMetadataError  "Distant resource does not seem to be on huggingface.co"
        -> LocalEntryNotFoundError

    (file_download.py:1568 and :1661). file_download.py re-raises a raw SSLError
    straight out of the worker thread (:1600-1602), so ONE blip on ONE of the
    parallel workers aborts the entire snapshot -- that is exactly what killed
    the 8/10 run. Blaming the mirror's headers, not the code.

    ModelScope serves the same official Qwen repos from
    cdn-lfs-cn-1.modelscope.cn (domestic CDN, Range/206 supported), byte-identical
    to the HF copy. Cross-checked 2026-09-16 -- these three agree:
        hf-mirror  X-Linked-Etag
        ModelScope X-Linked-Etag
        ModelScope LFS object path  lfs-objects/f4/7f/7117...  (content-addressed)
    NOTE: huggingface.co itself answered 502 from this network, so the official
    endpoint could NOT be used as a third source. If you have an independent
    channel, re-verify the three hashes in $Weight maps below.

    Default source is therefore plain curl.exe against ModelScope: no API
    metadata validation, resumable, no huggingface_hub in the loop.
    Use -Source hf to get the old huggingface_hub behaviour (now with retries).

    IMPORTANT - run this OUTSIDE the agent sandbox. The sandbox is throttled,
    which turns a 1.5 GB fetch into a multi-hour job.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\download_base_models.ps1
    powershell -ExecutionPolicy Bypass -File scripts\download_base_models.ps1 -Target all
    powershell -ExecutionPolicy Bypass -File scripts\download_base_models.ps1 -Source hf
    powershell -ExecutionPolicy Bypass -File scripts\download_base_models.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [ValidateSet("student", "teacher", "all")]
    [string]$Target = "student",

    # Defaults to <repo>/models
    [string]$ModelDir = "",

    # modelscope : curl.exe against cdn-lfs-cn-1.modelscope.cn (default, resumable)
    # hf         : huggingface_hub CLI against $Endpoint (kept as a fallback)
    [ValidateSet("modelscope", "hf")]
    [string]$Source = "modelscope",

    [string]$Endpoint = "https://hf-mirror.com",

    # Use the official endpoint instead of the mirror (implies -Source hf).
    [switch]$Direct,

    # -Source hf only: concurrent download workers. The flakiness scales with
    # concurrency, so keep this low.
    [int]$MaxWorkers = 2,

    # Outer retry attempts per model.
    [int]$Retries = 6,

    # Skip the sha256 check (only do the cheap size/existence check).
    [switch]$SkipHash,

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
    $Source = "hf"
}

# ---- tool discovery ------------------------------------------------------
function Find-Tool {
    param([string[]]$Candidates, [string]$Name)
    foreach ($c in $Candidates) {
        if (Test-Path $c) { return $c }
    }
    $onPath = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -ne $onPath) { return $onPath.Source }
    return $null
}

$Curl = Find-Tool -Candidates @((Join-Path $env:SystemRoot "System32\curl.exe")) -Name "curl.exe"
$Hf = Find-Tool -Candidates @(
    (Join-Path $RepoRoot ".venv\Scripts\hf.exe"),
    (Join-Path $RepoRoot ".venv\Scripts\huggingface-cli.exe")
) -Name "hf"
$Py = Find-Tool -Candidates @((Join-Path $RepoRoot ".venv\Scripts\python.exe")) -Name "python"

if ($Source -eq "modelscope" -and $null -eq $Curl) {
    throw "curl.exe not found. Windows 10 1803+ ships it at %SystemRoot%\System32\curl.exe; otherwise re-run with -Source hf"
}
if ($Source -eq "hf" -and $null -eq $Hf) {
    throw "hf CLI not found. Activate the project venv or run: pip install -U huggingface_hub"
}

# ---- model manifest ------------------------------------------------------
# Weight hashes are sha256 of the file content (== the LFS object hash).
# Other/ = everything that is not a weight file; these are small, so they get a
# cheap existence check and are covered by the functional smoke test at the end.
$student = @{
    Repo   = "Qwen/Qwen3-0.6B"
    Dir    = "Qwen3-0.6B"
    Bytes  = 1503300328
    Weight = [ordered]@{
        "model.safetensors" = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
    }
    Other  = @(
        ".gitattributes", "LICENSE", "README.md", "config.json", "generation_config.json",
        "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"
    )
}
$teacher = @{
    Repo   = "Qwen/Qwen3-1.7B"
    Dir    = "Qwen3-1.7B"
    Bytes  = 4063515592
    Weight = [ordered]@{
        "model-00001-of-00002.safetensors" = "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5"
        "model-00002-of-00002.safetensors" = "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"
    }
    Other  = @(
        ".gitattributes", "LICENSE", "README.md", "config.json", "generation_config.json",
        "merges.txt", "model.safetensors.index.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json"
    )
}

$models = @()
if ($Target -eq "student" -or $Target -eq "all") { $models += $student }
if ($Target -eq "teacher" -or $Target -eq "all") { $models += $teacher }

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

# Env for the hf fallback. Xet is disabled because hf-mirror redirects Xet-backed
# blobs to cas-bridge.xethub.hf.co, which we do not want to depend on.
$env:HF_ENDPOINT = $Endpoint
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_ETAG_TIMEOUT = "30"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "60"

Write-Host ""
Write-Host "==========================================" -ForegroundColor DarkGray
Write-Host "GenRetrieval v2 - base model download"
Write-Host "  source   : $Source"
if ($Source -eq "hf") { Write-Host "  endpoint : $Endpoint" }
Write-Host "  model dir: $ModelDir"
Write-Host "  target   : $Target"
Write-Host "  retries  : $Retries"
Write-Host "==========================================" -ForegroundColor DarkGray

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLower()
}

function Get-ModelScopeUrl {
    param([string]$Repo, [string]$File)
    return "https://modelscope.cn/models/$Repo/resolve/master/$File"
}

# ---- download one file via curl against ModelScope ------------------------
function Invoke-CurlFile {
    param([string]$Url, [string]$Dest, [int]$Attempts)
    for ($a = 1; $a -le $Attempts; $a++) {
        # -C - resumes a partial file; --retry* absorbs transient TLS/dns blips.
        & $Curl -L -C - --retry 5 --retry-delay 5 --retry-all-errors `
                --connect-timeout 30 --progress-bar -o $Dest $Url
        if ($LASTEXITCODE -eq 0) { return $true }
        Write-Host ("  curl attempt {0}/{1} failed (exit {2})" -f $a, $Attempts, $LASTEXITCODE) -ForegroundColor Yellow
        if ($a -lt $Attempts) { Start-Sleep -Seconds (3 * $a) }
    }
    return $false
}

foreach ($m in $models) {
    $dest = Join-Path $ModelDir $m.Dir
    # decimal GB (1e9), so this matches the 1.50 GB / 4.06 GB figures in UPGRADE_PLAN 5.1.1
    $sizeText = "{0:N2} GB" -f ($m.Bytes / 1e9)

    Write-Host ""
    Write-Host "[$($m.Repo)]  ->  $dest   ($sizeText)" -ForegroundColor Cyan
    Write-Host ("-" * 60) -ForegroundColor DarkGray
    New-Item -ItemType Directory -Force -Path $dest | Out-Null

    if ($DryRun) {
        if ($Source -eq "hf") {
            Write-Host "  [dry-run] would run: `"$Hf`" download $($m.Repo) --local-dir `"$dest`" --max-workers $MaxWorkers" -ForegroundColor DarkYellow
        } else {
            foreach ($f in ($m.Other + @($m.Weight.Keys))) {
                Write-Host ("  [dry-run] {0}  <-  {1}" -f $f, (Get-ModelScopeUrl -Repo $m.Repo -File $f)) -ForegroundColor DarkYellow
            }
        }
        continue
    }

    $t0 = Get-Date

    if ($Source -eq "hf") {
        # ---- huggingface_hub path, with an outer retry loop ------------------
        # Without this loop a single TLS blip aborts the whole snapshot; the
        # `.cache/huggingface/download/*.incomplete` file makes each retry resume.
        $ok = $false
        for ($a = 1; $a -le $Retries; $a++) {
            Write-Host ("  attempt {0}/{1} ..." -f $a, $Retries) -ForegroundColor DarkGray
            & $Hf download $m.Repo --local-dir $dest --max-workers $MaxWorkers
            if ($LASTEXITCODE -eq 0) { $ok = $true; break }
            Write-Host ("  attempt {0} failed (exit {1}); partial files are kept, next attempt resumes" -f $a, $LASTEXITCODE) -ForegroundColor Yellow
            if ($a -lt $Retries) { Start-Sleep -Seconds (5 * $a) }
        }
        if (-not $ok) {
            throw "download failed: $($m.Repo) after $Retries attempts. If it always dies on the same file, retry with -Source modelscope."
        }
    }
    else {
        # ---- ModelScope + curl path -----------------------------------------
        $todo = @()
        foreach ($f in $m.Other) { $todo += @{ File = $f; Hash = $null } }
        foreach ($f in $m.Weight.Keys) { $todo += @{ File = $f; Hash = $m.Weight[$f] } }

        foreach ($item in $todo) {
            $f = $item.File
            $p = Join-Path $dest $f
            $url = Get-ModelScopeUrl -Repo $m.Repo -File $f

            if (Test-Path $p) {
                if ($null -ne $item.Hash) {
                    if ((Get-Sha256 $p) -eq $item.Hash) {
                        Write-Host ("  skip  {0}  (sha256 ok)" -f $f) -ForegroundColor DarkGray
                        continue
                    }
                    Write-Host ("  redo  {0}  (sha256 mismatch, resuming)" -f $f) -ForegroundColor Yellow
                }
                else {
                    if ((Get-Item $p).Length -gt 0) {
                        Write-Host ("  skip  {0}" -f $f) -ForegroundColor DarkGray
                        continue
                    }
                }
            }

            Write-Host ("  get   {0}" -f $f) -ForegroundColor Gray
            if (-not (Invoke-CurlFile -Url $url -Dest $p -Attempts $Retries)) {
                throw "download failed: $($m.Repo)/$f"
            }
        }
    }

    # ---- verify ------------------------------------------------------------
    Write-Host ""
    foreach ($f in $m.Weight.Keys) {
        $p = Join-Path $dest $f
        if (-not (Test-Path $p)) { throw "weights missing after download: $f" }
        $len = (Get-Item $p).Length
        if ($SkipHash) {
            Write-Host ("  OK  {0}  ({1:N0} B, hash not checked)" -f $f, $len) -ForegroundColor Green
            continue
        }
        $h = Get-Sha256 $p
        if ($h -ne $m.Weight[$f]) {
            throw ("sha256 mismatch for {0}`n  expected {1}`n  actual   {2}`n  `nThis means the bytes on disk are not the bytes we pinned. Do NOT train on it.`nIf Qwen re-tagged the repo, re-run with -SkipHash only after re-verifying upstream." -f $f, $m.Weight[$f], $h)
        }
        Write-Host ("  OK  {0}  ({1:N0} B = {2:N2} GB, sha256 ok)" -f $f, $len, ($len / 1e9)) -ForegroundColor Green
    }

    $dt = (Get-Date) - $t0
    Write-Host ("  elapsed {0:hh\:mm\:ss}" -f $dt) -ForegroundColor DarkGray
}

# ---- functional smoke test ----------------------------------------------
Write-Host ""
Write-Host "==========================================" -ForegroundColor DarkGray
Write-Host "Done."
foreach ($m in $models) {
    Write-Host ("  {0}  ->  {1}" -f $m.Repo, (Join-Path $ModelDir $m.Dir))
}
Write-Host "==========================================" -ForegroundColor DarkGray

if (-not $DryRun -and $null -ne $Py) {
    Write-Host ""
    Write-Host "Smoke test (config + tokenizer + shard index, no weights in VRAM):" -ForegroundColor Yellow
    foreach ($m in $models) {
        $dest = Join-Path $ModelDir $m.Dir
        $code = @"
import json, os
from transformers import AutoConfig, AutoTokenizer
p = r'$dest'
cfg = AutoConfig.from_pretrained(p)
tok = AutoTokenizer.from_pretrained(p)
print('  ' + os.path.basename(p) + ': vocab=' + str(cfg.vocab_size) + ' hidden=' + str(cfg.hidden_size) + ' layers=' + str(cfg.num_hidden_layers))
print('    eos=' + str(tok.eos_token_id) + '  <|im_end|>=' + str(tok.convert_tokens_to_ids('<|im_end|>')))
idx = os.path.join(p, 'model.safetensors.index.json')
if os.path.exists(idx):
    wm = json.load(open(idx))['weight_map']
    shards = sorted(set(wm.values()))
    missing = [s for s in shards if not os.path.exists(os.path.join(p, s))]
    print('    index shards=' + str(shards) + ' missing=' + str(missing))
"@
        & $Py -c $code
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  smoke test FAILED for $dest" -ForegroundColor Red
        }
    }
}
Write-Host ""
