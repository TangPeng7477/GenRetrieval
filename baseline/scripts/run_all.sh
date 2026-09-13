#!/usr/bin/env bash
# 一键复现全部 baseline（双域）。用法：
#   bash baseline/scripts/run_all.sh            # 全跑（数小时）
#   bash baseline/scripts/run_all.sh quick      # 冒烟（1 轮 + 2 万样本，分钟级）
#   bash baseline/scripts/run_all.sh zero       # 只跑零训练组（约 2 分钟）
#
# 注意：在项目根目录执行；Windows 上用 ./.venv/Scripts/python.exe，
#       脚本会自动探测 python 路径。
set -euo pipefail

cd "$(dirname "$0")/../.."   # 回到项目根

if [ -x "./.venv/Scripts/python.exe" ]; then
  PY="./.venv/Scripts/python.exe"          # Windows
elif [ -x "./.venv/bin/python" ]; then
  PY="./.venv/bin/python"                  # Linux / 云
else
  PY="python"
fi
echo "[run_all] python = $PY"

MODE="${1:-full}"

case "$MODE" in
  quick)
    "$PY" -m baseline.run --model pop,itemknn,content_ann,sid_prefix --domain all --quick
    "$PY" -m baseline.run --model sid_gr,sasrec,bprmf,gru4rec --domain all --beam 20 --quick
    ;;
  zero)
    "$PY" -m baseline.run --model pop,itemknn,content_ann,sid_prefix --domain all
    ;;
  full)
    # 先零训练组拿标尺（分钟级），再上可训练组（小时级）
    "$PY" -m baseline.run --model pop,itemknn,content_ann,sid_prefix --domain all
    "$PY" -m baseline.run --model sid_gr,sasrec,bprmf,gru4rec --domain all --beam 20
    ;;
  *)
    echo "未知模式：$MODE（可选 full / quick / zero）" >&2
    exit 1
    ;;
esac

echo "[run_all] 汇总结果表"
"$PY" -m baseline.scripts.summarize --out baseline
echo "[run_all] 完成，见 baseline/RESULTS.md"
