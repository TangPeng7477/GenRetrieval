#!/usr/bin/env bash
# ============================================================================
# M1-1: 下载 Amazon Reviews 2023 (Industrial_and_Scientific) 数据
# ----------------------------------------------------------------------------
# 说明：
#   * 官方 huggingface.co 在国内不可达（实测超时），统一走 hf-mirror.com
#   * 只下两样东西：
#       1) benchmark/5core/timestamp/{train,valid,test}.csv  —— 官方 5-core 切分（~24MB）
#       2) raw/meta_categories/meta_<Category>.jsonl          —— metadata，图像 URL 唯一来源（~1.05GB）
#     不下 raw/review_categories/<Category>.jsonl（2.2GB 全量评论），因为官方 5core
#     切分已经等价于 k-core 结果，无需自己再跑 k-core。
#   * 支持断点续传（curl -C -），重复执行不会重下。
#
# 用法：
#   bash scripts/data/download_amazon23.sh                 # 默认 Industrial_and_Scientific
#   CATEGORY=Video_Games bash scripts/data/download_amazon23.sh
# ============================================================================
set -euo pipefail

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RAW="${ROOT}/data/Amazon23/raw"
mkdir -p "${RAW}"

echo "=============================================="
echo " category : ${CATEGORY}"
echo " mirror   : ${MIRROR}"
echo " target   : ${RAW}"
echo "=============================================="

# ---------- 1) 官方 5core 切分（小文件，直接下） ----------
for split in train valid test; do
  OUT="${RAW}/${CATEGORY}_5core_timestamp.${split}.csv"
  if [ -s "${OUT}" ]; then
    echo "[skip] ${split} 已存在 ($(wc -c < "${OUT}") bytes)"
    continue
  fi
  echo "[get ] 5core/${split}"
  curl -L --fail --retry 5 --retry-delay 3 -C - \
    "${MIRROR}/${REPO}/benchmark/5core/timestamp/${CATEGORY}.${split}.csv" \
    -o "${OUT}"
done

# ---------- 2) metadata（大文件，断点续传） ----------
META="${RAW}/meta_${CATEGORY}.jsonl"
EXPECT_BYTES="${EXPECT_BYTES:-1130006344}"   # I&S 实测 1.13GB（HF tree API 报的是 1077.7MB，以实测为准）
if [ -s "${META}" ]; then
  CUR=$(wc -c < "${META}")
  echo "[info] metadata 当前 ${CUR} bytes (期望约 ${EXPECT_BYTES})"
  if [ "${CUR}" -ge "${EXPECT_BYTES}" ]; then
    echo "[skip] metadata 已完成"
  else
    echo "[get ] metadata 续传中 ..."
    curl -L --fail --retry 5 --retry-delay 3 -C - \
      "${MIRROR}/${REPO}/raw/meta_categories/meta_${CATEGORY}.jsonl" \
      -o "${META}"
  fi
else
  echo "[get ] metadata 下载中（约 1.05GB，断点续传已开启）..."
  curl -L --fail --retry 5 --retry-delay 3 -C - \
    "${MIRROR}/${REPO}/raw/meta_categories/meta_${CATEGORY}.jsonl" \
    -o "${META}"
fi

echo "=============================================="
ls -la "${RAW}"
echo "done."
