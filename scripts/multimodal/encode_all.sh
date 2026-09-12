#!/usr/bin/env bash
# =============================================================================
# 多模态编码总入口（串行执行）
# =============================================================================
# 为什么必须串行：
#   本机 RTX 3050 Ti 只有 4GB 显存。Qwen3-Embedding-0.6B(1.19GB) 与
#   SigLIP(0.42GB) 并行时，两个进程各有独立 CUDA context，加上激活与
#   caching allocator 的碎片，总需求超过 4GB → 显存疯狂换页(thrashing)：
#   GPU 利用率显示 100%，但进度 4 分钟不动。
#   实测：并行时文本编码 2 item/s；串行后可正常发挥。
#
# 断点续跑：已存在的产物自动跳过（FORCE=1 强制重跑）
#
# 用法:
#   bash scripts/multimodal/encode_all.sh                 # 两域，补齐缺失
#   SHORTS=IandS bash scripts/multimodal/encode_all.sh    # 只跑一个域
#   FORCE=1 bash scripts/multimodal/encode_all.sh         # 全部重跑
# =============================================================================
set -euo pipefail

SHORTS="${SHORTS:-IandS VG}"
BATCH_IMG="${BATCH_IMG:-128}"
BATCH_TXT="${BATCH_TXT:-64}"
FORCE="${FORCE:-0}"
FIELDS="${FIELDS:-title+features+category}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 缓解 4GB 卡的显存碎片（不减少峰值占用，但显著降低 OOM 概率）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
[ -x "$PY" ] || { echo "找不到 python: $PY"; exit 1; }

echo "=============================================="
echo " 多模态编码（串行）  SHORTS=$SHORTS"
echo " HF_ENDPOINT=$HF_ENDPOINT"
echo "=============================================="

for S in $SHORTS; do
    D="data/Amazon23/$S/emb"
    mkdir -p "$D"

    # ---------- 图像 ----------
    if [ "$FORCE" = "1" ] || [ ! -f "$D/emb_image_siglip.npy" ]; then
        echo ""
        echo "########## [$S] 图像编码 ##########"
        "$PY" scripts/multimodal/encode_image.py --short "$S" --batch_size "$BATCH_IMG"
    else
        echo ""
        echo "########## [$S] 图像编码 已存在，跳过 ##########"
    fi

    # ---------- 文本 ----------
    if [ "$FORCE" = "1" ] || ! ls "$D"/emb_text_*.npy >/dev/null 2>&1; then
        echo ""
        echo "########## [$S] 文本编码 ##########"
        "$PY" scripts/multimodal/encode_text.py --short "$S" \
            --fields "$FIELDS" --batch_size "$BATCH_TXT" --token_budget "${TOK_BUDGET:-4096}"
    else
        echo ""
        echo "########## [$S] 文本编码 已存在，跳过 ##########"
    fi
done

echo ""
echo "=============================================="
echo " 全部编码完成，产物："
for S in $SHORTS; do
    ls -lh "data/Amazon23/$S/emb/" 2>/dev/null | tail -n +2
done
echo "=============================================="
