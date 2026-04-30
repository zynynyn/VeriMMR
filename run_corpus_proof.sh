#!/usr/bin/env bash
# 语料库完整证明：ZAC 串联 + zkLLM 全量证明（双卡并行）
#
# 用法：
#   bash run_corpus_proof.sh              # 前台阻塞，直到全部完成
#   nohup bash run_corpus_proof.sh \
#       > logs/run_corpus_proof.log 2>&1 & # 后台运行（推荐）
#
# 耗时预估：
#   ZAC 串联     ~5 min
#   zkLLM 全量   ~56 h（双卡并行，~152 图/GPU × ~22 min/图）

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CORPUS_JSONL="corpora/image.jsonl"
EMBEDDING_NPY="embedding/embedding.npy"
ZAC_OUTPUT="output/phase1/fingerprint.json"
WORKDIR="zkllm-workdir/jina-v4"
LOG_DIR="logs"

mkdir -p "$LOG_DIR" "$(dirname "$ZAC_OUTPUT")"

echo "========================================================"
echo "  语料库完整证明流水线"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ── Step 1: ZAC 串联重跑（n=2 filters）────────────────────────────────────────
echo ""
echo "[Step 1] ZAC 串联证明（n=2 filters）"
echo "  输出: $ZAC_OUTPUT"
echo "  日志: $LOG_DIR/zac_cascade.log"

python script/phase1_corpus_fingerprint.py \
    --zac-only \
    --corpus-jsonl  "$CORPUS_JSONL" \
    --embedding-npy "$EMBEDDING_NPY" \
    --output        "$ZAC_OUTPUT" \
    --n-filters     2 \
    2>&1 | tee "$LOG_DIR/zac_cascade.log"

if [ $? -ne 0 ]; then
    echo "[Step 1] ✗ ZAC 证明失败，终止" >&2
    exit 1
fi
echo "[Step 1] ✓ ZAC 证明完成  $(date '+%H:%M:%S')"

# ── Step 2: zkLLM 全量证明（双卡并行）─────────────────────────────────────────
echo ""
echo "[Step 2] zkLLM 全量证明（Conv3d + ViT 32blocks + PatchMerger + Pooling + LLM 36层）"
echo "  GPU0 → worker 0（偶数图）: $LOG_DIR/corpus_full_gpu0.log"
echo "  GPU1 → worker 1（奇数图）: $LOG_DIR/corpus_full_gpu1.log"
echo "  预计耗时：~56 h"

CUDA_VISIBLE_DEVICES=0 python script/build_corpus_full_proof.py \
    --worker-id 0 --num-workers 2 \
    --corpus    "$CORPUS_JSONL" \
    --workdir   "$WORKDIR" \
    2>&1 | tee "$LOG_DIR/corpus_full_gpu0.log" &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python script/build_corpus_full_proof.py \
    --worker-id 1 --num-workers 2 \
    --corpus    "$CORPUS_JSONL" \
    --workdir   "$WORKDIR" \
    2>&1 | tee "$LOG_DIR/corpus_full_gpu1.log" &
PID1=$!

echo "  worker 0 PID=$PID0,  worker 1 PID=$PID1"
echo "  等待两个 worker 完成..."

wait $PID0
RC0=$?
wait $PID1
RC1=$?

echo ""
echo "========================================================"
echo "  全量证明结束  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  worker 0 (GPU0): $([ $RC0 -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL (rc='$RC0')')"
echo "  worker 1 (GPU1): $([ $RC1 -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL (rc='$RC1')')"
echo "========================================================"

if [ $RC0 -ne 0 ] || [ $RC1 -ne 0 ]; then
    exit 1
fi
