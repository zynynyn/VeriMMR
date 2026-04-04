#!/usr/bin/env bash
# ============================================================
# C1 实验全流程脚本
# 运行方式：
#   nohup bash script/run_c1_experiments.sh > logs/c1_master.log 2>&1 &
#
# 执行顺序：
#   1. slidevqa  attack_verify（重新下载，完成后清理缓存）
#   2. docvqa    attack_verify（重新下载，完成后清理缓存）
#   3. chartvqa  recall（--no-clean 保留缓存）
#      chartvqa  attack_verify（复用缓存，完成后清理）
#   4. infovqa   recall（--no-clean 保留缓存）
#      infovqa   attack_verify（复用缓存，完成后清理）
#
# 备注：slidevqa 和 docvqa 的纯 recall 结果已有，无需重跑。
# ============================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

PYTHON="python"
RECALL_SCRIPT="$REPO_DIR/script/experiment_c1_recall.py"
ATTACK_SCRIPT="$REPO_DIR/script/experiment_c1_attack_verify.py"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

banner() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "  $(ts)"
    echo "============================================================"
}

run_step() {
    local label="$1"
    local log_file="$2"
    shift 2
    banner "$label → $log_file"
    "$PYTHON" "$@" 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo "[ERROR] $label 失败（exit $rc），中止流程。" >&2
        exit $rc
    fi
    echo "[OK] $label 完成  $(ts)"
}

# ── 1. SlideVQA attack_verify ────────────────────────────────────────────────
run_step "SlideVQA attack_verify" \
    "$LOG_DIR/c1_attack_slidevqa.log" \
    "$ATTACK_SCRIPT" --dataset slidevqa
# attack_verify 默认 clean_cache=True，无需额外清理

# ── 2. DocVQA attack_verify ──────────────────────────────────────────────────
run_step "DocVQA attack_verify" \
    "$LOG_DIR/c1_attack_docvqa.log" \
    "$ATTACK_SCRIPT" --dataset docvqa

# ── 3. ChartQA ───────────────────────────────────────────────────────────────
run_step "ChartQA recall (--no-clean)" \
    "$LOG_DIR/c1_recall_chartvqa.log" \
    "$RECALL_SCRIPT" --dataset chartvqa --no-clean

run_step "ChartQA attack_verify" \
    "$LOG_DIR/c1_attack_chartvqa.log" \
    "$ATTACK_SCRIPT" --dataset chartvqa
# attack_verify 默认清理缓存

# ── 4. InfoVQA ───────────────────────────────────────────────────────────────
run_step "InfoVQA recall (--no-clean)" \
    "$LOG_DIR/c1_recall_infovqa.log" \
    "$RECALL_SCRIPT" --dataset infovqa --no-clean

run_step "InfoVQA attack_verify" \
    "$LOG_DIR/c1_attack_infovqa.log" \
    "$ATTACK_SCRIPT" --dataset infovqa

# ── 完成 ──────────────────────────────────────────────────────────────────────
banner "全部 C1 实验完成"
echo "日志目录：$LOG_DIR"
ls -lh "$LOG_DIR"/c1_*.log
