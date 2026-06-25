#!/usr/bin/env bash
# Foreground corrected-v3.1 derag-rubric diagnostics.
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_LOG_DIR="${ZHJG_LOG_DIR:-$ZHJG_WORK_DIR/logs}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export V31_RUN_ID="${V31_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export CORRECTED_V31_LOG_DIR="${CORRECTED_V31_LOG_DIR:-$ZHJG_LOG_DIR/corrected_v31/$V31_RUN_ID}"
export ZHJG_CONSOLE_LOG_LEVEL="${ZHJG_CONSOLE_LOG_LEVEL:-WARNING}"
export V31_JUDGE_K="${V31_JUDGE_K:-3}"
export V31_KIMI_WORKERS="${V31_KIMI_WORKERS:-3}"

: "${DASHSCOPE_API_KEY:?Please export DASHSCOPE_API_KEY before running corrected-v3.1 diagnostics.}"

echo "===== corrected-v3.1 derag-rubric diagnostics ====="
echo "run_id=$V31_RUN_ID"
echo "Target: trace-free think + grounded facts + answer consistency"
echo "Plan: E0 repeated derag judge -> E1 paired readout -> E2 trace surgery sanity check"
echo "No GPU training will be launched."
echo "Monitor: V31_RUN_ID=$V31_RUN_ID bash scripts/monitor_corrected_v31.sh"
echo "Logs: $CORRECTED_V31_LOG_DIR"
exec "$ZHJG_ENV/bin/python" -X utf8 run_corrected_v31.py
