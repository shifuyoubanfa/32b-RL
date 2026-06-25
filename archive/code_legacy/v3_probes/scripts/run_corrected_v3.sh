#!/usr/bin/env bash
# Foreground corrected-v3 zero-training diagnostics.
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_LOG_DIR="${ZHJG_LOG_DIR:-$ZHJG_WORK_DIR/logs}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export V3_RUN_ID="${V3_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export CORRECTED_V3_LOG_DIR="${CORRECTED_V3_LOG_DIR:-$ZHJG_LOG_DIR/corrected_v3/$V3_RUN_ID}"
export ZHJG_CONSOLE_LOG_LEVEL="${ZHJG_CONSOLE_LOG_LEVEL:-WARNING}"
export V3_JUDGE_K="${V3_JUDGE_K:-3}"
export V3_KIMI_WORKERS="${V3_KIMI_WORKERS:-3}"

: "${DASHSCOPE_API_KEY:?Please export DASHSCOPE_API_KEY before running corrected-v3 diagnostics.}"

echo "===== corrected-v3 zero-training diagnostics ====="
echo "run_id=$V3_RUN_ID"
echo "Plan: E0 repeated judge + paired readout -> E1 placebo -> E2 rewrite ceiling -> E3 pairwise controls"
echo "No GPU training will be launched."
echo "Monitor: V3_RUN_ID=$V3_RUN_ID bash scripts/monitor_corrected_v3.sh"
echo "Logs: $CORRECTED_V3_LOG_DIR"
exec "$ZHJG_ENV/bin/python" -X utf8 run_corrected_v3.py
