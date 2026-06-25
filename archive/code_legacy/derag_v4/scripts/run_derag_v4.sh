#!/usr/bin/env bash
# Foreground derag_v4 training chain.
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_LOG_DIR="${ZHJG_LOG_DIR:-$ZHJG_WORK_DIR/logs}"
export ZHJG_OUTPUT_DIR="${ZHJG_OUTPUT_DIR:-$ZHJG_WORK_DIR/output}"
export ZHJG_CKPT_DIR="${ZHJG_CKPT_DIR:-$ZHJG_WORK_DIR/ckpts}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export DERAG_V4_RUN_ID="${DERAG_V4_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export ZHJG_CONSOLE_LOG_LEVEL="${ZHJG_CONSOLE_LOG_LEVEL:-WARNING}"

: "${DASHSCOPE_API_KEY:?Please export DASHSCOPE_API_KEY before running derag_v4.}"

echo "===== derag_v4 DPO -> GRPO chain ====="
echo "run_id=$DERAG_V4_RUN_ID"
echo "Plan:"
echo "  RFT merged base -> residual rewrite -> binary-vote Stage1 gate/SFT"
echo "  -> binary anchor calibration + trace/fact votes + arbiter/repair + blind spot-check sheet"
echo "  -> explicit deterministic fallback if Kimi binary calibration is unfit"
echo "  -> G1-1 density probe"
echo "  -> on-policy DPO with retries -> derag_v4 GRPO with retries -> reports"
echo "G1-1 K list: ${DERAG_V4_G11_KS:-16,32,64,128}"
echo "Logs: $ZHJG_LOG_DIR/derag_v4/$DERAG_V4_RUN_ID"
echo "Outputs: $ZHJG_OUTPUT_DIR/derag_v4/$DERAG_V4_RUN_ID"
echo "Monitor:"
echo "  DERAG_V4_RUN_ID=$DERAG_V4_RUN_ID bash scripts/monitor_derag_v4.sh"
echo ""

exec "$ZHJG_ENV/bin/python" -X utf8 run_derag_v4.py
