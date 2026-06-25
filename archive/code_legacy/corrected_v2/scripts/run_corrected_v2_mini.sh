#!/usr/bin/env bash
# Foreground corrected-v2 mini run. Open another terminal and run monitor_corrected_v2_mini.sh.
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_LOG_DIR="${ZHJG_LOG_DIR:-$ZHJG_WORK_DIR/logs}"
export CORRECTED_V2_LOG_DIR="${CORRECTED_V2_LOG_DIR:-$ZHJG_LOG_DIR/corrected_v2}"
export ZHJG_LOG_FILE="${ZHJG_LOG_FILE:-$CORRECTED_V2_LOG_DIR/pipeline.log}"
export ZHJG_FILE_LOG_LEVEL="${ZHJG_FILE_LOG_LEVEL:-INFO}"
export ZHJG_CONSOLE_LOG_LEVEL="${ZHJG_CONSOLE_LOG_LEVEL:-WARNING}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export VLLM_ENV="${VLLM_ENV:-/home/nvme02/biyh/vllm_env}"
export DEEPSPEED="${DEEPSPEED:-zero3}"
export TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
export VLLM_GPUS="${VLLM_GPUS:-0,1}"
export VLLM_SERVE_GPU_UTIL="${VLLM_SERVE_GPU_UTIL:-0.88}"
export V2_SCORE_QUERIES="${V2_SCORE_QUERIES:-800}"
export V2_KIMI_WORKERS="${V2_KIMI_WORKERS:-3}"
export V2_MIN_TRAIN_PAIRS="${V2_MIN_TRAIN_PAIRS:-80}"
export V2_HELDOUT_PAIRS="${V2_HELDOUT_PAIRS:-25}"
export V2_REJUDGE_PAIRS="${V2_REJUDGE_PAIRS:-0}"
export V2_STABLE_MIN_TRAIN_PAIRS="${V2_STABLE_MIN_TRAIN_PAIRS:-50}"
export V2_STABLE_HELDOUT_PAIRS="${V2_STABLE_HELDOUT_PAIRS:-10}"
export V2_STABLE_MIN_INITIAL_H_MARGIN="${V2_STABLE_MIN_INITIAL_H_MARGIN:-0.40}"
export V2_STABLE_MIN_REJUDGE_H_MARGIN="${V2_STABLE_MIN_REJUDGE_H_MARGIN:-0.05}"
export V2_STABLE_MIN_MEAN_H_MARGIN="${V2_STABLE_MIN_MEAN_H_MARGIN:-0.15}"
export DPO_BETA="${DPO_BETA:-0.1}"
export DPO_LR="${DPO_LR:-5e-6}"
export DPO_EPOCHS="${DPO_EPOCHS:-3}"
export DPO_GA="${DPO_GA:-2}"
export PMI_ENABLED=0

: "${DASHSCOPE_API_KEY:?Please export DASHSCOPE_API_KEY before running corrected-v2 mini.}"

echo "===== corrected-v2 mini DPO ====="
echo "Plan X: reuse 60_dpo_rollout -> Kimi score -> strict pairs -> full rejudge -> stable pairs -> mini DPO -> eval"
echo "Monitor in another terminal: bash scripts/monitor_corrected_v2_mini.sh"
echo "Kimi scoring: queries=$V2_SCORE_QUERIES workers=$V2_KIMI_WORKERS"
echo "Stable gates: train>=$V2_STABLE_MIN_TRAIN_PAIRS heldout=$V2_STABLE_HELDOUT_PAIRS initial_h>=$V2_STABLE_MIN_INITIAL_H_MARGIN rejudge_h>=$V2_STABLE_MIN_REJUDGE_H_MARGIN"
echo "DPO: lr=$DPO_LR beta=$DPO_BETA epochs=$DPO_EPOCHS ga=$DPO_GA"
echo "Logs: $CORRECTED_V2_LOG_DIR"
exec "$ZHJG_ENV/bin/python" -X utf8 run_corrected_v2_mini.py
