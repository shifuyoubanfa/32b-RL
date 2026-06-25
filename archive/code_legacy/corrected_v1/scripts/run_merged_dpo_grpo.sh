#!/usr/bin/env bash
# Foreground one-click corrected chain. Open another terminal and run monitor_merged_dpo_grpo.sh.
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_LOG_DIR="${ZHJG_LOG_DIR:-$ZHJG_WORK_DIR/logs}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export GRPO_ENV="${GRPO_ENV:-/home/nvme02/conda/grpo_env}"
export VLLM_ENV="${VLLM_ENV:-/home/nvme02/biyh/vllm_env}"
export DEEPSPEED="${DEEPSPEED:-zero3}"
export TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
export GRPO_GPUS="${GRPO_GPUS:-0,1,2,3,4,5,6,7}"
export GRPO_STEPS="${GRPO_STEPS:-50}"
export GRPO_LR="${GRPO_LR:-5e-7}"
export GRPO_BETA="${GRPO_BETA:-0.08}"
export VLLM_TP="${VLLM_TP:-8}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.5}"
export VLLM_MAX_LEN="${VLLM_MAX_LEN:-6144}"
export VLLM_SERVE_GPU_UTIL="${VLLM_SERVE_GPU_UTIL:-0.88}"
export PMI_ENABLED=0

: "${DASHSCOPE_API_KEY:?Please export DASHSCOPE_API_KEY before running the evaluation chain.}"

echo "===== Corrected merged-base chain ====="
echo "V1 + final RFT LoRA -> merged RFT base -> DPO -> merged DPO base -> two GRPO branches -> reports"
echo "Monitor in another terminal: bash scripts/monitor_merged_dpo_grpo.sh"
echo "GRPO: steps=$GRPO_STEPS lr=$GRPO_LR beta=$GRPO_BETA"
exec "$ZHJG_ENV/bin/python" -X utf8 run_merged_dpo_grpo.py
