#!/usr/bin/env bash
# Run a new GRPO experiment from RFT with the frozen RFT adapter as the KL reference.
# Uses an independent tag so it cannot overwrite the currently running conservative_v2.
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

: "${DASHSCOPE_API_KEY:?Please export DASHSCOPE_API_KEY before evaluation.}"

export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export GRPO_ENV="${GRPO_ENV:-/home/nvme02/conda/grpo_env}"
export DEEPSPEED="${DEEPSPEED:-zero3}"

export GRPO_BASES="${GRPO_V3_BASES:-rft}"
export GRPO_RUN_TAG="${GRPO_V3_RUN_TAG:-ref_rft_v3}"

export GRPO_STEPS="${GRPO_V3_STEPS:-50}"
export GRPO_LR="${GRPO_V3_LR:-5e-7}"
export GRPO_BETA="${GRPO_V3_BETA:-0.08}"

export USE_VLLM=true
export VLLM_TP="${GRPO_V3_VLLM_TP:-8}"
export VLLM_GPU_UTIL="${GRPO_V3_VLLM_GPU_UTIL:-0.5}"
export VLLM_MAX_LEN="${GRPO_V3_VLLM_MAX_LEN:-6144}"
export GRPO_K="${GRPO_V3_K:-8}"
export GRPO_PDBS="${GRPO_V3_PDBS:-1}"
export GRPO_GA="${GRPO_V3_GA:-8}"
export GRPO_MAX_COMPLETION="${GRPO_V3_MAX_COMPLETION:-1024}"
export VLLM_SERVE_GPU_UTIL="${GRPO_V3_SERVE_GPU_UTIL:-0.88}"

PY="$ZHJG_ENV/bin/python"
echo "[ref-rft-v3] base=$GRPO_BASES tag=$GRPO_RUN_TAG steps=$GRPO_STEPS lr=$GRPO_LR beta=$GRPO_BETA"
echo "[ref-rft-v3] grpo.sh will use the resolved RFT adapter for both --adapters and --ref_adapters."
exec "$PY" -X utf8 run.py --only grpo
