#!/usr/bin/env bash
# Start a static vLLM service for any full model, optionally with one LoRA adapter.
set -euo pipefail

MODEL_DIR="$1"
SERVED_NAME="$2"
LORA_NAME="${3:-}"
LORA_PATH="${4:-}"
VLLM_ENV="${VLLM_ENV:-/home/nvme02/biyh/vllm_env}"
VLLM_GPUS="${VLLM_GPUS:-0,1}"
PORT="${VLLM_PORT:-8000}"
LOG_DIR="${ZHJG_LOG_DIR:-/home/nvme01/zhjg/logs}"
LOG="$LOG_DIR/merged_chain_vllm.log"
PIDF="$LOG_DIR/merged_chain_vllm.pid"
TP="$(echo "$VLLM_GPUS" | tr ',' '\n' | grep -c .)"
GPU_UTIL="${VLLM_SERVE_GPU_UTIL:-0.88}"
LORA_ARGS=()
STARTED_PID=""

cleanup_failed_start() {
  rc=$?
  trap - EXIT INT TERM
  if [ "$rc" -ne 0 ] && [ -n "$STARTED_PID" ]; then
    # 本脚本用 setsid 启动，PID 同时是进程组 ID；只清自己刚起的组。
    kill -TERM -- "-$STARTED_PID" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$STARTED_PID" 2>/dev/null || true
    rm -f -- "$PIDF"
  fi
  exit "$rc"
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup_failed_start EXIT

[ -f "$MODEL_DIR/config.json" ] || { echo "[serve-model] missing model: $MODEL_DIR"; exit 1; }
[ -x "$VLLM_ENV/bin/vllm" ] || { echo "[serve-model] vllm not found: $VLLM_ENV/bin/vllm"; exit 1; }
if [ -n "$LORA_NAME" ] || [ -n "$LORA_PATH" ]; then
  [ -n "$LORA_NAME" ] && [ -f "$LORA_PATH/adapter_config.json" ] \
    || { echo "[serve-model] invalid LoRA name/path: $LORA_NAME $LORA_PATH"; exit 1; }
  LORA_ARGS=(--enable-lora --max-loras 1 --max-lora-rank 16 --lora-modules "$LORA_NAME=$LORA_PATH")
fi

mkdir -p "$LOG_DIR"
export NCCL_RAS_ENABLE=0
echo "[serve-model] model=$MODEL_DIR served_name=$SERVED_NAME lora=${LORA_NAME:-none} GPU=$VLLM_GPUS TP=$TP"
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" setsid "$VLLM_ENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization "$GPU_UTIL" \
  "${LORA_ARGS[@]}" \
  --port "$PORT" > "$LOG" 2>&1 &
STARTED_PID=$!
echo "$STARTED_PID" > "$PIDF"
sleep 4
kill -0 "$(cat "$PIDF")" 2>/dev/null || { echo "[serve-model] vLLM exited"; tail -n 30 "$LOG"; exit 1; }
echo "[serve-model] loading; pid=$(cat "$PIDF") log=$LOG"
trap - EXIT INT TERM
