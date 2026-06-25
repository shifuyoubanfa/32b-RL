#!/usr/bin/env bash
# 起【静态 V1】vLLM 服务（数据构建 / 评测 / RFT·DPO rollout 用）。
# 用 vllm_env 环境（独立于训练 zhjg_rl）；占 GPU 0,1（TP=2），训练用 2-7。
# --enable-lora + 起服务时 --lora-modules 预声明 adapter（vLLM 0.20.1 运行时加载端点不稳，改预声明）。
#   传 VLLM_LORA="名字=adapter目录" 即把该 adapter 以 <名字> 暴露；评测/rollout 用 --model <名字> 请求。
set -euo pipefail

V1_DIR="${V1_DIR:-/home/nvme01/zhjg/V1-32B/checkpoint-1500}"
VLLM_ENV="${VLLM_ENV:-/home/nvme02/biyh/vllm_env}"
VLLM_GPUS="${VLLM_GPUS:-0,1}"
PORT="${VLLM_PORT:-8000}"
LOG_DIR="${ZHJG_LOG_DIR:-/home/nvme01/zhjg/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/vllm.log"
PIDF="$LOG_DIR/vllm.pid"
TP="$(echo "$VLLM_GPUS" | tr ',' '\n' | grep -c .)"
# 与 GRPO colocate 使用的 VLLM_GPU_UTIL 分开，避免评测调低/调高显存比例时误伤训练侧。
GPU_UTIL="${VLLM_SERVE_GPU_UTIL:-0.90}"

# 新版 NCCL RAS 在这台机器上曾导致多进程退出/初始化阶段互等；关闭仅影响诊断线程，不影响推理。
export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"

LORA_OPT=""
[ -n "${VLLM_LORA:-}" ] && LORA_OPT="--lora-modules $VLLM_LORA"   # 值无空格（路径无空格），直接词分割
echo "[serve] V1 vLLM：GPU=$VLLM_GPUS TP=$TP port=$PORT gpu_util=$GPU_UTIL 日志=$LOG ${VLLM_LORA:+预声明adapter=$VLLM_LORA}"
# setsid 起新会话：$! 即会话/进程组 leader，停止时 kill -TERM -<PID> 可回收 TP worker 等全部子进程。
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
setsid "$VLLM_ENV/bin/vllm" serve "$V1_DIR" \
  --served-model-name v1 \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-lora --max-loras 4 --max-lora-rank 16 \
  $LORA_OPT \
  --port "$PORT" > "$LOG" 2>&1 &
echo $! > "$PIDF"
PID="$(cat "$PIDF")"
echo "[serve] PID/PGID=$PID（停止：kill -TERM -$PID）"
sleep 4
if ! kill -0 "$PID" 2>/dev/null; then
  echo "[serve] ❌ vLLM 秒退（OOM/端口占用/权重路径错？），日志末尾："; tail -n 20 "$LOG"; exit 1
fi
echo "[serve] 进程存活，加载中（run.py 会 poll /v1/models 直到就绪）"
