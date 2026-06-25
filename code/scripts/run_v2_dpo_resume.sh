#!/usr/bin/env bash
# 本地服务器接回 AutoDL DPO adapter：CPU合并 → 等两张空卡 → vLLM 500题 → Kimi三分。
# 前台运行本脚本；另开终端运行 monitor_v2_dpo_resume.sh。只续 2s-2s-2s 这一条正式叶。
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_LOG_DIR="${ZHJG_LOG_DIR:-$ZHJG_WORK_DIR/logs}"
export ZHJG_OUTPUT_DIR="${ZHJG_OUTPUT_DIR:-$ZHJG_WORK_DIR/output}"
export ZHJG_CKPT_DIR="${ZHJG_CKPT_DIR:-$ZHJG_WORK_DIR/ckpts}"
export ZHJG_MODEL_DIR="${ZHJG_MODEL_DIR:-$ZHJG_WORK_DIR/models}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export VLLM_ENV="${VLLM_ENV:-/home/nvme02/biyh/vllm_env}"
export V2_TAG=derag2
# 不继承旧终端里可能残留的 V2 输出/日志位置；正式叶只允许原 run_v2 路径。
export ZHJG_V2_OUTPUT_DIR="$ZHJG_OUTPUT_DIR/derag2"
export ZHJG_LOG_FILE="$ZHJG_LOG_DIR/v2_dpo_resume/pipeline.log"
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export VLLM_PORT=8000
export KIMI_MODEL="kimi/kimi-k2.6"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# 下载回来的 checkpoint 保持现状；正式合并模型与评测产物由 run_v2 原 lineage 计算，不能在这里改名。
export V2_DPO_DOWNLOADED_ADAPTER="${V2_DPO_DOWNLOADED_ADAPTER:-$ZHJG_CKPT_DIR/v2-dpo-derag2-lora/checkpoint-108}"

# 默认自动等任意两张真正空闲卡；若同事明确归还某两张，可启动前 export V2_DPO_EVAL_GPUS=0,1。
export V2_DPO_GPU_USED_MAX_MIB="${V2_DPO_GPU_USED_MAX_MIB:-2048}"
export V2_DPO_GPU_UTIL_MAX="${V2_DPO_GPU_UTIL_MAX:-5}"
export V2_DPO_GPU_WAIT_INTERVAL="${V2_DPO_GPU_WAIT_INTERVAL:-60}"
export V2_DPO_GPU_WAIT_TIMEOUT="${V2_DPO_GPU_WAIT_TIMEOUT:-0}"  # 0=一直等，不抢卡
export V2_DPO_GPU_STABLE_SAMPLES="${V2_DPO_GPU_STABLE_SAMPLES:-3}"
export VLLM_SERVE_GPU_UTIL="${VLLM_SERVE_GPU_UTIL:-0.88}"

mkdir -p "$ZHJG_LOG_DIR/v2_dpo_resume"
exec 9>"$ZHJG_LOG_DIR/v2_dpo_resume/runner.lock"
if ! flock -n 9; then
  printf '已有一个 V2 DPO resume 在运行；锁：%s\n' "$ZHJG_LOG_DIR/v2_dpo_resume/runner.lock" >&2
  exit 2
fi
export V2_DPO_LOCK_HELD=1

printf '%s\n' "===== V2 derag2 DPO 本地续跑（正式叶 2s-2s-2s）====="
printf 'adapter : %s\n' "$V2_DPO_DOWNLOADED_ADAPTER"
printf 'LoRA规范: %s\n' "$ZHJG_CKPT_DIR/v2-dpo-2sigma-2s-2s-lora"
printf 'base    : %s\n' "$ZHJG_MODEL_DIR/v2-rft-2sigma-2s-merged"
printf 'merged  : %s\n' "$ZHJG_MODEL_DIR/v2-dpo-2sigma-2s-2s-merged"
printf 'outputs : %s\n' "$ZHJG_V2_OUTPUT_DIR/v2-2s-2s-2s_{infer,scores,report,summary}.*"
printf 'logs    : %s\n' "$ZHJG_LOG_DIR/v2_dpo_resume/"
printf 'GPU     : %s（used<=%sMiB、util<=%s%%、无计算PID，连续%s次；超时0=一直等）\n' \
  "${V2_DPO_EVAL_GPUS:-auto}" "$V2_DPO_GPU_USED_MAX_MIB" "$V2_DPO_GPU_UTIL_MAX" "$V2_DPO_GPU_STABLE_SAMPLES"
printf 'monitor : cd %s && bash scripts/monitor_v2_dpo_resume.sh\n\n' "$CODE_ROOT"

exec "$ZHJG_ENV/bin/python" -X utf8 run_v2_dpo_resume.py
