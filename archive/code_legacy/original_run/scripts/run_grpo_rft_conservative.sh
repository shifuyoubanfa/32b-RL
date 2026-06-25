#!/usr/bin/env bash
# 从当前最佳 RFT checkpoint 做一次独立的保守 GRPO v2，并自动评测。
# 不覆盖首轮 GRPO：checkpoint/report 均写入 conservative_v2 后缀的新路径。
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

: "${DASHSCOPE_API_KEY:?请先 export DASHSCOPE_API_KEY=...，评测 Kimi 判分需要它}"

export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export GRPO_ENV="${GRPO_ENV:-/home/nvme02/conda/grpo_env}"
export DEEPSPEED="${DEEPSPEED:-zero3}"

# 使用 GRPO_V2_* 覆盖，避免继承当前终端残留的首轮/调试变量。
# 仅从已验证最优的 RFT 出发；run tag 保证不覆盖首轮结果。
export GRPO_BASES="${GRPO_V2_BASES:-rft}"
export GRPO_RUN_TAG="${GRPO_V2_RUN_TAG:-conservative_v2}"

# 首轮 100 步后 KL=0.7523 且 humanness/grounded/acc 均小幅回退。
# v2 收紧更新：学习率降 4 倍、单 KL 加强 2 倍、步数减半；其余跑通配置保持不变。
export GRPO_STEPS="${GRPO_V2_STEPS:-50}"
export GRPO_LR="${GRPO_V2_LR:-5e-7}"
export GRPO_BETA="${GRPO_V2_BETA:-0.08}"

# 锁定首轮已经跑通的训练/colocate 显存配方。
export USE_VLLM=true
export VLLM_TP="${GRPO_V2_VLLM_TP:-8}"
export VLLM_GPU_UTIL="${GRPO_V2_VLLM_GPU_UTIL:-0.5}"
export VLLM_MAX_LEN="${GRPO_V2_VLLM_MAX_LEN:-6144}"
export GRPO_K="${GRPO_V2_K:-8}"
export GRPO_PDBS="${GRPO_V2_PDBS:-1}"
export GRPO_GA="${GRPO_V2_GA:-8}"
export GRPO_MAX_COMPLETION="${GRPO_V2_MAX_COMPLETION:-1024}"

# 静态 vLLM 评测阶段给 CUDA graph 留余量；不改训练 colocate 已跑通的 VLLM_GPU_UTIL=0.5。
export VLLM_SERVE_GPU_UTIL="${GRPO_V2_SERVE_GPU_UTIL:-0.88}"

PY="$ZHJG_ENV/bin/python"
echo "[conservative-v2] base=$GRPO_BASES tag=$GRPO_RUN_TAG steps=$GRPO_STEPS lr=$GRPO_LR beta=$GRPO_BETA"
echo "[conservative-v2] train_vllm_util=$VLLM_GPU_UTIL serve_vllm_util=$VLLM_SERVE_GPU_UTIL tp=$VLLM_TP"
echo "[conservative-v2] 新产物不会覆盖首轮 GRPO；开始训练后自动推理、Kimi 判分并出报告。"
exec "$PY" -X utf8 run.py --only grpo
