#!/usr/bin/env bash
# GRPO on a full merged base model.
# A fresh LoRA is trained and the frozen reference is the same full base with that LoRA disabled.
set -euo pipefail

DATA="$1"
BASE_MODEL="$2"
OUT="$3"
CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GRPO_GPUS="${GRPO_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="$(echo "$GRPO_GPUS" | tr ',' '\n' | grep -c .)"
GRPO_ENV="${GRPO_ENV:-/home/nvme02/conda/grpo_env}"
SWIFT="$GRPO_ENV/bin/swift"
K="${GRPO_K:-8}"
STEPS="${GRPO_STEPS:-120}"
SMOKE="${GRPO_SMOKE:-0}"
[ "$SMOKE" = "1" ] && STEPS=2
BETA="${GRPO_BETA:-0.04}"
LR="${GRPO_LR:-1e-6}"
TEMP="${GRPO_TEMPERATURE:-1.0}"
TOPP="${GRPO_TOP_P:-0.95}"
DEEPSPEED="${DEEPSPEED:-zero3}"
DS_ARG=()
[ -n "$DEEPSPEED" ] && DS_ARG=(--deepspeed "$DEEPSPEED")

[ -f "$DATA" ] || { echo "[grpo-on-model] missing data: $DATA"; exit 1; }
[ -f "$BASE_MODEL/config.json" ] || { echo "[grpo-on-model] missing base: $BASE_MODEL"; exit 1; }
[ -x "$SWIFT" ] || { echo "[grpo-on-model] swift not found: $SWIFT"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_RAS_ENABLE=0
echo "[grpo-on-model] policy_start=$BASE_MODEL + fresh LoRA"
echo "[grpo-on-model] pi_ref=$BASE_MODEL (fresh LoRA disabled; no ref adapter ambiguity)"
REWARD_FUNC="${GRPO_REWARD_FUNC:-derag_v4}"
SAVE_STEPS="${GRPO_SAVE_STEPS:-25}"
echo "[grpo-on-model] data=$DATA out=$OUT K=$K steps=$STEPS beta=$BETA lr=$LR reward=$REWARD_FUNC gpus=$GRPO_GPUS smoke=$SMOKE"

CUDA_VISIBLE_DEVICES="$GRPO_GPUS" NPROC_PER_NODE="$NPROC" \
"$SWIFT" rlhf \
  --rlhf_type grpo \
  --model "$BASE_MODEL" \
  --model_type "${V1_MODEL_TYPE:-qwen2}" \
  --template "${V1_TEMPLATE:-qwen2_5}" \
  --train_type lora \
  --dataset "$DATA" \
  --external_plugins "$CODE_ROOT/swift/grpo_reward_plugin.py" \
  --reward_funcs "$REWARD_FUNC" \
  --num_generations "$K" \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size "${VLLM_TP:-$NPROC}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_UTIL:-0.5}" \
  --vllm_max_model_len "${VLLM_MAX_LEN:-6144}" \
  --move_model_batches 16 \
  --sleep_level 1 \
  --offload_model true \
  --offload_optimizer true \
  "${DS_ARG[@]}" \
  --scale_rewards group \
  --beta "$BETA" \
  --temperature "$TEMP" \
  --top_p "$TOPP" \
  --max_steps "$STEPS" \
  --learning_rate "$LR" \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max_length 4096 \
  --max_completion_length "${GRPO_MAX_COMPLETION:-1792}" \
  --per_device_train_batch_size "${GRPO_PDBS:-1}" \
  --gradient_accumulation_steps "${GRPO_GA:-8}" \
  --gradient_checkpointing true \
  --attn_impl sdpa \
  --logging_steps 1 \
  --save_steps "$SAVE_STEPS" --save_total_limit "${GRPO_SAVE_TOTAL_LIMIT:-8}" \
  --output_dir "$OUT"

echo done > "$OUT/.done"
echo "[grpo-on-model] complete -> $OUT"
