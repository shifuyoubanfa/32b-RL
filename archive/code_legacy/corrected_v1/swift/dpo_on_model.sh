#!/usr/bin/env bash
# DPO on a full merged base model.
# A fresh trainable LoRA is created; disabling it gives the exact frozen merged base as pi_ref.
set -euo pipefail

PAIRS="$1"
BASE_MODEL="$2"
OUT="$3"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)"
ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
SWIFT="$ZHJG_ENV/bin/swift"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_LR="${DPO_LR:-5e-6}"
DPO_EPOCHS="${DPO_EPOCHS:-1}"
DPO_MAX_LEN="${DPO_MAX_LEN:-4096}"
DEEPSPEED="${DEEPSPEED:-zero3}"
DS_ARG=()
[ -n "$DEEPSPEED" ] && DS_ARG=(--deepspeed "$DEEPSPEED")

[ -f "$PAIRS" ] || { echo "[dpo-on-model] missing pairs: $PAIRS"; exit 1; }
[ -f "$BASE_MODEL/config.json" ] || { echo "[dpo-on-model] missing base: $BASE_MODEL"; exit 1; }
[ -x "$SWIFT" ] || { echo "[dpo-on-model] swift not found: $SWIFT"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_RAS_ENABLE=0
echo "[dpo-on-model] policy_start=$BASE_MODEL + fresh LoRA"
echo "[dpo-on-model] pi_ref=$BASE_MODEL (fresh LoRA disabled; no ref adapter ambiguity)"
echo "[dpo-on-model] pairs=$PAIRS out=$OUT beta=$DPO_BETA lr=$DPO_LR gpus=$TRAIN_GPUS deepspeed=$DEEPSPEED"

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NPROC_PER_NODE="$NPROC" \
"$SWIFT" rlhf \
  --rlhf_type dpo \
  --model "$BASE_MODEL" \
  --model_type "${V1_MODEL_TYPE:-qwen2}" \
  --template "${V1_TEMPLATE:-qwen2_5}" \
  --train_type lora \
  --dataset "$PAIRS" \
  --torch_dtype bfloat16 \
  --beta "$DPO_BETA" \
  --num_train_epochs "$DPO_EPOCHS" \
  --learning_rate "$DPO_LR" \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max_length "$DPO_MAX_LEN" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  "${DS_ARG[@]}" \
  --logging_steps 1 \
  --save_strategy epoch --save_total_limit 2 \
  --output_dir "$OUT"

echo done > "$OUT/.done"
echo "[dpo-on-model] complete -> $OUT"
