#!/usr/bin/env bash
# SFT on a full merged base model.  Used by derag_v4 Stage1:
# RFT merged base + fresh LoRA -> s1 LoRA.
set -euo pipefail

TRAIN="$1"
VAL="$2"
BASE_MODEL="$3"
OUT="$4"
LR="${5:-5e-5}"
EPOCHS="${6:-2}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)"
ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
SWIFT="$ZHJG_ENV/bin/swift"
DEEPSPEED="${DEEPSPEED:-}"
DS_ARG=()
[ -n "$DEEPSPEED" ] && DS_ARG=(--deepspeed "$DEEPSPEED")

[ -f "$TRAIN" ] || { echo "[sft-on-model] missing train: $TRAIN"; exit 1; }
[ -f "$VAL" ] || { echo "[sft-on-model] missing val: $VAL"; exit 1; }
[ -f "$BASE_MODEL/config.json" ] || { echo "[sft-on-model] missing base: $BASE_MODEL"; exit 1; }
[ -x "$SWIFT" ] || { echo "[sft-on-model] swift not found: $SWIFT"; exit 1; }

echo "[sft-on-model] base=$BASE_MODEL train=$TRAIN val=$VAL out=$OUT lr=$LR epochs=$EPOCHS gpus=$TRAIN_GPUS deepspeed=${DEEPSPEED:-off}"

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NPROC_PER_NODE="$NPROC" \
"$SWIFT" sft \
  --model "$BASE_MODEL" \
  --model_type "${V1_MODEL_TYPE:-qwen2}" \
  --template "${V1_TEMPLATE:-qwen2_5}" \
  --train_type lora \
  --dataset "$TRAIN" \
  --val_dataset "$VAL" \
  --torch_dtype bfloat16 \
  --num_train_epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --warmup_ratio 0.05 \
  --lora_rank "${LORA_R:-16}" --lora_alpha "${LORA_ALPHA:-32}" --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max_length "${SFT_MAX_LEN:-4096}" \
  --per_device_train_batch_size "${SFT_PDBS:-1}" \
  --gradient_accumulation_steps "${SFT_GA:-8}" \
  --gradient_checkpointing true \
  "${DS_ARG[@]}" \
  --eval_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --save_total_limit "${SFT_SAVE_TOTAL_LIMIT:-3}" \
  --logging_steps "${SFT_LOGGING_STEPS:-5}" \
  --output_dir "$OUT"

echo done > "$OUT/.done"
echo "[sft-on-model] complete -> $OUT"
