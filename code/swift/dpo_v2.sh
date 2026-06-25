#!/usr/bin/env bash
# corrected-v2 mini DPO on the merged RFT base.
# Fresh LoRA is trained; disabling it gives the exact frozen merged base as pi_ref.
set -euo pipefail

PAIRS="$1"
BASE_MODEL="$2"
OUT="$3"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)"
ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
SWIFT="$ZHJG_ENV/bin/swift"
DEEPSPEED="${DEEPSPEED:-zero3}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_LR="${DPO_LR:-5e-6}"
DPO_EPOCHS="${DPO_EPOCHS:-2}"
DPO_GA="${DPO_GA:-2}"
DPO_MAX_LEN="${DPO_MAX_LEN:-4096}"
DPO_LOGGING_STEPS="${DPO_LOGGING_STEPS:-5}"
DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-5}"
DPO_SAVE_TOTAL_LIMIT="${DPO_SAVE_TOTAL_LIMIT:-8}"
DPO_RPO_ALPHA="${DPO_RPO_ALPHA:-1.0}"
DS_ARG=()
RPO_ARG=()

[ -f "$PAIRS" ] || { echo "[dpo-v2] missing pairs: $PAIRS"; exit 1; }
[ -f "$BASE_MODEL/config.json" ] || { echo "[dpo-v2] missing base: $BASE_MODEL"; exit 1; }
[ -x "$SWIFT" ] || { echo "[dpo-v2] swift not found: $SWIFT"; exit 1; }
[ -n "$DEEPSPEED" ] && DS_ARG=(--deepspeed "$DEEPSPEED")

if [ -n "$DPO_RPO_ALPHA" ]; then
  if "$SWIFT" rlhf --help 2>/dev/null | grep -q -- "--rpo_alpha"; then
    RPO_ARG=(--rpo_alpha "$DPO_RPO_ALPHA")
  else
    echo "[dpo-v2] FATAL: rpo_alpha=$DPO_RPO_ALPHA requested but current swift does not expose --rpo_alpha"
    exit 2
  fi
fi

PAIR_LINES="$(wc -l < "$PAIRS" | tr -d ' ')"
if [ "${DPO_AUTO_GA:-1}" = "1" ] && [ "$PAIR_LINES" -lt 400 ]; then
  DPO_GA=1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_RAS_ENABLE=0
echo "[dpo-v2] policy_start=$BASE_MODEL + fresh LoRA"
echo "[dpo-v2] pi_ref=$BASE_MODEL (fresh LoRA disabled; no ref adapter ambiguity)"
echo "[dpo-v2] pairs=$PAIRS lines=$PAIR_LINES out=$OUT beta=$DPO_BETA lr=$DPO_LR epochs=$DPO_EPOCHS ga=$DPO_GA gpus=$TRAIN_GPUS deepspeed=$DEEPSPEED rpo=${DPO_RPO_ALPHA:-off}"

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
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max_length "$DPO_MAX_LEN" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$DPO_GA" \
  --gradient_checkpointing true \
  "${DS_ARG[@]}" \
  "${RPO_ARG[@]}" \
  --logging_steps "$DPO_LOGGING_STEPS" \
  --save_strategy steps --save_steps "$DPO_SAVE_STEPS" --save_total_limit "$DPO_SAVE_TOTAL_LIMIT" \
  --output_dir "$OUT"

find "$OUT" -name trainer_state.json -print | sort > "$OUT/trainer_state_files.txt" || true
echo done > "$OUT/.done"
echo "[dpo-v2] complete -> $OUT"
