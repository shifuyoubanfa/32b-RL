#!/usr/bin/env bash
# 冷启动 SFT / RFT 续训：ms-swift LoRA 微调，多卡 DDP。
# 资源分工：vLLM 静态 V1 服务占 GPU 0,1（评测/后续 rollout 复用）；训练用 2-7 共 6 卡。
#   32B bf16 LoRA：base 65G < 单卡 80G → 每卡一份完整 V1+LoRA、数据并行（无需 deepspeed）。
#
# 用法: bash swift/sft.sh <train.jsonl> <val.jsonl> <output_dir> <lr> <epochs> [resume_adapter]
#
# ⚠️ swift 4.0.1 的精确旗标以服务器 `swift sft --help` 为准（版本差异），上线前 smoke 一轮确认。
set -euo pipefail

TRAIN="$1"; VAL="$2"; OUT="$3"; LR="$4"; EPOCHS="$5"; RESUME="${6:-}"
V1_DIR="${V1_DIR:-/home/nvme01/zhjg/V1-32B/checkpoint-1500}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3,4,5,6,7}"
NPROC="$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)"
SWIFT="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/swift"   # 全路径，避免 (base) 环境下 swift 不在 PATH

# 续训：从给定 adapter 权重【初始化】、起全新 optimizer/scheduler（RFT 是在新数据 CS_RFT_TRAIN 上、
# 用新 lr/epochs 的新训练，只想从冷启动权重出发）。用 --adapters（与 dpo.sh/grpo.sh 一致）。
# ⚠️ 不用 --resume_from_checkpoint：那是"恢复同一次训练"，会还原 scheduler/global_step 与新数据长度不匹配。
RESUME_ARG=()
if [ -n "$RESUME" ]; then RESUME_ARG=(--adapters "$RESUME"); fi

echo "[sft.sh] model=$V1_DIR train=$TRAIN val=$VAL out=$OUT lr=$LR epochs=$EPOCHS gpus=$TRAIN_GPUS nproc=$NPROC resume=${RESUME:-none}"

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NPROC_PER_NODE="$NPROC" \
"$SWIFT" sft \
  --model "$V1_DIR" \
  --model_type "${V1_MODEL_TYPE:-qwen2}" \
  --template "${V1_TEMPLATE:-qwen2_5}" \
  --train_type lora \
  --dataset "$TRAIN" \
  --val_dataset "$VAL" \
  --torch_dtype bfloat16 \
  --num_train_epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --save_total_limit 2 \
  --gradient_checkpointing true \
  --logging_steps 5 \
  --output_dir "$OUT" \
  "${RESUME_ARG[@]}"

echo done > "$OUT/.done"   # 仅训练成功(set -e 把关)才落完成标记，供 run.py has_adapter 判定
echo "[sft.sh] 完成 -> $OUT"
