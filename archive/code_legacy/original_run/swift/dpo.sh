#!/usr/bin/env bash
# DPO 偏好对齐：ms-swift rlhf --rlhf_type dpo，从 CS_RFT adapter 续训；πref=CS_RFT 冻结 adapter。
# 资源：训练用 GPU 2-7（vLLM 占 0,1）。
#
# 用法: bash swift/dpo.sh <pairs.jsonl> <cs_rft_adapter> <output_dir>
#
# ⚠️ swift 4.0.1 的 rlhf/dpo 精确旗标以 `swift rlhf --help` 为准，上线前 smoke 一轮确认。
set -euo pipefail

PAIRS="$1"; INIT_ADAPTER="$2"; OUT="$3"
REF_ADAPTER="${DPO_REF_ADAPTER:-$INIT_ADAPTER}"
V1_DIR="${V1_DIR:-/home/nvme01/zhjg/V1-32B/checkpoint-1500}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3,4,5,6,7}"
NPROC="$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)"
SWIFT="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/swift"   # 全路径，避免 (base) 下 swift 不在 PATH
DPO_BETA="${DPO_BETA:-0.1}"
DPO_LR="${DPO_LR:-5e-6}"
DPO_EPOCHS="${DPO_EPOCHS:-1}"
DPO_MAX_LEN="${DPO_MAX_LEN:-4096}"
# DPO 显存比 SFT 重（chosen+rejected 两序列 × policy+reference 两前向）。首次 DPO 在 step0 即 OOM。
# 先靠下面的 expandable_segments 回收碎片(实测有 3.48GB 碎片，足够补上缺的 3.92GB)；纯 DDP 默认即可。
# 若仍 OOM，设环境变量 DEEPSPEED=zero3 把 32B 分片到 6 卡(每卡只放 1/6 权重)，腾出大量显存——这是兜底大杀器(略慢)。
DEEPSPEED="${DEEPSPEED:-}"

# 减显存碎片（grpo.sh 已有、dpo.sh 之前漏了）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DS_ARG=""
[ -n "$DEEPSPEED" ] && DS_ARG="--deepspeed $DEEPSPEED"
echo "[dpo.sh] model=$V1_DIR init_adapter=$INIT_ADAPTER ref_adapter=$REF_ADAPTER pairs=$PAIRS out=$OUT beta=$DPO_BETA gpus=$TRAIN_GPUS deepspeed=${DEEPSPEED:-off} max_len=$DPO_MAX_LEN"

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NPROC_PER_NODE="$NPROC" \
"$SWIFT" rlhf \
  --rlhf_type dpo \
  --model "$V1_DIR" \
  --model_type "${V1_MODEL_TYPE:-qwen2}" \
  --template "${V1_TEMPLATE:-qwen2_5}" \
  --adapters "$INIT_ADAPTER" \
  --ref_adapters "$REF_ADAPTER" \
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
  $DS_ARG \
  --logging_steps 5 \
  --save_strategy epoch --save_total_limit 2 \
  --output_dir "$OUT"

echo done > "$OUT/.done"   # 仅训练成功(set -e 把关)才落完成标记，供 run.py has_adapter 判定
echo "[dpo.sh] 完成 -> $OUT"
