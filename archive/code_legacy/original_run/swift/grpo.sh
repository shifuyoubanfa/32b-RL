#!/usr/bin/env bash
# GRPO 在线强化：ms-swift rlhf --rlhf_type grpo。
# - 策略起点 = DPO adapter；swift 内置 vLLM 做 rollout，并【每步把更新的 LoRA 权重同步进 vLLM】保 on-policy。
# - reward = 我们的插件 humanness（两级门控 + 表面项；answer 段靠 KL 守准）。
# - 资源：GRPO 阶段【先停掉静态 V1 vLLM 服务】，swift 用全 8 卡（colocate：vLLM 与训练共卡，swift 自管权重同步）。
#
# 用法: bash swift/grpo.sh <grpo_data.jsonl> <dpo_adapter> <output_dir>
#
# ⚠️【非对称 KL 是 swift 盲区】swift GRPO 只有单标量 --beta，无 think/answer 分段 KL。
#    第一版策略①：用单 KL(取 think0.02 与 answer0.1 之间的 0.04) + 奖励漂移门 + πref 锚守准（见技术方案 §4.2 三策）。
#    若实测 acc 滑，再 patch swift 的 GRPO trainer 加分段 KL（策略②）。
# ⚠️ swift 4.0.1 的 grpo 旗标(--num_generations/--use_vllm/--vllm_mode/--beta 等)以 `swift rlhf --help` 为准，先 smoke 2 步。
set -euo pipefail

DATA="$1"; INIT_ADAPTER="$2"; OUT="$3"
# For LoRA GRPO, ms-swift disables the active adapter when --ref_adapters is
# absent. Default the frozen KL reference to the exact adapter we start from.
REF_ADAPTER="${GRPO_REF_ADAPTER:-$INIT_ADAPTER}"
CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V1_DIR="${V1_DIR:-/home/nvme01/zhjg/V1-32B/checkpoint-1500}"
GRPO_GPUS="${GRPO_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="$(echo "$GRPO_GPUS" | tr ',' '\n' | grep -c .)"
# GRPO colocate 需 swift+vLLM 同环境；GRPO_ENV 指向"克隆 vllm_env 再装 swift"的专用环境（不设则用 ZHJG_ENV）。
# 只有 GRPO 这一步用它；run.py / eval / 其它步骤仍用 zhjg_rl，互不影响。
GRPO_ENV="${GRPO_ENV:-${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}}"
SWIFT="$GRPO_ENV/bin/swift"
K="${GRPO_K:-8}"
STEPS="${GRPO_STEPS:-200}"
BETA="${GRPO_BETA:-0.04}"          # 单 KL（think0.02~answer0.1 之间）
LR="${GRPO_LR:-2e-6}"
TEMP="${GRPO_TEMPERATURE:-1.0}"
TOPP="${GRPO_TOP_P:-0.95}"
SMOKE="${GRPO_SMOKE:-0}"           # =1 时只跑 2 步冒烟看显存/插件/权重同步
[ "$SMOKE" = "1" ] && STEPS=2
# 生成引擎：USE_VLLM=true 用内置 vLLM colocate(快，但需 zhjg_rl 里装 vllm)；
#           USE_VLLM=false 用 HF 原生生成(慢，但零依赖、不需装 vllm)——B 方案。
USE_VLLM="${USE_VLLM:-true}"
DEEPSPEED="${DEEPSPEED:-}"          # 设 zero3 把 32B 分片省显存（use_vllm=false 走 DDP 多半需要，否则 65GB/卡 OOM）
if [ "$USE_VLLM" = "true" ]; then
  # colocate 标准配方：offload 在 vLLM 生成时把策略模型挪到 CPU、腾 GPU 给 vLLM 的 KV cache，
  #   否则 vLLM 报 "No available memory for the cache blocks"（策略模型一直占着 GPU，KV 没地方放）。
  # 配 zero3：每 rank 只持 1/8 分片(~8GB)，offload 到 CPU 总量≈64GB（远小于不分片的 520GB）→ 不再 CPU-OOM。
  # ★根治 "No available memory for cache blocks"：--vllm_tensor_parallel_size 默认=1 → 每卡各放整个 65GB 模型 → KV 没空间。
  #   设 TP=卡数(8) → vLLM 跨 8 卡分片、每卡只放 ~8GB 权重 → 腾出大量 KV 空间。这是官方 32B/72B 示例的关键参数。
  # VLLM_TP 默认=卡数；move_model_batches 削 ZeRO3→vLLM 同步峰值。
  # ★VLLM_MAX_LEN 必须 > max_length(4096) + max_completion_length(1024)：否则满长 prompt 占满窗口、
  #   生成空间 max_tokens = vllm_max_model_len - prompt = 0 → vLLM 报 "max_tokens must be at least 1, got 0"。
  #   默认 6144（=4096 prompt 上限 + 2048 生成余量）；TP 分片后 KV 够。
  GEN_ARGS="--use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size ${VLLM_TP:-$NPROC} --vllm_gpu_memory_utilization ${VLLM_GPU_UTIL:-0.5} --vllm_max_model_len ${VLLM_MAX_LEN:-6144} --move_model_batches 16 --sleep_level 1 --offload_model true --offload_optimizer true"
else
  GEN_ARGS="--use_vllm false"      # HF 原生生成；不传 vLLM/offload 专属旗标
  if [ -z "$DEEPSPEED" ]; then
    DEEPSPEED="zero3"              # 纯 DDP 下 65GB/卡必 OOM → 强制 zero3 分片，绝不静默走 DDP
    echo "[grpo.sh] ⚠️ use_vllm=false 且未设 DEEPSPEED → 自动用 zero3 分片防 OOM（注意：HF生成+zero3 会很慢）"
  fi
fi
DS_ARG=""; [ -n "$DEEPSPEED" ] && DS_ARG="--deepspeed $DEEPSPEED"

echo "[grpo.sh] data=$DATA init=$INIT_ADAPTER ref=$REF_ADAPTER out=$OUT K=$K steps=$STEPS beta=$BETA gpus=$GRPO_GPUS smoke=$SMOKE use_vllm=$USE_VLLM deepspeed=${DEEPSPEED:-off} pdbs=${GRPO_PDBS:-1} ga=${GRPO_GA:-8} max_completion=${GRPO_MAX_COMPLETION:-1024} gpu_util=${VLLM_GPU_UTIL:-0.5}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 关掉 NCCL RAS 健康监控子系统：新版 NCCL 的 RAS 线程在多卡训练【退出时会互相等待、死锁】
#   （8 rank 互相 reconnect→timeout→declare DEAD 死循环，进程永不退出）→ grpo.sh 不返回 → 整条 pipeline 卡死。
#   RAS 纯诊断功能、关掉不影响训练，但能让训完后干净退出（否则两基础顺序跑会在第一个训完就卡住）。
export NCCL_RAS_ENABLE=0

CUDA_VISIBLE_DEVICES="$GRPO_GPUS" NPROC_PER_NODE="$NPROC" \
"$SWIFT" rlhf \
  --rlhf_type grpo \
  --model "$V1_DIR" \
  --model_type "${V1_MODEL_TYPE:-qwen2}" \
  --template "${V1_TEMPLATE:-qwen2_5}" \
  --adapters "$INIT_ADAPTER" \
  --ref_adapters "$REF_ADAPTER" \
  --train_type lora \
  --dataset "$DATA" \
  --external_plugins "$CODE_ROOT/swift/grpo_reward_plugin.py" \
  --reward_funcs humanness \
  --num_generations "$K" \
  $GEN_ARGS \
  $DS_ARG \
  --scale_rewards group \
  --beta "$BETA" \
  --temperature "$TEMP" \
  --top_p "$TOPP" \
  --max_steps "$STEPS" \
  --learning_rate "$LR" \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max_length 4096 \
  --max_completion_length "${GRPO_MAX_COMPLETION:-1024}" \
  --per_device_train_batch_size "${GRPO_PDBS:-1}" \
  --gradient_accumulation_steps "${GRPO_GA:-8}" \
  --gradient_checkpointing true \
  --attn_impl sdpa \
  --logging_steps 1 \
  --save_steps 50 --save_total_limit 3 \
  --output_dir "$OUT"

echo done > "$OUT/.done"   # 仅训练成功(set -e 把关)才落完成标记，供 run.py has_adapter 判定
echo "[grpo.sh] 完成 -> $OUT"
