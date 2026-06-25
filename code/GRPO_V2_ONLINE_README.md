# V2 DPO → 在线 GRPO 跑法

当前推荐起点是已经评测完成的 DPO merged：

```bash
/home/nvme01/zhjg/models/v2-dpo-2sigma-2s-2s-merged
```

GRPO 使用旧的已跑通 ms-swift colocate 链路，不重造训练器：

- `swift/grpo_on_model.sh`
- `GRPO_ENV=/home/nvme02/conda/grpo_env`
- `--rlhf_type grpo`
- `--use_vllm true --vllm_mode colocate`
- `--vllm_tensor_parallel_size 8`
- `--vllm_gpu_memory_utilization 0.5`
- `--move_model_batches 16`
- `--offload_model true --offload_optimizer true`
- `--deepspeed zero3`

## 一个终端启动

```bash
cd /mnt/pfs/zhjg/code || cd /home/nvme01/zhjg/code || exit 1

export ZHJG_WORK_DIR=/home/nvme01/zhjg
export ZHJG_ENV=/home/nvme02/conda/zhjg_rl
export GRPO_ENV=/home/nvme02/conda/grpo_env
export VLLM_ENV=/home/nvme02/biyh/vllm_env

export V2_TAG=derag2
export V2_GRPO_LINEAGE=2s-2s-2s
export V2_GRPO_BASE_MODEL=/home/nvme01/zhjg/models/v2-dpo-2sigma-2s-2s-merged

export GRPO_GPUS=0,1,2,3,4,5,6,7
export VLLM_TP=8
export VLLM_GPU_UTIL=0.5
export VLLM_MAX_LEN=6144

export V2_GRPO_REWARD_AUDIT=1
export V2_GRPO_KIMI_SMOKE=1
export V2_GRPO_SWIFT_SMOKE=1
export V2_GRPO_WARMUP_STEPS=30
export V2_GRPO_MAIN_STEPS=90
export GRPO_K=8
export GRPO_V2_KIMI_K=2
export GRPO_V2_KIMI_LOCK=1
export KIMI_CACHE_MIN_K=2

bash scripts/run_v2_grpo_online.sh
```

默认产物不会覆盖 DPO：

- 数据：`/home/nvme01/zhjg/output/derag2/70_grpo_data.v2-2s-2s-2s.jsonl`
- 日志：`/home/nvme01/zhjg/logs/v2_grpo_online/`
- Kimi smoke：`/home/nvme01/zhjg/logs/v2_grpo_online/kimi_smoke.ok`
- smoke LoRA：`/home/nvme01/zhjg/ckpts/v2-grpo-smoke-2s-2s-2s-lora`
- warmup LoRA/merged：`v2-grpo-warmup-2s-2s-2s-*`
- final LoRA/merged：`v2-grpo-2sigma-2s-2s-2s-*`
- 评测 tag：`v2-2s-2s-2s-grpo`

## 另一个终端监控

```bash
cd /mnt/pfs/zhjg/code || cd /home/nvme01/zhjg/code || exit 1
export ZHJG_WORK_DIR=/home/nvme01/zhjg
export ZHJG_ENV=/home/nvme02/conda/zhjg_rl
export V2_TAG=derag2
export V2_GRPO_LINEAGE=2s-2s-2s
bash scripts/monitor_v2_grpo_online.sh
```

## 训练逻辑

1. 先构造 train-only GRPO 数据，不包含冻结 500 验证题。
2. 跑 reward audit：检查 `user_prompt` 和 `v1_answers_json`，并验证 reward 顺序。
3. 跑 Kimi smoke：确认在线 reward/eval 用的 Kimi API 在占 GPU 前可用。
4. 跑 2 step swift smoke：确认 grpo_env、vLLM colocate、reward 插件、数据列都能通。
5. 短规则 warmup：默认 30 steps，只推格式、answer pool、规则 think。
6. 在线 Kimi-GRPO：默认 90 steps，每组采样 8 个，answer/format 是硬门，Kimi k=2 只在安全区内做 think 排序。
7. 合并最终 LoRA，然后按冻结 500 题跑原三分评测。

不要把 `V2_GRPO_SWIFT_SMOKE=0` 作为默认，除非你已经在同一环境里确认过 swift GRPO 能启动。
