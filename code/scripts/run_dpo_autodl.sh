#!/usr/bin/env bash
# run_dpo_autodl.sh ─ AutoDL 2×A800 上跑 derag2 DPO + transformers 验收推理（不合并权重）
#
# 用法（从含 code/ 的父目录执行）：
#   bash code/scripts/run_dpo_autodl.sh <dpo_pairs.jsonl>
#
# 顺序：① Set env → ② DPO 训练 → ③ 找 adapter → ④ transformers 推理 → ⑤ 打印下载路径
# 2026-06-22 起不再启动 AutoDL vLLM：该环境的 vllm _C 是 cu13 build、torch 是 cu128，已验真不可用。
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# § 0  参数检查
# ─────────────────────────────────────────────────────────────────────────────
PAIRS="${1:?用法: bash code/scripts/run_dpo_autodl.sh <dpo_pairs.jsonl>}"

# ─────────────────────────────────────────────────────────────────────────────
# § 1  环境变量
#       ⚠ PATH 必须第一条 —— conda-pack shebang 是 #!/usr/bin/env python3.11，
#         env 在 PATH 外时 swift 会 exit127，被 2>/dev/null 吞掉变成假警报。
#       ⚠ 不用 source activate —— set -u 下 CONDA_PREFIX unbound 会崩。
# ─────────────────────────────────────────────────────────────────────────────
echo "[run-dpo] ① Set env"
export PATH="/root/envs/zhjg_rl/bin:$PATH"      # 最关键：python3.11/swift 从这里找
export ZHJG_ENV=/root/envs/zhjg_rl
export ZHJG_WORK_DIR=/root/autodl-tmp/dpo       # 旧默认 /home/nvme01/zhjg，import config 即建目录→系统盘爆
export ZHJG_LOG_DIR=/root/autodl-tmp/dpo/logs   # PID 文件写在这里，旧默认路径在系统盘
export TRAIN_GPUS=0,1                            # 旧默认 0,1,2,3,4,5,6,7 → CUDA OOM
export DEEPSPEED=zero3
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export DPO_AUTO_GA=0   # 禁止自动降 GA（pairs<400 时会把 GA 降成 1，等效 batch 从 16 缩到 2）
export DPO_GA=8        # 2 卡 × 1 bsz × GA=8 = 16，与原配方 8 卡 × GA=2 等效
export DPO_RPO_ALPHA=1.0      # swift 4.0.1 rlhf --help 命中 --rpo_alpha，保留；之前假警报是 exit127 被吞
export DPO_SAVE_TOTAL_LIMIT=2 # 省磁盘：只保留最新 2 个 checkpoint

# 固定路径（路线B 不合并权重，产物只有 LoRA adapter）
BASE_MODEL=/root/autodl-tmp/dpo/rft_merged
OUT_DPO=/root/autodl-tmp/dpo/out_dpo
EVAL_FILE=/root/autodl-tmp/dpo/pkg/00_data_v2_eval.jsonl
INFER_OUT=/root/autodl-tmp/dpo/output/v2-dpo-derag2_infer.jsonl
MODEL_NAME=v2-dpo-derag2  # ← 非 v1！system_for() 据此给去检索腔 COLDSTART_SYSTEM_PROMPT
                          #   若命名为 v1，system_for 会给 RAG 腔 prompt，评测结果全错！

mkdir -p "$ZHJG_LOG_DIR" /root/autodl-tmp/dpo/output

echo "    TRAIN_GPUS=$TRAIN_GPUS  DPO_GA=$DPO_GA  DPO_AUTO_GA=$DPO_AUTO_GA  DPO_RPO_ALPHA=$DPO_RPO_ALPHA"
echo "    BASE=$BASE_MODEL"
echo "    PAIRS=$PAIRS"
echo "    OUT_DPO=$OUT_DPO"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# § 2  DPO 训练（只产 LoRA adapter，几百 MB；不合并，不占 ~65G 额外磁盘）
# ─────────────────────────────────────────────────────────────────────────────
echo "[run-dpo] ② DPO 训练开始（2 卡 zero3，GA=8，等效 16 批）"
bash code/swift/dpo_v2.sh \
  "$PAIRS" \
  "$BASE_MODEL" \
  "$OUT_DPO"
echo "[run-dpo] ② DPO 训练完成 → $OUT_DPO"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# § 3  找最优 adapter（用 config.resolve_adapter，不自己 find+sort 实现）
#       优先 trainer_state.json 的 best_model_checkpoint，否则取最大 ckpt 号
# ─────────────────────────────────────────────────────────────────────────────
echo "[run-dpo] ③ 定位最优 adapter"
ADAPTER_PATH="$(python -X utf8 -c "
import sys, os
sys.path.insert(0, 'code')
os.environ.setdefault('ZHJG_WORK_DIR', '/root/autodl-tmp/dpo')
from config import resolve_adapter
print(resolve_adapter('${OUT_DPO}'))
")"
echo "[run-dpo] ③ adapter=$ADAPTER_PATH"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# § 4  transformers 推理（底座 + LoRA 不合并；默认按 query 断点续跑）
#       ⚠ --model_name 必须非 v1，确保使用去检索腔系统提示。
# ─────────────────────────────────────────────────────────────────────────────
echo "[run-dpo] ④ transformers 推理（model=$MODEL_NAME，去检索腔 prompt，贪心 temp=0）"
python -X utf8 code/pipeline/infer_hf.py \
  --base       "$BASE_MODEL" \
  --adapter    "$ADAPTER_PATH" \
  --model_name "$MODEL_NAME" \
  --eval_file  "$EVAL_FILE" \
  --out        "$INFER_OUT"
echo "[run-dpo] ④ 推理完成 → $INFER_OUT"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# § 5  打印下载清单 + 本地 Kimi 三分评测参考命令
# ─────────────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════"
echo "[run-dpo] ⑤ 全链完成！需要下载回本地的文件："
echo ""
echo "  [A] LoRA adapter（几百 MB）："
echo "      $ADAPTER_PATH"
echo ""
echo "  [B] 推理结果（几 MB）："
echo "      $INFER_OUT"
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "本地 Kimi 三分评测（下载后在本地服务器执行，从含 code/ 的父目录）："
echo ""
echo "  # 建议把 [B] 放到本地 runs/v2-dpo-derag2_infer.jsonl"
echo "  export V2_TAG=derag2"
echo ""
echo "  python code/pipeline/step_v2_eval.py \\"
echo "    --infer   runs/v2-dpo-derag2_infer.jsonl \\"
echo "    --scores  output/derag2/derag2_dpo_scores.jsonl \\"
echo "    --report  output/derag2/derag2_dpo_report.md \\"
echo "    --summary output/derag2/derag2_dpo_summary.json \\"
echo "    --support output/derag2/152_v1_support.v2.jsonl \\"
echo "    --tag     derag2_dpo"
echo ""
echo "══════════════════════════════════════════════════════════════════"
