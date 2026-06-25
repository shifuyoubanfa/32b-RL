#!/usr/bin/env bash
# ============ 一键启动：从数据构建一路跑到 GRPO 结束 ============
# 在服务器上运行；用训练环境 zhjg_rl 的 python，推理用 vllm_env（脚本内部起服务）。
# 全程 INFO 日志进 logs/pipeline.log；本脚本控制台输出再 tee 一份。
#
# 用法：
#   export DASHSCOPE_API_KEY=sk-...           # Kimi(DashScope) key
#   bash scripts/run_all.sh                    # 全链
#   bash scripts/run_all.sh --from rft         # 断点续跑（从某阶段）
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

# ---- 环境（可用 export 覆盖）----
export V1_DIR="${V1_DIR:-/home/nvme01/zhjg/V1-32B/checkpoint-1500}"
export VLLM_ENV="${VLLM_ENV:-/home/nvme02/biyh/vllm_env}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_LOG_DIR="${ZHJG_LOG_DIR:-$ZHJG_WORK_DIR/logs}"
mkdir -p "$ZHJG_LOG_DIR"
PYBIN="$ZHJG_ENV/bin/python"

# ---- 预检（缺一不可，省得跑一半崩）----
echo "===== 预检 ====="
[ -f "$V1_DIR/config.json" ]            || { echo "❌ V1 权重不存在: $V1_DIR"; exit 1; }
[ -x "$VLLM_ENV/bin/vllm" ]             || { echo "❌ vllm_env 不可用: $VLLM_ENV"; exit 1; }
[ -x "$PYBIN" ]                         || { echo "❌ 训练环境 python 不存在: $PYBIN"; exit 1; }
[ -n "${DASHSCOPE_API_KEY:-}" ]         || { echo "❌ 请先 export DASHSCOPE_API_KEY=sk-..."; exit 1; }
[ -f "$CODE_ROOT/data/00_data_teacher_outputs.jsonl" ] \
    || { echo "❌ 缺缓存数据 data/00_data_teacher_outputs.jsonl（从 14B output/ 拷入）"; exit 1; }
nvidia-smi -L >/dev/null 2>&1           || { echo "❌ 看不到 GPU"; exit 1; }
[ -x "$ZHJG_ENV/bin/swift" ] || { echo "❌ swift 命令不可用（训练全靠它）: $ZHJG_ENV/bin/swift"; exit 1; }
CLEAN_BASE="${CLEAN_BASE_DIR:-/mnt/pfs/model/Qwen2.5-32B-Instruct}"
[ -f "$CLEAN_BASE/config.json" ] || echo "⚠️ 探针 clean_base 尺子缺失: $CLEAN_BASE（阶段3 才用；可设 CLEAN_BASE_DIR 或届时补）"
echo "✅ 预检通过：V1/vllm_env/zhjg_rl/swift/DASHSCOPE_API_KEY/缓存数据/GPU 均就绪"

echo "===== 一键启动全链（数据→冷启动→探针→RFT→DPO→GRPO）。日志：$ZHJG_LOG_DIR/pipeline.log ====="
echo "===== 另开一窗实时监督：bash scripts/monitor.sh ====="
"$PYBIN" -X utf8 run.py "$@" 2>&1 | tee -a "$ZHJG_LOG_DIR/run_all.console.log"
echo "===== 全链结束。最终模型：$ZHJG_WORK_DIR/ckpts/v1-32b-grpo-lora ；报告见 output/*_report.md ====="
