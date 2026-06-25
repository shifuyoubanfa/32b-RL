#!/usr/bin/env bash
# 页面1：derag_v5 X 重测（离线、零 GPU）。把 s153 的"判 think 干净"从规则换成 Kimi，得到真 X_kimi vs 规则 X。
# 前提：run_id 对应的探针已跑过（151_rft_samples + 152_v1_support 在 output/derag_v5_probe/$run_id/）。
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$CODE_ROOT/config.py" ] || { [ -f "$CODE_ROOT/code/config.py" ] && CODE_ROOT="$CODE_ROOT/code"; }
cd "$CODE_ROOT"

export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_LOG_DIR="${ZHJG_LOG_DIR:-$ZHJG_WORK_DIR/logs}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
# run_id 默认 main，与探针一致（读它的 151/152，结果写回同目录）。
export V5_RUN_ID="${V5_RUN_ID:-main}"
export V5X_LOG_DIR="${V5X_LOG_DIR:-$ZHJG_LOG_DIR/derag_v5_xrecheck/$V5_RUN_ID}"
export ZHJG_CONSOLE_LOG_LEVEL="${ZHJG_CONSOLE_LOG_LEVEL:-WARNING}"

# Kimi 重判参数（都有默认值，按需 export）
export V5X_K="${V5X_K:-2}"                # 每条 in-support 样本判几次取均值降噪
export V5X_TF_CLEAN="${V5X_TF_CLEAN:-0.70}"  # trace_free≥此 且 无结构痕迹 = 干净
export V5X_CAP="${V5X_CAP:-16}"           # 每题最多判几条 in-support 样本（早停于首个干净，cap 仅全脏时生效）

# Kimi(DashScope) key：与探针同一把，明文默认写死；已 export 则以 export 为准。离开内网请轮换。
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-sk-REDACTED-ROTATE-ME}"

echo "===== derag_v5 X 重测（规则X vs Kimi真X，离线零GPU）====="
echo "run_id=$V5_RUN_ID  读 output/derag_v5_probe/$V5_RUN_ID/{151,152}"
echo "参数 k=$V5X_K tf_clean=$V5X_TF_CLEAN cap=$V5X_CAP"
echo "不serve vLLM、不碰GPU、不重新生成采样。监控另开一页："
echo "  bash scripts/monitor_derag_v5_xrecheck.sh"
echo "日志=$V5X_LOG_DIR"
exec "$ZHJG_ENV/bin/python" -X utf8 run_derag_v5_xrecheck.py
