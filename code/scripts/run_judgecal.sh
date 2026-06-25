#!/usr/bin/env bash
# 页面1：judgecal 判官标定（离线、零 GPU、只调 Kimi）。
# 标定 Kimi"逐句换词复述识别"能力：实验一(该打几遍) + 实验二(reworded召回/legit_use误伤)。
# 数据 data/judgecal_sentences.jsonl 随代码带上（人工四类标签）。不 serve vLLM、不碰 GPU。
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$CODE_ROOT/config.py" ] || { [ -f "$CODE_ROOT/code/config.py" ] && CODE_ROOT="$CODE_ROOT/code"; }
cd "$CODE_ROOT"

export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_LOG_DIR="${ZHJG_LOG_DIR:-$ZHJG_WORK_DIR/logs}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export JUDGECAL_RUN_ID="${JUDGECAL_RUN_ID:-main}"
export JUDGECAL_LOG_DIR="${JUDGECAL_LOG_DIR:-$ZHJG_LOG_DIR/judgecal/$JUDGECAL_RUN_ID}"
export ZHJG_CONSOLE_LOG_LEVEL="${ZHJG_CONSOLE_LOG_LEVEL:-WARNING}"

# 标定参数（都有默认值，按需 export）
export JUDGECAL_KMAX="${JUDGECAL_KMAX:-16}"        # 每条 think 让 Kimi 判几遍（一次高 k 采集）
export JUDGECAL_WORKERS="${JUDGECAL_WORKERS:-3}"   # Kimi 并发（DashScope 易 429，宁慢勿被限流）
export JUDGECAL_LIMIT="${JUDGECAL_LIMIT:-0}"       # 0=全量；>0 只判前 N 条(冒烟)

# Kimi(DashScope) key：与既有流水线同一把，明文默认；已 export 则以 export 为准。离开内网请轮换。
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-sk-REDACTED-ROTATE-ME}"

echo "===== judgecal 判官标定（离线、零 GPU、只调 Kimi）====="
echo "run_id=$JUDGECAL_RUN_ID  数据=data/judgecal_sentences.jsonl"
echo "参数 k_max=$JUDGECAL_KMAX workers=$JUDGECAL_WORKERS limit=$JUDGECAL_LIMIT"
echo "产物 output/judgecal/$JUDGECAL_RUN_ID/{160,161,162}*；监控另开一页："
echo "  bash scripts/monitor_judgecal.sh"
echo "日志=$JUDGECAL_LOG_DIR"
exec "$ZHJG_ENV/bin/python" -X utf8 run_judgecal.py
