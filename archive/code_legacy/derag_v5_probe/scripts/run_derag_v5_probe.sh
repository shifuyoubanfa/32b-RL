#!/usr/bin/env bash
# 页面1：跑 derag_v5 headroom 探针（不训练，只采样+判定）。
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# 兜底：若脚本不在 code/ 内而在其上一级，定位真正含 config.py 的代码根
[ -f "$CODE_ROOT/config.py" ] || { [ -f "$CODE_ROOT/code/config.py" ] && CODE_ROOT="$CODE_ROOT/code"; }
cd "$CODE_ROOT"

export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_LOG_DIR="${ZHJG_LOG_DIR:-$ZHJG_WORK_DIR/logs}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
# run_id 顶死为固定值 main：run 和 monitor 都默认用它，不必每次复制时间戳。
# 想从头重跑（改了挑题阈值等）：先 rm -rf $output/derag_v5_probe/main $logs/derag_v5_probe/main。
export V5_RUN_ID="${V5_RUN_ID:-main}"
export V5_LOG_DIR="${V5_LOG_DIR:-$ZHJG_LOG_DIR/derag_v5_probe/$V5_RUN_ID}"
export ZHJG_CONSOLE_LOG_LEVEL="${ZHJG_CONSOLE_LOG_LEVEL:-WARNING}"

# 模型权重（按需改）：RFT-merged 是被测的当前模型；V1 是答案金标准
export V5_RFT_MERGED_DIR="${V5_RFT_MERGED_DIR:-/home/nvme01/zhjg/models/v1-32b-corrected-v1-rft-merged}"
export V5_V1_DIR="${V5_V1_DIR:-/home/nvme01/zhjg/V1-32B/checkpoint-1500}"
export V5_VLLM_GPUS="${V5_VLLM_GPUS:-0,1}"
# 规模（卡不要钱可放大）：扫多少训练题找病题 / RFT每题采样数 / V1每题采样数
export V5_TRAIN_CAP="${V5_TRAIN_CAP:-1000}"
export V5_RFT_K="${V5_RFT_K:-16}"
export V5_V1_N="${V5_V1_N:-8}"
# 挑病题方式：kimi=用 Kimi DERAG 判(能看见结构性念手册，默认) / rule=确定性规则(只抓字面词，会漏)
export V5_DETECT="${V5_DETECT:-kimi}"
export V5_DETECT_K="${V5_DETECT_K:-2}"          # 每题判几次取均值降噪
export V5_PROBLEM_TF="${V5_PROBLEM_TF:-0.70}"   # trace_free 低于此 或 有结构痕迹 = 病题

# Kimi(DashScope) key：明文默认值写死在此（探针自己的配置入口），不必每次 export；
# 若已 export DASHSCOPE_API_KEY 则以 export 的为准。注意：明文在文件里，仓库离开内网环境请轮换此 key。
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-sk-REDACTED-ROTATE-ME}"

echo "===== derag_v5 headroom probe ====="
echo "run_id=$V5_RUN_ID"
echo "RFT(被测当前模型)=$V5_RFT_MERGED_DIR"
echo "V1(答案金标准)   =$V5_V1_DIR"
echo "挑病题=$V5_DETECT(k=$V5_DETECT_K, tf<$V5_PROBLEM_TF)  扫训练题上限=$V5_TRAIN_CAP"
echo "RFT自采K=$V5_RFT_K  V1采样N=$V5_V1_N  GPU=$V5_VLLM_GPUS"
echo "不训练，只采样+判定。监控另开一页（run_id 已顶死=main，直接跑即可，不用复制）："
echo "  bash scripts/monitor_derag_v5_probe.sh"
echo "日志=$V5_LOG_DIR"
exec "$ZHJG_ENV/bin/python" -X utf8 run_derag_v5_probe.py
