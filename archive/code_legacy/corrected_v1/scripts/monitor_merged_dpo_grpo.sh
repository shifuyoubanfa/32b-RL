#!/usr/bin/env bash
# Clean monitor for the corrected merged-base chain.
set -u

WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_DIR="${ZHJG_LOG_DIR:-$WORK/logs}"
RAW_DIR="$LOG_DIR/merged_dpo_grpo"
STATE="$RAW_DIR/state.json"
OUT_DIR="${ZHJG_OUTPUT_DIR:-$WORK/output}"
PY="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/python"

while true; do
  clear
  echo "================ $(date '+%F %T')  MERGED DPO→GRPO ================"
  echo "--- 当前阶段 ---"
  if [ -f "$STATE" ]; then
    "$PY" - "$STATE" <<'PY'
import json, sys
s=json.load(open(sys.argv[1], encoding="utf-8"))
print(f"status : {s.get('status')}")
print(f"stage  : {s.get('stage')}")
print(f"pid    : {s.get('pid')}")
print(f"started: {s.get('started_at')}")
print(f"updated: {s.get('updated_at')}")
print("done   : " + " -> ".join(s.get("completed", [])))
print(f"log    : {s.get('raw_log')}")
PY
    CUR_LOG="$("$PY" -c "import json; print(json.load(open('$STATE')).get('raw_log') or '')" 2>/dev/null)"
  else
    echo "等待流水线创建状态文件：$STATE"
    CUR_LOG=""
  fi

  echo ""
  echo "--- 最近流水线事件 ---"
  tail -8 "$RAW_DIR/events.log" 2>/dev/null

  echo ""
  echo "--- 最新训练/合并信号 ---"
  if [ -n "$CUR_LOG" ] && [ -f "$CUR_LOG" ]; then
    grep -E "Train:|global_step|max_steps|'loss'|'reward'|'kl'|Capturing CUDA graphs|Application startup complete|\[merge\]|\[dpo-on-model\]|\[grpo-on-model\]|ERROR|FAIL|Traceback|complete" "$CUR_LOG" 2>/dev/null | tail -8
  else
    echo "当前阶段尚无训练/合并信号"
  fi

  echo ""
  echo "--- 相关进程 ---"
  pgrep -af 'run_merged_dpo_grpo|swift/cli/rlhf.py|torch.distributed.run|swift export|vllm serve' 2>/dev/null \
    | sed 's/[[:space:]][[:space:]]*/ /g' | cut -c1-180 | head -12

  echo ""
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{printf "GPU%-2s util=%3s%% mem=%6s/%s MiB\n",$1,$2,$3,$4}'

  echo ""
  echo "--- 已生成报告 ---"
  for f in "$OUT_DIR"/8[0-4]_*.md; do
    [ -f "$f" ] || continue
    printf "%s : " "$(basename "$f")"
    grep -E "humanness 均值|^\| RFT merged|^\| DPO on|^\| GRPO from" "$f" 2>/dev/null | head -1 | tr '\n' ' '
    echo
  done
  echo ""
  echo "Ctrl+C 退出监控"
  sleep 10
done
