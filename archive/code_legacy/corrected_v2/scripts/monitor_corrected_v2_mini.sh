#!/usr/bin/env bash
# Clean monitor for corrected-v2 mini.
set -u

WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_DIR="${ZHJG_LOG_DIR:-$WORK/logs}"
RAW_DIR="${CORRECTED_V2_LOG_DIR:-$LOG_DIR/corrected_v2}"
STATE="$RAW_DIR/state.json"
OUT_DIR="${ZHJG_OUTPUT_DIR:-$WORK/output}"
PY="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/python"

while true; do
  clear
  echo "================ $(date '+%F %T')  corrected-v2 mini ================"
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
print(f"summary: {s.get('summary')}")
PY
    CUR_LOG="$("$PY" -c "import json; print(json.load(open('$STATE')).get('raw_log') or '')" 2>/dev/null)"
  else
    echo "等待状态文件：$STATE"
    CUR_LOG=""
  fi

  echo ""
  echo "--- 最近事件 ---"
  tail -10 "$RAW_DIR/events.log" 2>/dev/null

  echo ""
  echo "--- 当前 raw 关键信号 ---"
  if [ -n "$CUR_LOG" ] && [ -f "$CUR_LOG" ]; then
    grep -E "RESULT|PROGRESS|\[dpo-v2\]|Train:|global_step|max_steps|loss|rewards/chosen|rewards/rejected|Application startup complete|ERROR|FAIL|Traceback|complete" "$CUR_LOG" 2>/dev/null | tail -10
  else
    echo "当前阶段尚无 raw 信号"
  fi

  echo ""
  echo "--- 相关进程 ---"
  pgrep -af 'run_corrected_v2_mini|step93_kimi|step94_build|step95_rejudge|step95b_filter|swift/cli/rlhf.py|torch.distributed.run|vllm serve' 2>/dev/null \
    | sed 's/[[:space:]][[:space:]]*/ /g' | cut -c1-180 | head -12

  echo ""
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{printf "GPU%-2s util=%3s%% mem=%6s/%s MiB\n",$1,$2,$3,$4}'

  echo ""
  echo "--- 关键产物 ---"
  for f in \
    "$OUT_DIR/94_corrected_v2_pair_report.md" \
    "$OUT_DIR/95b_corrected_v2_pair_rejudge_all_report.md" \
    "$OUT_DIR/95b_corrected_v2_stable_pair_report.md" \
    "$OUT_DIR/96_corrected_v2_mini_dpo_report.md" \
    "$OUT_DIR/97_corrected_v2_mini_summary.md"; do
    [ -f "$f" ] || continue
    printf "%s : " "$(basename "$f")"
    grep -E "status:|train pairs|heldout pairs|direction kept|stable pairs|mini DPO|\| corrected-v2 mini DPO|\| RFT merged base|RESULT|Δh" "$f" 2>/dev/null | head -2 | tr '\n' ' '
    echo
  done

  echo ""
  echo "Ctrl+C 退出监控"
  sleep 10
done
