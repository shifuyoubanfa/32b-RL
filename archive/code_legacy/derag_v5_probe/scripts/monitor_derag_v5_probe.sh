#!/usr/bin/env bash
# 页面2：详细监控 derag_v5 探针（状态 / 进度 / GPU / 产物行数 / 原始信号 / 结论）。
set -u

WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_DIR="${ZHJG_LOG_DIR:-$WORK/logs}"
# run_id 顶死为 main（与 run 脚本一致），监控直接 bash 即可，不必传 V5_RUN_ID
RUN_ID="${V5_RUN_ID:-main}"
RAW_DIR="${V5_LOG_DIR:-$LOG_DIR/derag_v5_probe/$RUN_ID}"
STATE="$RAW_DIR/state.json"
OUT_DIR="${ZHJG_OUTPUT_DIR:-$WORK/output}/derag_v5_probe/$RUN_ID"
PY="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/python"
INTERVAL="${V5_MONITOR_INTERVAL:-8}"

lc() { [ -f "$1" ] && wc -l < "$1" 2>/dev/null | tr -d ' ' || echo "-"; }

while true; do
  clear
  echo "================= $(date '+%F %T') derag_v5 headroom probe ================="
  echo "run_id: $RUN_ID   刷新间隔: ${INTERVAL}s"

  echo ""
  echo "--- 阶段状态 (state) ---"
  if [ -f "$STATE" ]; then
    "$PY" - "$STATE" <<'PY'
import json, sys, datetime
s = json.load(open(sys.argv[1], encoding="utf-8"))
for k in ["status", "stage", "pid", "started_at", "updated_at"]:
    print(f"  {k:11}: {s.get(k)}")
print("  completed  : " + " -> ".join(s.get("completed", [])) or "  completed  :")
PY
    CUR_LOG="$("$PY" -c "import json;print(json.load(open('$STATE')).get('raw_log') or '')" 2>/dev/null)"
  else
    echo "  等待 $STATE 出现（probe 刚启动）"; CUR_LOG=""
  fi

  echo ""
  echo "--- GPU (0,1) ---"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
    | awk -F', ' '$1<=1{printf "  GPU%s  显存 %s / %s  利用率 %s\n",$1,$2,$3,$4}'

  echo ""
  echo "--- 产物行数（随步骤增长）---"
  printf "  150 病题(全量打分): %s 行 | 病题: %s 行\n" "$(lc "$OUT_DIR/150_problems.all.jsonl")" "$(lc "$OUT_DIR/150_problems.jsonl")"
  printf "  151 RFT自采样: %s 题 | 152 V1答案库: %s 题\n" "$(lc "$OUT_DIR/151_rft_samples.jsonl")" "$(lc "$OUT_DIR/152_v1_support.jsonl")"
  printf "  153 自救率明细: %s 题 | 154 改写明细: %s 题\n" "$(lc "$OUT_DIR/153_rft_headroom.jsonl")" "$(lc "$OUT_DIR/154_rewrite_headroom.jsonl")"

  echo ""
  echo "--- 最近事件 (events) ---"
  tail -12 "$RAW_DIR/events.log" 2>/dev/null | sed 's/^/  /'

  echo ""
  echo "--- 当前步骤进度/原始信号 ---"
  if [ -n "$CUR_LOG" ] && [ -f "$CUR_LOG" ]; then
    grep -E "RESULT|PROGRESS|ERROR|Traceback|NO_GO|NO-GO|GO|PASS|FAIL|it/s|verdict" "$CUR_LOG" 2>/dev/null | tail -14 | sed 's/^/  /'
  else
    echo "  无当前原始信号"
  fi

  echo ""
  echo "--- 结论 (跑完才有) ---"
  if [ -f "$OUT_DIR/159_probe_summary.md" ]; then
    grep -E "verdict|X=|Y=|病题|自救率|改写成功率" "$OUT_DIR/159_probe_summary.md" 2>/dev/null | head -8 | sed 's/^/  /'
  else
    echo "  未生成（探针未跑完）"
  fi

  echo ""
  echo "Ctrl+C 退出监控（不影响页面1运行）"
  sleep "$INTERVAL"
done
