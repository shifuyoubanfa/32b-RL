#!/usr/bin/env bash
# 页面2：监控 derag_v5 X 重测（状态 / 进度 / 产物 / 规则X vs Kimi真X 对照）。离线无 GPU，不看显存。
set -u

WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_DIR="${ZHJG_LOG_DIR:-$WORK/logs}"
RUN_ID="${V5_RUN_ID:-main}"
RAW_DIR="${V5X_LOG_DIR:-$LOG_DIR/derag_v5_xrecheck/$RUN_ID}"
STATE="$RAW_DIR/state.json"
OUT_DIR="${ZHJG_OUTPUT_DIR:-$WORK/output}/derag_v5_probe/$RUN_ID"
PY="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/python"
INTERVAL="${V5X_MONITOR_INTERVAL:-8}"

lc() { [ -f "$1" ] && wc -l < "$1" 2>/dev/null | tr -d ' ' || echo "-"; }

while true; do
  clear
  echo "================= $(date '+%F %T') derag_v5 X 重测（规则X vs Kimi真X）================="
  echo "run_id: $RUN_ID   刷新间隔: ${INTERVAL}s   （离线，纯 Kimi API，无 GPU）"

  echo ""
  echo "--- 阶段状态 (state) ---"
  if [ -f "$STATE" ]; then
    "$PY" - "$STATE" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
for k in ["status", "stage", "pid", "started_at", "updated_at"]:
    print(f"  {k:11}: {s.get(k)}")
print("  completed  : " + (" -> ".join(s.get("completed", [])) or "(无)"))
PY
    CUR_LOG="$("$PY" -c "import json;print(json.load(open('$STATE')).get('raw_log') or '')" 2>/dev/null)"
  else
    echo "  等待 $STATE 出现（重测刚启动）"; CUR_LOG=""
  fi

  echo ""
  echo "--- 产物行数 ---"
  printf "  153b 逐题(Kimi重判): %s 题\n" "$(lc "$OUT_DIR/153b_kimi_headroom.jsonl")"

  echo ""
  echo "--- 最近事件 (events) ---"
  tail -10 "$RAW_DIR/events.log" 2>/dev/null | sed 's/^/  /'

  echo ""
  echo "--- 当前进度/原始信号 ---"
  if [ -n "$CUR_LOG" ] && [ -f "$CUR_LOG" ]; then
    grep -E "RESULT|PROGRESS|ERROR|Traceback|it/s|%" "$CUR_LOG" 2>/dev/null | tail -10 | sed 's/^/  /'
  else
    echo "  无当前原始信号"
  fi

  echo ""
  echo "--- 结论：规则X vs Kimi真X（跑完才有）---"
  if [ -f "$OUT_DIR/153b_kimi_headroom.json" ]; then
    "$PY" - "$OUT_DIR/153b_kimi_headroom.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
for k in ("all", "eval", "train"):
    d = s.get(k, {})
    print(f"  {k:5}: 病题{d.get('n_problems')}  X_rule(虚高)={d.get('X_rule')}  X_kimi(真)={d.get('X_kimi')}")
print(f"  参数: {s.get('params')}")
PY
  else
    echo "  未生成（重测未跑完）"
  fi

  echo ""
  echo "Ctrl+C 退出监控（不影响页面1运行）"
  sleep "$INTERVAL"
done
