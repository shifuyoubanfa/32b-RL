#!/usr/bin/env bash
# 页面2：监控 judgecal 判官标定（状态 / 进度 / 产物 / 实验一二结论）。离线无 GPU，不看显存。
set -u

WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_DIR="${ZHJG_LOG_DIR:-$WORK/logs}"
RUN_ID="${JUDGECAL_RUN_ID:-main}"
RAW_DIR="${JUDGECAL_LOG_DIR:-$LOG_DIR/judgecal/$RUN_ID}"
STATE="$RAW_DIR/state.json"
OUT_DIR="${ZHJG_OUTPUT_DIR:-$WORK/output}/judgecal/$RUN_ID"
PY="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/python"
INTERVAL="${JUDGECAL_MONITOR_INTERVAL:-8}"

lc() { [ -f "$1" ] && wc -l < "$1" 2>/dev/null | tr -d ' ' || echo "-"; }

while true; do
  clear
  echo "================= $(date '+%F %T') judgecal 判官标定 ================="
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
    echo "  等待 $STATE 出现（标定刚启动）"; CUR_LOG=""
  fi

  echo ""
  echo "--- 产物行数 ---"
  printf "  160 装配后的 think 条数 : %s\n" "$(lc "$OUT_DIR/160_judgecal_items.jsonl")"
  printf "  161 已采集(判16遍) 条数 : %s\n" "$(lc "$OUT_DIR/161_sentence_judges.jsonl")"

  echo ""
  echo "--- 最近事件 (events) ---"
  tail -10 "$RAW_DIR/events.log" 2>/dev/null | sed 's/^/  /'

  echo ""
  echo "--- 当前进度/原始信号 ---"
  if [ -n "$CUR_LOG" ] && [ -f "$CUR_LOG" ]; then
    grep -E "RESULT|PROGRESS|ERROR|Traceback" "$CUR_LOG" 2>/dev/null | tail -10 | sed 's/^/  /'
  else
    echo "  无当前原始信号"
  fi

  echo ""
  echo "--- 结论：实验一(稳定遍数K) + 实验二(召回/误伤)（跑完才有）---"
  if [ -f "$OUT_DIR/162_judgecal_decision.json" ]; then
    "$PY" - "$OUT_DIR/162_judgecal_decision.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"  稳定遍数 K(四类最难)   : {s.get('chosen_K')}  (all_stable={s.get('all_stable')})")
print(f"  reworded 召回(头条)    : {s.get('reworded_recall')}")
print(f"  legit_use 误伤(头条)   : {s.get('legit_fp')}")
print(f"  判决                   : {'GO Kimi可当判官' if s.get('go') else 'NO-GO'}")
print(f"  误伤句数                : {s.get('n_false_positive_sentences')}")
PY
  else
    echo "  未生成（标定未跑完）"
  fi

  echo ""
  echo "完整报告：$OUT_DIR/162_judgecal_report.md"
  echo "Ctrl+C 退出监控（不影响页面1运行）"
  sleep "$INTERVAL"
done
