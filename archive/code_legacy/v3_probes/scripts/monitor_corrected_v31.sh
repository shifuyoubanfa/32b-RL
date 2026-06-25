#!/usr/bin/env bash
# Clean monitor for corrected-v3.1 derag diagnostics.
set -u

WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_DIR="${ZHJG_LOG_DIR:-$WORK/logs}"
RUN_ID="${V31_RUN_ID:-}"
if [ -z "$RUN_ID" ]; then
  RUN_ID="$(ls -1 "$LOG_DIR/corrected_v31" 2>/dev/null | tail -1)"
fi
RAW_DIR="${CORRECTED_V31_LOG_DIR:-$LOG_DIR/corrected_v31/$RUN_ID}"
STATE="$RAW_DIR/state.json"
OUT_DIR="${ZHJG_OUTPUT_DIR:-$WORK/output}/corrected_v31/$RUN_ID"
PY="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/python"

while true; do
  clear
  echo "================ $(date '+%F %T') corrected-v3.1 derag ================"
  echo "run_id: $RUN_ID"
  echo "--- state ---"
  if [ -f "$STATE" ]; then
    "$PY" - "$STATE" <<'PY'
import json, sys
s=json.load(open(sys.argv[1], encoding="utf-8"))
for k in ["status","stage","pid","started_at","updated_at","raw_log","summary"]:
    print(f"{k:10}: {s.get(k)}")
print("completed : " + " -> ".join(s.get("completed", [])))
PY
    CUR_LOG="$("$PY" -c "import json; print(json.load(open('$STATE')).get('raw_log') or '')" 2>/dev/null)"
  else
    echo "waiting for $STATE"
    CUR_LOG=""
  fi

  echo ""
  echo "--- recent events ---"
  tail -12 "$RAW_DIR/events.log" 2>/dev/null

  echo ""
  echo "--- current raw signal ---"
  if [ -n "$CUR_LOG" ] && [ -f "$CUR_LOG" ]; then
    grep -E "RESULT|PROGRESS|ERROR|Traceback|NO-GO|GO|PASS|FAIL" "$CUR_LOG" 2>/dev/null | tail -12
  else
    echo "no current raw signal"
  fi

  echo ""
  echo "--- artifacts ---"
  for f in \
    "$OUT_DIR/110_corrected_v31_derag_calibration.md" \
    "$OUT_DIR/111_corrected_v31_derag_paired_readout.md" \
    "$OUT_DIR/112_corrected_v31_trace_surgery_check.md" \
    "$OUT_DIR/113_corrected_v31_summary.md"; do
    [ -f "$f" ] || continue
    printf "%s : " "$(basename "$f")"
    grep -E "status:|verdict:|Δtrace_free|trace_free|Δgrounded|Δaccuracy|run_id|target" "$f" 2>/dev/null | head -2 | tr '\n' ' '
    echo
  done

  echo ""
  echo "Ctrl+C exit"
  sleep 10
done
