#!/usr/bin/env bash
# Clean monitor for corrected-v3 diagnostics.
set -u

WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_DIR="${ZHJG_LOG_DIR:-$WORK/logs}"
RUN_ID="${V3_RUN_ID:-}"
if [ -z "$RUN_ID" ]; then
  RUN_ID="$(ls -1 "$LOG_DIR/corrected_v3" 2>/dev/null | tail -1)"
fi
RAW_DIR="${CORRECTED_V3_LOG_DIR:-$LOG_DIR/corrected_v3/$RUN_ID}"
STATE="$RAW_DIR/state.json"
OUT_DIR="${ZHJG_OUTPUT_DIR:-$WORK/output}/corrected_v3/$RUN_ID"
PY="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/python"

while true; do
  clear
  echo "================ $(date '+%F %T') corrected-v3 diagnostics ================"
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
    grep -E "RESULT|PROGRESS|ERROR|Traceback|GOODHART|NO-GO|GO_|BIAS|PASS" "$CUR_LOG" 2>/dev/null | tail -12
  else
    echo "no current raw signal"
  fi

  echo ""
  echo "--- artifacts ---"
  for f in \
    "$OUT_DIR/100_corrected_v3_noise_calibration.md" \
    "$OUT_DIR/101_corrected_v3_mini_paired_readout.md" \
    "$OUT_DIR/102_corrected_v3_placebo_report.md" \
    "$OUT_DIR/103_corrected_v3_rewrite_probe.md" \
    "$OUT_DIR/104_corrected_v3_pairwise_controls.md" \
    "$OUT_DIR/105_corrected_v3_summary.md"; do
    [ -f "$f" ] || continue
    printf "%s : " "$(basename "$f")"
    grep -E "status:|verdict:|Δh|mean Δh|rewritten mean h|identical non-tie|short win|run_id" "$f" 2>/dev/null | head -2 | tr '\n' ' '
    echo
  done

  echo ""
  echo "Ctrl+C exit"
  sleep 10
done
