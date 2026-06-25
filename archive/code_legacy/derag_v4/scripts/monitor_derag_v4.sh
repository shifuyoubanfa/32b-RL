#!/usr/bin/env bash
# Clean monitor for derag_v4.
set -u

WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_DIR="${ZHJG_LOG_DIR:-$WORK/logs}"
OUT_BASE="${ZHJG_OUTPUT_DIR:-$WORK/output}"
PY="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/python"
RUN_ID="${DERAG_V4_RUN_ID:-}"
if [ -z "$RUN_ID" ]; then
  RUN_ID="$(ls -1 "$LOG_DIR/derag_v4" 2>/dev/null | tail -1)"
fi

LOG_ROOT="$LOG_DIR/derag_v4/$RUN_ID"
STATE="$LOG_ROOT/state.json"
OUT_DIR="$OUT_BASE/derag_v4/$RUN_ID"

while true; do
  clear
  echo "================ $(date '+%F %T') derag_v4 ================"
  echo "run_id: $RUN_ID"
  echo ""
  echo "--- state ---"
  if [ -f "$STATE" ]; then
    "$PY" - "$STATE" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
for k in ["status", "stage", "pid", "started_at", "updated_at", "raw_log", "summary", "reason", "next_plan"]:
    v = s.get(k)
    if v:
        print(f"{k:12}: {v}")
print("completed   : " + " -> ".join(s.get("completed", [])))
print("selected_dpo: " + str(s.get("selected_dpo_variant")))
print("selected_grpo: " + str(s.get("selected_grpo_variant")))
PY
    CUR_LOG="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8')).get('raw_log') or '')" "$STATE" 2>/dev/null)"
  else
    echo "waiting for $STATE"
    CUR_LOG=""
  fi

  echo ""
  echo "--- recent events ---"
  tail -14 "$LOG_ROOT/events.log" 2>/dev/null

  echo ""
  echo "--- current raw signal ---"
  if [ -n "$CUR_LOG" ] && [ -f "$CUR_LOG" ]; then
    grep -E "RESULT|PROGRESS|NO-GO|GO|PASS|FAIL|Traceback|ERROR|Train:|global_step|max_steps|'loss'|'reward'|'kl'|\\[sft-on-model\\]|\\[dpo-v2\\]|\\[grpo-on-model\\]|\\[merge\\]|Application startup complete|Capturing CUDA graphs" "$CUR_LOG" 2>/dev/null | tail -14
  else
    echo "no current raw signal"
  fi

  echo ""
  echo "--- related processes ---"
  pgrep -af 'run_derag_v4|swift/cli/rlhf.py|torch.distributed.run|swift export|vllm serve|VLLM::' 2>/dev/null \
    | sed 's/[[:space:]][[:space:]]*/ /g' | cut -c1-190 | head -14

  echo ""
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{printf "GPU%-2s util=%3s%% mem=%6s/%s MiB\n",$1,$2,$3,$4}'

  echo ""
  echo "--- gates / reports ---"
  for f in \
    "$OUT_DIR/125b_replay_report.json" \
    "$OUT_DIR/125a_anchor_calibration.json" \
    "$OUT_DIR/125_gate_rewrites.json" \
    "$OUT_DIR/125c_stage1_spotcheck_report.json" \
    "$OUT_DIR/126_dpo_seed_pools.json" \
    "$OUT_DIR"/126_g11_density_k*.json \
    "$OUT_DIR/127_dpo_pairs_meta.json" \
    "$OUT_DIR/130_dpo_variant_decisions.json" \
    "$OUT_DIR/133_grpo_variant_decisions.json" \
    "$OUT_DIR/138_final_det_report.md" \
    "$OUT_DIR/139_derag_v4_summary.md"; do
    [ -f "$f" ] || continue
    printf "%s : " "$(basename "$f")"
    if [[ "$f" == *.json ]]; then
      "$PY" - "$f" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
if isinstance(d, list):
    print(" | ".join(f"{x.get('label')} ok={x.get('ok')}" for x in d[:4]))
else:
    keys = ["status", "reason", "mode", "good_pass_rate", "bad_pass_rate", "balanced_accuracy",
            "l0_entrants", "passed_rewrites", "pass_sample", "fail_sample",
            "p_clean", "p_pair", "train_pairs", "heldout_pairs"]
    print(" ".join(f"{k}={d.get(k)}" for k in keys if k in d))
PY
    else
      grep -E "status:|trace_total|clean_rate|selected|PIPELINE|run_id" "$f" 2>/dev/null | head -2 | tr '\n' ' '
      echo
    fi
  done

  echo ""
  echo "Ctrl+C exit"
  sleep 10
done
