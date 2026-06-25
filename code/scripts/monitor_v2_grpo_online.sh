#!/usr/bin/env bash
# Clean monitor for V2 online GRPO.  Ctrl+C only exits this monitor.
set -u

WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_DIR="${ZHJG_LOG_DIR:-$WORK/logs}"
RAW_DIR="$LOG_DIR/v2_grpo_online"
STATE="$RAW_DIR/state.json"
OUT_DIR="${ZHJG_OUTPUT_DIR:-$WORK/output}/${V2_TAG:-derag2}"
CKPT_DIR="${ZHJG_CKPT_DIR:-$WORK/ckpts}"
MODEL_DIR="${ZHJG_MODEL_DIR:-$WORK/models}"
PY="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/python"
LINEAGE="${V2_GRPO_LINEAGE:-2s-2s-2s}"
EVAL_TAG="${V2_GRPO_EVAL_TAG:-v2-${LINEAGE}-grpo}"

while true; do
  clear
  echo "===== V2 online GRPO monitor · $(date '+%F %T') ====="
  echo "lineage: $LINEAGE | eval_tag: $EVAL_TAG | logs: $RAW_DIR"
  echo ""

  echo "--- 状态 ---"
  CUR_LOG=""
  if [ -f "$STATE" ]; then
    "$PY" - "$STATE" <<'PY'
import json, sys
s=json.load(open(sys.argv[1], encoding="utf-8"))
for k in ("status","stage","pid","started_at","updated_at","error"):
    if s.get(k) is not None:
        print(f"{k}: {s.get(k)}")
done=s.get("completed") or []
if done:
    print("done: " + " -> ".join(done[-8:]))
if s.get("raw_log"):
    print("raw_log: " + str(s.get("raw_log")))
PY
    CUR_LOG="$("$PY" - "$STATE" <<'PY' 2>/dev/null
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("raw_log") or "")
PY
)"
  else
    echo "等待状态文件: $STATE"
  fi

  echo ""
  echo "--- 固定产物 ---"
  DATA="${V2_GRPO_DATA:-$OUT_DIR/70_grpo_data.v2-${LINEAGE}.jsonl}"
  AUDIT="$RAW_DIR/reward_audit.json"
  KIMI_SMOKE="$RAW_DIR/kimi_smoke.ok"
  SMOKE_LORA="${V2_GRPO_SMOKE_LORA:-$CKPT_DIR/v2-grpo-smoke-${LINEAGE}-lora}"
  WARMUP_LORA="${V2_GRPO_WARMUP_LORA:-$CKPT_DIR/v2-grpo-warmup-${LINEAGE}-lora}"
  WARMUP_MERGED="${V2_GRPO_WARMUP_MERGED:-$MODEL_DIR/v2-grpo-warmup-${LINEAGE}-merged}"
  FINAL_LORA="${V2_GRPO_FINAL_LORA:-$CKPT_DIR/v2-grpo-2sigma-${LINEAGE}-lora}"
  FINAL_MERGED="${V2_GRPO_FINAL_MERGED:-$MODEL_DIR/v2-grpo-2sigma-${LINEAGE}-merged}"
  INFER="$OUT_DIR/${EVAL_TAG}_infer.jsonl"
  SCORES="$OUT_DIR/${EVAL_TAG}_scores.jsonl"
  REPORT="$OUT_DIR/${EVAL_TAG}_report.md"
  SUMMARY="$OUT_DIR/${EVAL_TAG}_summary.json"
  for pair in \
    "data:$DATA" \
    "reward_audit:$AUDIT" \
    "kimi_smoke:$KIMI_SMOKE" \
    "smoke_lora:$SMOKE_LORA" \
    "warmup_lora:$WARMUP_LORA" \
    "warmup_merged:$WARMUP_MERGED" \
    "final_lora:$FINAL_LORA" \
    "final_merged:$FINAL_MERGED" \
    "infer:$INFER" \
    "scores:$SCORES" \
    "report:$REPORT"; do
    name="${pair%%:*}"
    path="${pair#*:}"
    if [ -d "$path" ]; then
      done_mark=""; [ -f "$path/.done" ] && done_mark=" .done"
      size="$(du -sh "$path" 2>/dev/null | awk '{print $1}')"
      printf "%-14s ✅ %s  size=%s%s\n" "$name" "$path" "$size" "$done_mark"
    elif [ -f "$path" ]; then
      rows=""
      case "$path" in
        *.jsonl) rows=" rows=$(wc -l < "$path" 2>/dev/null | tr -d ' ')" ;;
      esac
      printf "%-14s ✅ %s%s\n" "$name" "$path" "$rows"
    else
      printf "%-14s ⏳ %s\n" "$name" "$path"
    fi
  done

  echo ""
  echo "--- 当前/最终三分 ---"
  if [ -f "$SUMMARY" ]; then
    "$PY" - "$SUMMARY" <<'PY'
import json, sys
s=json.load(open(sys.argv[1], encoding="utf-8"))
print(f"Kimi={s.get('clean_mean')}  规则={100*s.get('rule_pass_rate',0):.1f}%  在池={100*s.get('in_pool_rate',0):.1f}%  格式={100*s.get('format_pass_rate',0):.1f}%  SE={s.get('se')}")
PY
  else
    echo "尚无 summary；训练中看下方 raw log 的 reward/kl。"
  fi

  echo ""
  echo "--- Kimi预算 ---"
  # 计量表写在 OUTPUT_DIR 根（跨 V2_TAG 共享，见 kimi_budget.py），不带 /$V2_TAG 子目录。
  KB="${ZHJG_OUTPUT_DIR:-$WORK/output}/kimi_budget.json"
  if [ -f "$KB" ]; then
    "$PY" - "$KB" <<'PY'
import json, sys
s=json.load(open(sys.argv[1], encoding="utf-8"))
print(f"calls={s.get('calls',0)} yuan={s.get('yuan',0)} in={s.get('in_tokens',0)} out={s.get('out_tokens',0)} cache_hits={s.get('cache_hits',0)}")
PY
  else
    echo "暂无 $KB"
  fi

  echo ""
  echo "--- 最新训练/评测信号 ---"
  if [ -n "$CUR_LOG" ] && [ -f "$CUR_LOG" ]; then
    grep -E "Train:|global_step|max_steps|'loss'|'reward'|'kl'|Capturing CUDA graphs|Application startup complete|\[grpo-on-model\]|\[merge\]|Kimi|干净|规则|在池|ERROR|FAIL|Traceback|complete" "$CUR_LOG" 2>/dev/null | tail -10
  else
    echo "当前阶段暂无 raw log"
  fi

  echo ""
  echo "--- recent events ---"
  tail -14 "$RAW_DIR/events.log" 2>/dev/null

  echo ""
  echo "--- 相关进程 / GPU / 磁盘 ---"
  pgrep -af 'run_v2_grpo_online|swift/cli/rlhf.py|torch.distributed.run|swift export|vllm serve|step_v2_eval|step03_eval_infer' 2>/dev/null \
    | sed 's/[[:space:]][[:space:]]*/ /g' | cut -c1-180 | head -12
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{printf "GPU%-2s util=%3s%% used=%6s free=%6s / %6s MiB\n",$1,$2,$3,$4,$5}'
  df -h "$WORK" 2>/dev/null | tail -1 | awk '{print "disk: "$4" free ("$5" used) @ "$6}'

  echo ""
  echo "Ctrl+C 仅退出监控，不影响主流程 | 10s 刷新"
  sleep 10
done
