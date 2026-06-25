#!/usr/bin/env bash
# V2 derag2 DPO 本地续跑的独立干净监控；只读 logs/v2_dpo_resume，不刷旧 logs/v2。
set -u

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_ROOT="${ZHJG_LOG_DIR:-$WORK/logs}/v2_dpo_resume"
RAW="$LOG_ROOT/raw"
STATE="$LOG_ROOT/state.json"
EVENTS="$LOG_ROOT/events.log"
RESULT="$LOG_ROOT/result.md"
PY="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/python"

MODEL_DIR="${ZHJG_MODEL_DIR:-$WORK/models}"
CKPT_DIR="${ZHJG_CKPT_DIR:-$WORK/ckpts}"
OUTPUT_DIR="${ZHJG_OUTPUT_DIR:-$WORK/output}"
OUT="${ZHJG_V2_OUTPUT_DIR:-$OUTPUT_DIR/derag2}"
MERGED="$MODEL_DIR/v2-dpo-2sigma-2s-2s-merged"
CANONICAL_LORA="$CKPT_DIR/v2-dpo-2sigma-2s-2s-lora"
INFER="$OUT/v2-2s-2s-2s_infer.jsonl"
SCORES="$OUT/v2-2s-2s-2s_scores.jsonl"
REPORT="$OUT/v2-2s-2s-2s_report.md"
SUMMARY="$OUT/v2-2s-2s-2s_summary.json"
PROGRESS="$OUT/v2-2s-2s-2s_score_progress.jsonl"
INFER_PROV="$LOG_ROOT/infer_provenance.json"
EVAL_PROV="$LOG_ROOT/eval_provenance.json"
VLLM_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
GPU_USED_MAX="${V2_DPO_GPU_USED_MAX_MIB:-2048}"
GPU_UTIL_MAX="${V2_DPO_GPU_UTIL_MAX:-5}"
GPU_SAMPLES="${V2_DPO_GPU_STABLE_SAMPLES:-3}"

while true; do
  clear
  printf '===== V2 DPO resume monitor · %s =====\n' "$(date '+%F %T')"
  printf '正式叶: 2s-2s-2s | tag: v2-2s-2s-2s | V2_TAG: derag2\n'
  printf '日志: %s\n\n' "$LOG_ROOT"

  echo "--- 状态 ---"
  if [ -f "$STATE" ]; then
    "$PY" - "$STATE" <<'PY' 2>/dev/null || cat "$STATE"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for k in ("status", "stage", "pid", "selected_gpus", "requested_gpus", "updated_at", "error"):
    if d.get(k) not in (None, ""):
        print(f"{k}: {d[k]}")
PY
  else
    echo "尚未启动；主终端运行 bash scripts/run_v2_dpo_resume.sh"
  fi

  echo ""
  echo "--- 固定产物（原 run_v2 命名）---"
  if [ -f "$CANONICAL_LORA/.done" ] && [ -f "$CANONICAL_LORA/checkpoint-108/adapter_model.safetensors" ]; then
    printf 'LoRA:   ✅ %s/checkpoint-108\n' "$CANONICAL_LORA"
  else
    printf 'LoRA:   ⏳ %s\n' "$CANONICAL_LORA"
  fi
  if [ -f "$MERGED/.done" ] && [ -f "$MERGED/model.safetensors.index.json" ]; then
    size=$(du -sh "$MERGED" 2>/dev/null | awk '{print $1}')
    shards=$(find "$MERGED" -maxdepth 1 -type f -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
    printf 'merged: ✅ %s  size=%s shards=%s\n' "$MERGED" "${size:-?}" "$shards"
  elif [ -e "$MERGED.partial" ]; then
    printf 'merged: 🔄 partial=%s\n' "$MERGED.partial"
  else
    printf 'merged: ⏳ %s\n' "$MERGED"
  fi
  for spec in "infer:$INFER:500" "scores:$SCORES:500" "Kimi进度:$PROGRESS:500"; do
    name=${spec%%:*}; rest=${spec#*:}; path=${rest%:*}; target=${spec##*:}
    if [ -f "$path" ]; then
      n=$(wc -l < "$path" | tr -d ' ')
      printf '%-10s %4s/%s  %s\n' "$name" "$n" "$target" "$path"
    else
      printf '%-10s    0/%s  %s\n' "$name" "$target" "$path"
    fi
  done
  printf 'provenance infer=%s eval=%s（最终可信状态以续跑器严格校验为准）\n' \
    "$([ -f "$INFER_PROV" ] && echo yes || echo no)" "$([ -f "$EVAL_PROV" ] && echo yes || echo no)"
  if [ -f "$REPORT" ]; then printf 'report:    ✅ %s\n' "$REPORT"; else printf 'report:    ⏳ %s\n' "$REPORT"; fi

  echo ""
  echo "--- 三分曲线 ---"
  "$PY" - "$OUT" <<'PY' 2>/dev/null || echo "（summary 尚未齐）"
import json, os, sys
root = sys.argv[1]
names = ["v2-baseline-v1", "v2-sft-2s", "v2-rft-2s-2s", "v2-2s-2s-2s"]
for name in names:
    p = os.path.join(root, name + "_summary.json")
    if not os.path.exists(p):
        print(f"{name:20} —")
        continue
    d = json.load(open(p, encoding="utf-8"))
    fmt = d.get('format_pass_rate')
    fmt_text = f"  格式={fmt:.1%}" if fmt is not None else ""
    print(f"{name:20} Kimi={d['clean_mean']:.3f}  规则={d['rule_pass_rate']:.1%}  在池={d['in_pool_rate']:.1%}{fmt_text}  SE={d.get('se',0):.4f}")
PY

  echo ""
  echo "--- GPU（used<=${GPU_USED_MAX}MiB、util<=${GPU_UTIL_MAX}%、无计算PID，连续${GPU_SAMPLES}次）---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{printf "GPU%-2s util=%3s%% used=%6s free=%6s / %s MiB\n",$1,$2,$3,$4,$5}' \
    || echo "nvidia-smi 不可用"

  echo ""
  echo "--- vLLM（只认本叶服务名）---"
  served=$(curl -s --max-time 3 "$VLLM_URL/models" 2>/dev/null \
    | "$PY" -c "import json,sys
try: print(','.join(x.get('id','?') for x in json.load(sys.stdin).get('data',[])))
except Exception: pass" 2>/dev/null)
  if [ -n "$served" ]; then echo "serving: $served"; else echo "未服务（合并/等卡/Kimi阶段正常）"; fi

  echo ""
  echo "--- 当前干净信号 ---"
  CUR_LOG=$("$PY" - "$STATE" <<'PY' 2>/dev/null
import json, os, sys
if os.path.exists(sys.argv[1]): print(json.load(open(sys.argv[1], encoding="utf-8")).get("raw_log") or "")
PY
)
  if [ -n "$CUR_LOG" ] && [ -f "$CUR_LOG" ]; then
    grep -E "\[merge\]|Loading checkpoint|Application startup complete|评测推理|完成：|干净分|规则通过|在池率|RESULT|ERROR|FAIL|Traceback" \
      "$CUR_LOG" 2>/dev/null | tail -10
  else
    echo "（当前阶段无子进程 raw log；等卡时看上方状态/GPU）"
  fi

  echo ""
  echo "--- recent events ---"
  tail -15 "$EVENTS" 2>/dev/null || echo "（暂无 events）"

  if [ -f "$RESULT" ]; then
    echo ""
    echo "--- 最终结果 ---"
    cat "$RESULT"
  fi

  echo ""
  echo "--- 本链进程 / 磁盘 ---"
  pgrep -af 'run_v2_dpo_resume|v2_2s_2s_2s|v2-2s-2s-2s|merge_lora_model|swift export' 2>/dev/null \
    | sed 's/  */ /g' | cut -c1-180 | head -10 || true
  df -h "$WORK" 2>/dev/null | tail -1 | awk '{print "disk: "$4" free ("$5" used) @ "$6}'

  echo ""
  echo "Ctrl+C 仅退出监控，不影响主流程 | 10s 刷新"
  sleep 10
done
