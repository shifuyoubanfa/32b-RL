#!/usr/bin/env bash
# V2 训练管线监控：状态 + 二叉树进度(14 LoRA/8 叶) + 冷启动漏斗 + 采样进度 + Kimi 围栏 + GPU + vLLM + 三分曲线。
set -u

WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_DIR="${ZHJG_LOG_DIR:-$WORK/logs}"
OUT_BASE="${ZHJG_OUTPUT_DIR:-$WORK/output}"
CKPT_DIR="${ZHJG_CKPT_DIR:-$WORK/ckpts}"
MODEL_DIR="${ZHJG_MODEL_DIR:-$WORK/models}"
PY="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}/bin/python"
V2_OUT="$OUT_BASE/${V2_TAG:-v2}"   # 版本化：export V2_TAG=derag2 → 看新跑目录；默认 v2。日志/state 仍共享 v2 子目录
STATE="$LOG_DIR/v2/state.json"
EVENTS="$LOG_DIR/v2/events.log"
BUDGET="$OUT_BASE/kimi_budget.json"
VLLM_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"

# 数某类 .done 完成数：$1=名字 glob，$2=父目录
count_done() {
  local n=0 d
  while IFS= read -r d; do
    [ -f "$d/.done" ] && n=$((n + 1))
  done < <(find "$2" -maxdepth 1 -type d -name "$1" 2>/dev/null)
  echo "$n"
}

while true; do
  clear
  echo "================ $(date '+%F %T') V2 训练管线 ================"

  echo "--- state ---"
  if [ -f "$STATE" ]; then
    "$PY" - "$STATE" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
for k in ["status", "stage", "pid", "started_at", "updated_at", "raw_log", "error"]:
    v = s.get(k)
    if v:
        print(f"{k:11}: {v}")
print("completed  : " + " -> ".join(s.get("completed", [])[-12:]))
PY
    CUR_LOG="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('raw_log') or '')" "$STATE" 2>/dev/null)"
  else
    echo "等待 $STATE（run_v2 尚未启动？）"
    CUR_LOG=""
  fi

  echo ""
  echo "--- 二叉树进度 (LoRA / merged 完成数) ---"
  printf "SFT  LoRA %s/2  merged %s/2  （2sigma=广/质档·3sigma=精档；现回 σ 门选样）\n" "$(count_done 'v2-sft-*sigma-lora' "$CKPT_DIR")"  "$(count_done 'v2-sft-*sigma-merged' "$MODEL_DIR")"
  printf "RFT  LoRA %s/4  merged %s/4\n" "$(count_done 'v2-rft-*sigma*-lora' "$CKPT_DIR")" "$(count_done 'v2-rft-*sigma*-merged' "$MODEL_DIR")"
  printf "DPO  LoRA %s/8  merged %s/8\n" "$(count_done 'v2-dpo-*sigma*-lora' "$CKPT_DIR")" "$(count_done 'v2-dpo-*sigma*-merged' "$MODEL_DIR")"
  leaves=$(grep -l "答案在池率" "$V2_OUT"/v2-*_report.md 2>/dev/null | wc -l | tr -d ' ')
  printf "评测报告(三分齐全): %s\n" "$leaves"

  echo ""
  echo "--- 凑数据进度 (冷启动漏斗 + RFT/DPO 攒2σ，够了就停) ---"
  V2_COLDSTART_TARGET="${V2_COLDSTART_TARGET:-700}" V2_RFT_TARGET="${V2_RFT_TARGET:-200}" \
  V2_DPO_TARGET="${V2_DPO_TARGET:-900}" "$PY" - "$V2_OUT" <<'PY' 2>/dev/null || echo "（还没开始凑数据）"
import json, glob, os, sys
v2 = sys.argv[1]
tgt = {"cs": int(os.environ.get("V2_COLDSTART_TARGET", "700")),
       "rft": int(os.environ.get("V2_RFT_TARGET", "200")),
       "dpo": int(os.environ.get("V2_DPO_TARGET", "900"))}
def loadj(p):
    rows = []
    try:
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
    except OSError:
        pass
    return rows
def bar(cur, t, w=14):
    if t <= 0:
        return f"[不限] {cur}"
    n = int(min(1.0, cur / t) * w)
    return "[" + "#" * n + "-" * (w - n) + f"] {cur}/{t}" + ("  ✅够了" if cur >= t else "")
any_ = False
p = os.path.join(v2, "coldstart_progress.v2.jsonl")
if os.path.exists(p):
    any_ = True
    R = loadj(p)
    nat = sum(1 for r in R if r.get("natural"))
    rule = sum(1 for r in R if r.get("rule_ok"))
    fact = sum(1 for r in R if r.get("facts_ok"))
    k2 = sum(1 for r in R if r.get("s_clean") is not None)
    p2 = sum(1 for r in R if r.get("pass2"))
    p3 = sum(1 for r in R if r.get("pass3"))
    sc = [r["s_clean"] for r in R if r.get("s_clean") is not None]
    print(f"冷启动漏斗  处理{len(R)} → 改写{nat} → 规则过{rule} → facts过{fact} → k2过{k2} → 2σ {p2} / 3σ {p3}")
    if sc:
        print(f"            改写干净分(k16)均值 {sum(sc)/len(sc):.2f}（>6=破3.44天花板；旧版~3.4）")
    print(f"冷启动2σ目标 {bar(p2, tgt['cs'])}")
for f in sorted(glob.glob(os.path.join(v2, "rft_progress.*.v2.jsonl"))):
    any_ = True
    line = os.path.basename(f).split(".")[1]
    R = loadj(f)
    print(f"RFT  {line:8} {bar(sum(1 for r in R if r.get('pass2')), tgt['rft'])}  (已处理 {len(R)})")
for f in sorted(glob.glob(os.path.join(v2, "dpo_progress.*.v2.jsonl"))):
    any_ = True
    line = os.path.basename(f).split(".")[1]
    R = loadj(f)
    print(f"DPO  {line:8} {bar(sum(1 for r in R if r.get('pass2')), tgt['dpo'])}对 (已处理 {len(R)})")
if not any_:
    print("（还没开始凑数据）")
PY

  echo ""
  echo "--- 采样进度 (RFT 自采样 / DPO rollout：题数 × 候选) ---"
  "$PY" - "$V2_OUT" <<'PY' 2>/dev/null || echo "（还没开始自采样）"
import json, glob, os, sys
v2 = sys.argv[1]
def count(p):
    nq = nc = 0
    try:
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            nq += 1
            cand = r.get("samples") or r.get("candidates") or r.get("rollouts")
            if isinstance(cand, list):
                nc += len(cand)
            else:
                lists = [len(v) for v in r.values() if isinstance(v, list)]
                nc += max(lists) if lists else 0
    except OSError:
        pass
    return nq, nc
shown = False
for f in sorted(glob.glob(os.path.join(v2, "151_rft_selfsample.*.v2.jsonl"))):
    shown = True
    line = os.path.basename(f).split(".")[1]
    nq, nc = count(f)
    print(f"RFT自采样   {line:10} {nq} 题 × ~{(nc // nq) if nq else 0} 候选 = {nc} 段")
for f in sorted(glob.glob(os.path.join(v2, "60_dpo_rollout.*.v2.jsonl"))):
    shown = True
    line = os.path.basename(f).split(".")[1]
    nq, nc = count(f)
    print(f"DPO rollout {line:10} {nq} 题 × ~{(nc // nq) if nq else 0} 候选 = {nc} 段")
if not shown:
    print("（还没开始自采样/rollout）")
PY

  echo ""
  echo "--- Kimi 围栏 (改写+打分计量 / 预算) ---"
  if [ -f "$BUDGET" ]; then
    KIMI_BUDGET_YUAN="${KIMI_BUDGET_YUAN:-0}" "$PY" - "$BUDGET" <<'PY'
import json, os, sys
b = json.load(open(sys.argv[1], encoding="utf-8"))
cap = float(os.environ.get("KIMI_BUDGET_YUAN", "0"))
calls = b.get("calls", 0); it = b.get("in_tokens", 0); ot = b.get("out_tokens", 0)
yuan = b.get("yuan", 0.0); hits = b.get("cache_hits", 0)
print(f"Kimi 调用 {calls} 次（改写+判分；缓存命中 {hits} 次无损省算）")
print(f"token 入 {it/1e6:.2f}M / 出 {ot/1e6:.2f}M")
tail = (f" / 围栏 ¥{cap:.0f}（{yuan/cap:.0%}）" if cap > 0 else "（只计量、无硬闸）")
print(f"花费 ¥{yuan:.1f}{tail}")
PY
  else
    echo "尚无 $BUDGET（还没调 Kimi）"
  fi

  echo ""
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{printf "GPU%-2s util=%3s%% mem=%6s/%s MiB\n",$1,$2,$3,$4}' || echo "nvidia-smi 不可用"

  echo ""
  echo "--- vLLM (port 8000) ---"
  served=$(curl -s --max-time 3 "$VLLM_URL/models" 2>/dev/null \
    | "$PY" -c "import json,sys
try:
    d=json.load(sys.stdin); print(','.join(m.get('id','?') for m in d.get('data',[])))
except Exception:
    pass" 2>/dev/null)
  if [ -n "$served" ]; then echo "serving: $served"; else echo "未在服务（训练/合并阶段正常）"; fi

  echo ""
  echo "--- 三分曲线 (干净越高越像人 / 规则=去检索腔通过 / 在池=准确率；V1→SFT→RFT→DPO 应逐级抬) ---"
  "$PY" - "$V2_OUT" <<'PY' 2>/dev/null || echo "（还没有评测结果）"
import json, glob, os, sys
v2 = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(v2, "*_summary.json"))):
    try:
        rows.append(json.load(open(f, encoding="utf-8")))
    except Exception:
        continue
def order(d):
    t = d.get("tag", "")
    return (0 if "baseline" in t else 1 if "sft" in t else 2 if "rft" in t else 3, t)
if not rows:
    print("（还没有评测结果）")
for d in sorted(rows, key=order):
    t = d.get("tag", "?"); c = d.get("clean_mean", 0.0) or 0.0
    r = d.get("rule_pass_rate", 0.0) or 0.0; p = d.get("in_pool_rate", 0.0) or 0.0
    se = d.get("se", 0.0) or 0.0
    n = int(min(1.0, c / 10.0) * 10)
    print(f"{t:20} 干净 {c:4.1f} [" + "#" * n + "-" * (10 - n) + f"]  规则 {r:4.0%}  在池 {p:4.0%}  (SE±{se:.2f})")
print("真涨判据：两阶段干净分差 > ~3×SE≈0.15；在池率剪枝地板 0.85")
PY

  echo ""
  echo "--- recent events ---"
  tail -10 "$EVENTS" 2>/dev/null

  echo ""
  echo "--- current raw signal ---"
  if [ -n "$CUR_LOG" ] && [ -f "$CUR_LOG" ]; then
    grep -E "Train:|global_step|'loss'|eval_loss|干净分|在池率|漏斗|改写|\[sft-on-model\]|\[dpo-v2\]|\[merge\]|\[serve-model\]|RESULT|ERROR|Traceback|Application startup complete" "$CUR_LOG" 2>/dev/null | tail -8
  else
    echo "（无当前子进程日志）"
  fi

  echo ""
  echo "--- 进程 / 磁盘 ---"
  pgrep -af 'run_v2|step_v2|swift/cli/rlhf.py|torch.distributed.run|swift export|vllm serve' 2>/dev/null \
    | sed 's/  */ /g' | cut -c1-150 | head -8
  df -h "$WORK" 2>/dev/null | tail -1 | awk '{print "disk: "$4" free ("$5" used) @ "$6}'

  echo ""
  echo "Ctrl+C 退出监控(不影响训练) | 10s 刷新"
  sleep 10
done
