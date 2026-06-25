"""derag_v5 · X 重测补丁（离线、零 GPU）：把 s153 的"判 think 干净"从规则换成 Kimi，得到真 X_kimi vs 规则 X。

只读已在盘的 OUT_DIR/{151_rft_samples,152_v1_support}.jsonl + 153_rft_headroom.json，
跑 step153b（纯 Kimi API + CPU），输出 153b_kimi_headroom.* + 159b_xrecheck.md。
**不 serve vLLM、不碰 GPU、不重新生成任何采样**——所以不会动到任何训练显存设置。

前提：run_id 对应的探针已经跑过（151/152 在 OUT_DIR）。默认 run_id=main，与探针一致。

页面1：bash scripts/run_derag_v5_xrecheck.sh
页面2：bash scripts/monitor_derag_v5_xrecheck.sh
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _find_code_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here, here / "code", here.parent / "code"):
        if (cand / "config.py").exists():
            return cand
    return here


ROOT = _find_code_root()
sys.path.insert(0, str(ROOT))

from config import KIMI_API_KEY_ENV, LOG_DIR, OUTPUT_DIR  # noqa: E402

PY = Path(os.environ.get("ZHJG_ENV", "/home/nvme02/conda/zhjg_rl")) / "bin" / "python"
RUN_ID = os.environ.get("V5_RUN_ID", "main")
OUT_DIR = Path(OUTPUT_DIR) / "derag_v5_probe" / RUN_ID
LOG_ROOT = Path(os.environ.get("V5X_LOG_DIR", str(Path(LOG_DIR) / "derag_v5_xrecheck" / RUN_ID)))
EVENT_LOG = LOG_ROOT / "events.log"
STATE_FILE = LOG_ROOT / "state.json"
RAW_DIR = LOG_ROOT / "raw"

SAMPLES = OUT_DIR / "151_rft_samples.jsonl"
SUPPORT = OUT_DIR / "152_v1_support.jsonl"
RULE_JSON = OUT_DIR / "153_rft_headroom.json"
OUT_153B = OUT_DIR / "153b_kimi_headroom.jsonl"
OUT_153B_JSON = OUT_DIR / "153b_kimi_headroom.json"
REPORT_MD = OUT_DIR / "159b_xrecheck.md"

SIGNAL_WORDS = ("RESULT", "PROGRESS", "ERROR", "Traceback")
_state = {"status": "starting", "stage": "preflight", "completed": [], "run_id": RUN_ID, "out_dir": str(OUT_DIR)}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def emit(msg: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{now()} | {msg}"
    print(line, flush=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_state(**updates) -> None:
    _state.update(updates, updated_at=now())
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def latest_signal(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 128 * 1024))
        text = f.read().decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if any(w in ln for w in SIGNAL_WORDS)]
    return lines[-1][-500:] if lines else ""


def run(stage: str, cmd: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_log = RAW_DIR / f"{stage}.log"
    emit(f"START {stage}")
    emit("CMD   " + " ".join(cmd))
    with raw_log.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {now()} START {' '.join(cmd)} =====\n"); f.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                stdout=f, stderr=subprocess.STDOUT, text=True)
        save_state(status="running", stage=stage, pid=proc.pid, raw_log=str(raw_log), started_at=now())
        last, last_beat, started = "", 0.0, time.time()
        while proc.poll() is None:
            time.sleep(10)
            sig = latest_signal(raw_log)
            if sig and sig != last:
                emit(f"{stage} | {sig}"); last = sig
            elif time.time() - last_beat >= 60:
                emit(f"{stage} | alive pid={proc.pid} elapsed={int(time.time()-started)}s"); last_beat = time.time()
        rc = proc.returncode
        f.write(f"===== {now()} END rc={rc} =====\n")
    if rc != 0:
        emit(f"FAIL {stage} rc={rc}; see {raw_log}")
        save_state(status="failed", stage=stage, returncode=rc)
        raise SystemExit(rc)
    _state["completed"].append(stage)
    save_state(status="running", pid=None)
    emit(f"END   {stage}")


def preflight() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not PY.exists():
        raise SystemExit(f"missing python env: {PY}")
    if not os.environ.get(KIMI_API_KEY_ENV):
        raise SystemExit(f"{KIMI_API_KEY_ENV} required (Kimi 重判要用)")
    miss = [str(p) for p in (SAMPLES, SUPPORT) if not p.exists()]
    if miss:
        raise SystemExit(f"缺探针产物（先跑出/下载 run_id={RUN_ID} 的 151/152 到 {OUT_DIR}）:\n" + "\n".join(miss))
    emit(f"preflight OK | run_id={RUN_ID} | 读 {SAMPLES.name}/{SUPPORT.name} | out={OUT_DIR}")


def write_report() -> None:
    s = json.loads(OUT_153B_JSON.read_text(encoding="utf-8"))
    a, ev, tr, pp = s["all"], s["eval"], s["train"], s["params"]

    def gap(x):
        xr = x.get("X_rule")
        return f"{(x['X_kimi'] - xr):+.3f}" if isinstance(xr, (int, float)) else "n/a"

    lines = [
        "# derag_v5 · X 重测（规则 X vs Kimi 真 X）",
        "",
        f"- run_id: `{RUN_ID}`",
        f"- 参数: k={pp['k']}（每样本 Kimi 判次取均值）, tf_clean={pp['tf_clean']}, cap={pp['cap']}",
        "- 只换了「判 think 干净」那把尺子（规则→Kimi）；「答案不漂 V1」仍用确定性 in-support。",
        "",
        "## 对照",
        "",
        "| split | 病题数 | X_rule(规则,虚高) | **X_kimi(真)** | 差 |",
        "|---|---:|---:|---:|---:|",
        f"| all   | {a['n_problems']} | {a.get('X_rule')} | **{a['X_kimi']}** | {gap(a)} |",
        f"| eval  | {ev['n_problems']} | {ev.get('X_rule')} | **{ev['X_kimi']}** | {gap(ev)} |",
        f"| train | {tr['n_problems']} | {tr.get('X_rule')} | **{tr['X_kimi']}** | {gap(tr)} |",
        "",
        f"- 本次 Kimi 判样本数(all): {a['samples_judged']}（×k={pp['k']} 次调用）",
        "",
        "## 判据（沿用探针口径，X 换成 X_kimi）",
        "- X_kimi ≥ 0.45 → GO_RL（去 RAG 真有正样本可学，整链值得跑）",
        "- X_kimi < 0.45 → 自采样信号弱，看 Y（154 Kimi 改写率）决定先 SFT 还是 NO_GO",
        "",
        "## 怎么读",
        "- X_rule 是旧的、被规则瞎判抬高的数（clean 闸几乎不卡人）；X_kimi 把「干净」换成 Kimi 语义判，才是真自救率。",
        "- 逐题明细见 `153b_kimi_headroom.jsonl`（每题 in-support 样本各自的 kimi_clean/tf/traces，pass_idx=救活它的样本号）。",
        "- ⚠️ Kimi 有噪声：看 X_kimi 的「量级/方向」（是 ~0.45 还是 ~0.85），不要抠小数点。真要训练时，痕迹 reward 不能直接用任何单把裁判在线打分。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("REPORT " + str(REPORT_MD))


def main() -> None:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("usage: python -X utf8 run_derag_v5_xrecheck.py")
        print("env: DASHSCOPE_API_KEY(必填) V5_RUN_ID(默认main) V5X_K(2) V5X_TF_CLEAN(0.70) V5X_CAP(16)")
        return
    save_state(status="running", stage="preflight")
    try:
        preflight()
        if OUT_153B_JSON.exists() and OUT_153B_JSON.stat().st_size > 0:
            emit("RESUME 跳过 s153b（153b_kimi_headroom.json 已存在）")
        else:
            run("s153b_kimi_clean_rescore", [
                str(PY), "-X", "utf8", "pipeline/step153b_kimi_clean_rescore.py",
                "--samples", str(SAMPLES), "--support", str(SUPPORT), "--rule_json", str(RULE_JSON),
                "--out", str(OUT_153B), "--out_json", str(OUT_153B_JSON)])
        write_report()
        save_state(status="complete", stage="done", pid=None, summary=str(REPORT_MD))
        emit(f"RESULT XRECHECK COMPLETE | report={REPORT_MD}")
    except BaseException as exc:
        save_state(status="failed", error=repr(exc))
        emit(f"XRECHECK FAILED | {exc!r}")
        raise


if __name__ == "__main__":
    main()
