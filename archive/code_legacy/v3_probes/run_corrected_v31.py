"""One-click corrected-v3.1 derag-rubric diagnostics.

Narrowed goal:
1. Remove visible RAG/retrieval/reference traces from think.
2. Keep answer consistent with V1 gold.
3. Keep facts, numbers and conclusions grounded in references.

No GPU training is launched here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import KIMI_API_KEY_ENV, LOG_DIR, OUTPUT_DIR  # noqa: E402

PY = Path(os.environ.get("ZHJG_ENV", "/home/nvme02/conda/zhjg_rl")) / "bin" / "python"
RUN_ID = os.environ.get("V31_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
OUT_DIR = Path(OUTPUT_DIR) / "corrected_v31" / RUN_ID
LOG_ROOT = Path(os.environ.get("CORRECTED_V31_LOG_DIR", str(Path(LOG_DIR) / "corrected_v31" / RUN_ID)))
EVENT_LOG = LOG_ROOT / "events.log"
STATE_FILE = LOG_ROOT / "state.json"
RAW_DIR = LOG_ROOT / "raw"

BASE_INFER = Path(os.environ.get("V31_BASE_INFER", str(Path(OUTPUT_DIR) / "80_corrected_v1_rft_merged_base_infer.jsonl")))
CAND_INFER = Path(os.environ.get("V31_CANDIDATE_INFER", str(Path(OUTPUT_DIR) / "96_corrected_v2_mini_dpo_infer.jsonl")))

SIGNAL_WORDS = ("RESULT", "PROGRESS", "ERROR", "Traceback", "NO-GO", "GO", "PASS", "FAIL")
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


def child_env() -> dict:
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "V31_RUN_ID": RUN_ID,
        "ZHJG_CONSOLE_LOG_LEVEL": os.environ.get("ZHJG_CONSOLE_LOG_LEVEL", "WARNING"),
        "ZHJG_LOG_FILE": str(LOG_ROOT / "pipeline.log"),
    }


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
        f.write(f"\n===== {now()} START {' '.join(cmd)} =====\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=f, stderr=subprocess.STDOUT, text=True)
        save_state(status="running", stage=stage, pid=proc.pid, raw_log=str(raw_log), command=cmd, started_at=now())
        last = ""
        last_beat = 0.0
        started = time.time()
        while proc.poll() is None:
            time.sleep(10)
            sig = latest_signal(raw_log)
            if sig and sig != last:
                emit(f"{stage} | {sig}")
                last = sig
            elif time.time() - last_beat >= 60:
                emit(f"{stage} | alive pid={proc.pid} elapsed={int(time.time()-started)}s")
                last_beat = time.time()
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
        raise SystemExit(f"{KIMI_API_KEY_ENV} is required")
    missing = [str(p) for p in (BASE_INFER, CAND_INFER) if not p.exists()]
    if missing:
        raise SystemExit("missing required v3.1 inputs:\n" + "\n".join(missing))
    emit(f"preflight OK | run_id={RUN_ID} | out={OUT_DIR} | logs={LOG_ROOT}")


def load_decision(name: str) -> dict:
    p = OUT_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "missing", "path": str(p)}


def write_summary() -> None:
    e0 = load_decision("110_corrected_v31_derag_calibration_decision.json")
    e1 = load_decision("111_corrected_v31_derag_paired_readout_decision.json")
    e2 = load_decision("112_corrected_v31_trace_surgery_check_decision.json")
    lines = [
        "# corrected-v3.1 derag-rubric diagnostics summary",
        "",
        f"- run_id: `{RUN_ID}`",
        f"- output_dir: `{OUT_DIR}`",
        "- target: trace-free think + grounded facts + answer consistency",
        "",
        "## Decisions",
        "",
        "| probe | status | key readout |",
        "|---|---|---|",
        f"| E0 derag calibration | {e0.get('status')} | paired Δtrace_free={e0.get('delta_trace_free', 'NA')} CI={e0.get('delta_trace_free_ci95', 'NA')} |",
        f"| E1 paired readout | {e1.get('verdict', e1.get('status'))} | Δtrace_free={e1.get('delta_trace_free', 'NA')} CI={e1.get('delta_trace_free_ci95', 'NA')} Δg={e1.get('delta_grounded', 'NA')} Δacc={e1.get('delta_accuracy_score', 'NA')} |",
        f"| E2 trace surgery | {e2.get('status')} | Δtrace_free={e2.get('mean_delta_trace_free', 'NA')} Δg={e2.get('mean_delta_grounded', 'NA')} Δacc={e2.get('mean_delta_accuracy_score', 'NA')} |",
        "",
        "## Next Rule",
        "",
        "- If E1 is GO, candidate has readable de-RAG improvement without guardrail loss.",
        "- If E2 is PASS_DERAG_READABLE, the narrowed judge can read visible trace removal.",
        "- Only after both are acceptable should we build DPO pairs with chosen=trace-free+grounded+accurate and rejected=trace-heavy without better accuracy/grounding.",
    ]
    path = OUT_DIR / "113_corrected_v31_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit(f"RESULT v31_summary -> {path}")
    save_state(status="complete", stage="done", pid=None, summary=str(path))


def main() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print("usage: python -X utf8 run_corrected_v31.py")
        print("Environment:")
        print("  DASHSCOPE_API_KEY=...        required")
        print("  V31_RUN_ID=...               optional output/log namespace")
        print("  V31_BASE_INFER=...           optional base infer override")
        print("  V31_CANDIDATE_INFER=...      optional candidate infer override")
        print("  V31_JUDGE_K=3                repeated Kimi judges")
        print("  V31_KIMI_WORKERS=3           Kimi concurrency")
        return
    save_state(status="running", stage="preflight")
    try:
        preflight()
        cal = OUT_DIR / "110_corrected_v31_derag_calibration.jsonl"
        run("e0_derag_calibration", [
            str(PY), "-X", "utf8", "pipeline/step110_derag_judge_calibration.py",
            "--base-infer", str(BASE_INFER),
            "--candidate-infer", str(CAND_INFER),
            "--out-dir", str(OUT_DIR),
        ])
        run("e1_derag_paired_readout", [
            str(PY), "-X", "utf8", "pipeline/step111_derag_paired_readout.py",
            "--calibration", str(cal),
            "--out-dir", str(OUT_DIR),
        ])
        run("e2_trace_surgery_check", [
            str(PY), "-X", "utf8", "pipeline/step112_derag_trace_surgery_check.py",
            "--calibration", str(cal),
            "--out-dir", str(OUT_DIR),
        ])
        write_summary()
    except BaseException as exc:
        save_state(status="failed", error=repr(exc))
        emit(f"PIPELINE FAILED | {exc!r}")
        raise


if __name__ == "__main__":
    main()
