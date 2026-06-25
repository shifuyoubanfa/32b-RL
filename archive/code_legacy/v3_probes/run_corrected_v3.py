"""One-click corrected-v3 zero-training diagnostics.

This runner implements the postmortem plan in 98:
E0 repeated judge calibration + paired readout
E1 placebo trace surgery
E2 controlled rewrite ceiling probe
E3 pairwise judge controls

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
RUN_ID = os.environ.get("V3_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
OUT_DIR = Path(OUTPUT_DIR) / "corrected_v3" / RUN_ID
LOG_ROOT = Path(os.environ.get("CORRECTED_V3_LOG_DIR", str(Path(LOG_DIR) / "corrected_v3" / RUN_ID)))
EVENT_LOG = LOG_ROOT / "events.log"
STATE_FILE = LOG_ROOT / "state.json"
RAW_DIR = LOG_ROOT / "raw"

BASE_INFER = Path(os.environ.get("V3_BASE_INFER", str(Path(OUTPUT_DIR) / "80_corrected_v1_rft_merged_base_infer.jsonl")))
BASE_JUDGE = Path(os.environ.get("V3_BASE_JUDGE", str(Path(OUTPUT_DIR) / "80_corrected_v1_rft_merged_base_judge.jsonl")))
CAND_INFER = Path(os.environ.get("V3_CANDIDATE_INFER", str(Path(OUTPUT_DIR) / "96_corrected_v2_mini_dpo_infer.jsonl")))
CAND_JUDGE = Path(os.environ.get("V3_CANDIDATE_JUDGE", str(Path(OUTPUT_DIR) / "96_corrected_v2_mini_dpo_judge.jsonl")))

SIGNAL_WORDS = ("RESULT", "PROGRESS", "ERROR", "Traceback", "GOODHART", "NO-GO", "GO_", "BIAS", "PASS")
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
        "V3_RUN_ID": RUN_ID,
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


def run(stage: str, cmd: list[str], optional: bool = False) -> int:
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
        started = time.time()
        last_beat = 0.0
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
        emit(f"{'WARN' if optional else 'FAIL'} {stage} rc={rc}; see {raw_log}")
        if not optional:
            save_state(status="failed", stage=stage, returncode=rc)
            raise SystemExit(rc)
    else:
        _state["completed"].append(stage)
        save_state(status="running", pid=None)
        emit(f"END   {stage}")
    return rc


def preflight() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not PY.exists():
        raise SystemExit(f"missing python env: {PY}")
    if not os.environ.get(KIMI_API_KEY_ENV):
        raise SystemExit(f"{KIMI_API_KEY_ENV} is required")
    required = [BASE_INFER, BASE_JUDGE, CAND_INFER]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing required v3 inputs:\n" + "\n".join(missing))
    emit(f"preflight OK | run_id={RUN_ID} | out={OUT_DIR} | logs={LOG_ROOT}")


def load_decision(name: str) -> dict:
    p = OUT_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "missing", "path": str(p)}


def write_summary() -> None:
    decisions = {
        "e0_noise": load_decision("100_corrected_v3_noise_calibration_decision.json"),
        "e0_readout": load_decision("101_corrected_v3_mini_paired_readout_decision.json"),
        "e1_placebo": load_decision("102_corrected_v3_placebo_decision.json"),
        "e2_rewrite": load_decision("103_corrected_v3_rewrite_probe_decision.json"),
        "e3_pairwise": load_decision("104_corrected_v3_pairwise_controls_decision.json"),
    }
    lines = [
        "# corrected-v3 zero-training diagnostics summary",
        "",
        f"- run_id: `{RUN_ID}`",
        f"- output_dir: `{OUT_DIR}`",
        "",
        "## Decisions",
        "",
        "| probe | status | key readout |",
        "|---|---|---|",
    ]
    e0 = decisions["e0_readout"]
    lines.append(f"| E0 paired readout | {e0.get('verdict', e0.get('status'))} | Δh={e0.get('delta_h', 'NA')} CI={e0.get('delta_h_ci95', 'NA')} |")
    e1 = decisions["e1_placebo"]
    lines.append(f"| E1 placebo | {e1.get('status')} | mean Δh={e1.get('mean_delta_h', 'NA')} |")
    e2 = decisions["e2_rewrite"]
    lines.append(f"| E2 rewrite ceiling | {e2.get('status')} | rewrite_h={e2.get('rewrite_mean_h', 'NA')} Δh={e2.get('mean_delta_h', 'NA')} |")
    e3 = decisions["e3_pairwise"]
    lines.append(f"| E3 pairwise controls | {e3.get('status')} | identical_non_tie={e3.get('identical_non_tie_rate', 'NA')} short_win={e3.get('short_win_rate', 'NA')} |")
    lines += [
        "",
        "## Next Read",
        "",
        "- If E1 is GOODHART-SUSPECTED, stop optimizing single humanness mean before rubric revision.",
        "- If E2 is GO_REWRITE_CEILING, consider v3.1 controlled-rewrite DPO with k=3 gates.",
        "- If E3 is BIAS-SUSPECTED, do not use pairwise judge as a hard gate until calibrated.",
    ]
    path = OUT_DIR / "105_corrected_v3_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit(f"RESULT v3_summary -> {path}")
    save_state(status="complete", stage="done", pid=None, summary=str(path))


def main() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print("usage: python -X utf8 run_corrected_v3.py")
        print("Environment:")
        print("  DASHSCOPE_API_KEY=...        required")
        print("  V3_RUN_ID=...                optional output/log namespace")
        print("  V3_BASE_INFER/V3_BASE_JUDGE  optional input overrides")
        print("  V3_CANDIDATE_INFER/V3_CANDIDATE_JUDGE optional input overrides")
        print("  V3_SKIP_PAIRWISE_CONTROLS=1  optional skip E3")
        return
    save_state(status="running", stage="preflight")
    try:
        preflight()
        run("e0_noise_calibration", [
            str(PY), "-X", "utf8", "pipeline/step100_judge_noise_calibration.py",
            "--base-infer", str(BASE_INFER),
            "--base-judge", str(BASE_JUDGE),
            "--candidate-infer", str(CAND_INFER),
            "--candidate-judge", str(CAND_JUDGE),
            "--out-dir", str(OUT_DIR),
        ])
        run("e0_paired_readout", [
            str(PY), "-X", "utf8", "pipeline/step101_paired_eval_stats.py",
            "--out-dir", str(OUT_DIR),
        ])
        run("e1_placebo_trace_surgery", [
            str(PY), "-X", "utf8", "pipeline/step102_placebo_trace_surgery.py",
            "--base-judge", str(BASE_JUDGE),
            "--out-dir", str(OUT_DIR),
        ])
        run("e2_rewrite_ceiling_probe", [
            str(PY), "-X", "utf8", "pipeline/step103_rewrite_ceiling_probe.py",
            "--base-judge", str(BASE_JUDGE),
            "--out-dir", str(OUT_DIR),
        ])
        if os.environ.get("V3_SKIP_PAIRWISE_CONTROLS", "0") != "1":
            run("e3_pairwise_controls", [
                str(PY), "-X", "utf8", "pipeline/step104_pairwise_judge_controls.py",
                "--base-judge", str(BASE_JUDGE),
                "--out-dir", str(OUT_DIR),
            ])
        write_summary()
    except BaseException as exc:
        save_state(status="failed", error=repr(exc))
        emit(f"PIPELINE FAILED | {exc!r}")
        raise


if __name__ == "__main__":
    main()
