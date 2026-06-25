"""One-click corrected-v2 mini experiment.

Plan X:
1) Reuse final-RFT 60_dpo_rollout candidate pool.
2) Kimi-score 120 queries x 8 candidates.
3) Build strict v2 DPO pairs + heldout.
4) Rejudge a pair sample.
5) Train mini DPO only if data gates pass.
6) Evaluate RFT merged base + v2 mini DPO LoRA without merging.

Logs are isolated under logs/corrected_v2; this script prints only compact
stage transitions, heartbeats, and high-value RESULT/PROGRESS lines.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CKPT_DIR, DPO_ROLLOUT, LOG_DIR, OUTPUT_DIR, VLLM_BASE_URL, resolve_adapter, stage_eval_paths  # noqa: E402
from pipeline import vllm_client  # noqa: E402

PY = Path(os.environ.get("ZHJG_ENV", "/home/nvme02/conda/zhjg_rl")) / "bin" / "python"
WORK_DIR = Path(os.environ.get("ZHJG_WORK_DIR", "/home/nvme01/zhjg"))
MODEL_DIR = Path(os.environ.get("ZHJG_MODEL_DIR", str(WORK_DIR / "models")))
RAW_DIR = Path(os.environ.get("CORRECTED_V2_LOG_DIR", str(Path(LOG_DIR) / "corrected_v2")))
STATE_FILE = RAW_DIR / "state.json"
EVENT_LOG = RAW_DIR / "events.log"
SCRIPTS = ROOT / "scripts"
SWIFT = ROOT / "swift"

RFT_MERGED = Path(os.environ.get("RFT_MERGED_MODEL_DIR", str(MODEL_DIR / "v1-32b-corrected-v1-rft-merged")))
DPO_V2_LORA = Path(os.environ.get("CORRECTED_V2_DPO_LORA_DIR", str(Path(CKPT_DIR) / "v1-32b-corrected-v2-mini-dpo-lora")))

SCORES = Path(OUTPUT_DIR) / "93_corrected_v2_rollout_scores.jsonl"
PAIRS = Path(OUTPUT_DIR) / "94_corrected_v2_dpo_pairs.jsonl"
PAIR_DECISION = Path(OUTPUT_DIR) / "94_corrected_v2_pair_decision.json"
REJUDGE_ROWS = Path(OUTPUT_DIR) / "95b_corrected_v2_pair_rejudge_all.jsonl"
REJUDGE_REPORT = Path(OUTPUT_DIR) / "95b_corrected_v2_pair_rejudge_all_report.md"
REJUDGE_DECISION = Path(OUTPUT_DIR) / "95b_corrected_v2_pair_rejudge_all_decision.json"
STABLE_PAIRS = Path(OUTPUT_DIR) / "95b_corrected_v2_stable_dpo_pairs.jsonl"
STABLE_REPORT = Path(OUTPUT_DIR) / "95b_corrected_v2_stable_pair_report.md"
STABLE_DECISION = Path(OUTPUT_DIR) / "95b_corrected_v2_stable_pair_decision.json"
BASE_REPORT = Path(OUTPUT_DIR) / "80_corrected_v1_rft_merged_base_report.md"
MINI_TAG = "96_corrected_v2_mini_dpo"
SUMMARY = Path(OUTPUT_DIR) / "97_corrected_v2_mini_summary.md"

SIGNAL_RE = re.compile(
    r"RESULT|PROGRESS|\[dpo-v2\]|Train:|global_step|max_steps|loss|rewards/chosen|rewards/rejected|"
    r"Application startup complete|ERROR|FAIL|Traceback|complete",
    re.IGNORECASE,
)
REPORT_PATTERNS = {
    "humanness": r"humanness 均值：\*\*([0-9.]+)\*\*",
    "grounded": r"grounded\(忠于参考\) 均值：\*\*([0-9.]+)\*\*",
    "accuracy": r"准确率\(平均分,漂移\)：\*\*([0-9.]+)\*\*",
    "correct": r"correct%：\*\*([0-9.]+)%",
    "correct_partial": r"correct\+partial%：\*\*([0-9.]+)%",
}
_state = {"status": "starting", "stage": "preflight", "completed": []}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def emit(message: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{now()} | {message}"
    print(line, flush=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_state(**updates) -> None:
    _state.update(updates, updated_at=now())
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def complete(stage: str) -> None:
    if stage not in _state["completed"]:
        _state["completed"].append(stage)
    save_state(status="running")


def latest_signal(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 131072))
        text = f.read().decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if SIGNAL_RE.search(line)]
    return lines[-1][-500:] if lines else ""


def child_env(extra: dict | None = None) -> dict:
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "ZHJG_LOG_FILE": str(RAW_DIR / "pipeline.log"),
        "ZHJG_FILE_LOG_LEVEL": "INFO",
        "ZHJG_CONSOLE_LOG_LEVEL": "WARNING",
        "PMI_ENABLED": "0",
    }
    if extra:
        env.update(extra)
    return env


def run(stage: str, cmd: list[str], *, env: dict | None = None, optional: bool = False) -> int:
    raw_log = RAW_DIR / "raw" / f"{stage}.log"
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    emit(f"START {stage}")
    emit("CMD   " + " ".join(cmd))
    with raw_log.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {now()} START {' '.join(cmd)} =====\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env(env), stdout=f, stderr=subprocess.STDOUT, text=True)
        started = time.time()
        save_state(stage=stage, status="running", pid=proc.pid, raw_log=str(raw_log), command=cmd, started_at=now())
        last_line = ""
        last_heartbeat = 0.0
        while proc.poll() is None:
            time.sleep(10)
            line = latest_signal(raw_log)
            if line and line != last_line:
                emit(f"{stage} | {line}")
                last_line = line
            elif time.time() - last_heartbeat >= 60:
                emit(f"{stage} | alive pid={proc.pid} elapsed={int(time.time() - started)}s")
                last_heartbeat = time.time()
        rc = proc.returncode
        f.write(f"===== {now()} END rc={rc} =====\n")
    if rc != 0:
        emit(f"{'WARN' if optional else 'FAIL'} {stage} rc={rc}; see {raw_log}")
        save_state(status="failed", returncode=rc)
        if not optional:
            raise SystemExit(rc)
    else:
        complete(stage)
        emit(f"END   {stage}")
    return rc


def stop_static_vllm() -> None:
    for base in (Path(LOG_DIR), RAW_DIR):
        for name in ("vllm.pid", "merged_chain_vllm.pid"):
            pidf = base / name
            if not pidf.exists():
                continue
            try:
                pid = int(pidf.read_text().strip())
                emit(f"stop static vLLM process group {pid} ({pidf})")
                os.killpg(pid, signal.SIGTERM)
                time.sleep(8)
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except (ValueError, ProcessLookupError, PermissionError):
                pass
            pidf.unlink(missing_ok=True)
    for _ in range(12):
        if not vllm_client.health():
            return
        time.sleep(5)
    raise RuntimeError(f"port still serves after stopping known vLLM groups: {VLLM_BASE_URL}")


def serve_model(model_dir: Path, served_name: str, adapter_root: Path | None = None) -> None:
    stop_static_vllm()
    adapter = Path(resolve_adapter(str(adapter_root))) if adapter_root else None
    base_name = f"{served_name}_base" if adapter else served_name
    args = ["bash", str(SCRIPTS / "serve_model_vllm.sh"), str(model_dir), base_name]
    if adapter:
        args.extend([served_name, str(adapter)])
    run(f"serve_{served_name}", args, env={"ZHJG_LOG_DIR": str(RAW_DIR)})
    emit(f"WAIT  vLLM {served_name}")
    save_state(stage=f"wait_{served_name}", status="running", pid=None, raw_log=str(RAW_DIR / "merged_chain_vllm.log"))
    vllm_client.wait_ready(max_wait=1800)
    emit(f"READY vLLM {served_name}")


def report_complete(path: Path) -> bool:
    if not path.exists() or not path.stat().st_size:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(marker in text for marker in ("humanness 均值", "grounded(忠于参考) 均值", "correct+partial%"))


def adapter_complete(path: Path) -> bool:
    if not (path / ".done").exists():
        return False
    adapter = Path(resolve_adapter(str(path)))
    if not (adapter / "adapter_config.json").exists():
        raise RuntimeError(f"done marker exists but adapter is missing: {path}")
    return True


def move_interrupted(path: Path) -> None:
    if path.exists() and not (path / ".done").exists():
        moved = path.with_name(f"{path.name}.interrupted-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        emit(f"preserve interrupted output: {path} -> {moved}")
        path.rename(moved)


def evaluate() -> Path:
    infer, judge, report = map(Path, stage_eval_paths(MINI_TAG))
    if report_complete(report):
        emit(f"SKIP  eval report exists: {report}")
        return report
    if report.exists():
        moved = report.with_name(f"{report.stem}.interrupted-{datetime.now().strftime('%Y%m%d-%H%M%S')}{report.suffix}")
        emit(f"preserve incomplete report: {report} -> {moved}")
        report.rename(moved)
    serve_model(RFT_MERGED, MINI_TAG, DPO_V2_LORA)
    run(f"{MINI_TAG}_infer", [str(PY), "-X", "utf8", "pipeline/step03_eval_infer.py", "--model", MINI_TAG, "--out", str(infer)])
    run(f"{MINI_TAG}_judge", [str(PY), "-X", "utf8", "pipeline/step04_judge.py", "--in", str(infer), "--out", str(judge)])
    run(f"{MINI_TAG}_report", [str(PY), "-X", "utf8", "pipeline/step05_report.py", "--in", str(judge), "--out", str(report), "--tag", MINI_TAG])
    stop_static_vllm()
    return report


def parse_report(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    row = {}
    for key, pattern in REPORT_PATTERNS.items():
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"cannot parse {key} from {path}")
        row[key] = float(m.group(1))
    return row


def write_summary(mini_report: Path) -> None:
    base = parse_report(BASE_REPORT)
    mini = parse_report(mini_report)
    pair_decision = json.loads(PAIR_DECISION.read_text(encoding="utf-8"))
    rejudge_decision = json.loads(REJUDGE_DECISION.read_text(encoding="utf-8"))
    stable_decision = json.loads(STABLE_DECISION.read_text(encoding="utf-8"))
    dh = mini["humanness"] - base["humanness"]
    dacc = mini["accuracy"] - base["accuracy"]
    lines = [
        "# corrected-v2 mini DPO summary",
        "",
        "| model | humanness | grounded | acc | correct% | correct+partial% | Δh vs RFT | Δacc vs RFT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| RFT merged base | {base['humanness']:.3f} | {base['grounded']:.3f} | {base['accuracy']:.3f} | {base['correct']:.1f}% | {base['correct_partial']:.1f}% | +0.000 | +0.000 |",
        f"| corrected-v2 mini DPO | {mini['humanness']:.3f} | {mini['grounded']:.3f} | {mini['accuracy']:.3f} | {mini['correct']:.1f}% | {mini['correct_partial']:.1f}% | {dh:+.3f} | {dacc:+.3f} |",
        "",
        "## Data Gates",
        f"- pair build: {pair_decision['status']} train={pair_decision['train_pairs']} heldout={pair_decision['heldout_pairs']} headroom={pair_decision['headroom_text']} hard_negative={100*pair_decision['hard_negative_rate']:.1f}%",
        f"- rejudge: {rejudge_decision['status']} direction={100*rejudge_decision['direction_rate']:.1f}% sampled={rejudge_decision['sampled_pairs']}",
        f"- stable filter: {stable_decision['status']} train={stable_decision['train_pairs']} heldout={stable_decision['heldout_pairs']} stable={stable_decision['stable_pairs']}/{stable_decision['rejudged_pairs']} mean_h={stable_decision['mean_stable_rejudge_h_margin']:+.3f}",
        "",
        "## Readout",
        f"- paired target direction proxy: Δh={dh:+.3f}, Δacc={dacc:+.3f}.",
        "- This is a mini validation run; do not promote to full unless humanness moves up while guardrails stay healthy.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit(f"RESULT summary Δh={dh:+.3f} Δacc={dacc:+.3f} -> {SUMMARY}")


def preflight() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not PY.exists():
        raise SystemExit(f"missing python env: {PY}")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is required")
    if not (RFT_MERGED / "config.json").exists():
        raise SystemExit(f"missing RFT merged base; run corrected-v1 chain first: {RFT_MERGED}")
    if not (Path(DPO_ROLLOUT).exists() and Path(DPO_ROLLOUT).stat().st_size):
        raise SystemExit(f"missing final-RFT rollout pool: {DPO_ROLLOUT}")
    if not BASE_REPORT.exists():
        raise SystemExit(f"missing RFT merged baseline report: {BASE_REPORT}")
    if "corrected-v2" not in DPO_V2_LORA.name:
        raise SystemExit(f"refusing non-corrected-v2 output dir: {DPO_V2_LORA}")
    free_gib = shutil.disk_usage(WORK_DIR).free / 1024**3
    active = subprocess.run(
        ["bash", "-lc", "pgrep -af '[s]wift/cli/rlhf.py|[t]orch.distributed.run' || true"],
        capture_output=True, text=True,
    ).stdout.strip()
    if active:
        raise SystemExit("other training processes are active:\n" + active)
    stop_static_vllm()
    emit(f"preflight OK | RFT_MERGED={RFT_MERGED} | rollout={DPO_ROLLOUT} | free={free_gib:.1f}GiB")


def main() -> None:
    save_state(status="running", stage="preflight", completed=[])
    try:
        preflight()
        run("score_rollout_candidates", [str(PY), "-X", "utf8", "pipeline/step93_kimi_score_rollouts.py",
                                         "--rollout", DPO_ROLLOUT, "--out", str(SCORES)])
        run("build_v2_dpo_pairs", [str(PY), "-X", "utf8", "pipeline/step94_build_dpo_pairs_v2.py",
                                   "--scores", str(SCORES)])
        pair_decision = json.loads(PAIR_DECISION.read_text(encoding="utf-8"))
        if pair_decision.get("status") != "GO":
            save_state(status="no_go", stage="pair_build", pid=None, raw_log=None, summary=str(Path(OUTPUT_DIR) / "94_corrected_v2_pair_report.md"))
            emit(f"NO-GO pair_build | see {Path(OUTPUT_DIR) / '94_corrected_v2_pair_report.md'}")
            return

        run("rejudge_v2_pairs_all", [str(PY), "-X", "utf8", "pipeline/step95_rejudge_pairs_v2.py",
                                      "--meta", str(Path(OUTPUT_DIR) / "94_corrected_v2_dpo_pairs_meta.jsonl"),
                                      "--out", str(REJUDGE_ROWS),
                                      "--report", str(REJUDGE_REPORT),
                                      "--decision", str(REJUDGE_DECISION)])
        rejudge_decision = json.loads(REJUDGE_DECISION.read_text(encoding="utf-8"))
        if rejudge_decision.get("status") != "GO":
            emit(f"WARN  rejudge sample gate={rejudge_decision.get('status')} direction={100*rejudge_decision.get('direction_rate', 0):.1f}%; continue to stable filtering only")

        run("filter_stable_v2_pairs", [str(PY), "-X", "utf8", "pipeline/step95b_filter_stable_pairs_v2.py",
                                       "--rejudge", str(REJUDGE_ROWS),
                                       "--out", str(STABLE_PAIRS),
                                       "--report", str(STABLE_REPORT),
                                       "--decision", str(STABLE_DECISION)])
        stable_decision = json.loads(STABLE_DECISION.read_text(encoding="utf-8"))
        if stable_decision.get("status") != "GO":
            save_state(status="no_go", stage="stable_filter", pid=None, raw_log=None, summary=str(STABLE_REPORT))
            emit(f"NO-GO stable_filter | see {STABLE_REPORT}")
            return

        if adapter_complete(DPO_V2_LORA):
            emit(f"SKIP  DPO v2 training complete: {DPO_V2_LORA}")
        else:
            move_interrupted(DPO_V2_LORA)
            stop_static_vllm()
            run("train_v2_mini_dpo", ["bash", str(SWIFT / "dpo_v2.sh"), str(STABLE_PAIRS), str(RFT_MERGED), str(DPO_V2_LORA)])

        mini_report = evaluate()
        write_summary(mini_report)
        save_state(status="complete", stage="done", pid=None, raw_log=None, summary=str(SUMMARY))
        emit(f"PIPELINE COMPLETE | summary={SUMMARY}")
    except BaseException as exc:
        save_state(status="failed", error=repr(exc))
        emit(f"PIPELINE FAILED | {exc!r}")
        raise
    finally:
        try:
            stop_static_vllm()
        except Exception as exc:
            emit(f"WARN cleanup static vLLM: {exc!r}")


if __name__ == "__main__":
    main()
