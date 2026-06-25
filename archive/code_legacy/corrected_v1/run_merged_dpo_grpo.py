"""One-click corrected chain:

V1 + final RFT LoRA -> merged RFT base
  -> DPO with merged RFT base as frozen reference
  -> merge DPO LoRA -> merged DPO base
  -> GRPO from merged RFT base
  -> GRPO from merged DPO base
  -> evaluate every comparable model and write a comparison report.

Child process output is written to dedicated raw logs. This orchestrator prints
only stage transitions, heartbeats, and the latest useful metric line.
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

from config import (  # noqa: E402
    CKPT_DIR,
    CS_RFT_LORA_DIR,
    DPO_PAIRS,
    DPO_ROLLOUT,
    LOG_DIR,
    OUTPUT_DIR,
    V1_DIR,
    resolve_adapter,
    stage_eval_paths,
)
from pipeline import vllm_client  # noqa: E402

SCRIPTS = ROOT / "scripts"
SWIFT = ROOT / "swift"
PY = Path(os.environ.get("ZHJG_ENV", "/home/nvme02/conda/zhjg_rl")) / "bin" / "python"
WORK_DIR = Path(os.environ.get("ZHJG_WORK_DIR", "/home/nvme01/zhjg"))
MODEL_DIR = Path(os.environ.get("ZHJG_MODEL_DIR", str(WORK_DIR / "models")))
RAW_DIR = Path(LOG_DIR) / "merged_dpo_grpo"
STATE_FILE = RAW_DIR / "state.json"
EVENT_LOG = RAW_DIR / "events.log"
GRPO_DATA = Path(OUTPUT_DIR) / "70_grpo_data.jsonl"

RFT_MERGED = Path(
    os.environ.get("RFT_MERGED_MODEL_DIR", str(MODEL_DIR / "v1-32b-corrected-v1-rft-merged")))
DPO_LORA = Path(
    os.environ.get("MERGED_CHAIN_DPO_LORA_DIR", str(Path(CKPT_DIR) / "v1-32b-corrected-v1-rftmerged-dpo-lora")))
DPO_MERGED = Path(
    os.environ.get("DPO_MERGED_MODEL_DIR", str(MODEL_DIR / "v1-32b-corrected-v1-rft-dpo-merged")))
GRPO_RFT_LORA = Path(
    os.environ.get(
        "GRPO_RFT_MERGED_LORA_DIR", str(Path(CKPT_DIR) / "v1-32b-corrected-v1-grpo-from-rftmerged-lora")))
GRPO_DPO_LORA = Path(
    os.environ.get(
        "GRPO_DPO_MERGED_LORA_DIR", str(Path(CKPT_DIR) / "v1-32b-corrected-v1-grpo-from-dpomerged-lora")))
SUMMARY_REPORT = Path(OUTPUT_DIR) / "84_corrected_v1_merged_dpo_grpo_summary.md"

PROTECTED_HISTORICAL_DIRS = [
    Path(CKPT_DIR) / "v1-32b-coldstart-lora",
    Path(CKPT_DIR) / "v1-32b-cs-rft-lora",
    Path(CKPT_DIR) / "v1-32b-dpo-lora",
    Path(CKPT_DIR) / "v1-32b-grpo-lora",
    Path(CKPT_DIR) / "v1-32b-grpo-from-rft-lora",
    Path(CKPT_DIR) / "v1-32b-grpo-from-rft-lora-ref_rft_v3",
]
NEW_OUTPUT_DIRS = [RFT_MERGED, DPO_LORA, DPO_MERGED, GRPO_RFT_LORA, GRPO_DPO_LORA]

REPORTS = {
    "RFT merged base": "80_corrected_v1_rft_merged_base",
    "DPO on merged base": "81_corrected_v1_dpo_on_rft_merged",
    "GRPO from RFT merged": "82_corrected_v1_grpo_from_rft_merged",
    "GRPO from DPO merged": "83_corrected_v1_grpo_from_dpo_merged",
}

SIGNAL_RE = re.compile(
    r"Train:|global_step|max_steps|'loss'|'reward'|'kl'|Capturing CUDA graphs|Application startup complete|"
    r"\[merge\]|\[dpo-on-model\]|\[grpo-on-model\]|完成|complete|ERROR|FAIL|Traceback",
    re.IGNORECASE,
)
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


def run(stage: str, cmd: list[str], *, optional: bool = False, env: dict | None = None) -> None:
    raw_log = RAW_DIR / f"{stage}.log"
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    emit(f"START {stage}")
    emit("CMD   " + " ".join(cmd))
    with raw_log.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {now()} START {' '.join(cmd)} =====\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=full_env, stdout=f, stderr=subprocess.STDOUT, text=True)
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
        save_state(status="failed", returncode=rc)
        emit(f"{'WARN' if optional else 'FAIL'} {stage} rc={rc}; see {raw_log}")
        if optional:
            return
        raise SystemExit(rc)
    complete(stage)
    emit(f"END   {stage}")


def move_interrupted(path: Path) -> None:
    if path.exists() and not (path / ".done").exists():
        moved = path.with_name(f"{path.name}.interrupted-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        emit(f"preserve interrupted output: {path} -> {moved}")
        path.rename(moved)


def adapter_complete(path: Path) -> bool:
    if not (path / ".done").exists():
        return False
    adapter = Path(resolve_adapter(str(path)))
    if not (adapter / "adapter_config.json").exists():
        raise RuntimeError(f"done marker exists but adapter is missing: {path}")
    return True


def merged_model_complete(path: Path) -> bool:
    if not (path / ".done").exists():
        return False
    if not (path / "config.json").exists() or not any(path.glob("*.safetensors")):
        raise RuntimeError(f"done marker exists but merged model is incomplete: {path}")
    return True


def report_complete(path: Path) -> bool:
    if not path.exists() or not path.stat().st_size:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(marker in text for marker in (
        "humanness 均值", "grounded(忠于参考) 均值", "准确率(平均分,漂移)", "correct+partial%"))


def protect_historical_outputs() -> None:
    normalized_new = [p.resolve(strict=False) for p in NEW_OUTPUT_DIRS]
    normalized_old = {p.resolve(strict=False) for p in PROTECTED_HISTORICAL_DIRS}
    if len(set(normalized_new)) != len(normalized_new):
        raise SystemExit("new corrected-v1 output directories must be distinct")
    collisions = [
        str(p) for p in normalized_new
        if any(p == old or p.is_relative_to(old) or old.is_relative_to(p) for old in normalized_old)
    ]
    if collisions:
        raise SystemExit("refusing to use protected historical output directories:\n" + "\n".join(collisions))
    bad_names = [str(p) for p in NEW_OUTPUT_DIRS if "corrected-v1" not in p.name]
    if bad_names:
        raise SystemExit("new output directory names must contain corrected-v1:\n" + "\n".join(bad_names))


def stop_static_vllm() -> None:
    for name in ("vllm.pid", "merged_chain_vllm.pid"):
        pidf = Path(LOG_DIR) / name
        if not pidf.exists():
            continue
        try:
            pid = int(pidf.read_text().strip())
            emit(f"stop static vLLM process group {pid} ({name})")
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
    raise RuntimeError("port 8000 is still serving after stopping known vLLM process groups")


def serve_model(model_dir: Path, served_name: str, adapter: Path | None = None) -> None:
    stop_static_vllm()
    # vLLM model registry names must be unique: when serving an adapter, expose
    # the full base under a private name and the adapter under the requested name.
    base_served_name = f"{served_name}_base" if adapter else served_name
    args = ["bash", str(SCRIPTS / "serve_model_vllm.sh"), str(model_dir), base_served_name]
    if adapter:
        args.extend([served_name, str(adapter)])
    run(f"serve_{served_name}", args)
    save_state(stage=f"wait_{served_name}", status="running", pid=None, raw_log=str(Path(LOG_DIR) / "merged_chain_vllm.log"))
    emit(f"WAIT  vLLM {served_name}")
    vllm_client.wait_ready(max_wait=1800)
    emit(f"READY vLLM {served_name}")


def evaluate(label: str, tag: str, model_dir: Path, adapter_root: Path | None = None) -> Path:
    infer, judge, report = map(Path, stage_eval_paths(tag))
    if report_complete(report):
        emit(f"SKIP  eval {label}; report exists: {report}")
        return report
    if report.exists():
        moved = report.with_name(f"{report.stem}.interrupted-{datetime.now().strftime('%Y%m%d-%H%M%S')}{report.suffix}")
        emit(f"preserve incomplete report: {report} -> {moved}")
        report.rename(moved)
    adapter = Path(resolve_adapter(str(adapter_root))) if adapter_root else None
    model_name = re.sub(r"[^A-Za-z0-9_]", "_", tag)
    serve_model(model_dir, model_name, adapter)
    run(f"{tag}_infer", [str(PY), "-X", "utf8", "pipeline/step03_eval_infer.py",
                         "--model", model_name, "--out", str(infer)])
    run(f"{tag}_judge", [str(PY), "-X", "utf8", "pipeline/step04_judge.py",
                         "--in", str(infer), "--out", str(judge)])
    run(f"{tag}_report", [str(PY), "-X", "utf8", "pipeline/step05_report.py",
                          "--in", str(judge), "--out", str(report), "--tag", tag])
    stop_static_vllm()
    return report


def preflight() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    protect_historical_outputs()
    required = [Path(V1_DIR) / "config.json", PY, ROOT / "swift" / "dpo_on_model.sh",
                ROOT / "swift" / "grpo_on_model.sh"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("preflight missing:\n" + "\n".join(missing))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is required for evaluation")
    rft_adapter = Path(resolve_adapter(CS_RFT_LORA_DIR))
    if not (rft_adapter / "adapter_config.json").exists():
        raise SystemExit(f"final RFT adapter not found: {rft_adapter}")
    free_gib = shutil.disk_usage(WORK_DIR).free / 1024**3
    missing_merged = sum(not merged_model_complete(p) for p in (RFT_MERGED, DPO_MERGED))
    required_free_gib = 80 * missing_merged
    if free_gib < required_free_gib:
        raise SystemExit(
            f"need about {required_free_gib} GiB free for {missing_merged} missing merged model(s); "
            f"only {free_gib:.1f} GiB available")
    stop_static_vllm()
    active = subprocess.run(
        ["bash", "-lc", "pgrep -af '[s]wift/cli/rlhf.py|[t]orch.distributed.run' || true"],
        capture_output=True, text=True).stdout.strip()
    if active:
        raise SystemExit("other training processes are still active:\n" + active)
    emit(f"preflight OK | RFT adapter={rft_adapter} | free={free_gib:.1f}GiB")
    return rft_adapter


def ensure_dpo_pairs() -> None:
    pairs = Path(DPO_PAIRS)
    if pairs.exists() and pairs.stat().st_size:
        emit(f"SKIP  DPO pairs exist: {pairs}")
        return
    rollout = Path(DPO_ROLLOUT)
    if not rollout.exists() or not rollout.stat().st_size:
        serve_model(RFT_MERGED, "rft_merged_rollout")
        run("build_dpo_rollout", [str(PY), "-X", "utf8", "pipeline/step10_rollout.py",
                                  "--model", "rft_merged_rollout", "--out", str(rollout)])
        stop_static_vllm()
    run("build_dpo_pairs", [str(PY), "-X", "utf8", "pipeline/step12_build_dpo_pairs.py",
                            "--rollout", str(rollout), "--out", str(pairs)], env={"PMI_ENABLED": "0"})


def ensure_grpo_data() -> None:
    if GRPO_DATA.exists() and GRPO_DATA.stat().st_size:
        emit(f"SKIP  GRPO data exists: {GRPO_DATA}")
        return
    run("build_grpo_data", [str(PY), "-X", "utf8", "pipeline/step13_build_grpo_data.py", "--out", str(GRPO_DATA)])


def merge(stage: str, base: Path, adapter: Path, output: Path) -> None:
    if merged_model_complete(output):
        emit(f"SKIP  {stage}; merged model exists: {output}")
        return
    run(stage, ["bash", str(SCRIPTS / "merge_lora_model.sh"), str(base), str(adapter), str(output)])


def train_dpo() -> None:
    if adapter_complete(DPO_LORA):
        emit(f"SKIP  DPO training complete: {DPO_LORA}")
        return
    move_interrupted(DPO_LORA)
    stop_static_vllm()
    run("train_dpo_on_rft_merged",
        ["bash", str(SWIFT / "dpo_on_model.sh"), DPO_PAIRS, str(RFT_MERGED), str(DPO_LORA)])
    run("plot_dpo_on_rft_merged",
        [str(PY), "-X", "utf8", "pipeline/plot_loss.py", "--dir", str(DPO_LORA), "--tag", "dpo_on_rft_merged"],
        optional=True)


def train_grpo(stage: str, base: Path, output: Path) -> None:
    if adapter_complete(output):
        emit(f"SKIP  {stage}; training complete: {output}")
        return
    move_interrupted(output)
    stop_static_vllm()
    run(stage, ["bash", str(SWIFT / "grpo_on_model.sh"), str(GRPO_DATA), str(base), str(output)])
    run(f"plot_{stage}",
        [str(PY), "-X", "utf8", "pipeline/plot_loss.py", "--dir", str(output), "--tag", stage],
        optional=True)


def write_summary(reports: dict[str, Path]) -> None:
    cmd = [str(PY), "-X", "utf8", "pipeline/step14_compare_merged_chain.py", "--out", str(SUMMARY_REPORT)]
    for label, report in reports.items():
        cmd.extend(["--report", f"{label}={report}"])
    run("write_final_summary", cmd)


def main() -> None:
    save_state(status="running", stage="preflight", completed=[])
    reports = {}
    try:
        rft_adapter = preflight()
        merge("merge_v1_rft", Path(V1_DIR), rft_adapter, RFT_MERGED)
        reports["RFT merged base"] = evaluate("RFT merged base", REPORTS["RFT merged base"], RFT_MERGED)

        ensure_dpo_pairs()
        train_dpo()
        dpo_adapter = Path(resolve_adapter(str(DPO_LORA)))
        merge("merge_rft_dpo", RFT_MERGED, dpo_adapter, DPO_MERGED)
        reports["DPO on merged base"] = evaluate(
            "DPO on merged base", REPORTS["DPO on merged base"], DPO_MERGED)

        ensure_grpo_data()
        train_grpo("train_grpo_from_rft_merged", RFT_MERGED, GRPO_RFT_LORA)
        reports["GRPO from RFT merged"] = evaluate(
            "GRPO from RFT merged", REPORTS["GRPO from RFT merged"], RFT_MERGED, GRPO_RFT_LORA)

        train_grpo("train_grpo_from_dpo_merged", DPO_MERGED, GRPO_DPO_LORA)
        reports["GRPO from DPO merged"] = evaluate(
            "GRPO from DPO merged", REPORTS["GRPO from DPO merged"], DPO_MERGED, GRPO_DPO_LORA)

        write_summary(reports)
        save_state(status="complete", stage="done", pid=None, raw_log=None, summary=str(SUMMARY_REPORT))
        emit(f"PIPELINE COMPLETE | summary={SUMMARY_REPORT}")
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
