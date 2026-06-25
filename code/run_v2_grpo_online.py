"""V2 DPO -> GRPO online continuation.

Reuses the old working ms-swift GRPO path: ``swift/grpo_on_model.sh``.
Swift does online rollout, reward calls, and LoRA updates; this runner only
builds the V2 train-only dataset, launches two GRPO phases, merges, and runs
the frozen-500 V2 evaluation.
"""

from __future__ import annotations

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

from config import CKPT_DIR, KIMI_API_KEY_ENV, LOG_DIR, OUTPUT_DIR, resolve_adapter  # noqa: E402
from pipeline import vllm_client  # noqa: E402
from pipeline.v2_common import v2_eval_paths, v2_summary_path  # noqa: E402
from pipeline.v2_paths import V2_EVAL, V2_PROBLEMS_TRAIN, V2_V1_SUPPORT  # noqa: E402

SCRIPTS = ROOT / "scripts"
SWIFT = ROOT / "swift"
PY = Path(os.environ.get("ZHJG_ENV", "/home/nvme02/conda/zhjg_rl")) / "bin" / "python"
WORK_DIR = Path(os.environ.get("ZHJG_WORK_DIR", "/home/nvme01/zhjg"))
MODEL_DIR = Path(os.environ.get("ZHJG_MODEL_DIR", str(WORK_DIR / "models")))
RAW_DIR = Path(LOG_DIR) / "v2_grpo_online"
STATE_FILE = RAW_DIR / "state.json"
EVENT_LOG = RAW_DIR / "events.log"

LINEAGE = os.environ.get("V2_GRPO_LINEAGE", "2s-2s-2s")
BASE_MODEL = Path(os.environ.get("V2_GRPO_BASE_MODEL", str(MODEL_DIR / "v2-dpo-2sigma-2s-2s-merged")))
GRPO_DATA = Path(os.environ.get(
    "V2_GRPO_DATA",
    str(Path(OUTPUT_DIR) / os.environ.get("V2_TAG", "v2") / f"70_grpo_data.v2-{LINEAGE}.jsonl"),
))
REWARD_AUDIT = RAW_DIR / "reward_audit.json"
SMOKE_LORA = Path(os.environ.get("V2_GRPO_SMOKE_LORA", str(Path(CKPT_DIR) / f"v2-grpo-smoke-{LINEAGE}-lora")))
WARMUP_LORA = Path(os.environ.get("V2_GRPO_WARMUP_LORA", str(Path(CKPT_DIR) / f"v2-grpo-warmup-{LINEAGE}-lora")))
WARMUP_MERGED = Path(os.environ.get("V2_GRPO_WARMUP_MERGED", str(MODEL_DIR / f"v2-grpo-warmup-{LINEAGE}-merged")))
FINAL_LORA = Path(os.environ.get("V2_GRPO_FINAL_LORA", str(Path(CKPT_DIR) / f"v2-grpo-2sigma-{LINEAGE}-lora")))
FINAL_MERGED = Path(os.environ.get("V2_GRPO_FINAL_MERGED", str(MODEL_DIR / f"v2-grpo-2sigma-{LINEAGE}-merged")))
EVAL_TAG = os.environ.get("V2_GRPO_EVAL_TAG", f"v2-{LINEAGE}-grpo")
SERVED_NAME = re.sub(r"[^A-Za-z0-9_]", "_", EVAL_TAG)
WARMUP_STEPS = int(os.environ.get("V2_GRPO_WARMUP_STEPS", "30"))
MAIN_STEPS = int(os.environ.get("V2_GRPO_MAIN_STEPS", "90"))
EVAL_AFTER = os.environ.get("V2_GRPO_EVAL_AFTER", "1") == "1"
RUN_REWARD_AUDIT = os.environ.get("V2_GRPO_REWARD_AUDIT", "1") == "1"
REWARD_AUDIT_KIMI = os.environ.get("V2_GRPO_REWARD_AUDIT_KIMI", "0") == "1"
RUN_SWIFT_SMOKE = os.environ.get("V2_GRPO_SWIFT_SMOKE", "1") == "1"
RUN_KIMI_SMOKE = os.environ.get("V2_GRPO_KIMI_SMOKE", "1") == "1"

SIGNAL_RE = re.compile(
    r"Train:|global_step|max_steps|'loss'|'reward'|'kl'|Capturing CUDA graphs|"
    r"Application startup complete|\[merge\]|\[grpo-on-model\]|Kimi|干净|规则|在池|"
    r"complete|ERROR|FAIL|Traceback",
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
        f.seek(max(0, path.stat().st_size - 160000))
        text = f.read().decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if SIGNAL_RE.search(line)]
    return lines[-1][-700:] if lines else ""


def run(stage: str, cmd: list, *, env: dict | None = None, optional: bool = False) -> None:
    raw_log = RAW_DIR / f"{stage}.log"
    cmd = [str(c) for c in cmd]
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    emit(f"START {stage}")
    emit("CMD   " + " ".join(cmd))
    with raw_log.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {now()} START {' '.join(cmd)} =====\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=full_env, stdout=f, stderr=subprocess.STDOUT, text=True)
        started = time.time()
        save_state(stage=stage, status="running", pid=proc.pid, raw_log=str(raw_log),
                   command=cmd, started_at=now())
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


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def adapter_complete(path: Path) -> bool:
    if not (path / ".done").exists():
        return False
    adapter = Path(resolve_adapter(str(path)))
    if not (adapter / "adapter_config.json").exists():
        raise RuntimeError(f"done marker exists but adapter missing: {path}")
    return True


def merged_complete(path: Path) -> bool:
    if not (path / ".done").exists():
        return False
    if not (path / "config.json").exists() or not any(path.glob("*.safetensors")):
        raise RuntimeError(f"done marker exists but merged model incomplete: {path}")
    return True


def move_interrupted(path: Path) -> None:
    if path.exists() and not (path / ".done").exists():
        moved = path.with_name(f"{path.name}.interrupted-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        emit(f"preserve interrupted output: {path} -> {moved}")
        path.rename(moved)


def stop_static_vllm() -> None:
    for name in ("merged_chain_vllm.pid", "vllm.pid"):
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


def report_complete(summary: Path, report: Path) -> bool:
    if not summary.exists() or not report.exists():
        return False
    try:
        s = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(s.get("n", 0)) == 500 and report.stat().st_size > 0


def grpo_data_current(path: Path) -> bool:
    if not path.exists() or count_jsonl(path) <= 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                return bool(row.get("v1_answer_pool_trainable") is True and row.get("v1_answers_json"))
    except Exception:
        return False
    return False


def ensure_grpo_data() -> None:
    if GRPO_DATA.exists() and count_jsonl(GRPO_DATA) > 0:
        if grpo_data_current(GRPO_DATA):
            emit(f"SKIP  build GRPO data; exists rows={count_jsonl(GRPO_DATA)} | {GRPO_DATA}")
            return
        moved = GRPO_DATA.with_name(f"{GRPO_DATA.stem}.stale-unfiltered-{datetime.now().strftime('%Y%m%d-%H%M%S')}{GRPO_DATA.suffix}")
        emit(f"REBUILD GRPO data; old schema/unfiltered pool: {GRPO_DATA} -> {moved}")
        GRPO_DATA.rename(moved)
    run("build_v2_grpo_data", [
        PY, "-X", "utf8", "pipeline/step_v2_build_grpo_data.py",
        "--train", str(V2_PROBLEMS_TRAIN),
        "--support", str(V2_V1_SUPPORT),
        "--eval", str(V2_EVAL),
        "--out", str(GRPO_DATA),
    ])


def ensure_reward_audit() -> None:
    if not RUN_REWARD_AUDIT:
        emit("SKIP  V2 GRPO reward audit; V2_GRPO_REWARD_AUDIT=0")
        return
    if REWARD_AUDIT.exists():
        try:
            old = json.loads(REWARD_AUDIT.read_text(encoding="utf-8"))
            if old.get("ok") and int(old.get("rows_checked", 0)) > 0:
                emit(f"SKIP  V2 GRPO reward audit; complete: {REWARD_AUDIT}")
                return
        except Exception:
            pass
    cmd = [
        PY, "-X", "utf8", "pipeline/step_v2_grpo_reward_audit.py",
        "--data", str(GRPO_DATA),
        "--out", str(REWARD_AUDIT),
        "--n", os.environ.get("V2_GRPO_REWARD_AUDIT_N", "32"),
    ]
    if REWARD_AUDIT_KIMI:
        cmd.append("--kimi")
    run("v2_grpo_reward_audit", cmd)


def kimi_smoke() -> None:
    needs_kimi = (MAIN_STEPS > 0 and os.environ.get("GRPO_V2_USE_KIMI", "1") != "0") or EVAL_AFTER or REWARD_AUDIT_KIMI
    if not needs_kimi:
        emit("SKIP  Kimi smoke; Kimi not needed by this run")
        return
    if not RUN_KIMI_SMOKE:
        emit("SKIP  Kimi smoke; V2_GRPO_KIMI_SMOKE=0")
        return
    marker = RAW_DIR / "kimi_smoke.ok"
    if marker.exists():
        emit(f"SKIP  Kimi smoke; complete: {marker}")
        return
    emit("START Kimi smoke")
    from pipeline import kimi_client  # local import keeps non-Kimi dry runs light
    reply = kimi_client.smoke()
    marker.write_text(json.dumps({"ok": True, "reply": reply, "at": now()}, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(f"END   Kimi smoke | reply={reply[:40]!r}")


def swift_smoke() -> None:
    if not RUN_SWIFT_SMOKE:
        emit("SKIP  swift GRPO smoke; V2_GRPO_SWIFT_SMOKE=0")
        return
    if adapter_complete(SMOKE_LORA):
        emit(f"SKIP  swift GRPO smoke; complete: {SMOKE_LORA}")
        return
    move_interrupted(SMOKE_LORA)
    stop_static_vllm()
    env = {
        "GRPO_SMOKE": "1",
        "GRPO_REWARD_FUNC": "v2_rule_warmup",
        "GRPO_STEPS": "2",
        "GRPO_LR": os.environ.get("V2_GRPO_SMOKE_LR", "5e-7"),
        "GRPO_BETA": os.environ.get("V2_GRPO_SMOKE_BETA", "0.08"),
        "GRPO_K": os.environ.get("V2_GRPO_SMOKE_K", "2"),
        "GRPO_MAX_COMPLETION": os.environ.get("V2_GRPO_SMOKE_MAX_COMPLETION", "256"),
        "GRPO_SAVE_STEPS": "1",
        "GRPO_SAVE_TOTAL_LIMIT": "2",
        "GRPO_V2_USE_KIMI": "0",
    }
    run("swift_v2_grpo_smoke2", ["bash", str(SWIFT / "grpo_on_model.sh"), str(GRPO_DATA), str(BASE_MODEL), str(SMOKE_LORA)], env=env)


def train_grpo(stage: str, base: Path, out: Path, reward_func: str, steps: int,
               *, lr: str, beta: str, use_kimi: bool) -> None:
    if steps <= 0:
        emit(f"SKIP  {stage}; steps={steps}")
        return
    if adapter_complete(out):
        emit(f"SKIP  {stage}; trained: {out}")
        return
    move_interrupted(out)
    stop_static_vllm()
    env = {
        "GRPO_REWARD_FUNC": reward_func,
        "GRPO_STEPS": str(steps),
        "GRPO_LR": lr,
        "GRPO_BETA": beta,
        "GRPO_K": os.environ.get("GRPO_K", "8"),
        "GRPO_SAVE_STEPS": os.environ.get("GRPO_SAVE_STEPS", "25"),
        "GRPO_MAX_COMPLETION": os.environ.get("GRPO_MAX_COMPLETION", "1536"),
        "GRPO_V2_USE_KIMI": "1" if use_kimi else "0",
        "GRPO_V2_KIMI_K": os.environ.get("GRPO_V2_KIMI_K", "2"),
        "GRPO_V2_KIMI_REQUIRED": os.environ.get("GRPO_V2_KIMI_REQUIRED", "1"),
        "KIMI_CACHE_MIN_K": os.environ.get("KIMI_CACHE_MIN_K", "2"),
        "GRPO_V2_KIMI_LOCK": os.environ.get("GRPO_V2_KIMI_LOCK", "1"),
        "GRPO_V2_KIMI_MIN_INTERVAL": os.environ.get("GRPO_V2_KIMI_MIN_INTERVAL", "0.0"),
    }
    run(stage, ["bash", str(SWIFT / "grpo_on_model.sh"), str(GRPO_DATA), str(base), str(out)], env=env)


def merge(stage: str, base: Path, lora_root: Path, output: Path) -> None:
    if merged_complete(output):
        emit(f"SKIP  {stage}; merged: {output}")
        return
    move_interrupted(output)
    adapter = Path(resolve_adapter(str(lora_root)))
    run(stage, ["bash", str(SCRIPTS / "merge_lora_model.sh"), str(base), str(adapter), str(output)])


def serve_model(model_dir: Path, served_name: str) -> None:
    stop_static_vllm()
    run(f"serve_{served_name}", ["bash", str(SCRIPTS / "serve_model_vllm.sh"), str(model_dir), served_name])
    save_state(stage=f"wait_{served_name}", status="running", pid=None,
               raw_log=str(Path(LOG_DIR) / "merged_chain_vllm.log"))
    emit(f"WAIT  vLLM {served_name} ({model_dir})")
    vllm_client.wait_ready(max_wait=1800)
    emit(f"READY vLLM {served_name}")


def evaluate_final() -> None:
    infer, scores, report = v2_eval_paths(EVAL_TAG)
    summary = v2_summary_path(EVAL_TAG)
    if report_complete(summary, report):
        emit(f"SKIP  eval; complete: {report}")
        return
    for p in (infer, scores, summary, report):
        if p.exists():
            moved = p.with_name(f"{p.stem}.stale-{datetime.now().strftime('%Y%m%d-%H%M%S')}{p.suffix}")
            emit(f"preserve stale eval artifact: {p} -> {moved}")
            p.rename(moved)
    serve_model(FINAL_MERGED, SERVED_NAME)
    run("v2_grpo_infer", [
        PY, "-X", "utf8", "pipeline/step03_eval_infer.py",
        "--model", SERVED_NAME,
        "--eval_file", str(V2_EVAL),
        "--out", str(infer),
    ])
    run("v2_grpo_eval", [
        PY, "-X", "utf8", "pipeline/step_v2_eval.py",
        "--infer", str(infer),
        "--scores", str(scores),
        "--report", str(report),
        "--summary", str(summary),
        "--support", str(V2_V1_SUPPORT),
        "--tag", EVAL_TAG,
    ])
    stop_static_vllm()


def preflight() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        BASE_MODEL / "config.json", PY, SWIFT / "grpo_on_model.sh",
        SCRIPTS / "merge_lora_model.sh", SCRIPTS / "serve_model_vllm.sh",
        ROOT / "pipeline" / "step_v2_grpo_reward_audit.py",
        V2_PROBLEMS_TRAIN, V2_V1_SUPPORT, V2_EVAL,
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise SystemExit("preflight missing:\n" + "\n".join(missing))
    active = subprocess.run(
        ["bash", "-lc", "pgrep -af '[s]wift/cli/rlhf.py|[t]orch.distributed.run' || true"],
        capture_output=True, text=True,
    ).stdout.strip()
    if active:
        raise SystemExit("other training processes are active:\n" + active)
    needs_kimi = (MAIN_STEPS > 0 and os.environ.get("GRPO_V2_USE_KIMI", "1") != "0") or EVAL_AFTER or REWARD_AUDIT_KIMI
    if needs_kimi and not os.environ.get(KIMI_API_KEY_ENV, "").strip():
        raise SystemExit(f"missing {KIMI_API_KEY_ENV}; GRPO online/eval needs Kimi")
    free_gib = shutil.disk_usage(WORK_DIR).free / 1024 ** 3
    missing_merges = int(WARMUP_STEPS > 0 and not merged_complete(WARMUP_MERGED)) + int(not merged_complete(FINAL_MERGED))
    if free_gib < 80 * missing_merges:
        raise SystemExit(f"disk free {free_gib:.1f}GiB is too low for {missing_merges} merge(s)")
    stop_static_vllm()
    emit(
        "PREFLIGHT OK | "
        f"base={BASE_MODEL} | data={GRPO_DATA} | warmup_steps={WARMUP_STEPS} | "
        f"main_steps={MAIN_STEPS} | reward_audit={RUN_REWARD_AUDIT} | kimi_smoke={RUN_KIMI_SMOKE} | "
        f"swift_smoke={RUN_SWIFT_SMOKE} | "
        f"final={FINAL_MERGED} | eval_tag={EVAL_TAG} | free={free_gib:.1f}GiB"
    )


def main() -> None:
    save_state(status="running", stage="preflight", completed=[],
               base=str(BASE_MODEL), final_lora=str(FINAL_LORA), final_merged=str(FINAL_MERGED),
               data=str(GRPO_DATA), eval_tag=EVAL_TAG)
    try:
        preflight()
        ensure_grpo_data()
        ensure_reward_audit()
        kimi_smoke()
        swift_smoke()
        base_for_main = BASE_MODEL
        if WARMUP_STEPS > 0:
            train_grpo("train_v2_grpo_warmup", BASE_MODEL, WARMUP_LORA, "v2_rule_warmup", WARMUP_STEPS,
                       lr=os.environ.get("V2_GRPO_WARMUP_LR", "7e-7"),
                       beta=os.environ.get("V2_GRPO_WARMUP_BETA", "0.08"),
                       use_kimi=False)
            merge("merge_v2_grpo_warmup", BASE_MODEL, WARMUP_LORA, WARMUP_MERGED)
            base_for_main = WARMUP_MERGED
        train_grpo("train_v2_grpo_online", base_for_main, FINAL_LORA, "v2_online", MAIN_STEPS,
                   lr=os.environ.get("V2_GRPO_MAIN_LR", "7e-7"),
                   beta=os.environ.get("V2_GRPO_MAIN_BETA", "0.06"),
                   use_kimi=True)
        merge("merge_v2_grpo_final", base_for_main, FINAL_LORA, FINAL_MERGED)
        if EVAL_AFTER:
            evaluate_final()
        save_state(status="complete", stage="done", pid=None, raw_log=None,
                   final_lora=str(FINAL_LORA), final_merged=str(FINAL_MERGED),
                   report=str(v2_eval_paths(EVAL_TAG)[2]), summary=str(v2_summary_path(EVAL_TAG)))
        emit(f"V2 GRPO COMPLETE | final={FINAL_MERGED} | report={v2_eval_paths(EVAL_TAG)[2]}")
    except BaseException as exc:
        save_state(status="failed", error=repr(exc))
        emit(f"V2 GRPO FAILED | {exc!r}")
        raise
    finally:
        try:
            stop_static_vllm()
        except Exception as exc:
            emit(f"WARN cleanup static vLLM: {exc!r}")


if __name__ == "__main__":
    main()
