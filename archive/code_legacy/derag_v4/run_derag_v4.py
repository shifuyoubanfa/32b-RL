"""One-click derag_v4 DPO -> GRPO chain.

Goal:
  RFT merged base -> residual-trace rewrite SFT -> on-policy DPO ->
  derag_v4 GRPO -> deterministic reports.

This orchestrator is intentionally conservative about old artifacts:
all outputs live under output/logs/ckpts/models derag_v4/<run_id>.
If a gate fails, it either tries the planned fallback or stops with NO-GO.
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
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CKPT_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    SFT_EVAL,
    SFT_TRAIN,
    resolve_adapter,
)
from pipeline import vllm_client  # noqa: E402


RUN_ID = os.environ.get("DERAG_V4_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
WORK_DIR = Path(os.environ.get("ZHJG_WORK_DIR", "/home/nvme01/zhjg"))
MODEL_DIR = Path(os.environ.get("ZHJG_MODEL_DIR", str(WORK_DIR / "models")))
PY = Path(os.environ.get("ZHJG_ENV", "/home/nvme02/conda/zhjg_rl")) / "bin" / "python"
SCRIPTS = ROOT / "scripts"
SWIFT = ROOT / "swift"

RFT_MERGED = Path(os.environ.get(
    "DERAG_V4_RFT_MERGED",
    str(MODEL_DIR / "v1-32b-corrected-v1-rft-merged"),
))

OUT_DIR = Path(OUTPUT_DIR) / "derag_v4" / RUN_ID
LOG_ROOT = Path(LOG_DIR) / "derag_v4" / RUN_ID
RAW_DIR = LOG_ROOT / "raw"
STATE_FILE = LOG_ROOT / "state.json"
EVENT_LOG = LOG_ROOT / "events.log"
CKPT_ROOT = Path(CKPT_DIR) / "derag_v4" / RUN_ID
MODEL_ROOT = MODEL_DIR / "derag_v4" / RUN_ID

S1_LORA = CKPT_ROOT / "s1_rewrite_sft_lora"
S1_MERGED = MODEL_ROOT / "s1_rewrite_sft_merged"
S2_LORA_BASE = CKPT_ROOT / "s2_dpo_lora"
S2_MERGED_BASE = MODEL_ROOT / "s2_dpo_merged"
S3_LORA_BASE = CKPT_ROOT / "s3_grpo_lora"
S3_MERGED_BASE = MODEL_ROOT / "s3_grpo_merged"

SIGNAL_RE = re.compile(
    r"RESULT|PROGRESS|NO-GO|GO|PASS|FAIL|Traceback|ERROR|"
    r"Train:|global_step|max_steps|'loss'|'reward'|'kl'|"
    r"\[sft-on-model\]|\[dpo-v2\]|\[grpo-on-model\]|\[merge\]|"
    r"Application startup complete|Capturing CUDA graphs",
    re.IGNORECASE,
)

_state: dict[str, Any] = {
    "status": "starting",
    "stage": "preflight",
    "run_id": RUN_ID,
    "completed": [],
    "out_dir": str(OUT_DIR),
    "log_root": str(LOG_ROOT),
    "ckpt_root": str(CKPT_ROOT),
    "model_root": str(MODEL_ROOT),
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def emit(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{now()} | {message}"
    print(line, flush=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_state(**updates: Any) -> None:
    _state.update(updates, updated_at=now())
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def complete(stage: str) -> None:
    if stage not in _state["completed"]:
        _state["completed"].append(stage)
    save_state(status="running", pid=None)


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "DERAG_V4_RUN_ID": RUN_ID,
        "ZHJG_LOG_DIR": str(LOG_ROOT),
        "ZHJG_LOG_FILE": str(LOG_ROOT / "pipeline.log"),
        "ZHJG_CONSOLE_LOG_LEVEL": os.environ.get("ZHJG_CONSOLE_LOG_LEVEL", "WARNING"),
    }
    if extra:
        env.update(extra)
    return env


def latest_signal(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 192 * 1024))
        text = f.read().decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if SIGNAL_RE.search(ln)]
    return lines[-1][-700:] if lines else ""


def run(stage: str, cmd: list[str], *, env: dict[str, str] | None = None,
        allow_nogo: bool = False, optional: bool = False) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_log = RAW_DIR / f"{stage}.log"
    emit(f"START {stage}")
    emit("CMD   " + " ".join(map(str, cmd)))
    with raw_log.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {now()} START {' '.join(map(str, cmd))} =====\n")
        f.flush()
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=ROOT,
            env=child_env(env),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started = time.time()
        save_state(status="running", stage=stage, pid=proc.pid, raw_log=str(raw_log),
                   command=[str(x) for x in cmd], started_at=now())
        last = ""
        last_heartbeat = 0.0
        while proc.poll() is None:
            time.sleep(10)
            sig = latest_signal(raw_log)
            if sig and sig != last:
                emit(f"{stage} | {sig}")
                last = sig
            elif time.time() - last_heartbeat >= 60:
                emit(f"{stage} | alive pid={proc.pid} elapsed={int(time.time() - started)}s")
                last_heartbeat = time.time()
        rc = int(proc.returncode or 0)
        f.write(f"===== {now()} END rc={rc} =====\n")

    if rc == 2 and allow_nogo:
        emit(f"NO-GO {stage} rc=2; see {raw_log}")
        save_state(status="running", stage=stage, pid=None, returncode=rc, raw_log=str(raw_log))
        return rc
    if rc != 0:
        level = "WARN" if optional else "FAIL"
        emit(f"{level} {stage} rc={rc}; see {raw_log}")
        save_state(status="failed", stage=stage, pid=None, returncode=rc, raw_log=str(raw_log))
        if optional:
            return rc
        raise SystemExit(rc)
    complete(stage)
    emit(f"END   {stage}")
    return 0


def stop_no_go(stage: str, reason: str, next_plan: str) -> None:
    save_state(status="no_go", stage=stage, pid=None, reason=reason, next_plan=next_plan)
    emit(f"PIPELINE NO-GO | {stage} | {reason}")
    emit(f"NEXT PLAN | {next_plan}")
    write_summary(status="NO-GO", reason=reason)
    raise SystemExit(2)


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


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def stop_static_vllm() -> None:
    for base in (LOG_ROOT, Path(LOG_DIR)):
        for name in ("vllm.pid", "merged_chain_vllm.pid"):
            pidf = base / name
            if not pidf.exists():
                continue
            try:
                pid = int(pidf.read_text().strip())
                emit(f"stop vLLM process group {pid} ({pidf})")
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
    base_name = f"{served_name}_base" if adapter else served_name
    args = ["bash", str(SCRIPTS / "serve_model_vllm.sh"), str(model_dir), base_name]
    if adapter:
        args.extend([served_name, str(adapter)])
    run(f"serve_{served_name}", args)
    save_state(stage=f"wait_{served_name}", status="running",
               raw_log=str(LOG_ROOT / "merged_chain_vllm.log"), pid=None)
    emit(f"WAIT  vLLM {served_name}")
    vllm_client.wait_ready(max_wait=int(os.environ.get("DERAG_V4_VLLM_WAIT", "1800")))
    emit(f"READY vLLM {served_name}")


def merge(stage: str, base: Path, adapter_root: Path, output: Path) -> None:
    if merged_model_complete(output):
        emit(f"SKIP  {stage}; merged model exists: {output}")
        return
    move_interrupted(output)
    adapter = Path(resolve_adapter(str(adapter_root)))
    run(stage, ["bash", str(SCRIPTS / "merge_lora_model.sh"), str(base), str(adapter), str(output)])


def train_sft() -> None:
    if adapter_complete(S1_LORA):
        emit(f"SKIP  s1 SFT complete: {S1_LORA}")
        return
    move_interrupted(S1_LORA)
    stop_static_vllm()
    run("train_s1_rewrite_sft", [
        "bash", str(SWIFT / "sft_on_model.sh"),
        str(OUT_DIR / "125_s1_train.jsonl"),
        str(OUT_DIR / "125_s1_eval.jsonl"),
        str(RFT_MERGED),
        str(S1_LORA),
        os.environ.get("DERAG_V4_SFT_LR", "5e-5"),
        os.environ.get("DERAG_V4_SFT_EPOCHS", "2"),
    ])


def train_dpo_variant(label: str, env: dict[str, str]) -> tuple[Path, Path, Path]:
    lora = S2_LORA_BASE.with_name(f"{S2_LORA_BASE.name}_{label}")
    merged = S2_MERGED_BASE.with_name(f"{S2_MERGED_BASE.name}_{label}")
    if not adapter_complete(lora):
        move_interrupted(lora)
        stop_static_vllm()
        run(f"train_s2_dpo_{label}", [
            "bash", str(SWIFT / "dpo_v2.sh"),
            str(OUT_DIR / "127_dpo_train.jsonl"),
            str(S1_MERGED),
            str(lora),
        ], env=env)
    else:
        emit(f"SKIP  DPO {label}; adapter complete: {lora}")
    merge(f"merge_s2_dpo_{label}", S1_MERGED, lora, merged)
    infer = OUT_DIR / f"129_s2_{label}_eval_infer.jsonl"
    evaluate_model(f"s2_{label}", merged, infer)
    report = OUT_DIR / f"130_s2_{label}_det_report.md"
    run_report(f"report_s2_{label}", report, [
        ("s1", OUT_DIR / "128_s1_eval_infer.jsonl"),
        (f"s2_{label}", infer),
    ])
    return lora, merged, report


def train_grpo_variant(label: str, base_model: Path, env: dict[str, str]) -> tuple[Path, Path, Path]:
    lora = S3_LORA_BASE.with_name(f"{S3_LORA_BASE.name}_{label}")
    merged = S3_MERGED_BASE.with_name(f"{S3_MERGED_BASE.name}_{label}")
    if not adapter_complete(lora):
        move_interrupted(lora)
        stop_static_vllm()
        run(f"train_s3_grpo_{label}", [
            "bash", str(SWIFT / "grpo_on_model.sh"),
            str(OUT_DIR / "131_grpo_data.jsonl"),
            str(base_model),
            str(lora),
        ], env=env)
    else:
        emit(f"SKIP  GRPO {label}; adapter complete: {lora}")
    merge(f"merge_s3_grpo_{label}", base_model, lora, merged)
    infer = OUT_DIR / f"132_s3_{label}_eval_infer.jsonl"
    evaluate_model(f"s3_{label}", merged, infer)
    report = OUT_DIR / f"133_s3_{label}_det_report.md"
    run_report(f"report_s3_{label}", report, [
        ("s2_selected", OUT_DIR / "129_s2_selected_eval_infer.jsonl"),
        (f"s3_{label}", infer),
    ])
    return lora, merged, report


def evaluate_model(model_name: str, model_dir: Path, out: Path) -> None:
    if out.exists() and out.stat().st_size:
        emit(f"SKIP  eval {model_name}; infer exists: {out}")
        return
    served = re.sub(r"[^A-Za-z0-9_]", "_", f"derag_v4_{RUN_ID}_{model_name}")[-64:]
    serve_model(model_dir, served)
    run(f"eval_{model_name}", [
        str(PY), "-X", "utf8", "pipeline/step03_eval_infer.py",
        "--model", served,
        "--eval_file", str(SFT_EVAL),
        "--out", str(out),
    ])
    stop_static_vllm()


def run_report(stage: str, out: Path, items: list[tuple[str, Path]]) -> None:
    cmd = [str(PY), "-X", "utf8", "pipeline/step126_v4_report.py", "--out", str(out)]
    first = True
    for label, path in items:
        cmd.extend(["--base" if first else "--stage", f"{label}={path}"])
        first = False
    run(stage, cmd)


def report_rows(report: Path) -> list[dict[str, Any]]:
    return read_json(report.with_suffix(".json"), [])


def compare_gate(report: Path, *, max_answer_drop: float, min_trace_drop: int = 1,
                 min_clean_gain: float = 0.0) -> tuple[bool, str, dict[str, Any]]:
    rows = report_rows(report)
    if len(rows) < 2:
        return False, "report has fewer than 2 rows", {}
    prev, cand = rows[-2], rows[-1]
    answer_ok = cand.get("answer_score_mean", 0.0) >= prev.get("answer_score_mean", 0.0) - max_answer_drop
    trace_drop = int(cand.get("trace_total", 10**9)) <= int(prev.get("trace_total", 0)) - min_trace_drop
    clean_gain = float(cand.get("clean_rate", 0.0)) >= float(prev.get("clean_rate", 0.0)) + min_clean_gain
    ok = answer_ok and (trace_drop or clean_gain)
    reason = (
        f"answer_ok={answer_ok} trace_drop={trace_drop} clean_gain={clean_gain} "
        f"prev(clean={prev.get('clean_rate')}, trace={prev.get('trace_total')}, ans={prev.get('answer_score_mean')}) "
        f"cand(clean={cand.get('clean_rate')}, trace={cand.get('trace_total')}, ans={cand.get('answer_score_mean')})"
    )
    return ok, reason, {"prev": prev, "cand": cand}


def preflight() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    required = [
        PY,
        ROOT / "pipeline" / "reward_v3.py",
        ROOT / "pipeline" / "step121_pool_density_probe.py",
        ROOT / "pipeline" / "step124_rewrite_residual.py",
        ROOT / "pipeline" / "step124_build_dpo_pairs_v4.py",
        ROOT / "pipeline" / "step125a_anchor_calibration.py",
        ROOT / "pipeline" / "step125b_replay_gate_dryrun.py",
        ROOT / "pipeline" / "step125c_spotcheck_sheet.py",
        ROOT / "pipeline" / "step126_dpo_seed_pools.py",
        ROOT / "swift" / "sft_on_model.sh",
        ROOT / "swift" / "dpo_v2.sh",
        ROOT / "swift" / "grpo_on_model.sh",
        Path(SFT_TRAIN),
        Path(SFT_EVAL),
        RFT_MERGED / "config.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("preflight missing:\n" + "\n".join(missing))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is required for Kimi rewrite/gating")
    free_gib = shutil.disk_usage(WORK_DIR).free / 1024**3
    if free_gib < float(os.environ.get("DERAG_V4_MIN_FREE_GIB", "200")):
        raise SystemExit(f"free disk too low: {free_gib:.1f} GiB")
    stop_static_vllm()
    active = subprocess.run(
        ["bash", "-lc", "pgrep -af '[s]wift/cli/rlhf.py|[t]orch.distributed.run' || true"],
        capture_output=True, text=True,
    ).stdout.strip()
    if active:
        raise SystemExit("other training processes are still active:\n" + active)
    emit(f"preflight OK | run_id={RUN_ID} | free={free_gib:.1f}GiB | rft_merged={RFT_MERGED}")
    run("120_precheck", [
        str(PY), "-X", "utf8", "pipeline/step120_v4_precheck.py",
        "--run_id", RUN_ID,
        "--rft_merged", str(RFT_MERGED),
        "--out_dir", str(OUT_DIR),
    ])
    run("122_reward_preflight", [
        str(PY), "-X", "utf8", "pipeline/step122_reward_preflight.py",
        "--out", str(OUT_DIR / "122_reward_preflight.json"),
    ], allow_nogo=False)


def stage1_rewrite_sft() -> None:
    rft_train_infer = OUT_DIR / "123_rft_train_greedy.jsonl"
    rewrites = OUT_DIR / "124_rewrites.jsonl"
    trace_pool = OUT_DIR / "124_trace_pool.jsonl"
    reuse_dir = Path(os.environ.get(
        "DERAG_V4_STAGE1_REUSE_DIR",
        str(Path(OUTPUT_DIR) / "derag_v4" / "20260612_022656"),
    ))
    if os.environ.get("DERAG_V4_REUSE_STAGE1", "1") == "1" and reuse_dir != OUT_DIR:
        for name in ("123_rft_train_greedy.jsonl", "124_rewrites.jsonl", "124_trace_pool.jsonl"):
            src, dst = reuse_dir / name, OUT_DIR / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                emit(f"REUSE Stage1 artifact | {src} -> {dst}")
    if not rft_train_infer.exists():
        serve_model(RFT_MERGED, "derag_v4_rft_train")
        run("123_rft_train_greedy", [
            str(PY), "-X", "utf8", "pipeline/step03_eval_infer.py",
            "--model", "derag_v4_rft_train",
            "--eval_file", str(SFT_TRAIN),
            "--out", str(rft_train_infer),
        ])
        stop_static_vllm()
    else:
        emit(f"SKIP  RFT train greedy exists: {rft_train_infer}")

    if rewrites.exists() and trace_pool.exists():
        emit(f"SKIP  residual rewrite; reused/existing rows={count_lines(rewrites)}")
    else:
        run("124_rewrite_residual", [
            str(PY), "-X", "utf8", "pipeline/step124_rewrite_residual.py",
            "--infer", str(rft_train_infer),
            "--out", str(rewrites),
            "--pool_out", str(trace_pool),
            "--max_rewrites", os.environ.get("DERAG_V4_MAX_REWRITES", "0"),
            "--workers", os.environ.get("DERAG_V4_REWRITE_WORKERS", "3"),
        ])

    run("125b_gate_dryrun", [
        str(PY), "-X", "utf8", "pipeline/step125b_replay_gate_dryrun.py",
        "--rewrites", str(OUT_DIR / "124_rewrites.jsonl"),
        "--out", str(OUT_DIR / "125b_replay_report.json"),
    ])

    legacy_gate_rows = os.environ.get(
        "DERAG_V4_LEGACY_GATE_ROWS",
        str(Path(OUTPUT_DIR) / "derag_v4" / "20260612_022656" / "125_gate_rewrites.rows.jsonl"),
    )
    calibration_cmd = [
        str(PY), "-X", "utf8", "pipeline/step125a_anchor_calibration.py",
        "--mode", "binary",
        "--rewrites", str(OUT_DIR / "124_rewrites.jsonl"),
        "--out", str(OUT_DIR / "125a_anchor_calibration.json"),
        "--anchors_out", str(OUT_DIR / "125a_anchor_rows.jsonl"),
        "--workers", os.environ.get("DERAG_V4_JUDGE_WORKERS", "3"),
    ]
    if Path(legacy_gate_rows).exists():
        calibration_cmd.extend(["--legacy_rows", legacy_gate_rows])
    rc = run("125a_anchor_calibration", calibration_cmd, allow_nogo=True)
    degraded_binary_gate = rc == 2
    if degraded_binary_gate:
        emit(
            "DEGRADED Stage1 | judge_v4.1_bin calibration failed; "
            "continue with validated L0 deterministic gate + 60-row blind spot-check"
        )

    gate_cmd = [
        str(PY), "-X", "utf8", "pipeline/step125_gate_rewrites.py",
        "--rewrites", str(OUT_DIR / "124_rewrites.jsonl"),
        "--calibration", str(OUT_DIR / "125a_anchor_calibration.json"),
        "--anchors", str(OUT_DIR / "125a_anchor_rows.jsonl"),
        "--train_out", str(OUT_DIR / "125_s1_train.jsonl"),
        "--eval_out", str(OUT_DIR / "125_s1_eval.jsonl"),
        "--report", str(OUT_DIR / "125_gate_rewrites.json"),
        "--replay", os.environ.get("DERAG_V4_REPLAY", str(Path(OUTPUT_DIR) / "50_cs_rft_train.jsonl")),
        "--replay_n", os.environ.get("DERAG_V4_REPLAY_N", "150"),
        "--min_rewrites", os.environ.get("DERAG_V4_MIN_REWRITES", "400"),
        "--workers", os.environ.get("DERAG_V4_JUDGE_WORKERS", "3"),
    ]
    if degraded_binary_gate:
        gate_cmd.append("--degraded-deterministic")
    rc = run("125_gate_rewrites", gate_cmd, allow_nogo=True)
    if rc == 2:
        stop_no_go(
            "stage1_gate_rewrites",
            f"binary/deterministic Stage1 gate failed; report={OUT_DIR / '125_gate_rewrites.json'}",
            "Inspect L0 funnel, binary votes, repair recovery, phrase gate and output count. Do not loosen fact guards.",
        )

    gate_report = read_json(OUT_DIR / "125_gate_rewrites.json", {})
    pass_n = "60" if gate_report.get("status") == "DEGRADED-GO" else "30"
    run("125c_spotcheck_sheet", [
        str(PY), "-X", "utf8", "pipeline/step125c_spotcheck_sheet.py",
        "--gate_rows", str(OUT_DIR / "125_gate_rewrites.rows.jsonl"),
        "--sheet", str(OUT_DIR / "125c_stage1_spotcheck_sheet.md"),
        "--mapping", str(OUT_DIR / "125c_stage1_spotcheck_mapping.json"),
        "--report", str(OUT_DIR / "125c_stage1_spotcheck_report.json"),
        "--pass_n", pass_n,
        "--fail_n", "20",
    ])

    run("126_dpo_seed_pools", [
        str(PY), "-X", "utf8", "pipeline/step126_dpo_seed_pools.py",
        "--gate_rows", str(OUT_DIR / "125_gate_rewrites.rows.jsonl"),
        "--out", str(OUT_DIR / "126_dpo_seed_pools.jsonl"),
        "--report", str(OUT_DIR / "126_dpo_seed_pools.json"),
    ])

    train_sft()
    merge("merge_s1_rewrite_sft", RFT_MERGED, S1_LORA, S1_MERGED)
    evaluate_model("s1", S1_MERGED, OUT_DIR / "128_s1_eval_infer.jsonl")


def g11_probe() -> Path:
    k_list = [
        int(x) for x in os.environ.get("DERAG_V4_G11_KS", "16,32,64,128").split(",")
        if x.strip()
    ]
    selected_rollout = None
    selected_decision = None
    for k in k_list:
        rollout = OUT_DIR / f"126_s1_pool_rollout_k{k}.jsonl"
        decision = OUT_DIR / f"126_g11_density_k{k}.json"
        if not rollout.exists():
            serve_model(S1_MERGED, f"derag_v4_s1_k{k}")
            run(f"126_s1_pool_rollout_k{k}", [
                str(PY), "-X", "utf8", "pipeline/step10_rollout.py",
                "--model", f"derag_v4_s1_k{k}",
                "--pool", str(OUT_DIR / "124_trace_pool.jsonl"),
                "--out", str(rollout),
                "--k", str(k),
                "--temperature", os.environ.get("DERAG_V4_G11_TEMP", "0.9"),
                "--top_p", os.environ.get("DERAG_V4_G11_TOP_P", "0.95"),
                "--max_tokens", os.environ.get("DERAG_V4_G11_MAX_TOKENS", "1536"),
            ])
            stop_static_vllm()
        else:
            emit(f"SKIP  G1-1 rollout K={k} exists: {rollout}")
        rc = run(f"126_g11_density_k{k}", [
            str(PY), "-X", "utf8", "pipeline/step121_pool_density_probe.py",
            "--rollout", str(rollout),
            "--out", str(decision),
            "--mode", f"s1_k{k}",
            "--p_clean_min", os.environ.get("DERAG_V4_P_CLEAN_MIN", "0.60"),
            "--p_pair_min", os.environ.get("DERAG_V4_P_PAIR_MIN", "0.40"),
        ], allow_nogo=True)
        data = read_json(decision, {})
        emit(f"G1-1 K={k} status={data.get('status')} p_clean={data.get('p_clean')} p_pair={data.get('p_pair')}")
        if rc == 0:
            selected_rollout = rollout
            selected_decision = decision
            break
    if not selected_rollout:
        stop_no_go(
            "g1_1_pool_density",
            f"no K in {k_list} reached pool density gates",
            "Use the density rows to relax only the deterministic over-strict gate, or improve S1 rewrite SFT; do not enter DPO with sparse pairs.",
        )
    save_state(g11_rollout=str(selected_rollout), g11_decision=str(selected_decision))
    return selected_rollout


def build_dpo_pairs(rollout: Path) -> None:
    rc = run("127_build_dpo_pairs_v4", [
        str(PY), "-X", "utf8", "pipeline/step124_build_dpo_pairs_v4.py",
        "--rollout", str(rollout),
        "--train_out", str(OUT_DIR / "127_dpo_train.jsonl"),
        "--heldout_out", str(OUT_DIR / "127_dpo_heldout.jsonl"),
        "--meta_out", str(OUT_DIR / "127_dpo_pairs_meta.json"),
        "--min_pairs", os.environ.get("DERAG_V4_MIN_DPO_PAIRS", "160"),
        "--heldout_n", os.environ.get("DERAG_V4_HELDOUT_PAIRS", "40"),
        "--seed_pool", str(OUT_DIR / "126_dpo_seed_pools.jsonl"),
    ], allow_nogo=True)
    meta = read_json(OUT_DIR / "127_dpo_pairs_meta.json", {})
    emit(f"DPO pair build status={meta.get('status')} train={meta.get('train_pairs')} heldout={meta.get('heldout_pairs')}")
    if rc == 2:
        stop_no_go(
            "dpo_pair_build",
            f"pair build below threshold; meta={OUT_DIR / '127_dpo_pairs_meta.json'}",
            "Increase DERAG_V4_G11_KS upper bound or inspect pair funnel; do not train DPO if clean/trace contrast is too sparse.",
        )


def stage2_dpo() -> Path:
    variants = [
        ("a", {"DPO_LR": "5e-6", "DPO_BETA": "0.1", "DPO_EPOCHS": "2"}),
        ("b", {"DPO_LR": "1e-5", "DPO_BETA": "0.08", "DPO_EPOCHS": "2"}),
        ("c", {"DPO_LR": "3e-6", "DPO_BETA": "0.15", "DPO_EPOCHS": "2"}),
    ]
    selected = None
    scored: list[dict[str, Any]] = []
    for label, env in variants:
        lora, merged, report = train_dpo_variant(label, env)
        ok, reason, metrics = compare_gate(
            report,
            max_answer_drop=float(os.environ.get("DERAG_V4_MAX_ANSWER_DROP", "0.02")),
            min_trace_drop=int(os.environ.get("DERAG_V4_DPO_MIN_TRACE_DROP", "1")),
            min_clean_gain=float(os.environ.get("DERAG_V4_DPO_MIN_CLEAN_GAIN", "0.0")),
        )
        scored.append({"label": label, "lora": str(lora), "merged": str(merged),
                       "report": str(report), "ok": ok, "reason": reason, "metrics": metrics})
        emit(f"DPO variant {label} gate={ok} | {reason}")
        if ok:
            selected = (label, merged)
            break
    (OUT_DIR / "130_dpo_variant_decisions.json").write_text(
        json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
    if not selected:
        stop_no_go(
            "stage2_dpo_gate",
            "all DPO variants failed deterministic guardrail/improvement gate",
            "Inspect 130_dpo_variant_decisions.json; likely pair construction or S1 signal is not learnable enough.",
        )
    label, merged = selected
    shutil.copyfile(OUT_DIR / f"129_s2_{label}_eval_infer.jsonl", OUT_DIR / "129_s2_selected_eval_infer.jsonl")
    save_state(selected_dpo_variant=label, selected_s2_model=str(merged))
    return merged


def stage3_grpo(s2_model: Path) -> Path:
    run("131_build_grpo_data", [
        str(PY), "-X", "utf8", "pipeline/step13_build_grpo_data.py",
        "--pool", str(OUT_DIR / "124_trace_pool.jsonl"),
        "--out", str(OUT_DIR / "131_grpo_data.jsonl"),
    ])
    variants = [
        ("a", {"GRPO_LR": "1e-6", "GRPO_BETA": "0.04", "GRPO_STEPS": "120", "GRPO_K": "8"}),
        ("b", {"GRPO_LR": "5e-7", "GRPO_BETA": "0.08", "GRPO_STEPS": "120", "GRPO_K": "8"}),
        ("c", {"GRPO_LR": "2e-6", "GRPO_BETA": "0.04", "GRPO_STEPS": "160", "GRPO_K": "8"}),
    ]
    selected = None
    scored: list[dict[str, Any]] = []
    for label, env in variants:
        lora, merged, report = train_grpo_variant(label, s2_model, env)
        ok, reason, metrics = compare_gate(
            report,
            max_answer_drop=float(os.environ.get("DERAG_V4_MAX_ANSWER_DROP", "0.02")),
            min_trace_drop=int(os.environ.get("DERAG_V4_GRPO_MIN_TRACE_DROP", "1")),
            min_clean_gain=float(os.environ.get("DERAG_V4_GRPO_MIN_CLEAN_GAIN", "0.0")),
        )
        scored.append({"label": label, "lora": str(lora), "merged": str(merged),
                       "report": str(report), "ok": ok, "reason": reason, "metrics": metrics})
        emit(f"GRPO variant {label} gate={ok} | {reason}")
        if ok:
            selected = (label, merged)
            break
    (OUT_DIR / "133_grpo_variant_decisions.json").write_text(
        json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
    if not selected:
        stop_no_go(
            "stage3_grpo_gate",
            "all GRPO variants failed deterministic guardrail/improvement gate",
            "Inspect 133_grpo_variant_decisions.json and GRPO reward logs; next fallback is reward threshold tuning, not blind extra steps.",
        )
    label, merged = selected
    shutil.copyfile(OUT_DIR / f"132_s3_{label}_eval_infer.jsonl", OUT_DIR / "132_s3_selected_eval_infer.jsonl")
    save_state(selected_grpo_variant=label, selected_s3_model=str(merged))
    return merged


def write_summary(status: str = "PASS", reason: str = "") -> None:
    summary = OUT_DIR / "139_derag_v4_summary.md"
    lines = [
        "# derag_v4 run summary",
        "",
        f"- run_id: `{RUN_ID}`",
        f"- status: `{status}`",
        f"- reason: {reason or 'completed planned chain'}",
        f"- rft_merged: `{RFT_MERGED}`",
        f"- out_dir: `{OUT_DIR}`",
        f"- log_root: `{LOG_ROOT}`",
        "",
        "## Selected Artifacts",
        "",
        f"- S1 LoRA: `{S1_LORA}`",
        f"- S1 merged: `{S1_MERGED}`",
        f"- selected S2 model: `{_state.get('selected_s2_model')}`",
        f"- selected S3 model: `{_state.get('selected_s3_model')}`",
        f"- G1-1 decision: `{_state.get('g11_decision')}`",
        "",
        "## Reports",
        "",
    ]
    for p in (
        sorted(OUT_DIR.glob("125*.json"))
        + sorted(OUT_DIR.glob("*_det_report.md"))
        + sorted(OUT_DIR.glob("*variant_decisions.json"))
    ):
        lines.append(f"- `{p}`")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_state(summary=str(summary))
    emit(f"RESULT summary -> {summary}")


def main() -> None:
    save_state(status="running", stage="preflight", completed=[])
    try:
        preflight()
        stage1_rewrite_sft()
        rollout = g11_probe()
        build_dpo_pairs(rollout)
        s2_model = stage2_dpo()
        s3_model = stage3_grpo(s2_model)
        run_report("138_final_report", OUT_DIR / "138_final_det_report.md", [
            ("s1", OUT_DIR / "128_s1_eval_infer.jsonl"),
            ("s2", OUT_DIR / "129_s2_selected_eval_infer.jsonl"),
            ("s3", OUT_DIR / "132_s3_selected_eval_infer.jsonl"),
        ])
        save_state(selected_s3_model=str(s3_model))
        write_summary(status="PASS")
        save_state(status="complete", stage="done", pid=None)
        emit(f"PIPELINE COMPLETE | summary={OUT_DIR / '139_derag_v4_summary.md'}")
    except SystemExit:
        raise
    except BaseException as exc:
        save_state(status="failed", error=repr(exc), pid=None)
        emit(f"PIPELINE FAILED | {exc!r}")
        raise
    finally:
        try:
            stop_static_vllm()
        except Exception as exc:
            emit(f"WARN cleanup vLLM: {exc!r}")


if __name__ == "__main__":
    main()
