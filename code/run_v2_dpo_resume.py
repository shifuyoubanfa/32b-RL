"""derag2 的 2s-2s-2s DPO 叶：本地接回 AutoDL adapter 后继续合并、推理、三分评测。

只接这一条已经训练完的叶，不重跑任何 SFT/RFT/DPO 数据或训练。所有模型/评测产物严格复用
run_v2.py 原命名：
  base    = models/v2-rft-2sigma-2s-merged
  merged  = models/v2-dpo-2sigma-2s-2s-merged
  tag     = v2-2s-2s-2s
  outputs = output/derag2/v2-2s-2s-2s_{infer,scores,report,summary}.*

下载回来的 adapter 路径保留交接名 v2-dpo-derag2-lora/checkpoint-108，仅作为输入来源。
运行日志独立落 logs/v2_dpo_resume/，不污染旧 logs/v2/。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (GEN_MAX_NEW_TOKENS, JUDGE_TEMPERATURE, JUDGE_TOP_P, KIMI_BASE_URL,
                    KIMI_ENABLE_THINKING, KIMI_MODEL, LOG_DIR, VLLM_BASE_URL,
                    system_for)  # noqa: E402
from pipeline import vllm_client  # noqa: E402
from pipeline.reward import parse_think_answer_diagnostic  # noqa: E402
from pipeline.v2_common import v2_eval_paths, v2_lora_dir, v2_merged_dir, v2_summary_path  # noqa: E402
from pipeline.v2_paths import V2_EVAL, V2_V1_SUPPORT, eval_progress, qid_of, read_jsonl  # noqa: E402


WORK_DIR = Path(os.environ.get("ZHJG_WORK_DIR", "/home/nvme01/zhjg"))
PY = Path(os.environ.get("ZHJG_ENV", "/home/nvme02/conda/zhjg_rl")) / "bin" / "python"
VLLM_ENV = Path(os.environ.get("VLLM_ENV", "/home/nvme02/biyh/vllm_env"))
SCRIPTS = ROOT / "scripts"

# ★ 原 run_v2.py 的正式 lineage 命名，禁止另起 derag2_dpo 产物名。
TAG = "v2-2s-2s-2s"
SERVED_NAME = "v2_2s_2s_2s"  # run_v2.evaluate 对 TAG 做连字符→下划线后的名字
RFT_BASE = v2_merged_dir("rft", 2, "2s")
DPO_MERGED = v2_merged_dir("dpo", 2, "2s-2s")
DOWNLOADED_ADAPTER = Path(os.environ.get(
    "V2_DPO_DOWNLOADED_ADAPTER",
    str(WORK_DIR / "ckpts" / "v2-dpo-derag2-lora" / "checkpoint-108"),
))
CANONICAL_LORA = v2_lora_dir("dpo", 2, "2s-2s")
CANONICAL_ADAPTER = CANONICAL_LORA / "checkpoint-108"
INFER, SCORES, REPORT = map(Path, v2_eval_paths(TAG))
SUMMARY = Path(v2_summary_path(TAG))
PROGRESS = Path(eval_progress(TAG))

LOG_ROOT = Path(LOG_DIR) / "v2_dpo_resume"
RAW_DIR = LOG_ROOT / "raw"
EVENT_LOG = LOG_ROOT / "events.log"
STATE_FILE = LOG_ROOT / "state.json"
RESULT_JSON = LOG_ROOT / "result.json"
RESULT_MD = LOG_ROOT / "result.md"
MERGE_MANIFEST = DPO_MERGED / "dpo_resume_merge_manifest.json"
INFER_PROVENANCE = LOG_ROOT / "infer_provenance.json"
PROGRESS_BINDING = LOG_ROOT / "kimi_progress_binding.json"
EVAL_PROVENANCE = LOG_ROOT / "eval_provenance.json"
INFER_FORMAT_MANIFEST = LOG_ROOT / "infer_format_accounting_manifest.json"
LOCK_FILE = LOG_ROOT / "runner.lock"

EXPECTED_ADAPTER_SHA256 = "4d8b8ab244490b5e48b4b2f69bf2d32de459daa16c7dabacd8e723b930556dac"
EXPECTED_ADAPTER_CONFIG_SHA256 = "afd891c421f1f214ffb42daa03f1c1ae53c9fc95db2e60b1ac6c9195b0a9c990"
EXPECTED_MODEL_TOTAL_SIZE = 65527752704
EXPECTED_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
EXPECTED_KIMI_MODEL = "kimi/kimi-k2.6"
EXPECTED_KIMI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EXPECTED_FORMAT_FAILURE_QUERY = "员工1月入职，3月发放工资1.4万元，为什么需要缴纳个人所得税？"
EXPECTED_FORMAT_FAILURE_QID = qid_of(EXPECTED_FORMAT_FAILURE_QUERY)
HISTORICAL_CONTROLS = {
    "v2-baseline-v1": {"tag": "v2-baseline-v1", "clean_mean": 3.14,
                       "rule_pass_rate": 0.026, "in_pool_rate": 0.938},
    "v2-sft-2s": {"tag": "v2-sft-2s", "clean_mean": 4.408,
                  "rule_pass_rate": 0.454, "in_pool_rate": 0.83},
    "v2-rft-2s-2s": {"tag": "v2-rft-2s-2s", "clean_mean": 4.4887,
                     "rule_pass_rate": 0.456, "in_pool_rate": 0.85},
}

GPU_USED_MAX_MIB = int(os.environ.get("V2_DPO_GPU_USED_MAX_MIB", "2048"))
GPU_UTIL_MAX = int(os.environ.get("V2_DPO_GPU_UTIL_MAX", "5"))
GPU_WAIT_INTERVAL = int(os.environ.get("V2_DPO_GPU_WAIT_INTERVAL", "60"))
GPU_WAIT_TIMEOUT = int(os.environ.get("V2_DPO_GPU_WAIT_TIMEOUT", "0"))  # 0=一直等
GPU_STABLE_SAMPLES = int(os.environ.get("V2_DPO_GPU_STABLE_SAMPLES", "3"))
REQUESTED_GPUS = os.environ.get("V2_DPO_EVAL_GPUS", "").strip()

SIGNAL_RE = re.compile(
    r"\[merge\]|Loading checkpoint|Writing|Application startup complete|评测推理|完成：|"
    r"干净分|规则通过|在池率|RESULT|ERROR|FAIL|Traceback",
    re.IGNORECASE,
)
_state: dict = {"status": "starting", "stage": "preflight", "tag": TAG}
_launched_vllm = False
_vllm_pid: int | None = None
_lock_handle = None


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def emit(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{now()} | {message}"
    print(line, flush=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_state(**updates) -> None:
    _state.update(updates, updated_at=now())
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def acquire_lock() -> None:
    """launcher 已持 flock 时复用；直接 python 启动时也必须拿同一把独占锁。"""
    global _lock_handle
    if os.environ.get("V2_DPO_LOCK_HELD") == "1":
        return
    import fcntl
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    _lock_handle = LOCK_FILE.open("a+")
    try:
        fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"已有另一个 DPO resume 在运行（锁：{LOCK_FILE}）")


def latest_signal(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 131072))
        text = f.read().decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if SIGNAL_RE.search(line)]
    return lines[-1][-500:] if lines else ""


def run(stage: str, cmd: list[str | Path], *, env: dict | None = None) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_log = RAW_DIR / f"{stage}.log"
    argv = [str(x) for x in cmd]
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    emit(f"START {stage}")
    emit("CMD   " + " ".join(argv))
    with raw_log.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {now()} START {' '.join(argv)} =====\n")
        f.flush()
        proc = subprocess.Popen(argv, cwd=ROOT, env=full_env, stdout=f, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        started = time.time()
        save_state(status="running", stage=stage, pid=proc.pid, raw_log=str(raw_log), command=argv)
        last_signal = ""
        last_heartbeat = 0.0
        try:
            while proc.poll() is None:
                time.sleep(10)
                line = latest_signal(raw_log)
                if line and line != last_signal:
                    emit(f"{stage} | {line}")
                    last_signal = line
                elif time.time() - last_heartbeat >= 60:
                    emit(f"{stage} | alive pid={proc.pid} elapsed={int(time.time() - started)}s")
                    last_heartbeat = time.time()
        except BaseException:
            if proc.poll() is None:
                emit(f"STOP  child stage={stage} process_group={proc.pid}")
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
                except ProcessLookupError:
                    pass
            f.write(f"===== {now()} INTERRUPTED =====\n")
            raise
        rc = proc.returncode
        f.write(f"===== {now()} END rc={rc} =====\n")
    if rc != 0:
        save_state(status="failed", stage=stage, returncode=rc)
        emit(f"FAIL  {stage} rc={rc}; see {raw_log}")
        raise SystemExit(rc)
    emit(f"END   {stage}")


def archive(path: Path, label: str = "stale") -> Path | None:
    if not path.exists():
        return None
    moved = path.with_name(f"{path.name}.{label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    path.rename(moved)
    emit(f"ARCHIVE {path} -> {moved}")
    return moved


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_shape_ok(path: Path) -> bool:
    try:
        cfg = read_json(path / "config.json")
        idx = read_json(path / "model.safetensors.index.json")
    except Exception:
        return False
    return (cfg.get("model_type") == "qwen2"
            and cfg.get("hidden_size") == 5120
            and cfg.get("num_hidden_layers") == 64
            and cfg.get("vocab_size") == 152064
            and idx.get("metadata", {}).get("total_size") == EXPECTED_MODEL_TOTAL_SIZE
            and len(set(idx.get("weight_map", {}).values())) == 14
            and len(list(path.glob("*.safetensors"))) == 14)


def expected_merge_identity() -> dict:
    return {
        "tag": TAG,
        "rft_base": str(RFT_BASE),
        "rft_index_sha256": sha256(RFT_BASE / "model.safetensors.index.json"),
        "adapter": str(CANONICAL_ADAPTER),
        "adapter_sha256": sha256(CANONICAL_ADAPTER / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(CANONICAL_ADAPTER / "adapter_config.json"),
        "expected_total_size": EXPECTED_MODEL_TOTAL_SIZE,
    }


def merge_delta_proof(path: Path) -> dict:
    """抽验一层 q_proj：合并权重必须确实不同于 RFT base，防止“只复制底座”的假合并。"""
    import hashlib
    import torch
    from safetensors import safe_open

    base_idx = read_json(RFT_BASE / "model.safetensors.index.json")["weight_map"]
    merged_idx = read_json(path / "model.safetensors.index.json")["weight_map"]
    candidates = sorted(
        (k for k in base_idx if k.endswith("self_attn.q_proj.weight") and k in merged_idx),
        key=lambda k: int(re.search(r"layers\.(\d+)\.", k).group(1)),
    )
    if not candidates:
        raise RuntimeError("合并验真找不到共同 q_proj tensor")
    # 先探前/中/后层；若恰好被 bf16 舍入成相同，再逐层扫到第一个真实变化。
    preferred = [candidates[0], candidates[len(candidates) // 2], candidates[-1]]
    probe_order = preferred + [k for k in candidates if k not in preferred]
    key = ""
    base_tensor = merged_tensor = None
    for candidate in probe_order:
        with safe_open(str(RFT_BASE / base_idx[candidate]), framework="pt", device="cpu") as f:
            base_candidate = f.get_tensor(candidate)
        with safe_open(str(path / merged_idx[candidate]), framework="pt", device="cpu") as f:
            merged_candidate = f.get_tensor(candidate)
        if base_candidate.shape != merged_candidate.shape:
            raise RuntimeError(f"合并 tensor 形状变化: {candidate}")
        if not torch.equal(base_candidate, merged_candidate):
            key, base_tensor, merged_tensor = candidate, base_candidate, merged_candidate
            break
    if not key:
        raise RuntimeError("LoRA 合并后 64 层 q_proj 均与 RFT base 完全相同")
    max_abs = float((base_tensor.float() - merged_tensor.float()).abs().max().item())
    if not max_abs > 0:
        raise RuntimeError(f"LoRA 合并抽验 delta 非正: {key} max_abs={max_abs}")

    def tensor_sha(tensor) -> str:
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    return {"tensor": key, "base_tensor_sha256": tensor_sha(base_tensor),
            "merged_tensor_sha256": tensor_sha(merged_tensor), "max_abs_delta": max_abs}


def merged_complete(path: Path) -> bool:
    if not ((path / ".done").exists() and (path / "tokenizer_config.json").exists() and model_shape_ok(path)):
        return False
    manifest = path / "dpo_resume_merge_manifest.json"
    if not manifest.exists() or not CANONICAL_ADAPTER.exists():
        return False
    try:
        got = read_json(manifest)
        expected = expected_merge_identity()
    except Exception:
        return False
    if not all(got.get(k) == v for k, v in expected.items()):
        return False
    try:
        return got.get("merge_delta_proof") == merge_delta_proof(path)
    except Exception:
        return False


def validate_frozen_inputs() -> tuple[list[dict], list[dict]]:
    eval_rows = read_jsonl(V2_EVAL)
    support_rows = read_jsonl(V2_V1_SUPPORT)
    required_eval = {"qid", "query", "user_prompt", "answer", "reasoning", "split"}
    if len(eval_rows) != 500 or any(not required_eval <= set(r) for r in eval_rows):
        raise RuntimeError("冻结 eval 必须恰好 500 条且字段完整")
    eval_qids = [r.get("qid") for r in eval_rows]
    eval_queries = [r.get("query") for r in eval_rows]
    if (len(set(eval_qids)) != 500 or len(set(eval_queries)) != 500
            or any(not r.get("query") or not r.get("user_prompt") or not r.get("answer") for r in eval_rows)
            or any(r.get("split") != "eval" for r in eval_rows)
            or any(r.get("qid") != qid_of(r.get("query")) for r in eval_rows)):
        raise RuntimeError("冻结 eval 的 qid/query/split/非空字段校验失败")
    support_qids = [r.get("qid") for r in support_rows]
    if (len(support_rows) != 2239 or len(set(support_qids)) != 2239
            or any(not r.get("v1_answers") for r in support_rows)
            or not set(eval_qids) <= set(support_qids)):
        raise RuntimeError("V1 support 必须 2239 个唯一非空池并完整覆盖 eval")
    return eval_rows, support_rows


def normalize_generation_rows(rows: list[dict]) -> list[dict]:
    """Rebuild only fields deterministically derived from the preserved gen_text."""
    normalized = []
    for row in rows:
        parsed = parse_think_answer_diagnostic(row.get("gen_text") or "")
        item = dict(row)
        item.update(think=parsed["think"], answer=parsed["answer"],
                    format_ok=parsed["format_ok"], format_reason=parsed["format_reason"])
        normalized.append(item)
    return normalized


def infer_status(path: Path) -> dict:
    result = {"ok": False, "rows": 0, "empty_answer": 0, "empty_think": 0,
              "format_failures": 0, "format_failure_qids": [],
              "unaccounted_empty": 0, "reason": "missing"}
    if not path.exists():
        return result
    try:
        rows = read_jsonl(path)
        eval_rows, _ = validate_frozen_inputs()
    except Exception as exc:
        result["reason"] = f"read_failed:{exc!r}"
        return result
    result["rows"] = len(rows)
    result["empty_answer"] = sum(1 for r in rows if not (r.get("answer") or "").strip())
    result["empty_think"] = sum(1 for r in rows if not (r.get("think") or "").strip())
    result["format_failures"] = sum(1 for r in rows if r.get("format_ok") is False)
    result["format_failure_qids"] = [qid_of(r.get("query")) for r in rows if r.get("format_ok") is False]
    required = {"query", "user_prompt", "gold_answer", "gen_text", "think", "answer",
                "format_ok", "format_reason"}
    schema_ok = len(rows) == 500 and all(required <= set(r) for r in rows)
    aligned = schema_ok and all(
        out.get("query") == src.get("query")
        and out.get("user_prompt") == src.get("user_prompt")
        and out.get("gold_answer") == src.get("answer")
        for out, src in zip(rows, eval_rows)
    )
    generated = schema_ok and all((r.get("gen_text") or "").strip() for r in rows)
    derived_ok = schema_ok and all(
        r.get("think") == p["think"]
        and r.get("answer") == p["answer"]
        and r.get("format_ok") is p["format_ok"]
        and r.get("format_reason") == p["format_reason"]
        for r, p in ((row, parse_think_answer_diagnostic(row.get("gen_text") or "")) for row in rows)
    )
    # A non-empty raw generation may contain an explicitly accounted model
    # format failure.  It remains one of the frozen 500 items.  Any empty field
    # on a row claiming format_ok is an unaccounted corruption and is rejected.
    result["unaccounted_empty"] = sum(
        1 for r in rows if r.get("format_ok") is True
        and (not (r.get("think") or "").strip() or not (r.get("answer") or "").strip()))
    accounted = schema_ok and all(
        ((r.get("format_ok") is True
          and bool((r.get("think") or "").strip())
          and bool((r.get("answer") or "").strip())
          and r.get("format_reason") == "ok")
         or (r.get("format_ok") is False
             and bool((r.get("think") or "").strip())
             and r.get("format_reason") not in (None, "", "ok")))
        for r in rows)
    result["ok"] = bool(aligned and generated and derived_ok and accounted
                        and result["unaccounted_empty"] == 0)
    result["reason"] = "ok" if result["ok"] else "schema/alignment/derived/accounting failure"
    return result


def infer_identity() -> dict:
    import hashlib
    infer_contract_files = [
        ROOT / "pipeline" / "step03_eval_infer.py",
        ROOT / "pipeline" / "reward.py",
        ROOT / "pipeline" / "vllm_client.py",
    ]
    return {
        **expected_merge_identity(),
        "served_name": SERVED_NAME,
        "merged": str(DPO_MERGED),
        "merged_index_sha256": sha256(DPO_MERGED / "model.safetensors.index.json"),
        "eval": str(V2_EVAL),
        "eval_sha256": sha256(V2_EVAL),
        "support_sha256": sha256(V2_V1_SUPPORT),
        "system_prompt_sha256": hashlib.sha256(system_for(SERVED_NAME).encode("utf-8")).hexdigest(),
        "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": GEN_MAX_NEW_TOKENS},
        "infer_contract_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in infer_contract_files},
    }


def kimi_identity() -> dict:
    """把断点绑定到原 Kimi 裁判模型和完整三分实现，禁止跨口径混用 progress。"""
    contract_files = [
        ROOT / "pipeline" / "step_v2_eval.py",
        ROOT / "pipeline" / "v2_common.py",
        ROOT / "pipeline" / "reward.py",
        ROOT / "pipeline" / "judgecal_common.py",
        ROOT / "pipeline" / "kimi_client.py",
        ROOT / "pipeline" / "rules_v6.py",
    ]
    return {
        "kimi_model": KIMI_MODEL,
        "kimi_base_url": KIMI_BASE_URL,
        "kimi_enable_thinking": KIMI_ENABLE_THINKING,
        "judge_temperature": JUDGE_TEMPERATURE,
        "judge_top_p": JUDGE_TOP_P,
        "judge_contract_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in contract_files},
    }


def infer_trusted() -> bool:
    status = infer_status(INFER)
    if not status["ok"] or not INFER_PROVENANCE.exists():
        return False
    try:
        prov = read_json(INFER_PROVENANCE)
        identity = infer_identity()
    except Exception:
        return False
    manifest_ok = True
    if prov.get("format_accounting_manifest_sha256"):
        manifest_ok = (INFER_FORMAT_MANIFEST.exists()
                       and prov["format_accounting_manifest_sha256"] == sha256(INFER_FORMAT_MANIFEST))
    return (all(prov.get(k) == v for k, v in identity.items())
            and prov.get("infer_sha256") == sha256(INFER)
            and prov.get("rows") == 500
            and prov.get("format_failures") == status["format_failures"]
            and prov.get("format_failure_qids") == status["format_failure_qids"]
            and manifest_ok)


def scores_strict() -> bool:
    if not SCORES.exists():
        return False
    try:
        rows = read_jsonl(SCORES)
        eval_rows, _ = validate_frozen_inputs()
        infer_rows = read_jsonl(INFER)
    except Exception:
        return False
    expected_qids = {r["qid"] for r in eval_rows}
    infer_format = {qid_of(r.get("query")): (r.get("format_ok"), r.get("format_reason"))
                    for r in infer_rows}
    return (len(rows) == 500 and {r.get("qid") for r in rows} == expected_qids
            and all(r.get("clean_n") == 3 and r.get("clean_score") is not None
                    and not r.get("no_pool") and r.get("in_pool") is not None
                    and isinstance(r.get("rule_pass"), bool)
                    and isinstance(r.get("format_ok"), bool)
                    and (r.get("format_ok") or r.get("rule_pass") is False)
                    and (not r.get("empty_answer")
                         or (r.get("in_pool") is False and r.get("answer_reason") == "empty_answer"))
                    and (r.get("format_ok"), r.get("format_reason")) == infer_format.get(r.get("qid"))
                    for r in rows))


def report_complete() -> bool:
    if not (REPORT.exists() and SUMMARY.exists() and EVAL_PROVENANCE.exists()
            and infer_trusted() and scores_strict()):
        return False
    try:
        text = REPORT.read_text(encoding="utf-8", errors="replace")
        summary = read_json(SUMMARY)
        prov = read_json(EVAL_PROVENANCE)
    except Exception:
        return False
    infer_check = infer_status(INFER)
    summary_ok = (summary.get("tag") == TAG and summary.get("n") == 500
                  and summary.get("n_valid") == 500 and summary.get("n_pool") == 500
                  and int(summary.get("n_empty_answer", -1)) == infer_check["empty_answer"]
                  and int(summary.get("n_format_failure", -1)) == infer_check["format_failures"]
                  and summary.get("format_failure_qids") == infer_check["format_failure_qids"]
                  and int(summary.get("n_unaccounted_empty", -1)) == 0)
    expected_kimi = kimi_identity()
    prov_ok = (prov.get("tag") == TAG and prov.get("infer_sha256") == sha256(INFER)
               and prov.get("scores_sha256") == sha256(SCORES)
               and prov.get("summary_sha256") == sha256(SUMMARY)
               and prov.get("report_sha256") == sha256(REPORT)
               and prov.get("rows") == 500 and prov.get("clean_n") == 3
               and prov.get("format_failures") == infer_check["format_failures"]
               and prov.get("format_failure_qids") == infer_check["format_failure_qids"]
               and all(prov.get(k) == v for k, v in expected_kimi.items()))
    return (summary_ok and prov_ok
            and all(x in text for x in ("Kimi干净分", "规则去检索腔通过率", "答案在池率")))


def preflight() -> None:
    required = [
        RFT_BASE / "config.json",
        RFT_BASE / ".done",
        RFT_BASE / "tokenizer_config.json",
        DOWNLOADED_ADAPTER / "adapter_config.json",
        DOWNLOADED_ADAPTER / "adapter_model.safetensors",
        V2_EVAL,
        V2_V1_SUPPORT,
        PY,
        VLLM_ENV / "bin" / "vllm",
        SCRIPTS / "merge_lora_model.sh",
        SCRIPTS / "serve_model_vllm.sh",
        ROOT / "pipeline" / "step03_eval_infer.py",
        ROOT / "pipeline" / "step_v2_eval.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("preflight missing:\n" + "\n".join(missing))
    expected_v2_out = Path(os.environ.get("ZHJG_OUTPUT_DIR", str(WORK_DIR / "output"))) / "derag2"
    if V2_EVAL.parent.resolve() != expected_v2_out.resolve():
        raise SystemExit(f"V2 输出根被环境变量劫持：实际={V2_EVAL.parent}，期望={expected_v2_out}")
    validate_frozen_inputs()
    validate_historical_controls()
    if not model_shape_ok(RFT_BASE):
        raise SystemExit(f"RFT 底座结构/总尺寸/14分片校验失败: {RFT_BASE}")
    if (DOWNLOADED_ADAPTER / "adapter_model.safetensors").stat().st_size < 250_000_000:
        raise SystemExit("DPO adapter 权重尺寸异常（期望约 257M）")
    weight_sha = sha256(DOWNLOADED_ADAPTER / "adapter_model.safetensors")
    config_sha = sha256(DOWNLOADED_ADAPTER / "adapter_config.json")
    if weight_sha != EXPECTED_ADAPTER_SHA256 or config_sha != EXPECTED_ADAPTER_CONFIG_SHA256:
        raise SystemExit(
            "DPO adapter SHA256 与已审计 checkpoint-108 不一致:\n"
            f"weight={weight_sha}\nconfig={config_sha}")
    if (KIMI_MODEL != EXPECTED_KIMI_MODEL or KIMI_BASE_URL != EXPECTED_KIMI_BASE_URL
            or KIMI_ENABLE_THINKING is not False or JUDGE_TEMPERATURE != 0.0 or JUDGE_TOP_P != 0.7):
        raise SystemExit(
            "Kimi 裁判口径被环境变量改变，拒绝混评："
            f"model={KIMI_MODEL} base={KIMI_BASE_URL} thinking={KIMI_ENABLE_THINKING} "
            f"temperature={JUDGE_TEMPERATURE} top_p={JUDGE_TOP_P}")
    free_gib = shutil.disk_usage(WORK_DIR).free / 1024**3
    if free_gib < 100:
        raise SystemExit(f"nvme01 空间不足：{free_gib:.1f}GiB < 100GiB")
    # adapter 里的 AutoDL base_model_name_or_path 只是训练元数据；合并脚本会显式传本地 RFT_BASE。
    adapter_cfg = json.loads((DOWNLOADED_ADAPTER / "adapter_config.json").read_text(encoding="utf-8"))
    if (adapter_cfg.get("peft_type") != "LORA" or adapter_cfg.get("task_type") != "CAUSAL_LM"
            or adapter_cfg.get("r") != 16 or adapter_cfg.get("lora_alpha") != 32
            or set(adapter_cfg.get("target_modules") or []) != EXPECTED_TARGET_MODULES):
        raise SystemExit(f"DPO adapter 配置不符合本轮 r16/alpha32/7 modules: {adapter_cfg}")
    emit(f"PREFLIGHT OK | base={RFT_BASE} | downloaded_adapter={DOWNLOADED_ADAPTER} | canonical_lora="
         f"{CANONICAL_LORA} | adapter_declared_base={adapter_cfg.get('base_model_name_or_path')} | "
         f"adapter_sha256={weight_sha} | eval_sha256={sha256(V2_EVAL)} | support_sha256={sha256(V2_V1_SUPPORT)} | "
         f"output={DPO_MERGED} | tag={TAG} | free={free_gib:.1f}GiB")


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_canonical_lora() -> None:
    """把 AutoDL 交接目录原子发布到 run_v2 原本会产出的 LoRA 目录，让旧编排器也识别为已训练。"""
    src_weight = DOWNLOADED_ADAPTER / "adapter_model.safetensors"
    if ((CANONICAL_LORA / ".done").exists()
            and (CANONICAL_ADAPTER / "adapter_config.json").exists()
            and (CANONICAL_ADAPTER / "adapter_model.safetensors").exists()):
        if (sha256(src_weight) != sha256(CANONICAL_ADAPTER / "adapter_model.safetensors")
                or sha256(DOWNLOADED_ADAPTER / "adapter_config.json")
                != sha256(CANONICAL_ADAPTER / "adapter_config.json")):
            raise RuntimeError(f"规范 LoRA 已存在但与下载 adapter 哈希不一致: {CANONICAL_LORA}")
        emit(f"SKIP  publish adapter; canonical LoRA complete: {CANONICAL_LORA}")
        return
    if CANONICAL_LORA.exists():
        moved = CANONICAL_LORA.with_name(
            f"{CANONICAL_LORA.name}.interrupted-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        CANONICAL_LORA.rename(moved)
        emit(f"preserve incomplete canonical LoRA: {CANONICAL_LORA} -> {moved}")
    tmp = CANONICAL_LORA.with_name(CANONICAL_LORA.name + ".partial")
    if tmp.exists():
        moved = tmp.with_name(f"{tmp.name}.interrupted-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        tmp.rename(moved)
        emit(f"preserve stale adapter partial: {tmp} -> {moved}")
    emit(f"PUBLISH adapter | {DOWNLOADED_ADAPTER} -> {CANONICAL_ADAPTER}")
    tmp.mkdir(parents=True)
    shutil.copytree(DOWNLOADED_ADAPTER, tmp / "checkpoint-108")
    copied = tmp / "checkpoint-108" / "adapter_model.safetensors"
    copied_cfg = tmp / "checkpoint-108" / "adapter_config.json"
    if (sha256(src_weight) != sha256(copied)
            or sha256(DOWNLOADED_ADAPTER / "adapter_config.json") != sha256(copied_cfg)):
        raise RuntimeError("下载 adapter 复制到规范目录后 SHA256 不一致")
    (tmp / ".done").write_text(now(), encoding="utf-8")
    tmp.rename(CANONICAL_LORA)
    emit(f"PUBLISH OK | canonical_lora={CANONICAL_LORA} | checkpoint=108")


def merge_dpo() -> None:
    if merged_complete(DPO_MERGED):
        emit(f"SKIP  merge; complete: {DPO_MERGED}")
        return
    if DPO_MERGED.exists():
        moved = DPO_MERGED.with_name(
            f"{DPO_MERGED.name}.interrupted-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        DPO_MERGED.rename(moved)
        emit(f"preserve incomplete merged output: {DPO_MERGED} -> {moved}")
    run("merge_dpo_2s_2s_2s", [
        "bash", SCRIPTS / "merge_lora_model.sh", RFT_BASE, CANONICAL_ADAPTER, DPO_MERGED,
    ], env={"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
    if not merged_complete(DPO_MERGED):
        # 新合并此时尚未写 provenance manifest；先验模型结构，再把 base/adapter 身份绑定进去。
        if not ((DPO_MERGED / ".done").exists() and (DPO_MERGED / "tokenizer_config.json").exists()
                and model_shape_ok(DPO_MERGED)):
            raise RuntimeError(f"合并完成但模型结构校验失败: {DPO_MERGED}")
        manifest = {**expected_merge_identity(),
                    "merged": str(DPO_MERGED),
                    "merged_index_sha256": sha256(DPO_MERGED / "model.safetensors.index.json"),
                    "merge_delta_proof": merge_delta_proof(DPO_MERGED),
                    "created_at": now()}
        MERGE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not merged_complete(DPO_MERGED):
        raise RuntimeError(f"合并 provenance 校验失败: {DPO_MERGED}")
    emit(f"MERGED OK | {DPO_MERGED} | shards=14")


def gpu_snapshot() -> list[dict]:
    cp = subprocess.run([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True)
    apps = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=False)
    if apps.returncode != 0:
        raise RuntimeError(
            "nvidia-smi compute-app 查询失败；为避免把同事占用卡误判为空闲，拒绝选卡: "
            + (apps.stderr or apps.stdout).strip())
    compute_uuids = set()
    for line in apps.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0].startswith("GPU-"):
            compute_uuids.add(parts[0])
    rows = []
    for line in cp.stdout.splitlines():
        vals = [x.strip() for x in line.split(",")]
        if len(vals) == 5:
            rows.append({"index": int(vals[0]), "uuid": vals[1], "used": int(vals[2]),
                         "free": int(vals[3]), "util": int(vals[4]),
                         "has_compute_pid": vals[1] in compute_uuids})
    return rows


def assert_selected_gpus_still_free(gpus: str) -> None:
    requested = {int(x) for x in gpus.split(",")}
    snap = {r["index"]: r for r in gpu_snapshot()}
    safe = all(i in snap and snap[i]["used"] <= GPU_USED_MAX_MIB
               and snap[i]["util"] <= GPU_UTIL_MAX and not snap[i]["has_compute_pid"]
               for i in requested)
    if len(requested) != 2 or not safe:
        detail = {i: snap.get(i) for i in requested}
        raise RuntimeError(f"启动 vLLM 前即时复核发现所选 GPU 已不再空闲，拒绝抢卡: {detail}")


def choose_free_gpus() -> str:
    requested = None
    if REQUESTED_GPUS:
        requested = [int(x) for x in REQUESTED_GPUS.split(",")]
        if len(requested) != 2 or len(set(requested)) != 2:
            raise SystemExit("V2_DPO_EVAL_GPUS 必须是两个不同编号，例如 0,1")
    save_state(status="waiting_gpu", stage="wait_for_2_free_gpus", pid=None, requested_gpus=REQUESTED_GPUS or "auto")
    emit(f"WAIT  GPU | need=2 | requested={REQUESTED_GPUS or 'auto'} | "
         f"free判据 used<={GPU_USED_MAX_MIB}MiB 且 util<={GPU_UTIL_MAX}% | 不会抢占/杀同事进程")
    started = time.time()
    last_emit = 0.0
    stable: dict[tuple[int, ...], int] = {}
    while True:
        snap = gpu_snapshot()
        free = [r for r in snap if r["used"] <= GPU_USED_MAX_MIB and r["util"] <= GPU_UTIL_MAX
                and not r["has_compute_pid"]]
        if requested is not None:
            by_id = {r["index"]: r for r in free}
            chosen = requested if all(i in by_id for i in requested) else []
        else:
            chosen = [r["index"] for r in free[:2]]
        if len(chosen) == 2:
            key = tuple(chosen)
            stable = {key: stable.get(key, 0) + 1}
            if stable[key] >= GPU_STABLE_SAMPLES:
                value = ",".join(map(str, chosen))
                emit(f"GPU READY | {value} | stable_samples={stable[key]} | no_compute_pid")
                save_state(status="running", stage="gpu_ready", selected_gpus=value)
                return value
        else:
            stable = {}
        elapsed = time.time() - started
        if GPU_WAIT_TIMEOUT > 0 and elapsed >= GPU_WAIT_TIMEOUT:
            raise TimeoutError(f"等待两张空卡超时 {int(elapsed)}s")
        if time.time() - last_emit >= 300:
            compact = " ".join(
                f"{r['index']}:{r['used']}MiB/{r['util']}%/pid={'Y' if r['has_compute_pid'] else 'N'}"
                for r in snap)
            emit(f"WAIT  GPU | elapsed={int(elapsed)}s | {compact}")
            last_emit = time.time()
        time.sleep(GPU_WAIT_INTERVAL)


def vllm_port() -> int:
    from urllib.parse import urlparse
    parsed = urlparse(VLLM_BASE_URL)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise RuntimeError(f"VLLM_BASE_URL 必须是本机 HTTP 服务，实际={VLLM_BASE_URL}")
    if parsed.path.rstrip("/") != "/v1":
        raise RuntimeError(f"VLLM_BASE_URL 必须以 /v1 结尾，实际={VLLM_BASE_URL}")
    url_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    serve_port = int(os.environ.get("VLLM_PORT", "8000"))
    if url_port != serve_port:
        raise RuntimeError(f"VLLM_BASE_URL 端口 {url_port} 与 VLLM_PORT {serve_port} 不一致")
    return serve_port


def port_in_use() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        return sock.connect_ex(("127.0.0.1", vllm_port())) == 0


def process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def is_own_vllm(pid: int) -> bool:
    cmd = process_cmdline(pid)
    return bool(cmd and "vllm" in cmd and str(DPO_MERGED) in cmd and SERVED_NAME in cmd)


def cleanup_stale_own_vllm() -> None:
    pidf = RAW_DIR / "merged_chain_vllm.pid"
    if not pidf.exists():
        return
    try:
        pid = int(pidf.read_text().strip())
    except ValueError:
        raise RuntimeError(f"本链 vLLM PID 文件损坏: {pidf}")
    if not Path(f"/proc/{pid}").exists():
        pidf.unlink(missing_ok=True)
        return
    if not is_own_vllm(pid):
        raise RuntimeError(f"PID 文件指向非本链进程，拒绝清理 pid={pid}: {process_cmdline(pid)}")
    emit(f"CLEAN stale own vLLM process_group={pid}")
    os.killpg(pid, signal.SIGTERM)
    time.sleep(8)
    if Path(f"/proc/{pid}").exists():
        if not is_own_vllm(pid):
            raise RuntimeError(f"SIGKILL 前 PID 身份变化，拒绝误杀 pid={pid}: {process_cmdline(pid)}")
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    pidf.unlink(missing_ok=True)


def start_vllm(gpus: str) -> None:
    global _launched_vllm, _vllm_pid
    assert_selected_gpus_still_free(gpus)
    if port_in_use():
        raise RuntimeError(f"127.0.0.1:{vllm_port()} 已被占用；为避免误杀别人的服务，本续跑器拒绝启动")
    env = {
        "VLLM_ENV": str(VLLM_ENV),
        "VLLM_GPUS": gpus,
        "VLLM_SERVE_GPU_UTIL": os.environ.get("VLLM_SERVE_GPU_UTIL", "0.88"),
        "VLLM_PORT": str(vllm_port()),
        "ZHJG_LOG_DIR": str(RAW_DIR),
    }
    run("serve_v2_2s_2s_2s", [
        "bash", SCRIPTS / "serve_model_vllm.sh", DPO_MERGED, SERVED_NAME,
    ], env=env)
    pidf = RAW_DIR / "merged_chain_vllm.pid"
    if not pidf.exists():
        raise RuntimeError("serve 脚本返回成功但没有本链 PID 文件")
    _vllm_pid = int(pidf.read_text().strip())
    if not is_own_vllm(_vllm_pid):
        raise RuntimeError(f"启动出的 PID 不是预期模型服务: pid={_vllm_pid} cmd={process_cmdline(_vllm_pid)}")
    _launched_vllm = True
    save_state(status="waiting_vllm", stage="wait_vllm", selected_gpus=gpus,
               raw_log=str(RAW_DIR / "merged_chain_vllm.log"))
    emit(f"WAIT  vLLM | model={SERVED_NAME} | gpu={gpus}")
    vllm_client.wait_ready(max_wait=1800)
    import requests
    response = requests.get(VLLM_BASE_URL.rstrip("/") + "/models", timeout=10)
    response.raise_for_status()
    model_ids = {m.get("id") for m in response.json().get("data", [])}
    if SERVED_NAME not in model_ids or not is_own_vllm(_vllm_pid):
        raise RuntimeError(f"vLLM ready 但模型/PID身份不对: models={model_ids} pid={_vllm_pid}")
    emit(f"READY vLLM | model={SERVED_NAME} | gpu={gpus}")


def stop_vllm() -> None:
    global _launched_vllm, _vllm_pid
    pidf = RAW_DIR / "merged_chain_vllm.pid"
    if not _launched_vllm and not pidf.exists():
        return
    if pidf.exists():
        try:
            pid = int(pidf.read_text().strip())
        except ValueError as exc:
            raise RuntimeError(f"本链 vLLM PID 文件损坏，拒绝清理: {pidf}") from exc
        if not Path(f"/proc/{pid}").exists():
            pidf.unlink(missing_ok=True)
        else:
            if not is_own_vllm(pid):
                raise RuntimeError(f"PID 文件指向非本链进程，拒绝终止 pid={pid}: {process_cmdline(pid)}")
            emit(f"STOP  vLLM process_group={pid}")
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(8)
            if Path(f"/proc/{pid}").exists():
                if not is_own_vllm(pid):
                    raise RuntimeError(
                        f"SIGKILL 前 PID 身份变化，拒绝误杀 pid={pid}: {process_cmdline(pid)}")
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            pidf.unlink(missing_ok=True)
    _launched_vllm = False
    _vllm_pid = None
    for _ in range(30):
        if not port_in_use():
            return
        time.sleep(1)
    raise RuntimeError(f"停止本链 vLLM 后端口 {vllm_port()} 仍被占用")


def smoke_infer() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    smoke_eval = RAW_DIR / "smoke3_eval.jsonl"
    smoke_out = RAW_DIR / "smoke3_infer.jsonl"
    rows = read_jsonl(V2_EVAL)[:3]
    with smoke_eval.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    run("smoke3_infer", [
        PY, "-X", "utf8", "pipeline/step03_eval_infer.py",
        "--model", SERVED_NAME, "--eval_file", smoke_eval, "--out", smoke_out,
    ])
    smoke_rows = read_jsonl(smoke_out)
    required = {"query", "user_prompt", "gold_answer", "gen_text", "think", "answer"}
    aligned = (len(smoke_rows) == 3 and all(required <= set(r) for r in smoke_rows)
               and all(out.get("query") == src.get("query")
                       and out.get("user_prompt") == src.get("user_prompt")
                       and out.get("gold_answer") == src.get("answer")
                       for out, src in zip(smoke_rows, rows)))
    empty = sum(1 for r in smoke_rows if not (r.get("answer") or "").strip())
    empty_think = sum(1 for r in smoke_rows if not (r.get("think") or "").strip())
    if not aligned or empty or empty_think:
        raise RuntimeError(f"smoke 失败 rows={len(smoke_rows)} aligned={aligned} "
                           f"empty_answer={empty} empty_think={empty_think}")
    emit("SMOKE OK | rows=3 | empty_think=0 | empty_answer=0 | system=去检索腔")


def archive_eval_artifacts(*, include_progress: bool, reason: str) -> None:
    emit(f"INVALIDATE eval artifacts | reason={reason} | include_progress={include_progress}")
    paths = [SCORES, REPORT, SUMMARY, EVAL_PROVENANCE]
    if include_progress:
        paths += [PROGRESS, PROGRESS_BINDING]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for path in paths:
        if path.exists():
            moved = path.with_name(f"{path.name}.stale-{stamp}")
            path.rename(moved)
            emit(f"ARCHIVE {path} -> {moved}")


def write_infer_provenance(*, format_manifest_sha256: str | None = None) -> None:
    status = infer_status(INFER)
    identity = {**infer_identity(), "infer_sha256": sha256(INFER), "rows": 500,
                "format_failures": status["format_failures"],
                "format_failure_qids": status["format_failure_qids"],
                "created_at": now()}
    if format_manifest_sha256:
        identity["format_accounting_manifest_sha256"] = format_manifest_sha256
    INFER_PROVENANCE.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")


def adopt_approved_existing_infer() -> None:
    """Adopt the already-generated frozen-500 file without any GPU regeneration.

    This path is deliberately narrow and opt-in.  It preserves every gen_text
    byte, atomically rebuilds only parser-derived fields, and accepts exactly the
    one user-inspected max-token format failure.
    """
    if infer_trusted() or not INFER.exists() or INFER_PROVENANCE.exists():
        return
    if os.environ.get("V2_DPO_APPROVE_FORMAT_ACCOUNTING") != "1":
        return
    expected_sha = os.environ.get("V2_DPO_EXPECT_INFER_SHA256", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise RuntimeError("批准复用现有infer时必须设置 V2_DPO_EXPECT_INFER_SHA256=原文件SHA256")
    source_sha = sha256(INFER)
    if source_sha.lower() != expected_sha:
        raise RuntimeError(f"现有infer SHA与用户批准值不一致: actual={source_sha} expected={expected_sha}")

    rows = read_jsonl(INFER)
    eval_rows, _ = validate_frozen_inputs()
    required = {"query", "user_prompt", "gold_answer", "gen_text", "think", "answer"}
    transport_ok = (len(rows) == 500 and all(required <= set(r) for r in rows)
                    and all((r.get("gen_text") or "").strip() for r in rows)
                    and all(out.get("query") == src.get("query")
                            and out.get("user_prompt") == src.get("user_prompt")
                            and out.get("gold_answer") == src.get("answer")
                            for out, src in zip(rows, eval_rows)))
    if not transport_ok:
        raise RuntimeError("批准的现有infer未通过500题逐行对齐/非空raw传输合同，拒绝采纳")

    normalized = normalize_generation_rows(rows)
    failures = [(i + 1, qid_of(r.get("query")), r.get("format_reason"))
                for i, r in enumerate(normalized) if not r.get("format_ok")]
    if failures != [(437, EXPECTED_FORMAT_FAILURE_QID, "missing_think_close+empty_answer")]:
        raise RuntimeError(f"现有infer格式失败清单不是已人工核验的唯一第437题，拒绝采纳: {failures}")
    for old, new in zip(rows, normalized):
        if qid_of(old.get("query")) != EXPECTED_FORMAT_FAILURE_QID:
            if old.get("think") != new.get("think") or old.get("answer") != new.get("answer"):
                raise RuntimeError(f"正常题重解析发生语义字段变化，拒绝采纳: query={old.get('query')!r}")
    if not normalized[436]["think"].strip() or normalized[436]["answer"] != "":
        raise RuntimeError("第437题必须保留非空残缺think且answer严格为空，拒绝伪造")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = INFER.with_name(f"{INFER.name}.pre_format_accounting-{stamp}")
    shutil.copy2(INFER, backup)
    tmp = INFER.with_suffix(INFER.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in normalized:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_rows = read_jsonl(tmp)
    if any(a.get("gen_text") != b.get("gen_text") for a, b in zip(rows, tmp_rows)):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("格式记账迁移改变了原始gen_text，已停止")
    status = infer_status(tmp)
    if not status["ok"] or status["format_failures"] != 1 or status["empty_answer"] != 1:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"格式记账迁移后合同失败: {status}")
    tmp.replace(INFER)
    archive_eval_artifacts(include_progress=True, reason="approved format-failure accounting migration")
    manifest = {
        "contract": "preserve_raw_account_model_format_failure_v1",
        "approved_by_env": True,
        "source_infer": str(backup),
        "source_infer_sha256": source_sha,
        "normalized_infer": str(INFER),
        "normalized_infer_sha256": sha256(INFER),
        "rows": 500,
        "generation_changed": False,
        "gold_or_v1_answer_injected": False,
        "format_failures": [{"line": line, "qid": qid, "reason": reason}
                            for line, qid, reason in failures],
        "created_at": now(),
    }
    INFER_FORMAT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_infer_provenance(format_manifest_sha256=sha256(INFER_FORMAT_MANIFEST))
    if not infer_trusted():
        raise RuntimeError("现有infer格式记账并写入provenance后仍不可信")
    emit(f"ADOPT INFER OK | rows=500 | raw_unchanged=yes | format_failure=1/500 | "
         f"empty_answer=1 | source_sha256={source_sha} | normalized_sha256={sha256(INFER)}")


def full_infer() -> None:
    status = infer_status(INFER)
    if infer_trusted():
        emit(f"SKIP  infer; trusted complete: {INFER} | rows=500")
        return
    if INFER.exists():
        archive(INFER, "untrusted")
    if INFER_PROVENANCE.exists():
        archive(INFER_PROVENANCE, "untrusted")
    archive_eval_artifacts(include_progress=True, reason="infer regenerated or provenance mismatch")
    INFER.parent.mkdir(parents=True, exist_ok=True)
    run("v2_2s_2s_2s_infer", [
        PY, "-X", "utf8", "pipeline/step03_eval_infer.py",
        "--model", SERVED_NAME, "--eval_file", V2_EVAL, "--out", INFER,
    ])
    status = infer_status(INFER)
    if not status["ok"]:
        raise RuntimeError(f"500 题推理传输/对齐/格式记账失败: {status} path={INFER}")
    write_infer_provenance()
    if not infer_trusted():
        raise RuntimeError("推理 provenance 写入后仍无法验证")
    emit(f"INFER OK | rows=500 | format_failure={status['format_failures']} | "
         f"empty_think={status['empty_think']} | empty_answer={status['empty_answer']} | "
         f"sha256={sha256(INFER)} | {INFER}")


def prepare_kimi_progress() -> None:
    current = {"tag": TAG, "infer_sha256": sha256(INFER), "eval_sha256": sha256(V2_EVAL),
               **kimi_identity()}
    if PROGRESS.exists():
        binding_ok = False
        if PROGRESS_BINDING.exists():
            try:
                old = read_json(PROGRESS_BINDING)
                binding_ok = all(old.get(k) == v for k, v in current.items())
            except Exception:
                pass
        if not binding_ok:
            archive_eval_artifacts(include_progress=True, reason="Kimi progress not bound to current infer")
    # 仅保留当前 infer 下已经完整做满 k=3 的题；clean_n<3 必须重判，不能拿 k=1/2 冒充 k=3。
    if PROGRESS.exists():
        eval_qids = {r["qid"] for r in validate_frozen_inputs()[0]}
        rows = read_jsonl(PROGRESS)
        kept = [r for r in rows if r.get("qid") in eval_qids and r.get("clean_n") == 3
                and r.get("clean_score") is not None and not r.get("no_pool")]
        if len(kept) != len(rows):
            tmp = PROGRESS.with_suffix(PROGRESS.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for row in kept:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp.replace(PROGRESS)
            emit(f"KIMI progress sanitize | kept={len(kept)} dropped={len(rows)-len(kept)}（未满k=3重判）")
    PROGRESS_BINDING.write_text(json.dumps({**current, "updated_at": now()}, ensure_ascii=False, indent=2),
                                encoding="utf-8")


def write_eval_provenance() -> None:
    status = infer_status(INFER)
    data = {"tag": TAG, "infer_sha256": sha256(INFER), "scores_sha256": sha256(SCORES),
            "summary_sha256": sha256(SUMMARY), "report_sha256": sha256(REPORT),
            "rows": 500, "clean_n": 3,
            "format_failures": status["format_failures"],
            "format_failure_qids": status["format_failure_qids"],
            **kimi_identity(), "created_at": now()}
    EVAL_PROVENANCE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def kimi_eval() -> None:
    if report_complete():
        emit(f"SKIP  Kimi eval; strictly verified report+summary: {SUMMARY}")
        return
    if not infer_trusted():
        raise RuntimeError("Kimi 评测前 infer 身份/冻结集/模型 provenance 不可信")
    for attempt in range(1, 4):
        prepare_kimi_progress()
        # 上一轮不完整 summary/scores/report 不得被误认完成；保留已绑定且满k=3的 progress 续跑。
        archive_eval_artifacts(include_progress=False, reason=f"strict Kimi attempt {attempt}")
        run(f"v2_2s_2s_2s_score_attempt{attempt}", [
            PY, "-X", "utf8", "pipeline/step_v2_eval.py",
            "--infer", INFER,
            "--scores", SCORES,
            "--report", REPORT,
            "--summary", SUMMARY,
            "--support", V2_V1_SUPPORT,
            "--tag", TAG,
        ])
        if scores_strict():
            summary = read_json(SUMMARY)
            if (summary.get("tag") == TAG and summary.get("n") == 500
                    and summary.get("n_valid") == 500 and summary.get("n_pool") == 500):
                write_eval_provenance()
                if report_complete():
                    emit(f"KIMI OK | attempt={attempt} | rows=500 | every clean_n=3 | {SUMMARY}")
                    return
        emit(f"KIMI RETRY | attempt={attempt} 未达到 500题×k3；下一轮只补未满k3题")
    raise RuntimeError(f"Kimi 连续3轮仍未完成严格 500题×k3: {REPORT}")


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_reference_summary(path: Path, expected: dict) -> dict:
    data = read_summary(path)
    if (data.get("tag") != expected["tag"] or data.get("n") != 500
            or data.get("n_valid") != 500 or data.get("n_pool") != 500
            or not 0.04 <= float(data.get("se", -1)) <= 0.06):
        raise RuntimeError(f"历史对照 summary 身份/样本数/SE 异常: {path}: {data}")
    for key in ("clean_mean", "rule_pass_rate", "in_pool_rate"):
        if abs(float(data.get(key, -999)) - expected[key]) > 0.0005:
            raise RuntimeError(
                f"历史对照 summary 的 {key} 偏离已审计值: {path}: "
                f"actual={data.get(key)} expected={expected[key]}")
    return data


def validate_historical_controls() -> dict[str, dict]:
    """历史三阶段必须也是同一冻结500题、每题Kimi k=3，才能与新 DPO 作因果比较。"""
    eval_qids = {r["qid"] for r in validate_frozen_inputs()[0]}
    checked = {}
    for stem, expected in HISTORICAL_CONTROLS.items():
        summary_path = SUMMARY.with_name(stem + "_summary.json")
        scores_path = SUMMARY.with_name(stem + "_scores.jsonl")
        if not scores_path.exists():
            raise RuntimeError(f"缺历史对照 scores，无法证明同为500题×k3: {scores_path}")
        rows = read_jsonl(scores_path)
        if (len(rows) != 500 or {r.get("qid") for r in rows} != eval_qids
                or any(r.get("clean_n") != 3 or r.get("clean_score") is None for r in rows)):
            raise RuntimeError(f"历史对照不是冻结500题×Kimi k3: {scores_path}")
        checked[stem] = validate_reference_summary(summary_path, expected)
    emit("HISTORICAL CONTROLS OK | baseline/SFT/RFT each frozen-500 × Kimi-k3")
    return checked


def write_result() -> None:
    baseline_path = SUMMARY.with_name("v2-baseline-v1_summary.json")
    sft_path = SUMMARY.with_name("v2-sft-2s_summary.json")
    rft_path = SUMMARY.with_name("v2-rft-2s-2s_summary.json")
    for p in (baseline_path, sft_path, rft_path, SUMMARY):
        if not p.exists():
            raise RuntimeError(f"缺对照 summary: {p}")
    controls = validate_historical_controls()
    base = controls["v2-baseline-v1"]
    sft = controls["v2-sft-2s"]
    rft = controls["v2-rft-2s-2s"]
    dpo = read_summary(SUMMARY)
    if not report_complete():
        raise RuntimeError("写最终结果前严格评测合同未通过")
    infer_check = infer_status(INFER)
    se = float(dpo.get("se") or 0.05)
    final_threshold = 3 * se
    # 严格沿用 step_v2_eval/接力提示的项目判据：两阶段差 >~3×SE≈0.15。
    diff_threshold = 3 * se
    delta_base = dpo["clean_mean"] - base["clean_mean"]
    delta_rft = dpo["clean_mean"] - rft["clean_mean"]
    final_pass = (delta_base > final_threshold
                  and dpo["rule_pass_rate"] >= sft["rule_pass_rate"]
                  and dpo["in_pool_rate"] >= 0.85
                  and int(dpo.get("n_unaccounted_empty", -1)) == 0)
    if delta_rft > diff_threshold:
        incremental = "DPO_TRUE_GAIN"
    elif delta_rft < -diff_threshold:
        incremental = "DPO_TRUE_REGRESSION"
    else:
        incremental = "DPO_NEUTRAL_WITHIN_NOISE"
    result = {
        "tag": TAG,
        "baseline": base,
        "sft": sft,
        "rft": rft,
        "dpo": dpo,
        "delta_clean_vs_baseline": round(delta_base, 4),
        "delta_clean_vs_rft": round(delta_rft, 4),
        "final_project_pass": final_pass,
        "dpo_incremental_verdict": incremental,
        "final_clean_threshold": round(final_threshold, 4),
        "incremental_diff_threshold": round(diff_threshold, 4),
        "empty_answer": infer_check["empty_answer"],
        "empty_think": infer_check["empty_think"],
        "format_failures": infer_check["format_failures"],
        "format_failure_qids": infer_check["format_failure_qids"],
        "format_pass_rate": dpo.get("format_pass_rate"),
        "rule_floor_from_sft": sft["rule_pass_rate"],
    }
    RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    RESULT_MD.write_text(
        "\n".join([
            f"# V2 DPO 本地接回结果 · {TAG}", "",
            "| 阶段 | Kimi干净分 | 规则通过率 | 答案在池率 |",
            "|---|---:|---:|---:|",
            f"| V1 baseline | {base['clean_mean']:.3f} | {base['rule_pass_rate']:.1%} | {base['in_pool_rate']:.1%} |",
            f"| SFT 2s | {sft['clean_mean']:.3f} | {sft['rule_pass_rate']:.1%} | {sft['in_pool_rate']:.1%} |",
            f"| RFT 2s-2s | {rft['clean_mean']:.3f} | {rft['rule_pass_rate']:.1%} | {rft['in_pool_rate']:.1%} |",
            f"| DPO 2s-2s-2s | {dpo['clean_mean']:.3f} | {dpo['rule_pass_rate']:.1%} | {dpo['in_pool_rate']:.1%} |",
            "",
            f"- 最终三件套：{'PASS' if final_pass else 'FAIL'}",
            f"- DPO 相对 RFT：{incremental}（Δclean={delta_rft:+.3f}；沿用项目 3×SE≈{diff_threshold:.3f}）",
            f"- 格式完整性：format_failure={infer_check['format_failures']}/500，"
            f"empty_think={infer_check['empty_think']}，empty_answer={infer_check['empty_answer']}；"
            "已在规则think与规则answer主分母中记失败，不重采样、不补答案",
            f"- 正式报告：`{REPORT}`",
        ]) + "\n", encoding="utf-8")
    emit(f"RESULT | final={'PASS' if final_pass else 'FAIL'} | incremental={incremental} | "
         f"DPO clean={dpo['clean_mean']:.3f} rule={dpo['rule_pass_rate']:.1%} in_pool={dpo['in_pool_rate']:.1%} | "
         f"result={RESULT_MD}")


def main() -> None:
    global _launched_vllm
    acquire_lock()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt(f"signal {signum}")))
    # 上次结果只能代表上次执行；本次必须由已绑定 provenance 的产物重新生成。
    archive(RESULT_JSON, "previous_run")
    archive(RESULT_MD, "previous_run")
    save_state(status="running", stage="preflight", pid=os.getpid(), paths={
        "base": str(RFT_BASE), "downloaded_adapter": str(DOWNLOADED_ADAPTER),
        "canonical_lora": str(CANONICAL_LORA), "merged": str(DPO_MERGED),
        "infer": str(INFER), "summary": str(SUMMARY), "report": str(REPORT),
    })
    emit(f"RESUME START | tag={TAG} | pid={os.getpid()} | logs={LOG_ROOT}")
    try:
        preflight()
        publish_canonical_lora()
        merge_dpo()                       # CPU 合并，可在 8 卡仍被占时完成
        cleanup_stale_own_vllm()          # 只清 PID/cmdline 均确认属于本正式叶的遗留服务
        adopt_approved_existing_infer()   # 只重建派生字段；原始500条gen_text不变
        if report_complete():             # 已评完的断点直接汇总，不再等 GPU
            emit("SKIP  GPU/infer/eval; existing report complete")
        else:
            if not infer_trusted():
                if os.environ.get("V2_DPO_NO_GPU") == "1":
                    raise RuntimeError("V2_DPO_NO_GPU=1 且现有infer仍不可信；拒绝等待GPU或重新推理")
                gpus = choose_free_gpus()  # 没空卡就干净等待，不抢同事进程
                start_vllm(gpus)
                smoke_infer()
                full_infer()
                stop_vllm()
            kimi_eval()                   # 纯 Kimi，不占 GPU
        write_result()
        save_state(status="complete", stage="done", pid=None, raw_log=None)
        emit("RESUME COMPLETE")
    except KeyboardInterrupt:
        save_state(status="stopped", stage=_state.get("stage"), pid=None, error="KeyboardInterrupt")
        emit("RESUME STOPPED by user")
        raise
    except BaseException as exc:
        save_state(status="failed", stage=_state.get("stage"), pid=None, error=repr(exc))
        emit(f"RESUME FAILED | {exc!r}")
        raise
    finally:
        try:
            stop_vllm()
        except BaseException as exc:
            emit(f"WARN  final vLLM cleanup failed safely: {exc!r}")


if __name__ == "__main__":
    main()
