"""judgecal · 判官标定实验编排器（离线、零 GPU、只调 Kimi）。

新版实验设计【模块1】：标定 Kimi"逐句换词复述识别"能力。
回答两个问题（见 docs/journal/109_judgecal_module_design.md）：
  实验一·判得跳不跳 → 该打几遍(稳定遍数 K，按四类最难那类取)；
  实验二·判得准不准 → reworded 召回 / legit_use 误伤（四类标记率）。

链路：step160(校验装配) → step161(每条 think 让 Kimi 判 16 遍) → step162(读两遍出报告)。
**不 serve vLLM、不碰 GPU、不重新生成任何采样**——和 X 重测补丁一样安全，不动训练显存设置。

数据 data/judgecal_sentences.jsonl 已随代码带上（本地构造、人工四类标签）。

页面1：bash scripts/run_judgecal.sh
页面2：bash scripts/monitor_judgecal.sh
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

from config import DATA_DIR, KIMI_API_KEY_ENV, LOG_DIR, OUTPUT_DIR  # noqa: E402

PY = Path(os.environ.get("ZHJG_ENV", "/home/nvme02/conda/zhjg_rl")) / "bin" / "python"
RUN_ID = os.environ.get("JUDGECAL_RUN_ID", "main")
OUT_DIR = Path(OUTPUT_DIR) / "judgecal" / RUN_ID
LOG_ROOT = Path(os.environ.get("JUDGECAL_LOG_DIR", str(Path(LOG_DIR) / "judgecal" / RUN_ID)))
EVENT_LOG = LOG_ROOT / "events.log"
STATE_FILE = LOG_ROOT / "state.json"
RAW_DIR = LOG_ROOT / "raw"

DATASET = Path(DATA_DIR) / "judgecal_sentences.jsonl"
ITEMS = OUT_DIR / "160_judgecal_items.jsonl"
JUDGES = OUT_DIR / "161_sentence_judges.jsonl"
REPORT_MD = OUT_DIR / "162_judgecal_report.md"
DECISION = OUT_DIR / "162_judgecal_decision.json"

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
        raise SystemExit(f"{KIMI_API_KEY_ENV} required（Kimi 逐句判分要用）")
    if not DATASET.exists():
        raise SystemExit(f"缺标定数据集: {DATASET}（应随代码带上）")
    emit(f"preflight OK | run_id={RUN_ID} | 数据={DATASET.name} | out={OUT_DIR}")


def main() -> None:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("usage: python -X utf8 run_judgecal.py")
        print("env: DASHSCOPE_API_KEY(必填) JUDGECAL_RUN_ID(默认main) JUDGECAL_KMAX(16) JUDGECAL_WORKERS(3)")
        return
    save_state(status="running", stage="preflight")
    try:
        preflight()
        run("s160_build_dataset", [
            str(PY), "-X", "utf8", "pipeline/step160_build_judgecal_dataset.py",
            "--in", str(DATASET), "--out", str(ITEMS)])
        if JUDGES.exists() and JUDGES.stat().st_size > 0:
            emit("RESUME 跳过 s161（161_sentence_judges.jsonl 已存在；要重判先删它）")
        else:
            run("s161_judge_sentences", [
                str(PY), "-X", "utf8", "pipeline/step161_judge_sentences.py",
                "--items", str(ITEMS), "--out", str(JUDGES)])
        run("s162_analyze", [
            str(PY), "-X", "utf8", "pipeline/step162_analyze_judgecal.py",
            "--judges", str(JUDGES), "--out_md", str(REPORT_MD), "--out_json", str(DECISION)])
        save_state(status="complete", stage="done", pid=None, summary=str(REPORT_MD))
        emit(f"RESULT JUDGECAL COMPLETE | report={REPORT_MD}")
    except BaseException as exc:
        save_state(status="failed", error=repr(exc))
        emit(f"JUDGECAL FAILED | {exc!r}")
        raise


if __name__ == "__main__":
    main()
