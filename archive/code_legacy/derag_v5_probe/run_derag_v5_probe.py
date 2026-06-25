"""一键 derag_v5 headroom 探针（不训练、只采样+判定，回答"该不该烧卡"）。

流程：
  serve RFT-merged → step150 找病题 + step151 RFT自采16遍 → 停
  serve 原始V1     → step152 V1答8遍建答案库 → 停
  (CPU) step153 评自采样自救率X → (Kimi) step154 改写成功率Y → step159 汇总+生死判断

页面1：bash scripts/run_derag_v5_probe.sh      （跑）
页面2：bash scripts/monitor_derag_v5_probe.sh  （看运行记录）

复用现有链路：vllm 由 scripts/serve_v1_vllm.sh(vllm_env) 起、zhjg_rl 走 HTTP；
探测器/事实抽取复用 reward_v3；改写 prompt 复用 step06；Kimi 走 kimi_client。
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
    """定位含 config.py 的代码根，兼容"解压进 code/"或"解压到 code/ 上一级"两种情况。"""
    here = Path(__file__).resolve().parent
    for cand in (here, here / "code", here.parent / "code"):
        if (cand / "config.py").exists():
            return cand
    return here


ROOT = _find_code_root()
sys.path.insert(0, str(ROOT))

from config import KIMI_API_KEY_ENV, LOG_DIR, OUTPUT_DIR, V1_DIR, SFT_TRAIN, SFT_EVAL  # noqa: E402
from pipeline import vllm_client  # noqa: E402

PY = Path(os.environ.get("ZHJG_ENV", "/home/nvme02/conda/zhjg_rl")) / "bin" / "python"
RUN_ID = os.environ.get("V5_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
OUT_DIR = Path(OUTPUT_DIR) / "derag_v5_probe" / RUN_ID
LOG_ROOT = Path(os.environ.get("V5_LOG_DIR", str(Path(LOG_DIR) / "derag_v5_probe" / RUN_ID)))
EVENT_LOG = LOG_ROOT / "events.log"
STATE_FILE = LOG_ROOT / "state.json"
RAW_DIR = LOG_ROOT / "raw"

RFT_MERGED_DIR = os.environ.get("V5_RFT_MERGED_DIR", "/home/nvme01/zhjg/models/v1-32b-corrected-v1-rft-merged")
V1_MODEL_DIR = os.environ.get("V5_V1_DIR", V1_DIR)
VLLM_GPUS = os.environ.get("V5_VLLM_GPUS", "0,1")
VLLM_PIDF = Path(os.environ.get("ZHJG_LOG_DIR", LOG_DIR)) / "vllm.pid"

SIGNAL_WORDS = ("RESULT", "PROGRESS", "ERROR", "Traceback", "NO_GO", "NO-GO", "GO", "PASS", "FAIL", "verdict")
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
    return {**os.environ, "PYTHONUNBUFFERED": "1",
            "ZHJG_CONSOLE_LOG_LEVEL": os.environ.get("ZHJG_CONSOLE_LOG_LEVEL", "WARNING")}


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
        proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=f, stderr=subprocess.STDOUT, text=True)
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


_SERVED = {"dir": None}   # 当前在服务的模型目录（停服务时按它精准 pkill）
GPU_FREE_MIB = 4000       # 低于此视为"显存已释放"（空闲卡通常 <100MiB）


def gpu_max_used_mib() -> int:
    """本探针用的那几张卡(VLLM_GPUS)里，已用显存的最大值(MiB)。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=30)
    except Exception:
        return 0
    want = {g.strip() for g in VLLM_GPUS.split(",")}
    mx = 0
    for ln in out.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 2 and parts[0] in want:
            try:
                mx = max(mx, int(parts[1]))
            except ValueError:
                pass
    return mx


def _our_gpu_uuids() -> set[str]:
    """VLLM_GPUS 那几张卡的 UUID（用于只杀本探针占用的卡上的进程，不碰其它卡的作业）。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=30)
    except Exception:
        return set()
    want = {g.strip() for g in VLLM_GPUS.split(",")}
    uu = set()
    for ln in out.strip().splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) == 2 and parts[0] in want:
            uu.add(parts[1])
    return uu


def kill_gpu_procs() -> list[str]:
    """按 nvidia-smi 上报的 PID 精准杀掉占用本探针 GPU 的进程（兜底 vLLM spawn worker 命令行不含 'vllm'、
    pkill -f 抓不到的情况）。返回被杀 PID 列表。"""
    uuids = _our_gpu_uuids()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True, timeout=30)
    except Exception:
        return []
    pids = []
    for ln in out.strip().splitlines():
        if not ln.strip():
            continue
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) == 2 and (not uuids or parts[1] in uuids):
            pids.append(parts[0])
    for pid in pids:
        subprocess.run(["bash", "-c", f"kill -9 {pid} 2>/dev/null || true"])
    return pids


def wait_gpu_free(max_wait: int = 150) -> None:
    """轮询等显存真正释放（起下一个模型前兜底，防上一个没停干净导致新模型抢不到显存）。"""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if gpu_max_used_mib() < GPU_FREE_MIB:
            return
        time.sleep(5)
    emit(f"WARN GPU {max_wait}s 未降到 {GPU_FREE_MIB}MiB 以下（当前 {gpu_max_used_mib()}MiB），仍尝试起服务")


def serve(model_dir: str, tag: str) -> None:
    wait_gpu_free(120)   # 起服务前确认显存干净（兜底上一个 stop 没停彻底）
    emit(f"SERVE {tag} <- {model_dir} (GPU={VLLM_GPUS})")
    env = {**os.environ, "V1_DIR": model_dir, "VLLM_GPUS": VLLM_GPUS,
           "ZHJG_LOG_DIR": os.environ.get("ZHJG_LOG_DIR", LOG_DIR)}
    rc = subprocess.run(["bash", "scripts/serve_v1_vllm.sh"], cwd=ROOT, env=env).returncode
    if rc != 0:
        raise SystemExit(f"serve {tag} 启动失败 rc={rc}")
    _SERVED["dir"] = model_dir
    vllm_client.wait_ready(max_wait=1800)
    emit(f"SERVE {tag} ready")


def stop_vllm() -> None:
    pid = VLLM_PIDF.read_text().strip() if VLLM_PIDF.exists() else ""
    if pid:
        subprocess.run(["bash", "-c", f"kill -TERM -{pid} 2>/dev/null || true"])
        emit(f"STOP  vLLM pgid={pid}（TERM，轮询等显存释放）")
    # 轮询等显存释放；30s 还不掉就升级 -9（pgid + 按模型路径精准 pkill，只杀本探针的进程）
    t0, escalated = time.time(), False
    while time.time() - t0 < 180:
        time.sleep(5)
        if gpu_max_used_mib() < GPU_FREE_MIB:
            break
        if not escalated and time.time() - t0 > 30:
            if pid:
                subprocess.run(["bash", "-c", f"kill -KILL -{pid} 2>/dev/null || true"])
            # 关键兜底：vLLM spawn worker 命令行不含 'vllm'/模型路径，pkill -f 抓不到 →
            # 直接按 nvidia-smi 上报的 PID 精准杀（只杀本探针 GPU 上的进程）
            killed = kill_gpu_procs()
            emit(f"STOP  TERM 未释放 → 升级 KILL -9（pgid={pid} + 按GPU占用PID精准杀={killed}）")
            escalated = True
    used = gpu_max_used_mib()
    emit(f"STOP  vLLM 显存现 {used}MiB " + ("（已释放）" if used < GPU_FREE_MIB else "（⚠ 仍占用）"))
    _SERVED["dir"] = None
    try:
        VLLM_PIDF.unlink()
    except OSError:
        pass


def preflight() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not PY.exists():
        raise SystemExit(f"missing python env: {PY}")
    if not os.environ.get(KIMI_API_KEY_ENV):
        raise SystemExit(f"{KIMI_API_KEY_ENV} required (Kimi 改写要用)")
    miss = [p for p in (RFT_MERGED_DIR, V1_MODEL_DIR) if not Path(p).exists()]
    if miss:
        raise SystemExit("缺模型权重:\n" + "\n".join(miss))
    if not Path(SFT_EVAL).exists() and not Path(SFT_TRAIN).exists():
        raise SystemExit(f"缺题集: {SFT_EVAL} / {SFT_TRAIN}")
    emit(f"preflight OK | run_id={RUN_ID} | RFT={RFT_MERGED_DIR} | V1={V1_MODEL_DIR} | out={OUT_DIR}")


def main() -> None:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("usage: python -X utf8 run_derag_v5_probe.py")
        print("env: DASHSCOPE_API_KEY(必填) V5_RUN_ID V5_RFT_MERGED_DIR V5_V1_DIR V5_VLLM_GPUS")
        print("     V5_TRAIN_CAP(默认1000,0=全2015) V5_RFT_K(16) V5_V1_N(8)")
        return
    P = OUT_DIR
    problems = P / "150_problems.jsonl"
    rft_samples = P / "151_rft_samples.jsonl"
    v1_support = P / "152_v1_support.jsonl"
    rft_head = P / "153_rft_headroom.jsonl"; rft_head_json = P / "153_rft_headroom.json"
    rw_head = P / "154_rewrite_headroom.jsonl"; rw_head_json = P / "154_rewrite_headroom.json"
    summ_md = P / "159_probe_summary.md"; summ_json = P / "159_probe_summary.json"
    def done(p: Path) -> bool:
        return p.exists() and p.stat().st_size > 0

    save_state(status="running", stage="preflight")
    try:
        preflight()
        # 窗口1：RFT-merged（150/151 都已存在则整窗口跳过，不重起 RFT、不重跑 20min）
        if done(problems) and done(rft_samples):
            emit("RESUME 跳过 RFT 窗口（150_problems + 151_rft_samples 已存在）")
        else:
            serve(RFT_MERGED_DIR, "RFT-merged")
            if done(problems):
                emit("RESUME 跳过 s150（产物已存在）")
            else:
                run("s150_select_problems", [str(PY), "-X", "utf8", "pipeline/step150_select_problems.py",
                                             "--out", str(problems)])
            if done(rft_samples):
                emit("RESUME 跳过 s151（产物已存在）")
            else:
                run("s151_rft_selfsample", [str(PY), "-X", "utf8", "pipeline/step151_rft_selfsample.py",
                                            "--problems", str(problems), "--out", str(rft_samples)])
            stop_vllm()
        # 窗口2：原始 V1
        if done(v1_support):
            emit("RESUME 跳过 V1 窗口（152_v1_support 已存在）")
        else:
            serve(V1_MODEL_DIR, "V1-checkpoint")
            run("s152_v1_support", [str(PY), "-X", "utf8", "pipeline/step152_v1_support.py",
                                    "--problems", str(problems), "--out", str(v1_support)])
            stop_vllm()
        # CPU + Kimi（同样幂等跳过）
        if done(rft_head_json):
            emit("RESUME 跳过 s153")
        else:
            run("s153_score_selfsample", [str(PY), "-X", "utf8", "pipeline/step153_score_selfsample.py",
                                          "--samples", str(rft_samples), "--support", str(v1_support),
                                          "--out", str(rft_head), "--out_json", str(rft_head_json)])
        if done(rw_head_json):
            emit("RESUME 跳过 s154")
        else:
            run("s154_kimi_rewrite", [str(PY), "-X", "utf8", "pipeline/step154_kimi_rewrite_check.py",
                                      "--support", str(v1_support),
                                      "--out", str(rw_head), "--out_json", str(rw_head_json)])
        run("s159_report", [str(PY), "-X", "utf8", "pipeline/step159_probe_report.py",
                            "--rft_headroom", str(rft_head_json), "--rewrite_headroom", str(rw_head_json),
                            "--out_md", str(summ_md), "--out_json", str(summ_json), "--run_id", RUN_ID])
        save_state(status="complete", stage="done", pid=None, summary=str(summ_md))
        emit(f"RESULT PROBE COMPLETE | summary={summ_md}")
    except BaseException as exc:
        stop_vllm()
        save_state(status="failed", error=repr(exc))
        emit(f"PROBE FAILED | {exc!r}")
        raise


if __name__ == "__main__":
    main()
