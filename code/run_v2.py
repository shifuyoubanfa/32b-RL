"""V2 一键编排器：V1 重开 → 冷启动 SFT(2σ/3σ) → RFT(2σ/3σ) → DPO(2σ/3σ)，跑到 DPO 结束。

8 叶 / 14-LoRA 二叉树（节点 = 2 SFT + 4 RFT + 8 DPO）：
  ROOT_BASE(=V1) ──SFT 2σ/3σ──► sft_merged ──RFT 2σ/3σ──► rft_merged ──DPO 2σ/3σ──► dpo_merged(叶)
每节点：建数据 → swift 训 LoRA → 合并 → (下阶段 serve+rollout) → 三分评测；靠 .done + 完整性校验断点续跑。

骨架照搬 run_merged_dpo_grpo.py（已跑通多次）：emit/save_state/run/serve_model/stop_static_vllm/merge/
adapter_complete/merged_model_complete/report_complete/move_interrupted。GPU 互斥：训练/合并前必停 vLLM。
贯穿：answer-lock（answer 永远 V1 原版、只训 think）、选样 k=16/评测 k=2、复用 judgecal 打分提示词。
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
    V1_DIR, LOG_DIR, COLDSTART_LR, COLDSTART_EPOCHS, RFT_LR, RFT_EPOCHS, resolve_adapter,
)
from pipeline import vllm_client  # noqa: E402
from pipeline.v2_common import (  # noqa: E402
    v2_lora_dir, v2_merged_dir, v2_eval_paths, v2_summary_path, V2_OUTPUT_DIR)
from pipeline.v2_paths import (  # noqa: E402
    V2_TRAIN, V2_EVAL, V2_PROBLEMS, V2_PROBLEMS_TRAIN, V2_V1_SUPPORT,
    V2_RFT_SELFSAMPLE_K, V2_DPO_ROLLOUT_K, nonempty, read_jsonl,
    coldstart_train, coldstart_eval, rft_selfsample, rft_train, dpo_rollout, dpo_pairs,
)

SCRIPTS = ROOT / "scripts"
SWIFT = ROOT / "swift"
PY = Path(os.environ.get("ZHJG_ENV", "/home/nvme02/conda/zhjg_rl")) / "bin" / "python"
WORK_DIR = Path(os.environ.get("ZHJG_WORK_DIR", "/home/nvme01/zhjg"))
RAW_DIR = Path(LOG_DIR) / "v2"
STATE_FILE = RAW_DIR / "state.json"
EVENT_LOG = RAW_DIR / "events.log"

# 二叉树根 base：V2「一切以 V1 重开」→ 冷启动 SFT 的 base = V1 本体（可用 ZHJG_V2_ROOT_BASE 覆盖）。
ROOT_BASE = Path(os.environ.get("ZHJG_V2_ROOT_BASE", str(V1_DIR)))
SIGMAS = (2, 3)
EVAL_INTERMEDIATE = os.environ.get("V2_EVAL_INTERMEDIATE", "1") == "1"  # 也评 SFT/RFT 中间态（看分阶段曲线）
REPORT_MARKERS = ("Kimi干净分", "规则去检索腔通过率", "答案在池率")
# 剪枝：中间节点（SFT/RFT）评测【确证效果很差】就抛弃整条子树，省下其下游 RFT/DPO 的全部 token。
# 只在明确超噪声时剪（宁可不剪也别误杀好枝）；剪枝需要中间评测，故 PRUNE 开则强制评 SFT/RFT。
PRUNE = os.environ.get("V2_PRUNE", "1") == "1"
PRUNE_INPOOL_FLOOR = float(os.environ.get("V2_PRUNE_INPOOL_FLOOR", "0.85"))  # 答案在池率绝对地板，跌破=准确率塌方
PRUNE_ACC_DROP = float(os.environ.get("V2_PRUNE_ACC_DROP", "0.05"))          # 在池率比父掉超此=准确率显著退（>2σ噪声）

SIGNAL_RE = re.compile(
    r"Train:|global_step|max_steps|'loss'|'reward'|'kl'|eval_loss|Capturing CUDA graphs|"
    r"Application startup complete|\[merge\]|\[sft-on-model\]|\[dpo-v2\]|\[serve-model\]|"
    r"干净分|在池率|完成|complete|RESULT|PRUNE|ERROR|FAIL|Traceback",
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


def run(stage: str, cmd: list, *, optional: bool = False, env: dict | None = None) -> None:
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
        save_state(stage=stage, status="running", pid=proc.pid, raw_log=str(raw_log), command=cmd, started_at=now())
        last_line, last_heartbeat = "", 0.0
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


# ---------------------- 完整性校验 / 断点续跑 ----------------------

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


# 数据步用 sentinel 标记完成（与 adapter/merged 的 .done 范式一致；不能用 size>0，因合法空 σ 桶=0 字节会被
# 误判"未完成"反复重烧 Kimi，半成品又会被误判"完成"——见审查 B3/S1/S3）。
def marked(name: str) -> bool:
    return (Path(V2_OUTPUT_DIR) / f".{name}.done").exists()


def mark(name: str) -> None:
    p = Path(V2_OUTPUT_DIR) / f".{name}.done"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(now(), encoding="utf-8")


def report_complete(path: Path) -> bool:
    if not path.exists() or not path.stat().st_size:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(m in text for m in REPORT_MARKERS)


# ---------------------- vLLM 顺序起停 ----------------------

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


def serve_model(model_dir: Path, served_name: str = "v1") -> None:
    """起静态 vLLM 服务一个全量模型（V2 全程用全量 merged，不用 base+adapter）。
    数据构建步(step01/151/152/10)不传 --model，默认查 'v1' → 这些步把目标模型 serve 成 'v1'。"""
    stop_static_vllm()
    run(f"serve_{served_name}", ["bash", str(SCRIPTS / "serve_model_vllm.sh"), str(model_dir), served_name])
    save_state(stage=f"wait_{served_name}", status="running", pid=None,
               raw_log=str(Path(LOG_DIR) / "merged_chain_vllm.log"))
    emit(f"WAIT  vLLM {served_name} ({model_dir})")
    vllm_client.wait_ready(max_wait=1800)
    emit(f"READY vLLM {served_name}")


def merge(stage: str, base: Path, lora_dir: Path, output: Path) -> None:
    if merged_model_complete(output):
        emit(f"SKIP  {stage}; merged exists: {output}")
        return
    if output.exists():                       # merge 脚本拒绝已存在的输出，先挪走半成品
        move_interrupted(output)
    adapter = Path(resolve_adapter(str(lora_dir)))
    run(stage, ["bash", str(SCRIPTS / "merge_lora_model.sh"), str(base), str(adapter), str(output)])


# ---------------------- 训练 ----------------------

def train_sft(stage: str, train_file, val_file, base: Path, out: Path, lr, epochs) -> None:
    if adapter_complete(out):
        emit(f"SKIP  {stage}; trained: {out}")
        return
    move_interrupted(out)
    stop_static_vllm()
    run(stage, ["bash", str(SWIFT / "sft_on_model.sh"), str(train_file), str(val_file),
                str(base), str(out), str(lr), str(epochs)])


def train_dpo(stage: str, pairs_file, base: Path, out: Path) -> None:
    if adapter_complete(out):
        emit(f"SKIP  {stage}; trained: {out}")
        return
    move_interrupted(out)
    stop_static_vllm()
    run(stage, ["bash", str(SWIFT / "dpo_v2.sh"), str(pairs_file), str(base), str(out)])


def sft_node(stage: str, train_file, val_file, base: Path, lora: Path, merged: Path, lr, epochs):
    """训一个 SFT/RFT 节点并合并；空训练桶（σ 选不出样本，合法）→ 返回 None，调用方跳过其整条子树。"""
    if merged_model_complete(merged):
        return merged                       # 续跑：已合并直接用
    if not nonempty(train_file):
        emit(f"SKIP  {stage}: 空训练桶 {train_file}（σ 选不出样本，该节点及子树降级跳过）")
        return None
    if not nonempty(val_file):              # 共享自然腔 eval 空 → swift load_best 会崩，显式拦
        raise SystemExit(f"{stage}: 验证集为空 {val_file}（冷启动 eval 没产出，先查 step_v2_coldstart）")
    train_sft(stage, train_file, val_file, base, lora, lr, epochs)
    merge(f"merge_{stage}", base, lora, merged)
    return merged


def dpo_node(stage: str, pairs_file, base: Path, lora: Path, merged: Path):
    """训一个 DPO 叶并合并；空偏好对 → 返回 None，调用方跳过该叶。"""
    if merged_model_complete(merged):
        return merged
    if not nonempty(pairs_file):
        emit(f"SKIP  {stage}: 空偏好对 {pairs_file}（σ 选不出对，该叶降级跳过）")
        return None
    train_dpo(stage, pairs_file, base, lora)
    merge(f"merge_{stage}", base, lora, merged)
    return merged


# ---------------------- 评测（三分：Kimi k=2 / 规则 / 漂移）----------------------

def evaluate(tag: str, merged: Path) -> Path:
    infer, scores, report = map(Path, v2_eval_paths(tag))
    if report_complete(report) and v2_summary_path(tag).exists():  # summary 也在才算评过（剪枝要读它）
        emit(f"SKIP  eval {tag}; report+summary exist: {report}")
        return report
    model_name = re.sub(r"[^A-Za-z0-9_]", "_", tag)      # 非 'v1' → step03 自动用去检索腔 system
    serve_model(merged, model_name)
    run(f"{tag}_infer", [PY, "-X", "utf8", "pipeline/step03_eval_infer.py",
                         "--model", model_name, "--eval_file", str(V2_EVAL), "--out", str(infer)])
    stop_static_vllm()                                    # 打分纯 Kimi，放开 GPU
    run(f"{tag}_score", [PY, "-X", "utf8", "pipeline/step_v2_eval.py",
                         "--infer", str(infer), "--scores", str(scores), "--report", str(report),
                         "--summary", str(v2_summary_path(tag)), "--support", str(V2_V1_SUPPORT), "--tag", tag])
    return report


def evaluate_baseline(tag: str = "v2-baseline-v1") -> None:
    """V1 原始模型在 500 验证上的三分基线：剪枝时 SFT 节点的父参照，也是项目"三基线"。V1 用 RAG 腔。"""
    infer, scores, report = map(Path, v2_eval_paths(tag))
    if report_complete(report) and v2_summary_path(tag).exists():
        emit(f"SKIP  baseline eval; report+summary exist: {report}")
        return
    serve_model(Path(V1_DIR), "v1")                       # served-name 'v1' → step03 用 RAG 腔 SYSTEM_PROMPT
    run(f"{tag}_infer", [PY, "-X", "utf8", "pipeline/step03_eval_infer.py",
                         "--model", "v1", "--eval_file", str(V2_EVAL), "--out", str(infer)])
    stop_static_vllm()
    run(f"{tag}_score", [PY, "-X", "utf8", "pipeline/step_v2_eval.py",
                         "--infer", str(infer), "--scores", str(scores), "--report", str(report),
                         "--summary", str(v2_summary_path(tag)), "--support", str(V2_V1_SUPPORT), "--tag", tag])


def eval_summary(tag: str) -> dict | None:
    p = v2_summary_path(tag)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def node_too_bad(cur_tag: str, parent_tag: str, cross_prompt: bool = False) -> tuple:
    """中间节点（SFT/RFT）是否【确证效果很差】、可剪掉整条子树。只在明确超噪声时判 True，宁可不剪也别误杀好枝。

    cross_prompt=True（SFT vs V1 基线）：V1 是 RAG 腔自评、在池率贴顶，而健康 de-RAG SFT 本就会比它低几个点
        （grounding tradeoff，健康区间约 [0.85,0.95)），跨腔做差会误剪好枝——故【只用绝对地板 0.85】兜准确率塌方，
        不比父、不看干净分差。
    cross_prompt=False（RFT vs 本线 SFT）：父子同 de-RAG 腔、同 V1 池，差分同口径有效，三红线全用。
    """
    cur, par = eval_summary(cur_tag), eval_summary(parent_tag)
    if not cur or not par:
        return False, "缺评测摘要，不剪（信息不全宁可保留）"
    # 红线1 准确率塌方（绝对地板，与提示词分布无关，两种比较都用）
    if cur["in_pool_rate"] < PRUNE_INPOOL_FLOOR:
        return True, f"答案在池率 {cur['in_pool_rate']:.1%} < 地板 {PRUNE_INPOOL_FLOOR:.0%}（准确率塌方）"
    if cross_prompt:
        return False, ""   # 跨腔(RAG↔de-RAG)：acc/clean 差分不可比，只信绝对地板，过了就不剪
    # 同腔同池（RFT vs SFT）：差分红线有效。
    # 在池率=确定性规则(answer_in_v1_pool)+贪心推理 → 无打分方差，不是噪声门；
    # acc_drop 0.05 是【人定的容差】= 500 题里翻了 25 道才算真退（确定性、直接数题数）。
    acc_drop = par["in_pool_rate"] - cur["in_pool_rate"]
    if acc_drop > PRUNE_ACC_DROP:
        return True, (f"答案在池率 {cur['in_pool_rate']:.1%} 比父 {par['in_pool_rate']:.1%} 掉 {acc_drop:.1%}"
                      f" > 容差 {PRUNE_ACC_DROP:.0%}（同腔准确率显著退）")
    se = cur.get("se") or 0.05    # se 只给【有噪声的】干净分用
    clean_drop = par["clean_mean"] - cur["clean_mean"]
    thr = 3 * se * (2 ** 0.5)                  # 两均值之差的 3σ（SE_diff≈√2·SE），比单均值 3σ 更严、更不易误剪
    if clean_drop > thr and cur["rule_pass_rate"] <= par["rule_pass_rate"]:
        return True, (f"Kimi干净分 {cur['clean_mean']:.2f} 比父 {par['clean_mean']:.2f} 真降 {clean_drop:.2f}"
                      f"(>3σ_diff≈{thr:.2f}) 且规则通过率没升（拟人反退、这轮纯亏）")
    return False, ""


# ---------------------- 上游（共享，只跑一次）----------------------

def build_upstream() -> None:
    # V1 重产 think/answer（serve V1 as 'v1'，RAG 腔）→ 切 1739/500 → 建 V1 答案池。marker 只在整步成功后写。
    from config import V1_OUTPUTS
    if not marked("v1_build"):
        serve_model(Path(V1_DIR), "v1")                   # step01 自身按 query 增量续跑；被 kill 不写 marker→下次续跑补全
        run("v1_build", [PY, "-X", "utf8", "pipeline/step01_build_v1_data.py",
                         "--in", "data/00_data_teacher_outputs.jsonl", "--out", str(V1_OUTPUTS)])
        mark("v1_build")
    if not marked("v2_split"):
        run("v2_split", [PY, "-X", "utf8", "pipeline/step_v2_split.py"])
        mark("v2_split")
    if not marked("v1_support"):
        serve_model(Path(V1_DIR), "v1")                   # step152 用 SYSTEM_PROMPT、查 'v1'
        run("v1_support", [PY, "-X", "utf8", "pipeline/step152_v1_support.py",
                           "--problems", str(V2_PROBLEMS), "--out", str(V2_V1_SUPPORT), "--n", "8"])
        mark("v1_support")
    # 池覆盖审计：个别 qid 采样失败会让池不全却被 marked 完成 → 缺池题在答案门被全判漂、污染 RFT/eval/剪枝
    n_pool, n_prob = len(read_jsonl(V2_V1_SUPPORT)), len(read_jsonl(V2_PROBLEMS))
    if n_prob and n_pool < n_prob:
        emit(f"WARN V1 答案池覆盖 {n_pool}/{n_prob}（缺 {n_prob - n_pool} 题）——缺池题会在答案门被判漂、踢出训练/污染评测；"
             f"缺口大就删 {Path(V2_OUTPUT_DIR) / '.v1_support.done'} + {V2_V1_SUPPORT} 重建池")
    stop_static_vllm()


def build_coldstart_data() -> None:
    if marked("coldstart"):
        emit("SKIP  coldstart data done")
        return
    stop_static_vllm()                                    # 改写+打分纯 Kimi，不占 GPU
    run("coldstart_data", [PY, "-X", "utf8", "pipeline/step_v2_coldstart.py"])
    mark("coldstart")


def build_rft_data(sft_lin: str, sft_merged: Path) -> None:
    if marked(f"rft_data_{sft_lin}"):
        emit(f"SKIP  RFT data done: {sft_lin}")
        return
    if not nonempty(rft_selfsample(sft_lin)):
        serve_model(sft_merged, "v1")                     # step151 查 'v1'，COLDSTART 腔（显式）
        run(f"rft_selfsample_{sft_lin}", [PY, "-X", "utf8", "pipeline/step151_rft_selfsample.py",
                                          "--problems", str(V2_PROBLEMS_TRAIN),  # train-only 池（含 gold_answer，不泄漏 eval）
                                          "--out", str(rft_selfsample(sft_lin)), "--k", str(V2_RFT_SELFSAMPLE_K)])
    stop_static_vllm()
    run(f"rft_select_{sft_lin}", [PY, "-X", "utf8", "pipeline/step_v2_rft_select.py",
                                  "--selfsample", str(rft_selfsample(sft_lin)), "--lineage", sft_lin])
    mark(f"rft_data_{sft_lin}")


def build_dpo_data(rft_lin: str, rft_merged: Path) -> None:
    if marked(f"dpo_data_{rft_lin}"):
        emit(f"SKIP  DPO pairs done: {rft_lin}")
        return
    if not nonempty(dpo_rollout(rft_lin)):
        serve_model(rft_merged, "v1")                     # step10 传 --model v1；用 COLDSTART 腔
        run(f"dpo_rollout_{rft_lin}", [PY, "-X", "utf8", "pipeline/step10_rollout.py",
                                       "--model", "v1", "--pool", str(V2_TRAIN), "--out", str(dpo_rollout(rft_lin)),
                                       "--k", str(V2_DPO_ROLLOUT_K), "--max_tokens", "1536"])
    stop_static_vllm()
    run(f"dpo_pairs_{rft_lin}", [PY, "-X", "utf8", "pipeline/step_v2_dpo_pairs.py",
                                 "--rollout", str(dpo_rollout(rft_lin)), "--lineage", rft_lin])
    mark(f"dpo_data_{rft_lin}")


# ---------------------- preflight ----------------------

def preflight() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    required = [Path(V1_DIR) / "config.json", Path(ROOT_BASE) / "config.json", PY,
                SWIFT / "sft_on_model.sh", SWIFT / "dpo_v2.sh",
                SCRIPTS / "merge_lora_model.sh", SCRIPTS / "serve_model_vllm.sh"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("preflight missing:\n" + "\n".join(missing))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is required (Kimi 改写/打分)")
    free_gib = shutil.disk_usage(WORK_DIR).free / 1024**3
    # 镜像 run_derag_v4.preflight：环境变量 flat 下限、硬 raise（14 个 merged 各 ~65G，边跑可删父级）
    min_free = float(os.environ.get("V2_MIN_FREE_GIB", "200"))
    if free_gib < min_free:
        raise SystemExit(f"free disk too low: {free_gib:.1f} GiB < {min_free} GiB（放宽设 V2_MIN_FREE_GIB）")
    stop_static_vllm()
    active = subprocess.run(["bash", "-lc", "pgrep -af '[s]wift/cli/rlhf.py|[t]orch.distributed.run' || true"],
                            capture_output=True, text=True).stdout.strip()
    if active:
        raise SystemExit("other training processes are still active:\n" + active)
    emit(f"preflight OK | ROOT_BASE={ROOT_BASE} | free={free_gib:.1f}GiB(min {min_free}) | "
         f"RFT_K={V2_RFT_SELFSAMPLE_K} DPO_K={V2_DPO_ROLLOUT_K} eval_intermediate={EVAL_INTERMEDIATE} | "
         f"prune={PRUNE}(在池率地板{PRUNE_INPOOL_FLOOR:.0%}/掉幅容差{PRUNE_ACC_DROP:.0%})")


# ---------------------- 主流程：8 叶 / 14-LoRA 二叉树 ----------------------

def main() -> None:
    save_state(status="running", stage="preflight", completed=[])
    try:
        preflight()
        build_upstream()
        build_coldstart_data()

        base_tag = "v2-baseline-v1"                       # SFT 剪枝的父参照 + 项目三基线
        if PRUNE or EVAL_INTERMEDIATE:
            evaluate_baseline(base_tag)

        for s_sft in SIGMAS:                              # 2 个 SFT 节点
            sft_lin = f"{s_sft}s"
            sft_tag = f"v2-sft-{sft_lin}"
            sft_merged = sft_node(f"sft_{sft_lin}", coldstart_train(s_sft), coldstart_eval(), ROOT_BASE,
                                  v2_lora_dir("sft", s_sft), v2_merged_dir("sft", s_sft),
                                  COLDSTART_LR, COLDSTART_EPOCHS)
            if sft_merged is None:                        # 空 σ 桶 → 整条 SFT 子树降级跳过（3σ 选不出=预期，非异常）
                continue
            if PRUNE or EVAL_INTERMEDIATE:                # 剪枝需要评测，故 PRUNE 开则强制评中间态
                evaluate(sft_tag, sft_merged)
            if PRUNE:
                bad, why = node_too_bad(sft_tag, base_tag, cross_prompt=True)  # SFT vs V1：跨腔，只信地板
                if bad:
                    emit(f"PRUNE {sft_tag}: {why} → 抛弃整条 SFT 子树（省下其 RFT/DPO 全部 token）")
                    continue

            build_rft_data(sft_lin, sft_merged)           # 该 SFT 线自采样+选样（产 2σ/3σ 两桶）
            for s_rft in SIGMAS:                          # 每 SFT 下 2 个 RFT = 4
                rft_lin = f"{sft_lin}-{s_rft}s"
                rft_tag = f"v2-rft-{rft_lin}"
                rft_merged = sft_node(f"rft_{rft_lin}", rft_train(sft_lin, s_rft), coldstart_eval(), sft_merged,
                                      v2_lora_dir("rft", s_rft, sft_lin), v2_merged_dir("rft", s_rft, sft_lin),
                                      RFT_LR, RFT_EPOCHS)
                if rft_merged is None:                    # 空 RFT 桶 → 该 DPO 子树跳过
                    continue
                if PRUNE or EVAL_INTERMEDIATE:
                    evaluate(rft_tag, rft_merged)
                if PRUNE:
                    bad, why = node_too_bad(rft_tag, sft_tag, cross_prompt=False)  # RFT vs 本线 SFT：同腔，差分有效
                    if bad:
                        emit(f"PRUNE {rft_tag}: {why} → 抛弃其 DPO 子树（省下 2 个 DPO 叶）")
                        continue

                build_dpo_data(rft_lin, rft_merged)       # 该 RFT 线 rollout+构对（产 2σ/3σ 两桶）
                for s_dpo in SIGMAS:                      # 每 RFT 下 2 个 DPO = 8 叶
                    dpo_lin = f"{rft_lin}-{s_dpo}s"
                    dpo_merged = dpo_node(f"dpo_{dpo_lin}", dpo_pairs(rft_lin, s_dpo), rft_merged,
                                          v2_lora_dir("dpo", s_dpo, rft_lin), v2_merged_dir("dpo", s_dpo, rft_lin))
                    if dpo_merged is None:                # 空偏好对 → 跳过该叶
                        continue
                    evaluate(f"v2-{dpo_lin}", dpo_merged)  # 叶必评（三分）

        save_state(status="complete", stage="done", pid=None, raw_log=None)
        emit("V2 PIPELINE COMPLETE | 8 叶 / 14 LoRA 跑到 DPO 结束")
    except BaseException as exc:
        save_state(status="failed", error=repr(exc))
        emit(f"V2 PIPELINE FAILED | {exc!r}")
        raise
    finally:
        try:
            stop_static_vllm()
        except Exception as exc:
            emit(f"WARN cleanup static vLLM: {exc!r}")


if __name__ == "__main__":
    main()
