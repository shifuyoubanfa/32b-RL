"""32B（公司 V1）推理人性化 RL —— 全链一键编排。

从数据构建一路跑到 GRPO，含 vLLM 生命周期管理 + 各阶段 adapter 热加载评测。
所有 INFO 日志同时进 logs/pipeline.log（实时监督）。每阶段产出存在则跳过（断点续跑）。

阶段：
  data   阶段0  本地 V1 重产 think/answer + 切 train/eval
  base   阶段1  V1 基线评测（定 humanness0/verbatim_base）
  cs     阶段2  Kimi 改写全2014 + 打分 + 筛冷启动集 + swift SFT + 评测
  probe  阶段3  PMI 尺子探针 AUC 选优（AUC≥0.7 才放行）
  rft    阶段4  冷启动模型全池 rollout + 选样(开PMI) + swift 续训 + 评测
  dpo    阶段5  CS_RFT 新 rollout 构对 + swift DPO + 评测
  grpo   阶段6  swift GRPO(内置 vLLM+权重同步) + 评测

用法：
  python run.py                  # 全链
  python run.py --from rft       # 从某阶段开始
  python run.py --only base      # 只跑某阶段
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    V1_OUTPUTS, SFT_TRAIN, SFT_EVAL, SEEDS_RAW, SEEDS_SCORED, COLDSTART_TRAIN, COLDSTART_EVAL,
    PROBE_REPORT, CS_ROLLOUT, CS_RFT_TRAIN, DPO_ROLLOUT, DPO_PAIRS, OUTPUT_DIR, LOG_DIR,
    COLDSTART_LORA_DIR, CS_RFT_LORA_DIR, DPO_LORA_DIR, GRPO_LORA_DIR,
    GRPO_FROM_RFT_LORA_DIR, GRPO_FROM_DPO_LORA_DIR,
    COLDSTART_LR, COLDSTART_EPOCHS, RFT_LR, RFT_EPOCHS, stage_eval_paths,
    resolve_adapter, has_adapter, PMI_ENABLED,
)
from pipeline.logger import get_logger
from pipeline import vllm_client

log = get_logger("run")
PY = [sys.executable, "-X", "utf8"]
SWIFT_DIR = ROOT / "swift"
SCRIPTS = ROOT / "scripts"
GRPO_DATA = os.path.join(OUTPUT_DIR, "70_grpo_data.jsonl")
TRAIN_GPUS = os.environ.get("TRAIN_GPUS", "2,3,4,5,6,7")
# PMI 尺子(transformers device_map=auto)只用训练卡，避开 GPU 0,1 上的 vLLM（否则抢卡 OOM）
PMI_ENV = {"CUDA_VISIBLE_DEVICES": TRAIN_GPUS}


# ---------------- 通用执行 ----------------
def _run(cmd, desc, optional=False, env=None):
    log.info("===== START %s :: %s", desc, " ".join(str(c) for c in cmd))
    t0 = time.time()
    full_env = {**os.environ, **(env or {})}
    rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=full_env).returncode
    dt = time.time() - t0
    if rc != 0:
        if optional:
            log.warning("===== WARN  %s exit=%d (%.0fs) [optional]", desc, rc, dt)
            return
        log.error("===== FAIL  %s exit=%d (%.0fs)", desc, rc, dt)
        raise SystemExit(rc)
    log.info("===== END   %s (%.0fs)", desc, dt)


def py(script, *args, desc="", optional=False, env=None):
    _run(PY + [f"pipeline/{script}", *args], desc or script, optional, env=env)


def sh(script, *args, desc="", env=None):
    _run(["bash", str(script), *args], desc or Path(script).name, env=env)


def exists(*paths) -> bool:
    return all(Path(p).exists() and Path(p).stat().st_size > 0 for p in paths)


# adapter 路径解析(resolve_adapter)、是否已训(has_adapter) 统一在 config 实现，全链共用，见 config.py。


# ---------------- vLLM 生命周期 ----------------
_vllm_cur = {"lora": ""}   # 记当前 vLLM 服务里预声明的 adapter（"名字=路径"），用于判断是否需重启


def serve_vllm(lora_name=None, lora_root=None):
    """起 V1 vLLM。给 (lora_name, lora_root) 则起服务时用 --lora-modules 预声明该 adapter
    （vLLM 0.20.1 运行时加载端点 404，改预声明）；所需 adapter 与当前不一致时自动重启。"""
    want = f"{lora_name}={resolve_adapter(lora_root)}" if (lora_name and lora_root) else ""
    if vllm_client.health() and _vllm_cur["lora"] == want:
        log.info("vLLM 已在运行且 adapter 一致(%s)，复用", lora_name or "纯V1")
        return
    stop_vllm()
    sh(SCRIPTS / "serve_v1_vllm.sh",
       desc="起 V1 vLLM" + (f"(+adapter:{lora_name})" if want else ""),
       env=({"VLLM_LORA": want} if want else None))
    vllm_client.wait_ready()
    _vllm_cur["lora"] = want


def stop_vllm():
    _vllm_cur["lora"] = ""   # 停了就清记录，下次 serve 必重启
    pidf = Path(LOG_DIR) / "vllm.pid"
    if pidf.exists():
        pid = pidf.read_text().strip()
        log.info("停止 vLLM（按进程组，回收 TP worker/EngineCore 子进程）PID/PGID=%s", pid)
        subprocess.run(["bash", "-c",
                        f"kill -TERM -{pid} 2>/dev/null; sleep 8; kill -KILL -{pid} 2>/dev/null; "
                        f"pkill -KILL -f 'vllm serve' 2>/dev/null || true"])
        pidf.unlink(missing_ok=True)
    # poll 直到端口/显存真正释放（最多 ~60s），而非固定 sleep
    for _ in range(12):
        if not vllm_client.health():
            log.info("vLLM 已停止")
            return
        time.sleep(5)
    log.warning("vLLM 端口仍可达，可能未完全释放（后续起服务可能复用旧实例，留意）")


def eval_stage(model_name, tag, lora_path=None):
    """某模型在 224 验收集上评测：起带该 adapter 的 vLLM → infer → judge → report。
    lora_path=None 即纯 V1 基线（model_name='v1'）；否则起服务时预声明 adapter，按 model_name 请求。"""
    infer, judge, report = stage_eval_paths(tag)
    if exists(report):
        log.info("跳过评测 %s（已存在 %s）", tag, report)
        return
    serve_vllm(model_name if lora_path else None, lora_path)
    baseline = ["--baseline"] if tag == "10_base" else []
    py("step03_eval_infer.py", "--model", model_name, "--out", infer, desc=f"eval推理:{tag}")
    py("step04_judge.py", "--in", infer, "--out", judge, desc=f"Kimi判分:{tag}")
    py("step05_report.py", "--in", judge, "--out", report, "--tag", tag, *baseline, desc=f"报告:{tag}")


# ---------------- 各阶段 ----------------
def stage_data():
    if not exists(V1_OUTPUTS):
        serve_vllm(); py("step01_build_v1_data.py", desc="阶段0:本地V1重产")
    if not exists(SFT_TRAIN, SFT_EVAL):
        py("step02_split_sft.py", desc="阶段0:切train/eval")


def stage_base():
    eval_stage("v1", "10_base")   # eval_stage 内部会起纯 V1 vLLM


def stage_cs():
    if not exists(SEEDS_RAW):
        py("step06_rewrite_seeds.py", desc="阶段2:Kimi改写全2014")
    if not exists(SEEDS_SCORED):
        py("step07_score_seeds.py", desc="阶段2:Kimi打分")
    if not exists(COLDSTART_TRAIN, COLDSTART_EVAL):
        py("step08_build_coldstart.py", desc="阶段2:构冷启动集")
    if not has_adapter(COLDSTART_LORA_DIR):
        sh(SWIFT_DIR / "sft.sh", COLDSTART_TRAIN, COLDSTART_EVAL, COLDSTART_LORA_DIR,
           str(COLDSTART_LR), str(COLDSTART_EPOCHS), desc="阶段2:冷启动SFT")
    py("plot_loss.py", "--dir", COLDSTART_LORA_DIR, "--tag", "cs", desc="画 cs 损失曲线", optional=True)
    eval_stage("coldstart", "20_coldstart", lora_path=COLDSTART_LORA_DIR)


def stage_probe():
    if not PMI_ENABLED:
        log.info("PMI 关闭（探针实测 grounding 下反相关 AUC<0.5、表面项 AUC≈0.99）→ 跳过探针；"
                 "RFT/DPO/GRPO 的 humanness 奖励统一用表面项（检索腔关键词+照抄率），同 14B。")
        return
    if not exists(PROBE_REPORT):
        stop_vllm()  # 探针要加载尺子模型上训练卡，避免与 vLLM 抢；算完再起
        py("step09_probe_pmi.py", desc="阶段3:PMI尺子探针选优")


def stage_rft():
    if not exists(CS_ROLLOUT):
        serve_vllm("coldstart", COLDSTART_LORA_DIR)   # 起带冷启动 adapter 的 vLLM 做 rollout
        py("step10_rollout.py", "--model", "coldstart", "--out", CS_ROLLOUT, desc="阶段4:冷启动全池rollout")
    if not exists(CS_RFT_TRAIN):
        # step11 的 PMI 尺子用训练卡(2-7)，避开 0,1 上的 vLLM
        py("step11_select_rft.py", "--rollout", CS_ROLLOUT, "--out", CS_RFT_TRAIN,
           "--stage_adapter", resolve_adapter(COLDSTART_LORA_DIR), desc="阶段4:RFT选样(开PMI)", env=PMI_ENV)
    if not has_adapter(CS_RFT_LORA_DIR):
        sh(SWIFT_DIR / "sft.sh", CS_RFT_TRAIN, COLDSTART_EVAL, CS_RFT_LORA_DIR,
           str(RFT_LR), str(RFT_EPOCHS), resolve_adapter(COLDSTART_LORA_DIR), desc="阶段4:RFT续训")
    py("plot_loss.py", "--dir", CS_RFT_LORA_DIR, "--tag", "rft", desc="画 rft 损失曲线", optional=True)
    eval_stage("cs_rft", "50_cs_rft", lora_path=CS_RFT_LORA_DIR)


def stage_dpo():
    if not exists(DPO_ROLLOUT):
        serve_vllm("cs_rft", CS_RFT_LORA_DIR)
        py("step10_rollout.py", "--model", "cs_rft", "--out", DPO_ROLLOUT, desc="阶段5:CS_RFT新rollout")
    if not exists(DPO_PAIRS):
        py("step12_build_dpo_pairs.py", "--rollout", DPO_ROLLOUT, "--out", DPO_PAIRS,
           "--stage_adapter", resolve_adapter(CS_RFT_LORA_DIR), desc="阶段5:构DPO对", env=PMI_ENV)
    if not has_adapter(DPO_LORA_DIR):
        sh(SWIFT_DIR / "dpo.sh", DPO_PAIRS, resolve_adapter(CS_RFT_LORA_DIR), DPO_LORA_DIR, desc="阶段5:swift DPO")
    py("plot_loss.py", "--dir", DPO_LORA_DIR, "--tag", "dpo", desc="画 dpo 损失曲线", optional=True)
    eval_stage("dpo", "60_dpo", lora_path=DPO_LORA_DIR)


def stage_grpo():
    if not exists(GRPO_DATA):
        py("step13_build_grpo_data.py", "--out", GRPO_DATA, desc="阶段6:构GRPO数据")

    # 默认保持原行为：从 RFT/DPO 两个基础各跑一遍。
    # 调参重跑时可用 GRPO_BASES=rft + GRPO_RUN_TAG=<tag>：
    # - 只跑指定基础，避免无意重跑 DPO；
    # - 新产物写入带 tag 的独立目录/报告，不覆盖已跑通的首轮结果。
    all_bases = {
        "rft": (CS_RFT_LORA_DIR, GRPO_FROM_RFT_LORA_DIR),
        "dpo": (DPO_LORA_DIR, GRPO_FROM_DPO_LORA_DIR),
    }
    base_names = [x.strip() for x in os.environ.get("GRPO_BASES", "rft,dpo").split(",") if x.strip()]
    unknown = [x for x in base_names if x not in all_bases]
    if unknown:
        raise SystemExit(f"GRPO_BASES 含未知基础 {unknown}；可用: {list(all_bases)}")
    run_tag = os.environ.get("GRPO_RUN_TAG", "").strip()
    if run_tag and not re.fullmatch(r"[A-Za-z0-9_]+", run_tag):
        raise SystemExit("GRPO_RUN_TAG 只允许字母、数字、下划线（用于目录、vLLM model 名和报告名）")
    log.info("GRPO 运行选择：bases=%s run_tag=%s", base_names, run_tag or "<default>")

    succeeded = []
    for tag in base_names:
        base_dir, default_out_dir = all_bases[tag]
        out_dir = f"{default_out_dir}-{run_tag}" if run_tag else default_out_dir
        model_name = f"grpo_{tag}_{run_tag}" if run_tag else f"grpo_{tag}"
        report_tag = f"70_grpo_{tag}_{run_tag}" if run_tag else f"70_grpo_{tag}"
        try:
            if not has_adapter(out_dir):
                stop_vllm()  # GRPO 用全 8 卡（colocate vLLM 或 use_vllm=false），先停静态 vLLM 服务
                sh(SWIFT_DIR / "grpo.sh", GRPO_DATA, resolve_adapter(base_dir), out_dir,
                   desc=f"阶段6:GRPO(from {tag}, run_tag={run_tag or 'default'})")
            py("plot_loss.py", "--dir", out_dir, "--tag", model_name,
               desc=f"画 {model_name} 曲线(loss/reward/kl)", optional=True)
            eval_stage(model_name, report_tag, lora_path=out_dir)
            succeeded.append(tag)
        except SystemExit as e:
            # 一个基础崩了不拖累另一个（.done 可断点续跑）——无人值守时至少另一基础有机会出结果
            log.error("GRPO base=%s run_tag=%s 失败(exit=%s)，跳过该基础、继续下一个",
                      tag, run_tag or "default", getattr(e, "code", e))
            continue
    if not succeeded:
        raise SystemExit(f"所选 GRPO 基础全部失败：bases={base_names} run_tag={run_tag or 'default'}")


PHASES = [("data", stage_data), ("base", stage_base), ("cs", stage_cs), ("probe", stage_probe),
          ("rft", stage_rft), ("dpo", stage_dpo), ("grpo", stage_grpo)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--to", default=None)
    args = ap.parse_args()
    ids = [p for p, _ in PHASES]
    for x in (args.frm, args.only, args.to):
        if x and x not in ids:
            raise SystemExit(f"未知阶段 {x}，可用: {ids}")

    started = args.frm is None
    t0 = time.time()
    log.info("########## PIPELINE START from=%s only=%s to=%s ##########", args.frm, args.only, args.to)
    try:
        for pid, fn in PHASES:
            if args.only and pid != args.only:
                continue
            if not started:
                started = pid == args.frm
                if not started:
                    continue
            log.info("########## 阶段 [%s] ##########", pid)
            fn()
            if args.to and pid == args.to:
                break
    finally:
        stop_vllm()
    log.info("########## PIPELINE DONE elapsed=%.0fs ##########", time.time() - t0)


if __name__ == "__main__":
    main()
