"""一键串联：数据抽取 -> teacher 调用 -> SFT 数据 -> 7B 下载 -> 训练 -> 合并 -> 推理 -> 评测 -> 报告。

每个 stage 都先检查产出文件是否已存在；存在则跳过，便于断点续跑。
所有 stage 与本脚本的 INFO 级日志都写到 logs/pipeline.log。

常用命令：
    python run.py                           # 跑全部，已完成的 stage 自动跳过
    python run.py --force                   # 强制重跑全部
    python run.py --only 02_call_company    # 只跑某一步
    python run.py --from 04_train_sft       # 从某一步开始
    python run.py --skip 04_train_sft 04c_merge_lora  # 跳过指定步骤
    python run.py --list                    # 列出所有 stage
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    USABLE_QUERIES, SUMMARY_QUERIES, ALL_QUERIES, COMPANY_OUTPUTS,
    SFT_TRAIN, SFT_EVAL, STUDENT_LOCAL_DIR, STUDENT_LORA_DIR, STUDENT_MERGED_DIR,
    STUDENT_OUTPUTS, JUDGE_RESULTS, REPORT_MD,
    # ---- RL ----
    RL_ROLLOUT, RFT_TRAIN, CALIB_REPORT,
    RFT_LORA_DIR, DPO_LORA_DIR, GRPO_LORA_DIR,
    RFT_STUDENT_OUTPUTS, RFT_JUDGE_RESULTS, RFT_REPORT,
    DPO_STUDENT_OUTPUTS, DPO_JUDGE_RESULTS, DPO_REPORT,
    GRPO_STUDENT_OUTPUTS, GRPO_JUDGE_RESULTS, GRPO_REPORT,
    RFT_LR, RFT_EPOCHS,
    # ---- 冷启动种子 ----
    SEEDS_RAW, SEEDS_SCORED,
    # ---- 冷启动 → RFT 整链 ----
    COLDSTART_TRAIN, COLDSTART_EVAL, COLDSTART_LORA_DIR, COLDSTART_LR, COLDSTART_EPOCHS,
    CS_ROLLOUT, CS_RFT_TRAIN, CS_RFT_LORA_DIR, CS_RFT_OUTPUTS, CS_RFT_JUDGE, CS_RFT_REPORT,
    # ---- 扩 query + DPO/GRPO ----
    RL_POOL_EXPANDED, DPO_ROLLOUT,
)
from pipeline.logger import get_logger


log = get_logger("run")


def _cfg(*parts):
    return [os.path.join(*parts)]


# 每条 Stage：stage_id, 脚本, 产出文件(用于跳过判断), 描述, optional, manual, args
# - optional=True：失败只 warning 不 abort
# - manual=True：纯 bulk（无 --from/--only）运行时跳过；属 RL 阶段，需显式 --from/--only 才跑
class Stage:
    def __init__(self, sid, script, outputs, desc, optional=False, manual=False, args=None):
        self.sid, self.script, self.outputs, self.desc = sid, script, outputs, desc
        self.optional, self.manual, self.args = optional, manual, (args or [])


STAGES = [
    # ---------------- SFT 主线 ----------------
    Stage("01_extract_usable",   "pipeline/step01_extract_usable.py",   [USABLE_QUERIES],  "A模型纯样本 -> '可用' query"),
    Stage("01b_extract_summary", "pipeline/step01b_extract_summary.py", [SUMMARY_QUERIES], "一阶段模型输出 -> '模型总结问题'"),
    Stage("01c_merge_queries",   "pipeline/step01c_merge_queries.py",   [ALL_QUERIES],     "合并去重 -> query 池"),
    Stage("02_call_company",     "pipeline/step02_call_company_model.py", [COMPANY_OUTPUTS], "调 teacher 拿 think+answer"),
    Stage("03_build_sft",        "pipeline/step03_build_sft_dataset.py", [SFT_TRAIN, SFT_EVAL], "切分 train/eval"),
    Stage("04_download_student", "pipeline/step04_download_student.py", _cfg(STUDENT_LOCAL_DIR, "config.json"), "魔搭下载 14B base"),
    Stage("04_train_sft",        "pipeline/step04_train_sft.py",        _cfg(STUDENT_LORA_DIR, "adapter_config.json"), "LoRA SFT 蒸馏"),
    Stage("04c_merge_lora",      "pipeline/step04c_merge_lora.py",      _cfg(STUDENT_MERGED_DIR, "config.json"),
          "合并 LoRA（可选，OOM 会自动回退 base+LoRA）", optional=True),
    Stage("05_infer_student",    "pipeline/step05_infer_student.py",    [STUDENT_OUTPUTS], "eval 推理"),
    Stage("06_judge",            "pipeline/step06_judge_with_claude.py", [JUDGE_RESULTS],  "Kimi 评测"),
    Stage("07_report",           "pipeline/step07_report.py",           [REPORT_MD],      "SFT 报告"),

    # ---------------- 冷启动种子（manual：rewrite teacher think → 自然推导）----------------
    Stage("11_rewrite_seeds", "pipeline/step11_rewrite_seeds.py", [SEEDS_RAW],
          "种子①：Kimi 把 teacher think 改写成自然推导(一份两用)", manual=True),
    Stage("12_score_seeds",   "pipeline/step12_score_seeds.py", [SEEDS_SCORED],
          "种子②：Kimi 给自然 think 打 humanness(做'该高分'对照)", manual=True),

    # ---------------- 冷启动 → RFT 整链（manual）----------------
    Stage("14_build_coldstart", "pipeline/step14_build_coldstart.py", [COLDSTART_TRAIN, COLDSTART_EVAL],
          "冷启动数据：从高质量自然种子筛 SFT 集 + 切自然腔留出 eval", manual=True),
    Stage("cs_train",         "pipeline/step04_train_sft.py", _cfg(COLDSTART_LORA_DIR, "adapter_config.json"),
          "冷启动 SFT：让 14B 学会自然推理(自然腔 eval 早停留最优)", manual=True,
          args=["--train_file", COLDSTART_TRAIN, "--eval_file", COLDSTART_EVAL,
                "--output_dir", COLDSTART_LORA_DIR, "--lr", str(COLDSTART_LR), "--epochs", str(COLDSTART_EPOCHS)]),
    Stage("cs_rollout",       "pipeline/step08_rl_rollout.py", [CS_ROLLOUT],
          "冷启动模型自生成：每 query 采 K 个", manual=True,
          args=["--lora_dir", COLDSTART_LORA_DIR, "--out", CS_ROLLOUT]),
    Stage("cs_rft_build",     "pipeline/step08b_build_rft.py", [CS_RFT_TRAIN],
          "新奖励选样构 RFT 集(默认无PMI；加 --with-pmi 需手动)", manual=True,
          args=["--rollout", CS_ROLLOUT, "--out", CS_RFT_TRAIN]),
    Stage("cs_rft_train",     "pipeline/step04_train_sft.py", _cfg(CS_RFT_LORA_DIR, "adapter_config.json"),
          "RFT：在冷启动 adapter 上续训选出的好样本(自然腔 eval 早停留最优)", manual=True,
          args=["--train_file", CS_RFT_TRAIN, "--eval_file", COLDSTART_EVAL, "--init_adapter", COLDSTART_LORA_DIR,
                "--output_dir", CS_RFT_LORA_DIR, "--lr", str(RFT_LR), "--epochs", str(RFT_EPOCHS)]),
    Stage("cs_rft_infer",     "pipeline/step05_infer_student.py", [CS_RFT_OUTPUTS],
          "RFT 模型 eval 推理", manual=True, args=["--lora_dir", CS_RFT_LORA_DIR, "--out", CS_RFT_OUTPUTS]),
    Stage("cs_rft_judge",     "pipeline/step06_judge_with_claude.py", [CS_RFT_JUDGE],
          "RFT Kimi 评测", manual=True, args=["--in", CS_RFT_OUTPUTS, "--out", CS_RFT_JUDGE]),
    Stage("cs_rft_report",    "pipeline/step07_report.py", [CS_RFT_REPORT],
          "RFT 报告(对比 SFT 基线看 humanness 涨没涨)", manual=True, args=["--in", CS_RFT_JUDGE, "--out", CS_RFT_REPORT]),

    # ---------------- RL 阶段（manual：需 --from/--only 显式进入）----------------
    Stage("08c_calibrate",   "pipeline/step08c_calibrate.py", [CALIB_REPORT],
          "阶段0：用已有 05/06 校准本地打分器(零 Kimi 调用)", optional=True, manual=True),
    Stage("08_rl_rollout",   "pipeline/step08_rl_rollout.py", [RL_ROLLOUT],
          "阶段1a：对 SFT_TRAIN 采样K个并本地打分", manual=True),
    Stage("08b_build_rft",   "pipeline/step08b_build_rft.py", [RFT_TRAIN],
          "阶段1b：挑最佳样本构 RFT 数据", manual=True),
    Stage("08_rft_train",    "pipeline/step04_train_sft.py",  _cfg(RFT_LORA_DIR, "adapter_config.json"),
          "阶段1c：RFT 自蒸馏再训", manual=True,
          args=["--train_file", RFT_TRAIN, "--eval_file", SFT_EVAL,
                "--output_dir", RFT_LORA_DIR, "--lr", str(RFT_LR), "--epochs", str(RFT_EPOCHS)]),
    Stage("08d_rft_infer",   "pipeline/step05_infer_student.py", [RFT_STUDENT_OUTPUTS],
          "RFT 模型 eval 推理", manual=True, args=["--lora_dir", RFT_LORA_DIR, "--out", RFT_STUDENT_OUTPUTS]),
    Stage("08e_rft_judge",   "pipeline/step06_judge_with_claude.py", [RFT_JUDGE_RESULTS],
          "RFT Kimi 评测", manual=True, args=["--in", RFT_STUDENT_OUTPUTS, "--out", RFT_JUDGE_RESULTS]),
    Stage("08f_rft_report",  "pipeline/step07_report.py", [RFT_REPORT],
          "RFT 报告", manual=True, args=["--in", RFT_JUDGE_RESULTS, "--out", RFT_REPORT]),

    # ---------------- 扩 query → DPO → GRPO（manual）----------------
    Stage("15_new_queries",  "pipeline/step15_build_new_queries.py", [RL_POOL_EXPANDED],
          "扩 query：摄取生产新问题(去重验收集)+并入老池 → 扩充 RL 池(防过拟合/证泛化)", manual=True),
    Stage("20_dpo_rollout",  "pipeline/step08_rl_rollout.py", [DPO_ROLLOUT],
          "DPO 前置：RFT 模型在扩充池上自采(供构偏好对)", manual=True,
          args=["--lora_dir", CS_RFT_LORA_DIR, "--pool", RL_POOL_EXPANDED, "--out", DPO_ROLLOUT]),
    Stage("09_dpo_train",    "pipeline/step09_dpo.py", _cfg(DPO_LORA_DIR, "adapter_config.json"),
          "阶段2：DPO 偏好对齐(从 RFT adapter 续，扩充池 rollout 构对)", manual=True,
          args=["--rollout", DPO_ROLLOUT, "--init_lora", CS_RFT_LORA_DIR, "--out_dir", DPO_LORA_DIR]),
    Stage("09d_dpo_infer",   "pipeline/step05_infer_student.py", [DPO_STUDENT_OUTPUTS],
          "DPO 模型 eval 推理", manual=True, args=["--lora_dir", DPO_LORA_DIR, "--out", DPO_STUDENT_OUTPUTS]),
    Stage("09e_dpo_judge",   "pipeline/step06_judge_with_claude.py", [DPO_JUDGE_RESULTS],
          "DPO Kimi 评测", manual=True, args=["--in", DPO_STUDENT_OUTPUTS, "--out", DPO_JUDGE_RESULTS]),
    Stage("09f_dpo_report",  "pipeline/step07_report.py", [DPO_REPORT],
          "DPO 报告(对比 RFT 看 humanness/准确率)", manual=True, args=["--in", DPO_JUDGE_RESULTS, "--out", DPO_REPORT]),

    Stage("10_grpo_train",   "pipeline/step10_grpo.py", _cfg(GRPO_LORA_DIR, "adapter_config.json"),
          "阶段3：GRPO 在线强化(非对称 KL，从 DPO 续，扩充池在线采样)", manual=True,
          args=["--init_lora", DPO_LORA_DIR, "--pool", RL_POOL_EXPANDED, "--out_dir", GRPO_LORA_DIR]),
    Stage("10d_grpo_infer",  "pipeline/step05_infer_student.py", [GRPO_STUDENT_OUTPUTS],
          "GRPO 模型 eval 推理", manual=True, args=["--lora_dir", GRPO_LORA_DIR, "--out", GRPO_STUDENT_OUTPUTS]),
    Stage("10e_grpo_judge",  "pipeline/step06_judge_with_claude.py", [GRPO_JUDGE_RESULTS],
          "GRPO Kimi 评测", manual=True, args=["--in", GRPO_STUDENT_OUTPUTS, "--out", GRPO_JUDGE_RESULTS]),
    Stage("10f_grpo_report", "pipeline/step07_report.py", [GRPO_REPORT],
          "GRPO 报告", manual=True, args=["--in", GRPO_JUDGE_RESULTS, "--out", GRPO_REPORT]),
]


def stage_complete(outputs) -> bool:
    return all(Path(p).exists() and Path(p).stat().st_size > 0 for p in outputs)


def run_stage(st: "Stage") -> None:
    cmd = [sys.executable, "-X", "utf8", st.script, *st.args]
    log.info("===== START  %-16s (%s) =====", st.sid, " ".join([st.script, *st.args]))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        if st.optional:
            log.warning("===== WARN   %-16s exit=%d (%.1fs) [optional, 继续] =====",
                        st.sid, proc.returncode, elapsed)
            return
        log.error("===== FAIL   %-16s exit=%d  (%.1fs) =====", st.sid, proc.returncode, elapsed)
        raise SystemExit(proc.returncode)
    log.info("===== END    %-16s (%.1fs) =====", st.sid, elapsed)


def main():
    parser = argparse.ArgumentParser(description="RL 蒸馏项目一键运行")
    parser.add_argument("--force", action="store_true", help="忽略产出，强制重跑")
    parser.add_argument("--only", help="只跑指定 stage_id")
    parser.add_argument("--from", dest="from_stage", help="从指定 stage_id 开始")
    parser.add_argument("--to", dest="to_stage", help="跑到指定 stage_id 为止(含)，之后停。配 --from 圈定一段链")
    parser.add_argument("--skip", nargs="*", default=[], help="跳过的 stage_id")
    parser.add_argument("--include-rl", action="store_true",
                        help="bulk 运行时也跑 RL(manual) 阶段（默认 bulk 只跑 SFT 01-07）")
    parser.add_argument("--list", action="store_true", help="列出所有 stage 并退出")
    args = parser.parse_args()

    if args.list:
        for st in STAGES:
            done = "[v]" if stage_complete(st.outputs) else "[ ]"
            tags = []
            if st.optional:
                tags.append("optional")
            if st.manual:
                tags.append("RL/manual")
            tag = f"  ({', '.join(tags)})" if tags else ""
            print(f"{done} {st.sid:<18} {st.desc}{tag}")
        return

    known_ids = {st.sid for st in STAGES}
    for sid in [args.only, args.from_stage, args.to_stage, *args.skip]:
        if sid and sid not in known_ids:
            log.error("未知 stage_id: %s  (可用: %s)", sid, ", ".join(known_ids))
            raise SystemExit(2)

    # bulk = 没有 --only / --from。bulk 模式默认跳过 manual(RL) 阶段，除非 --include-rl。
    bulk = args.only is None and args.from_stage is None
    started = args.from_stage is None
    total_t0 = time.time()
    log.info("########## PIPELINE START   force=%s only=%s from=%s to=%s skip=%s include_rl=%s ##########",
             args.force, args.only, args.from_stage, args.to_stage, args.skip, args.include_rl)

    for st in STAGES:
        if args.only and st.sid != args.only:
            continue
        if not started:
            if st.sid == args.from_stage:
                started = True
            else:
                continue
        if st.sid in args.skip:
            log.info("---- SKIP   %-16s (by --skip)", st.sid)
            continue
        if st.manual and bulk and not args.include_rl:
            log.info("---- SKIP   %-16s (RL 阶段，bulk 默认跳过；用 --from/--only/--include-rl 进入)", st.sid)
            continue
        if not args.force and stage_complete(st.outputs):
            log.info("---- SKIP   %-16s (产出已存在: %s)", st.sid, st.outputs[0])
            if args.to_stage and st.sid == args.to_stage:
                break
            continue
        run_stage(st)
        if args.to_stage and st.sid == args.to_stage:
            log.info("---- STOP   到达 --to %s，停止。", args.to_stage)
            break

    log.info("########## PIPELINE DONE    elapsed=%.1fs ##########", time.time() - total_t0)


if __name__ == "__main__":
    main()
