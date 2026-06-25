"""Step 08b（RL 阶段1b）：从 rollout 结果里给每条 query 挑出"既准又自然"的样本，
构成新的 SFT 训练集（与 step03 同格式），供 step04 做 RFT 自蒸馏再训。

重要：用【当前 reward.py】对 rollout 里存的样本原文(text)重新打分，不信 rollout 当时
存的旧分数——这样改进奖励后只需重跑本步即可复用昂贵的 rollout，无需重新生成。

选样策略（对准 RFT 目标：保准确 + 提自然）：
- 先卡准确率硬门槛 R_acc ≥ RFT_ACC_FLOOR（压住准确率下滑）；
- 在合格样本里挑 R_human 最高的前 RFT_TOPN 个（直接朝"更自然"选，而非挑总分最高）；
- 该 query 无任何合格样本 -> 跳过（不强行喂坏数据，避免逼模型瞎编）。

输出 20_rft1_trainset.jsonl，每行：{query, reasoning, answer, messages:[system,user,assistant]}
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    RL_ROLLOUT, RFT_TRAIN, RFT_TOPN, RFT_ACC_FLOOR, SYSTEM_PROMPT,
    REWARD_TAU_ACC, REWARD_W_HUMAN, REWARD_W_ACC, THINK_MIN_CHARS, THINK_MAX_CHARS,
)
from pipeline.logger import get_logger
from pipeline import reward as R

log = get_logger("step08b_rft")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", default=RL_ROLLOUT)
    parser.add_argument("--out", default=RFT_TRAIN)
    parser.add_argument("--topn", type=int, default=RFT_TOPN)
    parser.add_argument("--acc_floor", type=float, default=RFT_ACC_FLOOR)
    parser.add_argument("--with-pmi", dest="with_pmi", action="store_true",
                        help="选样时把 PMI 结构信号并入 humanness(需加载 14B,慢+1~2h)。默认关，只用关键词+字符照抄。")
    args = parser.parse_args()

    if not Path(args.rollout).exists():
        log.error("找不到 rollout 文件 %s，请先跑 step08_rl_rollout.py", args.rollout)
        sys.exit(1)

    with open(args.rollout, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    log.info("读取 rollout %d 条 query；用当前 reward 重新打分，准确率门槛=%.2f，PMI=%s",
             len(recs), args.acc_floor, "开" if args.with_pmi else "关")

    # 可选：加载 14B base 算 PMI
    pmi_model = pmi_tok = None
    if args.with_pmi:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from config import STUDENT_LOCAL_DIR
        log.info("加载 14B base 算 PMI: %s", STUDENT_LOCAL_DIR)
        pmi_tok = AutoTokenizer.from_pretrained(STUDENT_LOCAL_DIR, trust_remote_code=True)
        if pmi_tok.pad_token is None:
            pmi_tok.pad_token = pmi_tok.eos_token
        pmi_model = AutoModelForCausalLM.from_pretrained(
            STUDENT_LOCAL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        ).eval()

    n_written = 0
    n_skip = 0
    human_kept = []
    with open(args.out, "w", encoding="utf-8") as fout:
        for ridx, rec in enumerate(recs, 1):
            user_prompt = rec.get("user_prompt") or ""
            gold = (rec.get("gold_answer") or "").strip()
            query = rec.get("query") or ""
            qa_user = "【问题】\n%s\n\n【标准答案】\n%s" % (query, gold)

            # 用当前 reward.py 对原文重新打分（可选并入 PMI）
            rescored = []
            for s in rec.get("samples", []):
                text = s.get("text")
                if not text:
                    continue
                s_pmi = None
                if args.with_pmi:
                    think, _ = R.parse_think_answer(text)
                    try:
                        s_pmi = R.pmi_cond(pmi_model, pmi_tok, SYSTEM_PROMPT, user_prompt, qa_user, think)
                    except Exception as e:
                        log.error("PMI 失败 query=%s...: %r", query[:24], e)
                        s_pmi = None
                sc = R.score_rollout(
                    text, user_prompt, gold,
                    tau_acc=REWARD_TAU_ACC, w_human=REWARD_W_HUMAN, w_acc=REWARD_W_ACC,
                    think_min=THINK_MIN_CHARS, think_max=THINK_MAX_CHARS, s_pmi=s_pmi,
                )
                rescored.append(sc)
            if args.with_pmi and ridx % 50 == 0:
                log.info("PMI 选样进度 [%d/%d]", ridx, len(recs))

            # 选样：先卡准确率门槛 + 格式合法，再挑最自然
            cand = [s for s in rescored if s.get("gate") == "ok" and s.get("R_acc", 0.0) >= args.acc_floor]
            if not cand:
                n_skip += 1
                continue
            cand.sort(key=lambda x: x.get("R_human", -1.0), reverse=True)

            for s in cand[: max(1, args.topn)]:
                reasoning = (s.get("think") or "").strip()
                answer = (s.get("answer") or "").strip()
                if not reasoning or not answer:
                    continue
                target_preview = f"<think>\n{reasoning}\n</think>\n\n<answer>\n{answer}\n</answer>"
                out = {
                    "query": rec.get("query"),
                    "reasoning": reasoning,
                    "answer": answer,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": target_preview},
                    ],
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_written += 1
                human_kept.append(s.get("R_human", 0.0))

    avg_h = sum(human_kept) / len(human_kept) if human_kept else 0.0
    log.info("RFT 数据：写入 %d 条（跳过 %d 条无合格样本的 query），入选样本平均 R_human=%.3f -> %s",
             n_written, n_skip, avg_h, args.out)
    if n_written == 0:
        log.warning("没有任何合格样本入选！RFT_ACC_FLOOR 可能太高，或 rollout 质量不行。")
    elif n_written < 100:
        log.warning("入选样本偏少(%d)，RFT 可能训不充分；可适当下调 RFT_ACC_FLOOR。", n_written)


if __name__ == "__main__":
    main()
