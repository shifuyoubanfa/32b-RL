"""阶段4：RFT 选样 —— 从 rollout 候选里挑"既守准又自然"的，构 swift SFT 续训集。

省资源策略：先用廉价表面项给所有候选打分、过门（gate=ok 且 R_acc≥RFT_ACC_FLOOR），
再【只对幸存候选】用 PMI 尺子算 s_pmi、重算 R_human、按降序选 TopN（避免对全部候选算 PMI）。
输出 CS_RFT_TRAIN（swift messages 格式）。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    CS_ROLLOUT, CS_RFT_TRAIN, RFT_ACC_FLOOR, RFT_TOPN, REWARD_TAU_ACC,
    THINK_MIN_CHARS, THINK_MAX_CHARS, COLDSTART_SYSTEM_PROMPT, PMI_ENABLED,
)
from pipeline import reward
from pipeline.logger import get_logger

log = get_logger("step11_select_rft")


def _assistant(think: str, answer: str) -> str:
    return f"<think>\n{think.strip()}\n</think>\n\n<answer>\n{answer.strip()}\n</answer>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", default=CS_ROLLOUT)
    ap.add_argument("--out", default=CS_RFT_TRAIN)
    ap.add_argument("--topn", type=int, default=RFT_TOPN)
    ap.add_argument("--with_pmi", action="store_true", default=PMI_ENABLED,
                    help="开 PMI（默认按 config.PMI_ENABLED）")
    ap.add_argument("--stage_adapter", default=None, help="PMI 尺子若用 pi_ref，指定上一阶段 adapter")
    args = ap.parse_args()

    with open(args.rollout, "r", encoding="utf-8") as f:
        rolls = [json.loads(l) for l in f if l.strip()]
    log.info("RFT 选样：%d query，PMI=%s", len(rolls), "开" if args.with_pmi else "关")

    # 1) 廉价表面项过门：每 query 收集 gate=ok 且 R_acc≥floor 的幸存候选
    survivors = []   # [(query, user_prompt, gold, think, answer, base_scored)]
    for r in rolls:
        up, gold = r.get("user_prompt", ""), r.get("gold_answer", "")
        for cand in r.get("candidates", []):
            sc = reward.score_rollout(cand, up, gold, tau_acc=REWARD_TAU_ACC,
                                      think_min=THINK_MIN_CHARS, think_max=THINK_MAX_CHARS, s_pmi=None)
            if sc["gate"] == "ok" and sc["R_acc"] >= RFT_ACC_FLOOR:
                survivors.append({"query": r.get("query"), "user_prompt": up, "gold": gold,
                                  "think": sc["think"], "answer": sc["answer"],
                                  "R_acc": sc["R_acc"]})
    log.info("过门幸存候选：%d（gate=ok 且 R_acc≥%.2f）", len(survivors), RFT_ACC_FLOOR)

    # 2) 只对幸存者算 PMI -> 重算 R_human
    if args.with_pmi and survivors:
        from pipeline import pmi_scorer
        pmi_scorer.load_ruler(stage_adapter=args.stage_adapter)
        for i, s in enumerate(survivors):
            sp = pmi_scorer.s_pmi(s["user_prompt"], s["query"], s["gold"], s["think"])
            r_human, _ = reward.humanness(s["think"], reward.extract_references(s["user_prompt"]), s_pmi=sp)
            s["s_pmi"], s["R_human"] = round(sp, 4), round(r_human, 4)
            if (i + 1) % 100 == 0:
                log.info("PMI 幸存者打分 %d/%d", i + 1, len(survivors))
    else:
        for s in survivors:
            r_human, _ = reward.humanness(s["think"], reward.extract_references(s["user_prompt"]))
            s["s_pmi"], s["R_human"] = None, round(r_human, 4)

    # 3) 每 query 按【综合奖励 R_acc·(0.5+0.5·R_human)】降序选 TopN。
    #    ★关键：不能只按 R_human 排（那会在过门后专挑"最自然"的、准确率往门槛线滑→acc 不升反降，
    #    这正是首版 RFT acc 0.803→0.784 的根因）。综合奖励=score_rollout 的 ok 分支总分，既守准又要自然。
    def _combined(x):
        return x["R_acc"] * (0.5 + 0.5 * x["R_human"])
    by_q = {}
    for s in survivors:
        by_q.setdefault(s["query"], []).append(s)
    rows = []
    for q, cands in by_q.items():
        cands.sort(key=_combined, reverse=True)
        for s in cands[: args.topn]:
            rows.append({"messages": [
                {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
                {"role": "user", "content": s["user_prompt"]},
                {"role": "assistant", "content": _assistant(s["think"], s["answer"])},
            ], "query": q})

    with open(args.out, "w", encoding="utf-8") as f:
        for s in rows:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info("RFT 训练集：%d 条（覆盖 %d query）-> %s", len(rows), len(by_q), args.out)


if __name__ == "__main__":
    main()
