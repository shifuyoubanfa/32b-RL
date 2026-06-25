"""阶段5：构 DPO 偏好对 —— 从 CS_RFT 的新 rollout 里挑 chosen/rejected。

规则（与 14B 一致）：同 query 的 gate=ok 候选里，chosen=reward 最高、rejected=R_human 最低，
二者答案都没漂（gate=ok），且 R_human 差 ≥ DPO_MARGIN，否则该 query 跳过。
省资源：先廉价过门，再只对幸存者算 PMI、重算 reward/R_human。
输出 swift DPO 格式：{"messages":[system,user,assistant(=chosen)], "rejected_response": rejected}。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    DPO_ROLLOUT, DPO_PAIRS, DPO_MARGIN, RFT_ACC_FLOOR, REWARD_TAU_ACC,
    THINK_MIN_CHARS, THINK_MAX_CHARS, COLDSTART_SYSTEM_PROMPT, PMI_ENABLED, CS_RFT_LORA_DIR,
)
from pipeline import reward
from pipeline.logger import get_logger

log = get_logger("step12_build_dpo_pairs")


def _full(think: str, answer: str) -> str:
    return f"<think>\n{think.strip()}\n</think>\n\n<answer>\n{answer.strip()}\n</answer>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", default=DPO_ROLLOUT)
    ap.add_argument("--out", default=DPO_PAIRS)
    ap.add_argument("--margin", type=float, default=DPO_MARGIN)
    ap.add_argument("--with_pmi", action="store_true", default=PMI_ENABLED)
    ap.add_argument("--stage_adapter", default=CS_RFT_LORA_DIR, help="PMI 尺子(pi_ref/stage)用的本阶段 adapter")
    args = ap.parse_args()

    with open(args.rollout, "r", encoding="utf-8") as f:
        rolls = [json.loads(l) for l in f if l.strip()]
    log.info("DPO 构对：%d query，PMI=%s margin=%.3f", len(rolls), "开" if args.with_pmi else "关", args.margin)

    # 1) 廉价过门：每 query 收集 gate=ok 且 R_acc≥floor 的候选
    per_q = {}
    for r in rolls:
        up, gold, q = r.get("user_prompt", ""), r.get("gold_answer", ""), r.get("query")
        for cand in r.get("candidates", []):
            sc = reward.score_rollout(cand, up, gold, tau_acc=REWARD_TAU_ACC,
                                      think_min=THINK_MIN_CHARS, think_max=THINK_MAX_CHARS, s_pmi=None)
            if sc["gate"] == "ok" and sc["R_acc"] >= RFT_ACC_FLOOR:
                per_q.setdefault(q, {"up": up, "gold": gold, "cands": []})
                per_q[q]["cands"].append({"think": sc["think"], "answer": sc["answer"], "R_acc": sc["R_acc"]})
    n_surv = sum(len(v["cands"]) for v in per_q.values())
    log.info("过门幸存候选 %d（覆盖 %d query，需≥2 才能构对）", n_surv, len(per_q))

    # 2) 只对幸存者算 PMI -> reward / R_human
    scorer = None
    if args.with_pmi:
        from pipeline import pmi_scorer
        pmi_scorer.load_ruler(stage_adapter=args.stage_adapter)
        scorer = pmi_scorer
    for q, v in per_q.items():
        for c in v["cands"]:
            sp = scorer.s_pmi(v["up"], q, v["gold"], c["think"]) if scorer else None
            r_human, _ = reward.humanness(c["think"], reward.extract_references(v["up"]), s_pmi=sp)
            c["R_human"] = r_human
            c["reward"] = c["R_acc"] * (0.5 + 0.5 * r_human)  # gate=ok 公式

    # 3) 每 query 选 chosen(综合奖励最高) / rejected(综合奖励最低)，差≥margin。
    # ★用综合奖励 reward=R_acc·(0.5+0.5·R_human) 排，不用纯 R_human：过门只保证"都≥floor"，
    # 不保证 chosen 不比 rejected 更不准——纯按 R_human 选会把"更自然但更不准"的当 chosen、DPO 学歪、伤准确率
    # （RFT 首版 acc 0.803→0.784 同源教训）。综合奖励含 humanness 项，相近准确率下仍favored 更自然者，故不丢"学更像人"。
    rows = []
    for q, v in per_q.items():
        cands = v["cands"]
        if len(cands) < 2:
            continue
        chosen = max(cands, key=lambda x: x["reward"])
        rejected = min(cands, key=lambda x: x["reward"])
        if chosen is rejected or (chosen["reward"] - rejected["reward"]) < args.margin:
            continue
        rows.append({
            "messages": [
                {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
                {"role": "user", "content": v["up"]},
                {"role": "assistant", "content": _full(chosen["think"], chosen["answer"])},
            ],
            "rejected_response": _full(rejected["think"], rejected["answer"]),
            "query": q,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for s in rows:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info("DPO 偏好对：%d 对（产出率 %.1f%%）-> %s",
             len(rows), 100 * len(rows) / max(1, len(per_q)), args.out)


if __name__ == "__main__":
    main()
