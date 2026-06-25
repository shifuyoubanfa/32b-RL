"""Step 14（冷启动·建数据）：从打过分的自然种子里筛出高质量的，构成 cold-start SFT 训练集。

筛选：facts_ok(改写没引入新数字) 且 kimi_facts_kept(Kimi 认为事实保留) 且 kimi_humanness ≥ 阈值。
产出与 step03/RFT 同格式：{query, reasoning(=自然think), answer, messages:[system,user,assistant]}，
供 step04 冷启动微调，让 14B 学会"自然推理"。

切分：把合格样本确定性地切成 train + 留出 eval(COLDSTART_EVAL)。
关键：eval 必须与训练【同分布】(自然腔 think)。若拿 SFT_EVAL(teacher 机器腔)当 eval，模型越训越自然、
eval_loss 反而升高，best-checkpoint/早停会朝反方向选——会把冷启动要做的风格迁移直接掐死(对抗验证已确认)。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    SEEDS_SCORED, COLDSTART_TRAIN, COLDSTART_EVAL, COLDSTART_EVAL_FRAC,
    SEED_HUMANNESS_MIN, SYSTEM_PROMPT,
)
from pipeline.logger import get_logger

log = get_logger("step14_coldstart")


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=SEEDS_SCORED)
    parser.add_argument("--out", default=COLDSTART_TRAIN)
    parser.add_argument("--eval_out", default=COLDSTART_EVAL)
    parser.add_argument("--eval_frac", type=float, default=COLDSTART_EVAL_FRAC,
                        help="留出 eval 比例(确定性切分：每 round(1/frac) 条抽 1 条进 eval)")
    parser.add_argument("--hmin", type=float, default=SEED_HUMANNESS_MIN)
    args = parser.parse_args()

    if not Path(args.in_path).exists():
        log.error("找不到 %s，请先跑 step12_score_seeds.py", args.in_path)
        sys.exit(1)

    with open(args.in_path, "r", encoding="utf-8") as f:
        seeds = [json.loads(l) for l in f if l.strip()]
    log.info("读取打分种子 %d 条；筛选门槛：facts_ok 且 kimi_facts_kept 且 humanness≥%.2f", len(seeds), args.hmin)

    # 先筛出全部合格样本，再确定性切分 train/eval
    kept = []
    drop_fact = drop_hum = drop_empty = 0
    for s in seeds:
        think = (s.get("natural_think") or "").strip()
        answer = (s.get("answer") or "").strip()
        up = s.get("user_prompt") or ""
        if not think or not answer or not up:
            drop_empty += 1
            continue
        if not s.get("facts_ok", True) or (s.get("kimi_facts_kept") is False):
            drop_fact += 1
            continue
        if _f(s.get("kimi_humanness")) < args.hmin:
            drop_hum += 1
            continue
        rec = {
            "query": s.get("query"),
            "reasoning": think,
            "answer": answer,
            "kimi_humanness": _f(s.get("kimi_humanness")),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": up},
                {"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n<answer>\n{answer}\n</answer>"},
            ],
        }
        kept.append(rec)

    # 确定性切分：每 step 条抽 1 条进 eval（自然腔同分布留出集），可复现、无随机
    frac = args.eval_frac if 0.0 < args.eval_frac < 0.5 else 0.10
    step = max(2, round(1.0 / frac))
    n_tr = n_ev = htr = hev = 0
    with open(args.out, "w", encoding="utf-8") as ftr, open(args.eval_out, "w", encoding="utf-8") as fev:
        for i, rec in enumerate(kept):
            h = rec.pop("kimi_humanness", 0.0)
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if i % step == 0:
                fev.write(line); n_ev += 1; hev += h
            else:
                ftr.write(line); n_tr += 1; htr += h

    log.info("冷启动数据：合格 %d 条（丢弃：事实%d / 低humanness%d / 空%d）",
             len(kept), drop_fact, drop_hum, drop_empty)
    log.info("切分：train=%d (avg humanness=%.3f) -> %s | eval=%d (avg=%.3f) -> %s",
             n_tr, (htr / n_tr if n_tr else 0.0), args.out,
             n_ev, (hev / n_ev if n_ev else 0.0), args.eval_out)
    if n_tr < 100:
        log.warning("训练样本偏少(%d)，可下调 --hmin。", n_tr)
    if n_ev < 20:
        log.warning("留出 eval 偏少(%d)，eval_loss 会偏噪声；可下调 --eval_frac 或扩种子。", n_ev)


if __name__ == "__main__":
    main()
