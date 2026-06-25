"""阶段2-③：从打分种子构冷启动数据 + 自然腔留出 eval + 探针低分对照。

- 入选(c)：facts_ok ∧ kimi_humanness ≥ SEED_HUMANNESS_MIN → 冷启动 SFT 集（swift messages 格式）。
- 自然腔 eval：从入选里确定性切 ~10%（绝不能用机器腔 SFT_EVAL 当 eval，否则越练越自然 eval_loss 反升）。
- 探针低分对照(PROBE_LOW)：原始机器腔 think（+ 低质量改写）→ 阶段3 探针选 PMI 尺子用。

swift SFT 数据格式：每行 {"messages":[system, user(含参考), assistant(<think>natural</think><answer>answer</answer>)]}。
assistant 段含开头 <think>（V1 模板不注入、模型自吐，训练目标必须带上，见技术方案 §4.8）。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    SEEDS_SCORED, COLDSTART_TRAIN, COLDSTART_EVAL, PROBE_LOW,
    SEED_HUMANNESS_MIN, SEED_FAITHFUL_MIN, COPY_RATIO_MAX,
    COLDSTART_EVAL_FRAC, COLDSTART_SYSTEM_PROMPT, seed_is_chosen,
)
from pipeline.logger import get_logger

log = get_logger("step08_build_coldstart")


def _assistant(think: str, answer: str) -> str:
    return f"<think>\n{think.strip()}\n</think>\n\n<answer>\n{answer.strip()}\n</answer>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=SEEDS_SCORED)
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]

    # 入选冷启动需【五道闸】：数字未臆造 ∧ 去检索腔(确定性) ∧ 像人 ∧ grounding忠实 ∧ 不照抄。
    # 其中 grounded(faithfulness) 是本次修复核心——把"自然但偏离/矛盾参考"的样本挡在训练集外。
    chosen, low = [], []
    rej = {"facts": 0, "trace": 0, "human": 0, "grounded": 0, "copy": 0}
    for r in recs:
        facts_ok = r.get("facts_ok", False)
        trace_ok = r.get("trace_hits", 0) == 0
        h = r.get("kimi_humanness", 0.0)
        g = r.get("grounded") or 0.0
        cp = r.get("copy_ratio", 0.0)
        if seed_is_chosen(r):   # 五道闸的唯一判据(config)，step09 探针正样本同源，绝不漂移
            chosen.append(r)
        else:
            low.append(r)
            if not facts_ok: rej["facts"] += 1
            elif not trace_ok: rej["trace"] += 1
            elif h < SEED_HUMANNESS_MIN: rej["human"] += 1
            elif g < SEED_FAITHFUL_MIN: rej["grounded"] += 1
            elif cp > COPY_RATIO_MAX: rej["copy"] += 1
    log.info("入选冷启动 %d / 候选 %d（门槛 facts_ok ∧ trace_hits=0 ∧ humanness≥%.2f ∧ grounded≥%.2f ∧ copy≤%.2f）",
             len(chosen), len(recs), SEED_HUMANNESS_MIN, SEED_FAITHFUL_MIN, COPY_RATIO_MAX)
    log.info("剔除分解：数字臆造 %d / 检索腔 %d / 不像人 %d / 不忠实参考 %d / 照抄 %d",
             rej["facts"], rej["trace"], rej["human"], rej["grounded"], rej["copy"])

    # 确定性切自然腔 eval（每 1/FRAC 抽 1）
    every = max(2, round(1 / COLDSTART_EVAL_FRAC))
    train_rows, eval_rows = [], []
    for i, r in enumerate(chosen):
        sample = {"messages": [
            {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
            {"role": "user", "content": r.get("user_prompt", "")},
            {"role": "assistant", "content": _assistant(r.get("natural_think", ""), r.get("answer", ""))},
        ], "query": r.get("query")}
        (eval_rows if i % every == 0 else train_rows).append(sample)

    for path, rows in ((COLDSTART_TRAIN, train_rows), (COLDSTART_EVAL, eval_rows)):
        with open(path, "w", encoding="utf-8") as f:
            for s in rows:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info("冷启动 train=%d -> %s ; 自然腔 eval=%d -> %s", len(train_rows), COLDSTART_TRAIN,
             len(eval_rows), COLDSTART_EVAL)

    # 探针低分对照：原始机器腔 think（每条都有），是最干净的"该低分"样本
    with open(PROBE_LOW, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({
                "query": r.get("query"), "user_prompt": r.get("user_prompt", ""),
                "think": r.get("original_think", ""), "answer": r.get("answer", ""),
                "label": "low",
            }, ensure_ascii=False) + "\n")
    log.info("探针低分对照(原始机器腔) %d -> %s", len(recs), PROBE_LOW)

    if len(train_rows) < 800:
        log.warning("冷启动入选 %d < 800：按技术方案应放宽改写轮次/重采，不放宽事实门。", len(train_rows))


if __name__ == "__main__":
    main()
