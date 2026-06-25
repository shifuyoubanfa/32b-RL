"""阶段0（续）：把 V1 自产 (think, answer) 切成 训练池 / 固定验收集。

输入：output/00_v1_outputs.jsonl（query, user_prompt, reasoning, answer）。
输出：00_data_sft_train.jsonl（2014，RL 训练池）/ 00_data_sft_eval.jsonl（224，全程冻结验收集）。
口径与 14B 一致：固定 seed=42 切 9:1，验收集绝不进训练。
样本字段：query / user_prompt(含参考) / reasoning(=think) / answer / messages(可读 preview)。
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import V1_OUTPUTS, SFT_TRAIN, SFT_EVAL, SYSTEM_PROMPT, TRAIN_EVAL_RATIO
from pipeline.logger import get_logger

log = get_logger("step02_split_sft")


def to_sample(rec: dict) -> dict | None:
    reasoning = (rec.get("reasoning") or "").strip()
    answer = (rec.get("answer") or "").strip()
    user_prompt = (rec.get("user_prompt") or "").strip()
    if not answer or not user_prompt:
        return None
    preview = f"<think>\n{reasoning}\n</think>\n\n<answer>\n{answer}\n</answer>"
    return {
        "query": rec.get("query"),
        "user_prompt": user_prompt,
        "reasoning": reasoning,
        "answer": answer,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": preview},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=V1_OUTPUTS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    samples = [s for s in (to_sample(r) for r in recs) if s]
    log.info("有效样本 %d / %d", len(samples), len(recs))

    random.Random(args.seed).shuffle(samples)
    n_train = int(len(samples) * TRAIN_EVAL_RATIO)
    train, eval_ = samples[:n_train], samples[n_train:]

    for path, rows in ((SFT_TRAIN, train), (SFT_EVAL, eval_)):
        with open(path, "w", encoding="utf-8") as f:
            for s in rows:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info("train=%d -> %s", len(train), SFT_TRAIN)
    log.info("eval =%d -> %s（固定验收集，全程冻结）", len(eval_), SFT_EVAL)


if __name__ == "__main__":
    main()
