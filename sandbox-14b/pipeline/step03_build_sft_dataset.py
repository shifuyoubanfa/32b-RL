"""Step 3: 把公司模型的输出转成 SFT 训练样本。

样本字段：
- query:     原始问题
- reasoning: teacher 的推理（=think 内容，单独存，训练时手工拼，避免被 R1 模板剥离）
- answer:    teacher 的答案
- messages:  [system, user, assistant]，其中 assistant 仅供人工查看；
             训练只用 system+user 构 prompt，target 由 reasoning/answer 手工拼。

为什么 reasoning/answer 单独存：DeepSeek-R1 系列的 chat_template 会自动删除 assistant
内容里的 <think>...</think> 段。若把推理塞进 assistant 再过模板，推理会在喂给模型前丢失。
step04 因此改为：prompt 走模板（它会注入 <think>），target = reasoning+</think>+<answer> 手工拼。

按比例切分 train / eval。
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    COMPANY_OUTPUTS,
    SFT_TRAIN,
    SFT_EVAL,
    SYSTEM_PROMPT,
    TRAIN_EVAL_RATIO,
)
from pipeline.logger import get_logger

log = get_logger("step03")


def to_sft_sample(rec: dict) -> dict | None:
    reasoning = (rec.get("reasoning_content") or "").strip()
    content = (rec.get("content") or "").strip()
    user_prompt = (rec.get("user_prompt") or "").strip()
    if not content or not user_prompt:
        return None
    # assistant 仅供人工查看，训练不用它（用下面的 reasoning/answer 字段手工拼 target）
    target_preview = f"<think>\n{reasoning}\n</think>\n\n<answer>\n{content}\n</answer>"
    return {
        "query": rec.get("query"),
        "reasoning": reasoning,
        "answer": content,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": target_preview},
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(COMPANY_OUTPUTS, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]

    samples = [s for s in (to_sft_sample(r) for r in recs) if s]
    log.info("有效样本: %d / %d", len(samples), len(recs))

    random.Random(args.seed).shuffle(samples)
    n_train = int(len(samples) * TRAIN_EVAL_RATIO)
    train, eval_ = samples[:n_train], samples[n_train:]

    with open(SFT_TRAIN, "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(SFT_EVAL, "w", encoding="utf-8") as f:
        for s in eval_:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    log.info("train=%d -> %s", len(train), SFT_TRAIN)
    log.info("eval =%d -> %s", len(eval_), SFT_EVAL)


if __name__ == "__main__":
    main()
