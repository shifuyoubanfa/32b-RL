"""阶段6-①：构 GRPO 数据集（全池 prompt + 奖励所需的 gold_answer/user_prompt 列）。

swift GRPO 从每条 prompt 在线采样 K 个 completion，用自定义 reward 打分。
reward 需要参考资料(算照抄) + V1 金标准答案(算漂移) → 作为额外列随数据带，reward 插件从 kwargs 读。
全池(2014)；swift 内部按需 shuffle/采样（配合 num_generations 组采样）。
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RL_POOL, OUTPUT_DIR, COLDSTART_SYSTEM_PROMPT
from pipeline.logger import get_logger

log = get_logger("step13_build_grpo_data")
GRPO_DATA = os.path.join(OUTPUT_DIR, "70_grpo_data.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=RL_POOL)
    ap.add_argument("--out", default=GRPO_DATA)
    args = ap.parse_args()

    with open(args.pool, "r", encoding="utf-8") as f:
        pool = [json.loads(l) for l in f if l.strip()]

    rows = []
    for r in pool:
        up = r.get("user_prompt", "")
        rows.append({
            "messages": [
                {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
                {"role": "user", "content": up},
            ],
            "gold_answer": r.get("answer", ""),
            "user_prompt": up,
            "query": r.get("query"),
        })
    with open(args.out, "w", encoding="utf-8") as f:
        for s in rows:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info("GRPO 数据：%d 条全池 prompt -> %s", len(rows), args.out)


if __name__ == "__main__":
    main()
