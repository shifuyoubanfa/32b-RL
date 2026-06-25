"""derag_v5 探针 · 第二步：RFT 对每道病题自由生成 16 遍（原始文本，先存不评）。

需 RFT-merged base 在 vLLM（与 step150 同一服务窗口）。COLDSTART 系统提示。
输出 151_rft_samples.jsonl：{qid, split, query, user_prompt, gold_answer, samples:[16 段 text]}
评分留到 step153（CPU，要先有 V1 答案库）。
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import COLDSTART_SYSTEM_PROMPT, RL_TEMPERATURE, RL_TOP_P, GEN_MAX_NEW_TOKENS
from pipeline import vllm_client
from pipeline.logger import get_logger

log = get_logger("step151_rft_selfsample")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=int(os.environ.get("V5_RFT_K", "16")))
    args = ap.parse_args()

    vllm_client.wait_ready()
    probs = [json.loads(l) for l in open(args.problems, encoding="utf-8") if l.strip()]
    log.info("PROGRESS RFT 自采样：%d 道病题 × K=%d", len(probs), args.k)

    def _gen(r):
        try:
            samples = vllm_client.gen_k(r["user_prompt"], k=args.k, system=COLDSTART_SYSTEM_PROMPT,
                                        temperature=RL_TEMPERATURE, top_p=RL_TOP_P,
                                        max_tokens=GEN_MAX_NEW_TOKENS)
        except Exception as e:
            log.warning("RFT 自采样失败 qid=%s: %r", r["qid"], e)
            return None
        return {"qid": r["qid"], "split": r["split"], "query": r["query"],
                "user_prompt": r["user_prompt"], "gold_answer": r["gold_answer"], "samples": samples}

    results = [x for x in vllm_client.map_concurrent(probs, _gen, desc="rft_selfsample") if x]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("RESULT RFT 自采样完成 %d 题 -> %s", len(results), args.out)


if __name__ == "__main__":
    main()
