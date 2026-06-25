"""derag_v5 探针 · 第三步：V1 对每道病题答 N=8 遍，建"V1 认可答案范围"（那把尺子）。

需【原始 V1】(checkpoint-1500) 在 vLLM（step150/151 的 RFT 服务已停、改服务 V1）。
V1 用其原生 RAG 系统提示(SYSTEM_PROMPT，还原 V1 真实答案分布)。
另存 1 条 V1 贪心输出作"canonical V1 think+answer"，供 step154 改写。
输出 152_v1_support.jsonl：{qid, support(结论/数字集合), v1_canonical_think, v1_canonical_answer, v1_answers:[8]}
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SYSTEM_PROMPT, GEN_MAX_NEW_TOKENS
from pipeline import vllm_client
from pipeline.v5_probe_common import build_support, parse
from pipeline.logger import get_logger

log = get_logger("step152_v1_support")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=int(os.environ.get("V5_V1_N", "8")))
    args = ap.parse_args()

    vllm_client.wait_ready()
    probs = [json.loads(l) for l in open(args.problems, encoding="utf-8") if l.strip()]
    log.info("PROGRESS V1 建答案库：%d 道病题 × (1 贪心 + N=%d 采样)", len(probs), args.n)

    def _v1(r):
        try:
            canonical = vllm_client.gen_one(r["user_prompt"], system=SYSTEM_PROMPT,
                                            temperature=0.0, max_tokens=GEN_MAX_NEW_TOKENS)
            samples = vllm_client.gen_k(r["user_prompt"], k=args.n, system=SYSTEM_PROMPT,
                                        temperature=0.8, top_p=0.95, max_tokens=GEN_MAX_NEW_TOKENS)
        except Exception as e:
            log.warning("V1 采样失败 qid=%s: %r", r["qid"], e)
            return None
        c_think, c_ans = parse(canonical)
        ans_list = [parse(s)[1] for s in samples]
        return {"qid": r["qid"], "split": r["split"], "query": r["query"],
                "user_prompt": r["user_prompt"], "gold_answer": r["gold_answer"],
                "v1_canonical_think": c_think, "v1_canonical_answer": c_ans,
                "v1_answers": ans_list, "support": build_support(ans_list)}

    results = [x for x in vllm_client.map_concurrent(probs, _v1, desc="v1_support") if x]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("RESULT V1 答案库完成 %d 题 -> %s", len(results), args.out)


if __name__ == "__main__":
    main()
