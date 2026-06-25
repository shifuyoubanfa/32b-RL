"""rollout（RFT / DPO 复用）：某模型在训练池上自采样 K 个候选。

全池 + 随机轮换（固定 seed 洗牌），不再死取前 N。候选生成走 vLLM（高吞吐）。
--model 是该模型在 vLLM 里的名字（adapter 名；orchestration 会先把 adapter 载入 vLLM）。
输出：{query, user_prompt, gold_answer, candidates:[文本×K]}。
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (RL_POOL, RL_K, RL_TEMPERATURE, RL_TOP_P, RL_GEN_MAX_NEW_TOKENS, SAMPLE_SEED,
                    COLDSTART_SYSTEM_PROMPT)
from pipeline import vllm_client
from pipeline.logger import get_logger

log = get_logger("step10_rollout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="vLLM 里的 adapter 名（被采样的模型）")
    ap.add_argument("--pool", default=RL_POOL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=RL_K)
    ap.add_argument("--max_queries", type=int, default=0, help="0=全池")
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--temperature", type=float, default=RL_TEMPERATURE)
    ap.add_argument("--top_p", type=float, default=RL_TOP_P)
    ap.add_argument("--max_tokens", type=int, default=RL_GEN_MAX_NEW_TOKENS)
    ap.add_argument("--chunk_k", type=int, default=0,
                    help="Split large K into smaller vLLM n calls. 0=use DERAG_V4_ROLLOUT_CHUNK_K or K.")
    ap.add_argument("--system_suffix", default="", help="可选 rollout 引导后缀；训练 messages 仍应使用部署 prompt")
    args = ap.parse_args()

    vllm_client.wait_ready()
    with open(args.pool, "r", encoding="utf-8") as f:
        pool = [json.loads(l) for l in f if l.strip()]
    random.Random(args.seed).shuffle(pool)            # 随机轮换
    if args.max_queries and args.max_queries > 0:
        pool = pool[: args.max_queries]
    if args.chunk_k <= 0:
        import os
        args.chunk_k = int(os.environ.get("DERAG_V4_ROLLOUT_CHUNK_K", str(args.k)))
    args.chunk_k = max(1, min(args.k, args.chunk_k))
    log.info("rollout：model=%s 池=%d K=%d temp=%.2f max_tokens=%d（全池=%s seed=%d guided=%s）",
             args.model, len(pool), args.k, args.temperature, args.max_tokens,
             args.max_queries == 0, args.seed, bool(args.system_suffix))

    def _roll(rec: dict) -> dict:
        up = rec.get("user_prompt") or ""
        sys_prompt = COLDSTART_SYSTEM_PROMPT + (("\n" + args.system_suffix.strip()) if args.system_suffix.strip() else "")
        cands = []
        remaining = args.k
        while remaining > 0:
            cur_k = min(args.chunk_k, remaining)
            cands.extend(vllm_client.gen_k(
                up, k=cur_k, model=args.model, system=sys_prompt,
                temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_tokens))
            remaining -= cur_k
        return {"query": rec.get("query"), "user_prompt": up,
                "gold_answer": rec.get("answer") or rec.get("gold_answer", ""), "candidates": cands}

    results = vllm_client.map_concurrent(pool, _roll, desc=f"rollout:{args.model}")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("完成：%d query × K=%d -> %s", len(results), args.k, args.out)


if __name__ == "__main__":
    main()
