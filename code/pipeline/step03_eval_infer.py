"""评测推理（所有阶段复用）：在固定 224 验收集上跑某个模型，产出 think/answer。

--model 指定 vLLM 服务里的模型名：
  - 'v1'        基线（原始 V1，阶段1）
  - 各阶段 adapter 名（vLLM --lora-modules 或动态加载后用该名请求；见 serve 脚本）
评测用贪心(temperature=0)保证可复现（同集两次跑分差异 <1%）。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SFT_EVAL, VLLM_MODEL, GEN_MAX_NEW_TOKENS, system_for
from pipeline import vllm_client
from pipeline.reward import parse_think_answer_diagnostic
from pipeline.logger import get_logger

log = get_logger("step03_eval_infer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=VLLM_MODEL, help="vLLM 服务里的模型/adapter 名")
    ap.add_argument("--eval_file", default=SFT_EVAL)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vllm_client.wait_ready()
    sys_prompt = system_for(args.model)   # 基线 v1 用 RAG 腔；训练后模型用去检索腔中性提示
    with open(args.eval_file, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    log.info("评测推理：%d 条（model=%s, system=%s）", len(recs), args.model,
             "RAG腔" if args.model == "v1" else "中性")

    def _infer(rec: dict) -> dict:
        up = rec.get("user_prompt") or ""
        raw = vllm_client.chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": up}],
            model=args.model, n=1, temperature=0.0, top_p=1.0, max_tokens=GEN_MAX_NEW_TOKENS,
        )[0]
        parsed = parse_think_answer_diagnostic(raw)
        return {"query": rec.get("query"), "user_prompt": up,
                "gold_answer": rec.get("answer", ""), "gen_text": raw,
                "think": parsed["think"], "answer": parsed["answer"],
                "format_ok": parsed["format_ok"], "format_reason": parsed["format_reason"]}

    results = vllm_client.map_concurrent(recs, _infer, desc=f"eval:{args.model}")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    empties = sum(1 for r in results if not (r["answer"] or "").strip())
    log.info("完成：%d -> %s（空答案 %d）", len(results), args.out, empties)


if __name__ == "__main__":
    main()
