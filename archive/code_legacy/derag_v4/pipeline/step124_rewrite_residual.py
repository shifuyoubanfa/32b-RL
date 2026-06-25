"""Stage1 derag_v4 residual-trace rewrite.

Input is a greedy inference jsonl from the RFT merged base on the training pool.
Only examples with deterministic residual traces are sent to Kimi.  The prompt
inherits the old grounding red lines and adds explicit anti-policy-list rules.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SEED_WORKERS  # noqa: E402
from pipeline import kimi_client, reward_v3, vllm_client  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402

log = get_logger("step124_rewrite_residual")

REWRITE_SYSTEM = (
    "你是中文税务推理改写专家。你的任务不是改答案，而是把模型 think 中暴露的检索/RAG 痕迹改掉。"
    "所有依据、口径、数字、结论必须完全来自给定参考资料和原答案，不能凭常识发挥。"
)

REWRITE_TEMPLATE = """下面是一道税务题的参考资料、用户问题、标准答案和一段原始 think。
请只改写 think，不输出 answer，不输出解释，不输出标签。

硬性要求：
1. 去掉“参考问答对/参考资料/检索结果/资料显示/问题1/原文提到/参考文件”等检索痕迹。
2. 不要机械罗列文件号、政策名、政策依据清单；如果必须提到政策依据，必须自然嵌入含“规定/按照/明确”等内容词的推理句里，每句最多一个文号。
3. 不要大段照搬参考原文，要用自己的话转述关键事实。
4. 依据、政策口径、数字、期限、税率、金额、会计科目、最终结论必须完全忠于参考资料和标准答案，不能新增、不能删除关键事实、不能与参考资料矛盾。
5. 必要的法规名、政策名、税率档、法定枚举、会计分录和表单栏次应原样保留；这些不是需要消除的 RAG 痕迹。
6. 禁止输出 URL、<img>、附件名；禁用“综上/综上所述/因此可知/需要注意的是”等模板化总结语。
7. 保持原答案含义不变；你只输出改写后的 think 正文。

【参考资料】
{reference}

【用户问题】
{query}

【标准答案】
{answer}

【原始 think】
{think}
"""


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.open(encoding="utf-8"):
        try:
            done.add(json.loads(line).get("query", ""))
        except Exception:
            continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool_out", required=True)
    ap.add_argument("--max_rewrites", type=int, default=0)
    ap.add_argument("--workers", type=int, default=SEED_WORKERS)
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.infer).open(encoding="utf-8") if l.strip()]
    residual = []
    for r in recs:
        text = r.get("gen_text") or f"<think>{r.get('think','')}</think><answer>{r.get('answer','')}</answer>"
        feat = reward_v3.candidate_features(text, r.get("user_prompt", ""), r.get("gold_answer", ""), r.get("query", ""))
        if feat["burden"] > 0 or feat["masked_copy"] > 0.35 or feat["customer_trace"] or feat["qa_trace"]:
            residual.append({**r, "deterministic_features": {k: feat.get(k) for k in (
                "trace_total", "trace_types", "trace_counts", "copy_ratio", "masked_copy",
                "citation_density", "burden", "customer_trace", "qa_trace", "fact_recall")}})
    if args.max_rewrites and args.max_rewrites > 0:
        residual = residual[:args.max_rewrites]

    pool_path = Path(args.pool_out)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with pool_path.open("w", encoding="utf-8") as f:
        for r in residual:
            f.write(json.dumps({
                "query": r.get("query"),
                "user_prompt": r.get("user_prompt", ""),
                "answer": r.get("gold_answer") or r.get("answer", ""),
                "gold_answer": r.get("gold_answer") or r.get("answer", ""),
                "source_trace_features": r.get("deterministic_features", {}),
            }, ensure_ascii=False) + "\n")
    log.warning("PROGRESS residual_trace_pool=%d/%d -> %s", len(residual), len(recs), pool_path)

    out = Path(args.out)
    done = _load_done(out)
    todo = [r for r in residual if r.get("query") not in done]
    lock = threading.Lock()
    fout = out.open("a", encoding="utf-8")

    def _rewrite(r: dict):
        query = r.get("query") or ""
        up = r.get("user_prompt") or ""
        think = r.get("think") or reward_v3.parse_think_answer(r.get("gen_text") or "")[0]
        answer = r.get("gold_answer") or r.get("answer") or ""
        refs = reward_v3.extract_references(up)
        try:
            natural = kimi_client.chat(
                [{"role": "system", "content": REWRITE_SYSTEM},
                 {"role": "user", "content": REWRITE_TEMPLATE.format(
                     reference=refs[:6000], query=query[:500], answer=answer[:3000], think=think[:5000])}],
                temperature=0.3, top_p=0.9, max_tokens=2048,
            ).strip()
        except Exception as exc:
            log.warning("rewrite failed query=%s... %r", query[:30], exc)
            return None
        original_nums = reward_v3.nums("\n".join([think, answer, up]))
        new_nums = reward_v3.nums(natural)
        rec = {
            "query": query,
            "user_prompt": up,
            "original_think": think,
            "natural_think": re.sub(r"^\s*<think>\s*|\s*</think>\s*$", "", natural).strip(),
            "answer": answer,
            "source_features": r.get("deterministic_features", {}),
            "introduced_nums": sorted(new_nums - original_nums),
            "missing_nums": sorted((reward_v3.nums(think) & reward_v3.nums(answer + up)) - new_nums),
        }
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        return rec

    results = vllm_client.map_concurrent(todo, _rewrite, workers=args.workers, desc="derag_v4_rewrite")
    fout.close()
    ok = sum(1 for r in results if r)
    log.warning("RESULT rewrite_residual success=%d todo=%d residual=%d out=%s", ok, len(todo), len(residual), out)


if __name__ == "__main__":
    main()
