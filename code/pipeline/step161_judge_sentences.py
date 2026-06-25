"""judgecal step161 · 一次性高 k 采集：每条 think 让 Kimi 打 0-10 干净分 K_MAX 遍。

这是"一次采集、读两遍"里那一次采集（见 109 设计文档）：
对每条 think（整段），让 Kimi 独立打 0-10 干净分 K_MAX(默认16) 遍。
下游 step162 从这 16 遍里：① 读分数稳不稳(实验一)；② 把分数按真实档位画分辨率曲线、找饱和拐点(实验二)。

只调 Kimi(DashScope) + CPU，不 serve vLLM、不碰 GPU。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_CALL_WORKERS
from pipeline.judgecal_common import judge_clean_score, think_text
from pipeline.logger import get_logger
from pipeline.v3_utils import read_jsonl, write_jsonl

log = get_logger("step161_judgecal")


def judge_item_k(item: dict, k_max: int) -> dict:
    """对一条 think 整段打 0-10 干净分 k_max 遍。"""
    think = think_text(item["sentences"])
    scores = []
    for _ in range(k_max):
        try:
            r = judge_clean_score(item["reference"], think)
            scores.append({"clean_score": r["clean_score"], "n_copied_est": r.get("n_copied_est")})
        except Exception as exc:  # 单遍失败不致命，记下来，下游按有效遍数处理
            scores.append({"error": repr(exc)})
    return {
        "item_id": item.get("item_id"),
        "topic": item.get("topic"),
        "true_level": item.get("true_level"),
        "level_idx": item.get("level_idx"),
        "anchor": item.get("anchor"),
        "true_copy_count": item.get("true_copy_count"),  # 留存客观事实标准，端到端可追溯
        "rule_has_rag_style": item.get("rule_has_rag_style"),  # 规则函数1 旁证(应全 False)
        "k_max": k_max,
        "scores": scores,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="step160 装配出的 160_judgecal_items.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k_max", type=int, default=int(os.environ.get("JUDGECAL_KMAX", "16")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("JUDGECAL_WORKERS", str(JUDGE_CALL_WORKERS))))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("JUDGECAL_LIMIT", "0")), help="0=全量；>0 只判前 N 条(冒烟)")
    args = ap.parse_args()

    items = read_jsonl(args.items)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"空 items: {args.items}")

    total_calls = len(items) * args.k_max
    print(f"RESULT judge_plan items={len(items)} k_max={args.k_max} workers={args.workers} "
          f"kimi_calls={total_calls} out={args.out}", flush=True)

    out_rows = []
    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(judge_item_k, it, args.k_max) for it in items]
        for fut in cf.as_completed(futs):
            out_rows.append(fut.result())
            done += 1
            n_err = sum(1 for r in out_rows for s in r["scores"] if s.get("error"))
            print(f"PROGRESS judge {done}/{len(items)} items "
                  f"failed_scores={n_err} rate={done/max(time.time()-t0,1e-3):.2f} items/s", flush=True)

    out_rows.sort(key=lambda r: r.get("item_id") or "")
    write_jsonl(args.out, out_rows)
    print(f"RESULT judge_complete items={len(out_rows)} out={args.out}", flush=True)
    log.info("采集完成：%d 条 × k=%d -> %s", len(out_rows), args.k_max, args.out)


if __name__ == "__main__":
    main()
