"""Build V2 GRPO dataset for ms-swift online GRPO.

Input is the frozen V2 train pool only.  The 500 eval questions never enter
this file.  Each row carries the prompt plus ``v1_answers_json`` so the reward
plugin can run the exact V2 answer-in-pool hard gate online without depending
on a hidden worker-local support path.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import COLDSTART_SYSTEM_PROMPT  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402
from pipeline.rules_v6 import answer_in_v1_pool  # noqa: E402
from pipeline.v2_paths import (  # noqa: E402
    V2_EVAL,
    V2_PROBLEMS_TRAIN,
    V2_V1_SUPPORT,
    load_support_index,
    qid_of,
    read_jsonl,
    write_jsonl,
)

log = get_logger("step_v2_build_grpo_data")


def pool_has_trainable_answer(v1_answers: list[str]) -> bool:
    """Whether this V1 answer pool can support the online answer hard gate.

    Historical eval keeps some non-empty/no-slot answers comparable=False for
    metric continuity.  Online GRPO should not train on pools with no concrete
    extractable polarity/number/date facts: every concrete answer would be
    rejected and generic answers would only learn to be vague.
    """
    for ans in v1_answers:
        ad = answer_in_v1_pool(ans, v1_answers)
        if ad.get("in_pool") and ad.get("comparable", True):
            return True
    return False


def build_rows(train_rows: list[dict], support: dict[str, dict], *, eval_qids: set[str] | None = None,
               shuffle_seed: int | None = 42,
               limit: int = 0) -> list[dict]:
    pool = list(train_rows)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(pool)
    if limit and limit > 0:
        pool = pool[:limit]

    rows = []
    leaked = []
    missing = []
    untrainable = []
    for r in pool:
        query = r.get("query") or ""
        qid = r.get("qid") or qid_of(query)
        split = r.get("split")
        if (split and split != "train") or (eval_qids and qid in eval_qids):
            leaked.append(qid)
            continue
        sup = support.get(qid) or {}
        raw_pool = [
            sup.get("v1_canonical_answer"),
            sup.get("gold_answer"),
            *(sup.get("v1_answers") or []),
        ]
        v1_answers = []
        seen_answers = set()
        for a in raw_pool:
            s = str(a or "").strip()
            if not s or s in seen_answers:
                continue
            seen_answers.add(s)
            v1_answers.append(s)
        if not v1_answers:
            missing.append(qid)
            continue
        if not pool_has_trainable_answer(v1_answers):
            untrainable.append(qid)
            continue
        user_prompt = r.get("user_prompt") or ""
        gold = r.get("gold_answer") or r.get("answer") or sup.get("gold_answer") or sup.get("v1_canonical_answer") or ""
        rows.append({
            "messages": [
                {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "qid": qid,
            "query": query,
            "user_prompt": user_prompt,
            "gold_answer": gold,
            "v1_answers_json": json.dumps(v1_answers, ensure_ascii=False),
            "v1_answer_pool_trainable": True,
            "v1_answer_pool_size": len(v1_answers),
        })

    if leaked:
        raise SystemExit(f"GRPO 数据混入非 train split，拒绝继续。示例 qid={leaked[:5]}")
    if missing:
        raise SystemExit(
            f"GRPO 数据有 {len(missing)} 条缺 V1 answer pool，拒绝在线训练；"
            f"请先补 {V2_V1_SUPPORT}。示例 qid={missing[:8]}")
    if untrainable:
        log.warning("跳过 %d 条 V1 answer pool 无可比较事实的训练题；示例 qid=%s",
                    len(untrainable), untrainable[:8])
    if not rows:
        raise SystemExit("GRPO 数据过滤后为空：所有训练题都缺可训练 V1 answer pool，拒绝继续。")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(V2_PROBLEMS_TRAIN))
    ap.add_argument("--support", default=str(V2_V1_SUPPORT))
    ap.add_argument("--eval", default=str(V2_EVAL), help="frozen eval jsonl; qid overlap is rejected")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle_seed", type=int, default=42,
                    help="shuffle train rows before writing; set negative to keep original order")
    ap.add_argument("--limit", type=int, default=0, help="debug/smoke: keep only first N after shuffle; 0=all")
    args = ap.parse_args()

    train_rows = read_jsonl(args.train)
    support = load_support_index(args.support)
    eval_rows = read_jsonl(args.eval) if args.eval else []
    if not train_rows:
        raise SystemExit(f"empty train pool: {args.train}")
    if not support:
        raise SystemExit(f"empty V1 support: {args.support}")
    eval_qids = {r.get("qid") or qid_of(r.get("query") or "") for r in eval_rows if (r.get("qid") or r.get("query"))}
    seed = args.shuffle_seed if args.shuffle_seed >= 0 else None
    rows = build_rows(train_rows, support, eval_qids=eval_qids, shuffle_seed=seed, limit=args.limit)
    write_jsonl(args.out, rows)
    log.info("V2 GRPO 数据：%d 条 train prompt -> %s（support=%s）", len(rows), args.out, args.support)


if __name__ == "__main__":
    main()
