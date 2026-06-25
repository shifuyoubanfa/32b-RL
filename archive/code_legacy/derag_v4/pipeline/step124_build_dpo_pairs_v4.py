"""Build derag_v4 DPO pairs from on-policy rollout.

Margins are deterministic: v4 trace burden and tax-aware masked-copy
differences only. Kimi may audit later, but cannot create a pair margin here.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import COLDSTART_SYSTEM_PROMPT  # noqa: E402
from pipeline import reward_v3  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402

log = get_logger("step124_build_dpo_pairs_v4")


def full(think: str, answer: str) -> str:
    return f"<think>\n{think.strip()}\n</think>\n\n<answer>\n{answer.strip()}\n</answer>"


def length_ok(a: str, b: str, max_ratio: float) -> bool:
    la, lb = max(1, len(a)), max(1, len(b))
    return abs(la / lb - 1.0) <= max_ratio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--train_out", required=True)
    ap.add_argument("--heldout_out", required=True)
    ap.add_argument("--meta_out", required=True)
    ap.add_argument("--min_pairs", type=int, default=160)
    ap.add_argument("--heldout_n", type=int, default=40)
    ap.add_argument("--max_len_ratio", type=float, default=0.30)
    ap.add_argument("--seed_pool", default="")
    ap.add_argument("--seed_pair_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rolls = [json.loads(l) for l in Path(args.rollout).open(encoding="utf-8") if l.strip()]
    pairs = []
    funnel = {"queries": len(rolls), "lt2_candidates": 0, "no_clean": 0, "no_rejected": 0,
              "length_fail": 0, "fact_balance_fail": 0, "paired": 0}
    for r in rolls:
        q = r.get("query") or ""
        up = r.get("user_prompt") or ""
        gold = r.get("gold_answer") or r.get("answer") or ""
        feats = []
        for i, cand in enumerate(r.get("candidates") or []):
            f = reward_v3.candidate_features(cand, up, gold, q)
            f["cand_id"] = i
            feats.append(f)
        if len(feats) < 2:
            funnel["lt2_candidates"] += 1
            continue
        clean = [f for f in feats if f["clean"] and f["answer_score"] >= 0.55]
        rejected = [f for f in feats if f["trace_heavy"] and f["answer_score"] >= 0.45]
        if not clean:
            funnel["no_clean"] += 1
            continue
        if not rejected:
            funnel["no_rejected"] += 1
            continue
        clean.sort(key=lambda x: (x["burden"], x["masked_copy"], -x["fact_recall"], len(x["think"])))
        rejected.sort(key=lambda x: (-x["burden"], -x["masked_copy"], -len(x["think"])))
        chosen = clean[0]
        reject = None
        for cand in rejected:
            if not reward_v3.margin_ok(chosen, cand):
                continue
            if not length_ok(chosen["think"], cand["think"], args.max_len_ratio):
                continue
            if chosen["fact_recall"] + 0.05 < cand["fact_recall"]:
                continue
            reject = cand
            break
        if reject is None:
            if any(not length_ok(chosen["think"], c["think"], args.max_len_ratio) for c in rejected):
                funnel["length_fail"] += 1
            else:
                funnel["fact_balance_fail"] += 1
            continue
        pair = {
            "messages": [
                {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
                {"role": "user", "content": up},
                {"role": "assistant", "content": full(chosen["think"], chosen["answer"])},
            ],
            "rejected_response": full(reject["think"], reject["answer"]),
            "query": q,
            "source": "derag_v4_on_policy",
            "margin_evidence": {
                "burden_chosen": chosen["burden"],
                "burden_rejected": reject["burden"],
                "burden_delta": reject["burden"] - chosen["burden"],
                "masked_copy_chosen": chosen["masked_copy"],
                "masked_copy_rejected": reject["masked_copy"],
                "masked_copy_delta": round(reject["masked_copy"] - chosen["masked_copy"], 4),
                "trace_counter_version": reward_v3.TRACE_RE_V4_VERSION,
                "chosen_cand_id": chosen["cand_id"],
                "rejected_cand_id": reject["cand_id"],
                "chosen_trace_types": chosen["trace_types"],
                "rejected_trace_types": reject["trace_types"],
            },
        }
        pairs.append(pair)
        funnel["paired"] += 1

    seed_added = 0
    if args.seed_pool and Path(args.seed_pool).exists() and pairs:
        seed_rows = [
            json.loads(line)
            for line in Path(args.seed_pool).open(encoding="utf-8")
            if line.strip()
        ]
        anchors = [row for row in seed_rows if row.get("tier") == "T1_anchor"]
        seed_cap = max(1, int(len(pairs) * args.seed_pair_ratio))
        for row in anchors[:seed_cap]:
            pairs.append({
                "messages": [
                    {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
                    {"role": "user", "content": row.get("user_prompt") or ""},
                    {"role": "assistant", "content": row.get("chosen") or ""},
                ],
                "rejected_response": row.get("rejected") or "",
                "query": row.get("query"),
                "source": "derag_v4_stage1_anchor_seed",
                "margin_evidence": row.get("margin_evidence") or {},
            })
            seed_added += 1

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    heldout = pairs[:args.heldout_n]
    train = pairs[args.heldout_n:]
    for path, data in ((args.train_out, train), (args.heldout_out, heldout)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    status = "GO" if len(train) >= args.min_pairs and len(heldout) >= min(args.heldout_n, 1) else "NO-GO"
    meta = {
        "status": status,
        "rollout": args.rollout,
        "train_pairs": len(train),
        "heldout_pairs": len(heldout),
        "funnel": funnel,
        "stage1_seed_pairs_added": seed_added,
        "thresholds": {"min_pairs": args.min_pairs, "heldout_n": args.heldout_n,
                       "max_len_ratio": args.max_len_ratio, "seed_pair_ratio": args.seed_pair_ratio},
    }
    Path(args.meta_out).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.warning("RESULT dpo_pairs_v4 status=%s train=%d heldout=%d paired=%d -> %s",
                status, len(train), len(heldout), len(pairs), args.meta_out)
    print(json.dumps(meta, ensure_ascii=False))
    if status != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
