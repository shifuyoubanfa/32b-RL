"""Build auditable Stage1 DPO seed pools from gate decisions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline import reward_v3  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402

log = get_logger("step126_dpo_seed_pools")


def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()] if p.exists() else []


def full(think: str, answer: str) -> str:
    return f"<think>\n{think.strip()}\n</think>\n\n<answer>\n{answer.strip()}\n</answer>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    rows = load_jsonl(args.gate_rows)
    seeds = []
    skipped = Counter()
    for row in rows:
        record = row.get("record") or {}
        chosen_think = record.get("natural_think") or ""
        source_think = record.get("original_think") or ""
        answer = record.get("answer") or ""
        features = row.get("features") or {}
        # Repaired and deterministic-degraded Stage1 rows are intentionally
        # excluded from DPO chosen seeds. Only clean binary votes may become a
        # preference anchor; the deterministic fast path remains independent.
        if row.get("pass") and row.get("l3_path") in ("auto", "arb", "bin_unanimous", "bin_majority"):
            source_features = reward_v3.candidate_features(
                full(source_think, answer),
                record.get("user_prompt", ""),
                answer,
                record.get("query", ""),
            )
            chosen_features = reward_v3.candidate_features(
                full(chosen_think, answer),
                record.get("user_prompt", ""),
                answer,
                record.get("query", ""),
            )
            delta_b = source_features["burden"] - chosen_features["burden"]
            if source_features["burden"] >= 1 and delta_b >= 1:
                seeds.append({
                    "tier": "T1_anchor",
                    "query": record.get("query"),
                    "user_prompt": record.get("user_prompt"),
                    "chosen": full(chosen_think, answer),
                    "rejected": full(source_think, answer),
                    "margin_evidence": {
                        "burden_chosen": chosen_features["burden"],
                        "burden_rejected": source_features["burden"],
                        "delta_burden": delta_b,
                        "delta_masked_copy": round(
                            source_features["masked_copy"] - chosen_features["masked_copy"], 4
                        ),
                        "trace_version": reward_v3.TRACE_RE_V4_VERSION,
                    },
                    "judge_certificate": {
                        "path": row.get("l3_path"),
                        "judge_version": row.get("judge_version"),
                        "votes": row.get("votes"),
                        "final": row.get("final"),
                        "fact": row.get("fact"),
                    },
                })
            else:
                skipped["no_anchor_margin"] += 1
        elif not row.get("pass") and features.get("burden", 0) >= 1:
            seeds.append({
                "tier": "T3_negative_only",
                "query": record.get("query"),
                "user_prompt": record.get("user_prompt"),
                "rejected": full(chosen_think, answer),
                "failure": row.get("decision_reason"),
                "features": features,
            })
        else:
            skipped["not_eligible"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        for row in seeds:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    tiers = Counter(row["tier"] for row in seeds)
    report = {
        "status": "PASS",
        "rows": len(seeds),
        "tiers": dict(tiers),
        "skipped": dict(skipped),
        "trace_version": reward_v3.TRACE_RE_V4_VERSION,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.warning("RESULT dpo_seed_pools tiers=%s -> %s", dict(tiers), out)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
