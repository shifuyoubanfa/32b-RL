"""Calibrate Stage1 Kimi judges on frozen good/bad anchors.

The default v4.1 mode uses binary evidence-backed votes. The old continuous
AUC mode remains available only for longitudinal comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_CALL_WORKERS  # noqa: E402
from pipeline import judge_common, reward_v3, vllm_client  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402

log = get_logger("step125a_anchor_calibration")


def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()] if p.exists() else []


def assistant(think: str, answer: str) -> str:
    return f"<think>\n{think.strip()}\n</think>\n\n<answer>\n{answer.strip()}\n</answer>"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def auc(good: list[float], bad: list[float]) -> float:
    if not good or not bad:
        return 0.0
    wins = 0.0
    for g in good:
        for b in bad:
            wins += 1.0 if g > b else 0.5 if g == b else 0.0
    return wins / (len(good) * len(bad))


def anchors_from(rewrites: list[dict], legacy: list[dict], good_n: int, bad_n: int) -> list[tuple[str, dict]]:
    good = [row.get("record") or row for row in legacy if row.get("pass")][:good_n]
    if len(good) < good_n:
        candidates = []
        for row in rewrites:
            feats = reward_v3.candidate_features(
                assistant(row.get("natural_think", ""), row.get("answer", "")),
                row.get("user_prompt", ""),
                row.get("answer", ""),
                row.get("query", ""),
            )
            if feats["l0_pass"] and feats["masked_copy"] <= 0.25 and feats["burden"] == 0:
                candidates.append((feats["masked_copy"], row))
        good = [row for _, row in sorted(candidates, key=lambda item: item[0])[:good_n]]

    bad_ranked = []
    for row in rewrites:
        feats = reward_v3.candidate_features(
            assistant(row.get("natural_think", ""), row.get("answer", "")),
            row.get("user_prompt", ""),
            row.get("answer", ""),
            row.get("query", ""),
        )
        hard = any(reason in feats["l0_reasons"] for reason in (
            "img_trace", "explicit_ref", "ref_enumeration", "raw_copy_cap",
        ))
        if hard:
            severity = (
                10 * int(feats["img_trace"])
                + 5 * int("explicit_ref" in feats["l0_reasons"])
                + 4 * int("ref_enumeration" in feats["l0_reasons"])
                + feats["copy_ratio"]
            )
            bad_ranked.append((severity, row))
    bad = [row for _, row in sorted(bad_ranked, key=lambda item: -item[0])[:bad_n]]
    return [("good", row) for row in good] + [("bad", row) for row in bad]


def features_for(row: dict, think: str) -> dict:
    return reward_v3.candidate_features(
        assistant(think, row.get("answer", "")),
        row.get("user_prompt", ""),
        row.get("answer", ""),
        row.get("query", ""),
    )


def trace_vote_value(vote: dict) -> str:
    if vote.get("error") or vote.get("confidence") == "low":
        return "uncertain"
    verdict = vote.get("verdict")
    if verdict == "traced":
        valid = [span for span in vote.get("trace_spans") or [] if span.get("verified")]
        return "traced" if valid else "uncertain"
    return "clean" if verdict == "clean" else "uncertain"


def binary_score(item: tuple[str, dict]) -> dict:
    label, row = item
    think = row.get("natural_think") or row.get("think") or ""
    feats = features_for(row, think)
    votes = [
        judge_common.judge_trace_bin_v4(
            row.get("query", ""), row.get("user_prompt", ""), think, feats, temperature=0.0,
        ),
        judge_common.judge_trace_bin_v4(
            row.get("query", ""), row.get("user_prompt", ""), think, feats, temperature=0.3,
        ),
    ]
    values = [trace_vote_value(vote) for vote in votes]
    arbiter = None
    if values[0] == values[1] and values[0] in ("clean", "traced"):
        final = values[0]
        path = "unanimous"
    else:
        arbiter = judge_common.judge_arbiter_v4(
            row.get("query", ""), row.get("user_prompt", ""), think, feats, votes, [],
        )
        arb_vote = "clean" if arbiter.get("verdict") == "pass" else "traced"
        clean_n = values.count("clean") + int(arb_vote == "clean")
        traced_n = values.count("traced") + int(arb_vote == "traced")
        final = "clean" if clean_n > traced_n else "traced"
        path = "majority"
    return {
        "label": label,
        "record": row,
        "query": row.get("query"),
        "think": think,
        "features": {k: feats.get(k) for k in (
            "l0_reasons", "copy_ratio", "masked_copy", "burden", "img_trace",
        )},
        "votes": votes,
        "arbiter": arbiter,
        "final": {"verdict": final, "path": path},
        "pass": final == "clean",
    }


def continuous_score(item: tuple[str, dict]) -> dict:
    label, row = item
    think = row.get("natural_think") or row.get("think") or ""
    feats = features_for(row, think)
    votes = [
        judge_common.judge_trace_v4(
            row.get("query", ""), row.get("user_prompt", ""), think, feats, temperature=0.0,
        ),
        judge_common.judge_trace_v4(
            row.get("query", ""), row.get("user_prompt", ""), think, feats, temperature=0.3,
        ),
    ]
    vals = [float(v["trace_free"]) for v in votes if not v.get("error")]
    return {
        "label": label,
        "record": row,
        "query": row.get("query"),
        "think": think,
        "features": {k: feats.get(k) for k in (
            "l0_reasons", "copy_ratio", "masked_copy", "burden", "img_trace",
        )},
        "votes": votes,
        "trace_mean": sum(vals) / len(vals) if vals else 0.0,
        "trace_min": min(vals) if vals else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites", required=True)
    ap.add_argument("--legacy_rows", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors_out", required=True)
    ap.add_argument("--mode", choices=("binary", "continuous"), default="binary")
    ap.add_argument("--good_n", type=int, default=60)
    ap.add_argument("--bad_n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=JUDGE_CALL_WORKERS)
    args = ap.parse_args()

    rewrites = load_jsonl(args.rewrites)
    legacy = load_jsonl(args.legacy_rows) if args.legacy_rows else []
    anchors = anchors_from(rewrites, legacy, args.good_n, args.bad_n)
    scorer = binary_score if args.mode == "binary" else continuous_score
    rows = vllm_client.map_concurrent(anchors, scorer, workers=args.workers, desc="anchor_calibration")

    if args.mode == "binary":
        good_rows = [row for row in rows if row["label"] == "good"]
        bad_rows = [row for row in rows if row["label"] == "bad"]
        good_pass = sum(bool(row.get("pass")) for row in good_rows) / max(1, len(good_rows))
        bad_pass = sum(bool(row.get("pass")) for row in bad_rows) / max(1, len(bad_rows))
        balanced = 0.5 * (good_pass + (1.0 - bad_pass))
        passed = len(good_rows) >= 30 and len(bad_rows) >= 20 and good_pass >= 0.85 and bad_pass <= 0.15
        decision = {
            "status": "PASS" if passed else "DEGRADED",
            "mode": "binary",
            "judge_version": judge_common.JUDGE_V4_BIN_VERSION,
            "good_n": len(good_rows),
            "bad_n": len(bad_rows),
            "good_pass_rate": round(good_pass, 4),
            "bad_pass_rate": round(bad_pass, 4),
            "balanced_accuracy": round(balanced, 4),
            "requirements": {"good_pass_rate_min": 0.85, "bad_pass_rate_max": 0.15},
            "fallback": None if passed else "l0_plus_deterministic_trace_then_spotcheck_60",
            "legacy_good_source": args.legacy_rows or "fallback_current_rewrites",
        }
    else:
        good_scores = [row["trace_mean"] for row in rows if row["label"] == "good"]
        bad_scores = [row["trace_mean"] for row in rows if row["label"] == "bad"]
        metric_auc = auc(good_scores, bad_scores)
        decision = {
            "status": "PASS" if len(good_scores) >= 30 and len(bad_scores) >= 20 and metric_auc >= 0.85 else "NO-GO",
            "mode": "continuous",
            "judge_version": judge_common.JUDGE_V4_VERSION,
            "good_n": len(good_scores),
            "bad_n": len(bad_scores),
            "auc": round(metric_auc, 4),
            "good_mean": round(sum(good_scores) / max(1, len(good_scores)), 4),
            "bad_mean": round(sum(bad_scores) / max(1, len(bad_scores)), 4),
            "good_p20": round(percentile(good_scores, 0.20), 4),
            "t_pass": round(max(0.85, percentile(good_scores, 0.20)), 4),
            "requirements": {"auc_min": 0.85, "t_pass_floor": 0.85},
            "legacy_good_source": args.legacy_rows or "fallback_current_rewrites",
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.anchors_out).open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.warning("RESULT anchor_calibration mode=%s status=%s -> %s", args.mode, decision["status"], out)
    print(json.dumps(decision, ensure_ascii=False))
    if decision["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
