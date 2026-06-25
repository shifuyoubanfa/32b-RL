"""Binary-vote Stage1 gate for derag_v4.

Normal path:
  L0 deterministic gate -> J-trace-bin k2 + J-fact-bin k1 ->
  unanimous/arbiter majority -> one targeted repair -> Stage1 SFT data.

Degraded path:
  If binary judge calibration fails, keep the RL chain alive with the already
  validated L0 deterministic gate and emit a larger human spot-check sheet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import COLDSTART_SYSTEM_PROMPT, JUDGE_CALL_WORKERS  # noqa: E402
from pipeline import judge_common, reward_v3, vllm_client  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402

log = get_logger("step125_gate_rewrites")


def assistant(think: str, answer: str) -> str:
    return f"<think>\n{think.strip()}\n</think>\n\n<answer>\n{answer.strip()}\n</answer>"


def load_json(path: str, default: Any = None) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()] if p.exists() else []


def cand_id(record: dict, think: str | None = None) -> str:
    body = "\n".join([
        record.get("query") or "",
        think if think is not None else record.get("natural_think") or "",
        record.get("answer") or "",
    ])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:20]


def compact_features(features: dict) -> dict:
    keys = (
        "l0_pass", "l0_reasons", "copy_ratio", "masked_copy", "enum_density",
        "citation_density", "standalone_citation_units", "citation_examples",
        "fact_recall", "introduced_nums", "grounding_floor_ok", "answer_score",
        "img_trace", "customer_trace", "qa_trace", "degen_soft", "extreme_degen",
        "degeneration", "burden", "repair_route", "trace_counts",
        "frozen_trace_counts", "frozen_trace_total",
    )
    return {key: features.get(key) for key in keys}


def trace_value(vote: dict) -> str:
    if vote.get("error") or vote.get("confidence") == "low":
        return "uncertain"
    if vote.get("verdict") == "clean":
        return "clean"
    if vote.get("verdict") == "traced":
        valid = [item for item in vote.get("trace_spans") or [] if item.get("verified")]
        return "traced" if valid else "uncertain"
    return "uncertain"


def fact_value(vote: dict) -> str:
    if vote.get("error") or vote.get("confidence") == "low":
        return "uncertain"
    if vote.get("fact_ok") is True:
        return "ok"
    if vote.get("fact_ok") is False:
        valid = [item for item in vote.get("fact_issues") or [] if item.get("verified")]
        return "bad" if valid else "uncertain"
    return "uncertain"


def repair_route(trace_votes: list[dict], fact_vote: dict, arbiter: dict | None, features: dict) -> str:
    if fact_value(fact_vote) == "bad":
        return "fact"
    if arbiter and arbiter.get("fix_type") not in (None, "", "none"):
        return str(arbiter["fix_type"])
    types = {
        span.get("type")
        for vote in trace_votes
        for span in vote.get("trace_spans") or []
        if span.get("verified")
    }
    if types & {"A", "B"}:
        return "trace"
    if types & {"C", "E"}:
        return "paraphrase"
    if "D" in types:
        return "citation"
    return str(features.get("repair_route") or "none")


def judge_once(record: dict, *, allow_repair: bool = True, degraded: bool = False) -> dict:
    think = record.get("natural_think") or ""
    features = reward_v3.candidate_features(
        assistant(think, record.get("answer", "")),
        record.get("user_prompt", ""),
        record.get("answer", ""),
        record.get("query", ""),
    )
    result: dict[str, Any] = {
        "cand_id": cand_id(record),
        "record": record,
        "features": compact_features(features),
        "judge_version": judge_common.JUDGE_V4_BIN_VERSION,
        "votes": {"trace": [], "fact": []},
        "fact": {"fact_ok": None, "issues": []},
        "final": {"verdict": "fail", "path": "fail"},
        "l3_path": "fail",
        "pass": False,
        "repair_round": int(record.get("repair_round") or 0),
    }
    if not features["l0_pass"]:
        result["decision_reason"] = "l0_hard_fail"
        return result

    if degraded:
        result.update({
            "pass": True,
            "l3_path": "deterministic_degraded",
            "decision_reason": "binary_judge_degraded_l0_pass",
            "final": {"verdict": "pass", "path": "deterministic_degraded"},
        })
        return result

    trace_votes = [
        judge_common.judge_trace_bin_v4(
            record.get("query", ""), record.get("user_prompt", ""), think, features, temperature=0.0,
        ),
        judge_common.judge_trace_bin_v4(
            record.get("query", ""), record.get("user_prompt", ""), think, features, temperature=0.3,
        ),
    ]
    fact_vote = judge_common.judge_fact_bin_v4(
        record.get("query", ""),
        record.get("user_prompt", ""),
        record.get("answer", ""),
        record.get("original_think", ""),
        think,
        temperature=0.0,
    )
    result["votes"] = {"trace": trace_votes, "fact": [fact_vote]}
    result["fact"] = {
        "fact_ok": fact_vote.get("fact_ok") is True and not fact_vote.get("error"),
        "issues": fact_vote.get("fact_issues") or [],
        "vote": fact_vote,
    }

    tv = [trace_value(vote) for vote in trace_votes]
    fv = fact_value(fact_vote)
    if tv == ["clean", "clean"] and fv == "ok":
        result.update({
            "pass": True,
            "l3_path": "bin_unanimous",
            "decision_reason": "binary_unanimous_clean_fact_ok",
            "final": {"verdict": "pass", "path": "bin_unanimous"},
        })
        return result

    arbiter = None
    if tv == ["traced", "traced"] and fv == "ok":
        result["decision_reason"] = "binary_unanimous_traced"
    else:
        arbiter = judge_common.judge_arbiter_v4(
            record.get("query", ""),
            record.get("user_prompt", ""),
            think,
            features,
            trace_votes,
            [fact_vote],
        )
        result["arbiter"] = arbiter
        arb_vote = "clean" if arbiter.get("verdict") == "pass" else "traced"
        clean_n = tv.count("clean") + int(arb_vote == "clean")
        traced_n = tv.count("traced") + int(arb_vote == "traced")
        majority = "clean" if clean_n > traced_n else "traced"
        if majority == "clean" and fv == "ok":
            result.update({
                "pass": True,
                "l3_path": "bin_majority",
                "decision_reason": "binary_majority_clean_fact_ok",
                "final": {"verdict": "pass", "path": "bin_majority"},
            })
            return result
        result["decision_reason"] = "binary_majority_fail_or_fact_issue"

    fix_type = repair_route(trace_votes, fact_vote, arbiter, features)
    result["fix_type"] = fix_type
    if not allow_repair or fix_type in (None, "", "none") or result["repair_round"] >= 1:
        return result

    issues = [
        item
        for vote in trace_votes
        for item in vote.get("trace_spans") or []
        if item.get("verified")
    ]
    issues.extend(item for item in fact_vote.get("fact_issues") or [] if item.get("verified"))
    try:
        repaired = judge_common.repair_think_v4(
            record.get("query", ""),
            record.get("user_prompt", ""),
            record.get("answer", ""),
            think,
            fix_type,
            issues,
        )
    except Exception as exc:
        result["repair_error"] = repr(exc)
        return result

    repaired = re.sub(r"^\s*<think>\s*|\s*</think>\s*$", "", repaired).strip()
    nums_same = reward_v3.nums(repaired) == reward_v3.nums(think)
    length_ratio = len(repaired) / max(1, len(think))
    result["repair_attempt"] = {
        "fix_type": fix_type,
        "think": repaired,
        "nums_same": nums_same,
        "length_ratio": round(length_ratio, 4),
    }
    if not nums_same or length_ratio < 0.60:
        result["decision_reason"] = "repair_deterministic_fail"
        return result

    repaired_record = dict(record)
    repaired_record["pre_repair_think"] = think
    repaired_record["natural_think"] = repaired
    repaired_record["repair_round"] = 1
    repaired_record["fix_type"] = fix_type
    after = judge_once(repaired_record, allow_repair=False)
    result["repair_result"] = after
    if after.get("pass"):
        after["l3_path"] = "bin_repaired"
        after["decision_reason"] = "binary_repair_pass"
        after["final"] = {"verdict": "pass", "path": "bin_repaired"}
        after["fix_type"] = fix_type
        return after
    return result


def sentence_starts(text: str) -> list[str]:
    return [
        re.sub(r"^[\s\d一二三四五六七八九十、.．)）(（]+", "", sent)[:2]
        for sent in reward_v3.split_sentences(text)
        if re.sub(r"^[\s\d一二三四五六七八九十、.．)）(（]+", "", sent)
    ]


def phrase_audit(passed: list[dict]) -> dict:
    new_texts = [row["record"].get("natural_think", "") for row in passed]
    old_texts = [row["record"].get("original_think", "") for row in passed]
    chars = max(1, sum(len(text) for text in new_texts))
    summary_per_k = 1000 * sum(len(re.findall(r"综上(?:所述)?", text)) for text in new_texts) / chars
    new_counts = Counter(start for text in new_texts for start in sentence_starts(text))
    old_counts = Counter(start for text in old_texts for start in sentence_starts(text))
    new_total, old_total = max(1, sum(new_counts.values())), max(1, sum(old_counts.values()))
    top6 = [item for item, _ in new_counts.most_common(6)]
    shifts = {
        item: round(new_counts[item] / new_total - old_counts[item] / old_total, 4)
        for item in top6
    }
    max_shift = max(shifts.values(), default=0.0)
    return {
        "pass": summary_per_k <= 0.15 and max_shift <= 0.03,
        "summary_phrase_per_1k": round(summary_per_k, 4),
        "top6_start_shift": shifts,
        "max_positive_shift": round(max_shift, 4),
        "thresholds": {"summary_phrase_per_1k_max": 0.15, "max_positive_shift": 0.03},
    }


def build_sft_rows(passed: list[dict], replay_rows: list[dict]) -> list[dict]:
    rows = []
    for gate_row in passed:
        record = gate_row["record"]
        rows.append({
            "messages": [
                {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
                {"role": "user", "content": record.get("user_prompt", "")},
                {"role": "assistant", "content": assistant(record.get("natural_think", ""), record.get("answer", ""))},
            ],
            "query": record.get("query"),
            "source": f"derag_v4_rw_{gate_row.get('l3_path')}",
            "cand_id": gate_row.get("cand_id"),
        })
    seen = {row.get("query") for row in rows}
    for row in replay_rows:
        if row.get("query") in seen:
            continue
        item = dict(row)
        item["source"] = item.get("source", "derag_v4_replay")
        rows.append(item)
        seen.add(item.get("query"))
    return rows


def filter_replay(rows: list[dict], limit: int) -> tuple[list[dict], Counter]:
    kept = []
    killed = Counter()
    for row in rows:
        if len(kept) >= max(0, limit):
            break
        messages = row.get("messages") or []
        user = next((m.get("content", "") for m in messages if m.get("role") == "user"), row.get("user_prompt", ""))
        output = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), row.get("response", ""))
        _, answer = reward_v3.parse_think_answer(output)
        features = reward_v3.candidate_features(output, user, answer, row.get("query", ""))
        if features["l0_pass"]:
            kept.append(row)
        else:
            killed.update(features["l0_reasons"])
    return kept, killed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--anchors", default="")
    ap.add_argument("--train_out", required=True)
    ap.add_argument("--eval_out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--replay", default="")
    ap.add_argument("--replay_n", type=int, default=150)
    ap.add_argument("--min_rewrites", type=int, default=400)
    ap.add_argument("--workers", type=int, default=JUDGE_CALL_WORKERS)
    ap.add_argument("--degraded-deterministic", action="store_true")
    # Retained for CLI compatibility with the previous package.
    ap.add_argument("--pilot_n", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--pilot_min", type=float, default=0.0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    records = load_jsonl(args.rewrites)
    calibration = load_json(args.calibration, {})
    if not args.degraded_deterministic and calibration.get("status") != "PASS":
        raise SystemExit("binary anchor calibration must PASS unless --degraded-deterministic is explicit")

    entrants, killed = [], []
    for record in records:
        features = reward_v3.candidate_features(
            assistant(record.get("natural_think", ""), record.get("answer", "")),
            record.get("user_prompt", ""),
            record.get("answer", ""),
            record.get("query", ""),
        )
        record["_features"] = features
        (entrants if features["l0_pass"] else killed).append(record)

    if args.degraded_deterministic:
        gated = [judge_once(row, degraded=True) for row in entrants]
    else:
        gated = vllm_client.map_concurrent(
            entrants,
            judge_once,
            workers=args.workers,
            desc="stage1_binary_gate",
        )
    for record in killed:
        gated.append(judge_once(record, degraded=args.degraded_deterministic))

    passed = [row for row in gated if row.get("pass")]
    non_repaired = [row for row in passed if row.get("l3_path") != "bin_repaired"]
    repaired = [row for row in passed if row.get("l3_path") == "bin_repaired"]
    repaired_cap = math.floor(len(non_repaired) * 0.25)
    selected_passed = non_repaired + repaired[:repaired_cap]
    phrase = phrase_audit(selected_passed)
    repaired_rate = sum(row.get("l3_path") == "bin_repaired" for row in selected_passed) / max(1, len(selected_passed))

    replay_source = load_jsonl(args.replay) if args.replay else []
    replay_rows, replay_kills = filter_replay(replay_source, args.replay_n)
    sft_rows = build_sft_rows(selected_passed, replay_rows)
    train = [row for index, row in enumerate(sft_rows) if index % 10 != 0]
    val = [row for index, row in enumerate(sft_rows) if index % 10 == 0]
    for path, data in ((args.train_out, train), (args.eval_out, val)):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as file:
            for row in data:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    if len(selected_passed) < args.min_rewrites:
        status, reason = "NO-GO", "passed_rewrites_below_400"
    elif repaired_rate > 0.20:
        status, reason = "NO-GO", "repaired_lineage_above_20pct"
    elif args.degraded_deterministic:
        status, reason = "DEGRADED-GO", "binary_judge_unfit_using_l0_deterministic_fallback"
    elif not phrase["pass"]:
        status, reason = "NO-GO", "phrase_gate_failed"
    else:
        status, reason = "GO", "binary_stage1_gate_passed"

    path_counts = Counter(row.get("l3_path") for row in gated)
    reason_counts = Counter(
        reason
        for row in gated
        for reason in (row.get("features") or {}).get("l0_reasons") or []
    )
    trace_votes = [
        trace_value(vote)
        for row in gated
        for vote in (row.get("votes") or {}).get("trace") or []
    ]
    report = {
        "status": status,
        "reason": reason,
        "mode": "deterministic_degraded" if args.degraded_deterministic else "binary_vote",
        "judge_version": judge_common.JUDGE_V4_BIN_VERSION,
        "trace_version": reward_v3.TRACE_RE_V4_VERSION,
        "input_rewrites": len(records),
        "l0_entrants": len(entrants),
        "l0_killed": len(killed),
        "l0_kills_by_reason": dict(reason_counts),
        "passed_before_lineage_cap": len(passed),
        "passed_rewrites": len(selected_passed),
        "path_counts": dict(path_counts),
        "trace_vote_counts": dict(Counter(trace_votes)),
        "lineage": {
            "repaired_available": len(repaired),
            "repaired_selected": min(len(repaired), repaired_cap),
            "repaired_rate": round(repaired_rate, 4),
            "repaired_rate_max": 0.20,
        },
        "phrase_gate": phrase,
        "phrase_gate_blocking": not args.degraded_deterministic,
        "human_spotcheck": {
            "status": "PENDING",
            "required_pass_sample": 60 if args.degraded_deterministic else 30,
            "required_fail_sample": 20,
            "note": "Generated by step125c; human result is recorded but does not silently rewrite this gate.",
        },
        "calibration": calibration,
        "replay_rows": len(replay_rows),
        "replay_kills_by_reason": dict(replay_kills),
        "train_rows": len(train),
        "eval_rows": len(val),
        "thresholds": {"min_rewrites": args.min_rewrites},
    }
    report_path = Path(args.report)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with report_path.with_suffix(".rows.jsonl").open("w", encoding="utf-8") as file:
        for row in gated:
            row["record"].pop("_features", None)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.warning("RESULT binary_gate status=%s reason=%s passed=%d l0=%d/%d -> %s",
                status, reason, len(selected_passed), len(entrants), len(records), report_path)
    print(json.dumps(report, ensure_ascii=False))
    if status == "NO-GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
