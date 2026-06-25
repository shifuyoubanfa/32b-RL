"""corrected-v2 Phase 1: build strict Kimi-aligned DPO pairs.

Input is step93 candidate scores. Output includes:
- train pairs for swift DPO
- heldout pairs for validation
- full metadata for audit/rejudge
- markdown + JSON decision report
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import COLDSTART_SYSTEM_PROMPT, OUTPUT_DIR

BAD_TRACES = {"ref_enumeration", "verbatim_copy", "explicit_ref"}
BAD_CHOSEN_PHRASES = (
    "从现有的回答来看",
    "现有的回答",
    "参考资料",
    "上述资料",
    "资料中",
    "资料显示",
    "检索到",
    "这里提供",
    "图片链接",
    "文件链接",
    "原文提到",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(xs) -> float:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return sum(vals) / len(vals) if vals else 0.0


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def acc_tier(label: str | None) -> int:
    return {"incorrect": 0, "partial": 1, "correct": 2}.get(label or "incorrect", 0)


def compact_len(text: str) -> int:
    return len("".join((text or "").split()))


def get_j(c: dict, key: str, default=None):
    return (c.get("judge") or {}).get(key, default)


def get_l(c: dict, key: str, default=None):
    return (c.get("local") or {}).get(key, default)


def traces(c: dict) -> set[str]:
    return set(get_j(c, "rag_traces", []) or [])


def bad_phrases(text: str) -> list[str]:
    text = text or ""
    return [p for p in BAD_CHOSEN_PHRASES if p in text]


def chosen_ok(c: dict, h_min: float) -> bool:
    j = c.get("judge") or {}
    if not j:
        return False
    h = j.get("humanness") or 0.0
    g = j.get("grounded") or 0.0
    acc = j.get("accuracy")
    acc_score = j.get("accuracy_score") or 0.0
    fact_raw = get_l(c, "a_fact_recall", 0.0)
    copy_raw = get_l(c, "h_copy_ratio", 1.0)
    fact = 0.0 if fact_raw is None else float(fact_raw)
    copy = 1.0 if copy_raw is None else float(copy_raw)
    return (
        h >= h_min
        and not (traces(c) & BAD_TRACES)
        and not bad_phrases(c.get("text") or "")
        and g >= 0.8
        and (acc == "correct" or (acc == "partial" and acc_score >= 0.6))
        and fact >= 0.7
        and copy <= 0.30
    )


def rejected_ok(r: dict, c: dict, margin: float, max_len_diff: float) -> bool:
    rj = r.get("judge") or {}
    cj = c.get("judge") or {}
    if not rj or not cj:
        return False
    rh = rj.get("humanness") or 0.0
    ch = cj.get("humanness") or 0.0
    rg = rj.get("grounded") or 0.0
    cg = cj.get("grounded") or 0.0
    if rh > ch - margin:
        return False
    if rg < 0.6 or cg < rg - 0.05:
        return False
    if acc_tier(rj.get("accuracy")) > acc_tier(cj.get("accuracy")):
        return False
    clen, rlen = compact_len(c.get("text") or ""), compact_len(r.get("text") or "")
    if max(clen, rlen) and abs(clen - rlen) / max(clen, rlen) > max_len_diff:
        return False
    return True


def pair_record(qid: str, chosen: dict, rejected: dict, mode: str) -> dict:
    cj, rj = chosen["judge"], rejected["judge"]
    hard_negative = bool((traces(rejected) & BAD_TRACES) and acc_tier(rj.get("accuracy")) == acc_tier(cj.get("accuracy")))
    clen, rlen = compact_len(chosen["text"]), compact_len(rejected["text"])
    return {
        "qid": qid,
        "query": chosen.get("query") or "",
        "user_prompt": chosen.get("user_prompt") or "",
        "gold_answer": chosen.get("gold_answer") or "",
        "chosen_text": chosen.get("text") or "",
        "rejected_text": rejected.get("text") or "",
        "chosen_judge": cj,
        "rejected_judge": rj,
        "chosen_local": chosen.get("local") or {},
        "rejected_local": rejected.get("local") or {},
        "chosen_idx": chosen.get("cand_idx"),
        "rejected_idx": rejected.get("cand_idx"),
        "chosen_bad_phrases": bad_phrases(chosen.get("text") or ""),
        "rejected_bad_phrases": bad_phrases(rejected.get("text") or ""),
        "mode": mode,
        "hard_negative": hard_negative,
        "h_margin": (cj.get("humanness") or 0.0) - (rj.get("humanness") or 0.0),
        "g_margin": (cj.get("grounded") or 0.0) - (rj.get("grounded") or 0.0),
        "acc_tier_margin": acc_tier(cj.get("accuracy")) - acc_tier(rj.get("accuracy")),
        "len_diff_ratio": abs(clen - rlen) / max(1, max(clen, rlen)),
    }


def to_swift_pair(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
            {"role": "user", "content": row["user_prompt"]},
            {"role": "assistant", "content": row["chosen_text"]},
        ],
        "rejected_response": row["rejected_text"],
        "query": row["query"],
    }


def baseline_h_by_query(path: Path) -> dict[str, float]:
    out = {}
    for r in read_jsonl(path):
        q = r.get("query")
        j = r.get("judge") or {}
        if q and j.get("humanness") is not None:
            out[q] = float(j["humanness"])
    return out


def build_pairs(candidates: list[dict], h_min: float, margin: float, max_len_diff: float) -> list[dict]:
    by_qid = defaultdict(list)
    for c in candidates:
        if c.get("judge"):
            by_qid[c.get("qid")].append(c)
    pairs = []
    for qid, cands in by_qid.items():
        chosens = [c for c in cands if chosen_ok(c, h_min)]
        if not chosens:
            continue
        chosens.sort(key=lambda c: (get_j(c, "humanness", 0.0), get_j(c, "grounded", 0.0), get_j(c, "accuracy_score", 0.0)), reverse=True)
        chosen = chosens[0]
        rejecteds = [r for r in cands if r is not chosen and rejected_ok(r, chosen, margin, max_len_diff)]
        if not rejecteds:
            continue
        rejecteds.sort(
            key=lambda r: (
                bool((traces(r) & BAD_TRACES) and acc_tier(get_j(r, "accuracy")) == acc_tier(get_j(chosen, "accuracy"))),
                acc_tier(get_j(r, "accuracy")) == acc_tier(get_j(chosen, "accuracy")),
                (get_j(chosen, "humanness", 0.0) or 0.0) - (get_j(r, "humanness", 0.0) or 0.0),
                -(get_j(r, "grounded", 0.0) or 0.0),
            ),
            reverse=True,
        )
        pairs.append(pair_record(qid, chosen, rejecteds[0], f"h_min_{h_min:.2f}"))
    return pairs


def make_report(decision: dict, pairs: list[dict], train: list[dict], heldout: list[dict]) -> str:
    lines = [
        "# corrected-v2 Phase 1 / step94 strict DPO pair report",
        "",
        f"- status: **{decision['status']}**",
        f"- reason: {decision['reason']}",
        f"- scored queries: {decision['scored_queries']}",
        f"- scored candidates with Kimi judge: {decision['scored_candidates']}",
        f"- all pairs: {len(pairs)}",
        f"- train pairs: {len(train)}",
        f"- heldout pairs: {len(heldout)}",
        f"- best-of-K headroom vs RFT base: {decision['headroom_text']}",
        f"- hard negative rate: {pct(decision['hard_negative_rate'])}",
        f"- mean length diff ratio: {decision['mean_len_diff']:.3f}",
        "",
        "## Pair Means",
        "",
        "| metric | chosen | rejected | margin |",
        "|---|---:|---:|---:|",
        f"| humanness | {mean([p['chosen_judge'].get('humanness') for p in pairs]):.3f} | {mean([p['rejected_judge'].get('humanness') for p in pairs]):.3f} | {mean([p['h_margin'] for p in pairs]):+.3f} |",
        f"| grounded | {mean([p['chosen_judge'].get('grounded') for p in pairs]):.3f} | {mean([p['rejected_judge'].get('grounded') for p in pairs]):.3f} | {mean([p['g_margin'] for p in pairs]):+.3f} |",
        f"| accuracy_score | {mean([p['chosen_judge'].get('accuracy_score') for p in pairs]):.3f} | {mean([p['rejected_judge'].get('accuracy_score') for p in pairs]):.3f} | {mean([(p['chosen_judge'].get('accuracy_score') or 0)-(p['rejected_judge'].get('accuracy_score') or 0) for p in pairs]):+.3f} |",
        "",
        "## Gates",
        "",
        "| gate | value | pass |",
        "|---|---:|---:|",
        f"| train pairs >= {decision['min_train_pairs']} | {len(train)} | {decision['enough_pairs']} |",
        f"| heldout pairs >= {decision['min_heldout_pairs']} | {len(heldout)} | {decision['enough_heldout']} |",
        f"| headroom >= {decision['min_headroom']:.2f} | {decision['headroom_text']} | {decision['headroom_ok']} |",
        f"| hard negatives >= {decision['min_hard_negative_rate']:.0%} | {pct(decision['hard_negative_rate'])} | {decision['hard_negative_ok']} |",
        "",
        "## Notes",
        "",
        "- Old 60_dpo_pairs.jsonl is not reused for training here.",
        "- This step uses Kimi-scored candidates from the existing final-RFT rollout pool.",
        "- If status is NO-GO, stop before DPO and inspect pair quality rather than forcing training.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(Path(OUTPUT_DIR) / "93_corrected_v2_rollout_scores.jsonl"))
    ap.add_argument("--out", default=str(Path(OUTPUT_DIR) / "94_corrected_v2_dpo_pairs.jsonl"))
    ap.add_argument("--heldout", default=str(Path(OUTPUT_DIR) / "94_corrected_v2_dpo_pairs_heldout.jsonl"))
    ap.add_argument("--meta", default=str(Path(OUTPUT_DIR) / "94_corrected_v2_dpo_pairs_meta.jsonl"))
    ap.add_argument("--report", default=str(Path(OUTPUT_DIR) / "94_corrected_v2_pair_report.md"))
    ap.add_argument("--decision", default=str(Path(OUTPUT_DIR) / "94_corrected_v2_pair_decision.json"))
    ap.add_argument("--baseline-judge", default=str(Path(OUTPUT_DIR) / "80_corrected_v1_rft_merged_base_judge.jsonl"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("V2_PAIR_SEED", "11")))
    ap.add_argument("--h-min", type=float, default=float(os.environ.get("V2_PAIR_H_MIN", "0.75")))
    ap.add_argument("--relaxed-h-min", type=float, default=float(os.environ.get("V2_PAIR_RELAXED_H_MIN", "0.70")))
    ap.add_argument("--margin", type=float, default=float(os.environ.get("V2_PAIR_MARGIN", "0.25")))
    ap.add_argument("--max-len-diff", type=float, default=float(os.environ.get("V2_PAIR_MAX_LEN_DIFF", "0.30")))
    ap.add_argument("--min-train-pairs", type=int, default=int(os.environ.get("V2_MIN_TRAIN_PAIRS", "80")))
    ap.add_argument("--heldout-size", type=int, default=int(os.environ.get("V2_HELDOUT_PAIRS", "25")))
    ap.add_argument("--min-headroom", type=float, default=float(os.environ.get("V2_MIN_HEADROOM", "0.05")))
    ap.add_argument("--min-hard-negative-rate", type=float, default=float(os.environ.get("V2_MIN_HARD_NEG_RATE", "0.25")))
    args = ap.parse_args()

    scores = [r for r in read_jsonl(Path(args.scores)) if r.get("judge")]
    if not scores:
        raise SystemExit(f"no judged score rows: {args.scores}")
    primary = build_pairs(scores, args.h_min, args.margin, args.max_len_diff)
    pairs = primary
    if len(primary) < args.min_train_pairs + args.heldout_size:
        relaxed = build_pairs(scores, args.relaxed_h_min, args.margin, args.max_len_diff)
        if len(relaxed) > len(primary):
            pairs = relaxed

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    heldout_n = min(args.heldout_size, max(0, len(pairs) - args.min_train_pairs))
    heldout_meta = pairs[:heldout_n]
    train_meta = pairs[heldout_n:]

    baseline = baseline_h_by_query(Path(args.baseline_judge))
    headroom_vals = []
    by_qid = defaultdict(list)
    for c in scores:
        by_qid[c.get("qid")].append(c)
    for cands in by_qid.values():
        q = cands[0].get("query")
        if q in baseline:
            best_h = max((get_j(c, "humanness", 0.0) or 0.0) for c in cands)
            headroom_vals.append(best_h - baseline[q])
    headroom = mean(headroom_vals) if headroom_vals else None
    hard_rate = sum(1 for p in pairs if p["hard_negative"]) / max(1, len(pairs))
    mean_len_diff = mean([p["len_diff_ratio"] for p in pairs])
    enough_pairs = len(train_meta) >= args.min_train_pairs
    enough_heldout = len(heldout_meta) >= min(args.heldout_size, 20)
    headroom_ok = True if headroom is None else headroom >= args.min_headroom
    hard_ok = hard_rate >= args.min_hard_negative_rate
    go = enough_pairs and enough_heldout and headroom_ok and hard_ok
    reason = "data gates passed" if go else "data gates failed; inspect report before training"
    decision = {
        "status": "GO" if go else "NO-GO",
        "reason": reason,
        "scored_queries": len(by_qid),
        "scored_candidates": len(scores),
        "all_pairs": len(pairs),
        "train_pairs": len(train_meta),
        "heldout_pairs": len(heldout_meta),
        "min_train_pairs": args.min_train_pairs,
        "min_heldout_pairs": min(args.heldout_size, 20),
        "enough_pairs": enough_pairs,
        "enough_heldout": enough_heldout,
        "headroom": headroom,
        "headroom_text": "NA" if headroom is None else f"{headroom:+.3f}",
        "min_headroom": args.min_headroom,
        "headroom_ok": headroom_ok,
        "hard_negative_rate": hard_rate,
        "min_hard_negative_rate": args.min_hard_negative_rate,
        "hard_negative_ok": hard_ok,
        "mean_len_diff": mean_len_diff,
    }

    write_jsonl(Path(args.out), [to_swift_pair(p) for p in train_meta])
    write_jsonl(Path(args.heldout), [to_swift_pair(p) for p in heldout_meta])
    write_jsonl(Path(args.meta), pairs)
    Path(args.report).write_text(make_report(decision, pairs, train_meta, heldout_meta), encoding="utf-8")
    Path(args.decision).write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"RESULT pair_build status={decision['status']} train={len(train_meta)} heldout={len(heldout_meta)} "
        f"headroom={decision['headroom_text']} hard_neg={pct(hard_rate)} report={args.report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
