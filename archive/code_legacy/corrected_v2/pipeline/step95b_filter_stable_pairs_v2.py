"""corrected-v2: keep only double-judge-stable DPO pairs.

This step is intentionally conservative. step94 finds likely good pairs from
the first Kimi pass; step95 rejudges them. This script writes the actual DPO
training set from pairs whose direction survives both passes.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import COLDSTART_SYSTEM_PROMPT, OUTPUT_DIR

BAD_STABLE_PHRASES = (
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
    vals = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if not math.isnan(v):
            vals.append(v)
    return sum(vals) / len(vals) if vals else 0.0


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def bad_phrases(text: str) -> list[str]:
    text = text or ""
    return [p for p in BAD_STABLE_PHRASES if p in text]


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


def keep_reason(row: dict, min_initial_h: float, min_rejudge_h: float, min_g_margin: float) -> tuple[bool, list[str]]:
    reasons = []
    if row.get("h_margin", 0.0) < min_initial_h:
        reasons.append("initial_h_margin_low")
    if row.get("rejudge_h_margin", 0.0) < min_rejudge_h:
        reasons.append("rejudge_h_margin_low")
    if row.get("rejudge_g_margin", 0.0) < min_g_margin:
        reasons.append("grounded_drop")
    if row.get("rejudge_acc_tier_margin", 0) < 0:
        reasons.append("accuracy_rank_loss")
    if bad_phrases(row.get("chosen_text") or ""):
        reasons.append("chosen_rag_phrase")
    if row.get("rejudge_error"):
        reasons.append("rejudge_error")
    return not reasons, reasons


def make_report(decision: dict, rows: list[dict], stable: list[dict], train: list[dict], heldout: list[dict]) -> str:
    fail_counts = {}
    for row in rows:
        for reason in row.get("stable_reject_reasons", []) or []:
            fail_counts[reason] = fail_counts.get(reason, 0) + 1
    fail_lines = [f"| {k} | {v} |" for k, v in sorted(fail_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if not fail_lines:
        fail_lines = ["| none | 0 |"]
    lines = [
        "# corrected-v2 Phase 1 / step95b stable pair filter report",
        "",
        f"- status: **{decision['status']}**",
        f"- reason: {decision['reason']}",
        f"- rejudged pairs: {decision['rejudged_pairs']}",
        f"- stable pairs: {decision['stable_pairs']} ({pct(decision['stable_rate'])})",
        f"- train pairs: {len(train)}",
        f"- heldout pairs: {len(heldout)}",
        f"- mean stable rejudge humanness margin: {decision['mean_stable_rejudge_h_margin']:+.3f}",
        "",
        "## Stable Gates",
        "",
        "| gate | value | pass |",
        "|---|---:|---:|",
        f"| train pairs >= {decision['min_train_pairs']} | {len(train)} | {decision['enough_train']} |",
        f"| heldout pairs >= {decision['min_heldout_pairs']} | {len(heldout)} | {decision['enough_heldout']} |",
        f"| mean stable rejudge h >= {decision['min_mean_stable_rejudge_h_margin']:.2f} | {decision['mean_stable_rejudge_h_margin']:+.3f} | {decision['mean_h_ok']} |",
        "",
        "## Rejection Reasons",
        "",
        "| reason | count |",
        "|---|---:|",
        *fail_lines,
        "",
        "Only stable train pairs are passed to DPO. The old step94 train file is kept as audit material, not used for training after this step.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejudge", default=str(Path(OUTPUT_DIR) / "95_corrected_v2_pair_rejudge.jsonl"))
    ap.add_argument("--out", default=str(Path(OUTPUT_DIR) / "95b_corrected_v2_stable_dpo_pairs.jsonl"))
    ap.add_argument("--heldout", default=str(Path(OUTPUT_DIR) / "95b_corrected_v2_stable_dpo_pairs_heldout.jsonl"))
    ap.add_argument("--meta", default=str(Path(OUTPUT_DIR) / "95b_corrected_v2_stable_pairs_meta.jsonl"))
    ap.add_argument("--report", default=str(Path(OUTPUT_DIR) / "95b_corrected_v2_stable_pair_report.md"))
    ap.add_argument("--decision", default=str(Path(OUTPUT_DIR) / "95b_corrected_v2_stable_pair_decision.json"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("V2_STABLE_SEED", "23")))
    ap.add_argument("--min-initial-h-margin", type=float, default=float(os.environ.get("V2_STABLE_MIN_INITIAL_H_MARGIN", "0.40")))
    ap.add_argument("--min-rejudge-h-margin", type=float, default=float(os.environ.get("V2_STABLE_MIN_REJUDGE_H_MARGIN", "0.05")))
    ap.add_argument("--min-rejudge-g-margin", type=float, default=float(os.environ.get("V2_STABLE_MIN_REJUDGE_G_MARGIN", "-0.05")))
    ap.add_argument("--min-train-pairs", type=int, default=int(os.environ.get("V2_STABLE_MIN_TRAIN_PAIRS", "50")))
    ap.add_argument("--heldout-size", type=int, default=int(os.environ.get("V2_STABLE_HELDOUT_PAIRS", "10")))
    ap.add_argument("--min-mean-h-margin", type=float, default=float(os.environ.get("V2_STABLE_MIN_MEAN_H_MARGIN", "0.15")))
    args = ap.parse_args()

    rows = read_jsonl(Path(args.rejudge))
    if not rows:
        raise SystemExit(f"missing rejudge rows: {args.rejudge}")
    stable = []
    annotated = []
    for row in rows:
        row = dict(row)
        keep, reasons = keep_reason(row, args.min_initial_h_margin, args.min_rejudge_h_margin, args.min_rejudge_g_margin)
        row["stable_pair_ok"] = keep
        row["stable_reject_reasons"] = reasons
        annotated.append(row)
        if keep:
            stable.append(row)

    random.Random(args.seed).shuffle(stable)
    heldout_n = min(args.heldout_size, max(0, len(stable) - args.min_train_pairs))
    heldout_meta = stable[:heldout_n]
    train_meta = stable[heldout_n:]
    mean_h = mean([r.get("rejudge_h_margin") for r in stable])
    enough_train = len(train_meta) >= args.min_train_pairs
    enough_heldout = len(heldout_meta) >= min(args.heldout_size, 5)
    mean_h_ok = mean_h >= args.min_mean_h_margin
    go = enough_train and enough_heldout and mean_h_ok
    decision = {
        "status": "GO" if go else "NO-GO",
        "reason": "stable pair gates passed" if go else "stable pair gates failed; score more queries or relax only after review",
        "rejudged_pairs": len(rows),
        "stable_pairs": len(stable),
        "stable_rate": len(stable) / max(1, len(rows)),
        "train_pairs": len(train_meta),
        "heldout_pairs": len(heldout_meta),
        "min_train_pairs": args.min_train_pairs,
        "min_heldout_pairs": min(args.heldout_size, 5),
        "enough_train": enough_train,
        "enough_heldout": enough_heldout,
        "mean_stable_rejudge_h_margin": mean_h,
        "min_mean_stable_rejudge_h_margin": args.min_mean_h_margin,
        "mean_h_ok": mean_h_ok,
        "min_initial_h_margin": args.min_initial_h_margin,
        "min_rejudge_h_margin": args.min_rejudge_h_margin,
        "min_rejudge_g_margin": args.min_rejudge_g_margin,
    }

    write_jsonl(Path(args.out), [to_swift_pair(r) for r in train_meta])
    write_jsonl(Path(args.heldout), [to_swift_pair(r) for r in heldout_meta])
    write_jsonl(Path(args.meta), annotated)
    Path(args.report).write_text(make_report(decision, annotated, stable, train_meta, heldout_meta), encoding="utf-8")
    Path(args.decision).write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"RESULT stable_filter status={decision['status']} stable={len(stable)}/{len(rows)} "
        f"train={len(train_meta)} heldout={len(heldout_meta)} mean_h={mean_h:+.3f} report={args.report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
