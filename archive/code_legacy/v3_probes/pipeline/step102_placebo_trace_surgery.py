"""corrected-v3 E1: placebo trace surgery.

Mechanically remove obvious RAG trigger phrases from fixed base outputs without
changing the substantive answer. If Kimi humanness rises substantially, the
rubric is vulnerable to surface-token Goodhart.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_CALL_WORKERS, OUTPUT_DIR
from pipeline.judge_common import aggregate_judges, judge_text
from pipeline.v3_utils import bootstrap_ci, file_fingerprint, mean, qid_for, read_jsonl, run_dir, trace_surgery, write_jsonl


def has_trace(row: dict) -> bool:
    j = row.get("judge") or {}
    if j.get("rag_traces"):
        return True
    text = row.get("gen_text") or ""
    return trace_surgery(text) != text


def judge_row(row: dict, k: int) -> dict:
    edited = trace_surgery(row.get("gen_text") or "")
    judges = []
    for _ in range(max(1, k)):
        try:
            judges.append(judge_text(row.get("query",""), row.get("user_prompt",""), row.get("gold_answer",""), edited))
        except Exception as exc:
            judges.append({"error": repr(exc)})
    out = {
        "qid": qid_for(row),
        "query": row.get("query") or "",
        "user_prompt": row.get("user_prompt") or "",
        "gold_answer": row.get("gold_answer") or "",
        "original_text": row.get("gen_text") or "",
        "edited_text": edited,
        "original_judge": row.get("judge") or {},
        "edited_judges": judges,
        "edited_agg": aggregate_judges(judges),
    }
    out["delta_h"] = (out["edited_agg"].get("humanness") or 0.0) - (out["original_judge"].get("humanness") or 0.0)
    out["delta_g"] = (out["edited_agg"].get("grounded") or 0.0) - (out["original_judge"].get("grounded") or 0.0)
    out["delta_acc"] = (out["edited_agg"].get("accuracy_score") or 0.0) - (out["original_judge"].get("accuracy_score") or 0.0)
    return out


def make_report(rows: list[dict], decision: dict) -> str:
    dh = [r["delta_h"] for r in rows]
    lo, hi = bootstrap_ci(dh)
    mean_dh = mean(dh)
    decision.update({
        "rows": len(rows),
        "mean_delta_h": mean_dh,
        "delta_h_ci95": [lo, hi],
        "mean_delta_g": mean([r["delta_g"] for r in rows]),
        "mean_delta_acc": mean([r["delta_acc"] for r in rows]),
        "goodhart_suspected": mean_dh >= decision["threshold_delta_h"],
        "status": "GOODHART-SUSPECTED" if mean_dh >= decision["threshold_delta_h"] else "not_triggered",
    })
    lines = [
        "# corrected-v3 E1 placebo trace surgery",
        "",
        f"- status: **{decision['status']}**",
        f"- rows: {len(rows)}",
        f"- mean Δh after mechanical trace deletion: {mean_dh:+.3f}",
        f"- bootstrap 95% CI: [{lo:+.3f}, {hi:+.3f}]",
        f"- mean Δgrounded: {decision['mean_delta_g']:+.3f}",
        f"- mean Δacc_score: {decision['mean_delta_acc']:+.3f}",
        f"- Goodhart threshold: {decision['threshold_delta_h']:+.3f}",
        "",
        "If this triggers, current humanness is too sensitive to superficial phrase deletion and should not be the sole optimization target.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-judge", default=str(Path(OUTPUT_DIR) / "80_corrected_v1_rft_merged_base_judge.jsonl"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=JUDGE_CALL_WORKERS)
    ap.add_argument("--threshold-delta-h", type=float, default=0.04)
    args = ap.parse_args()
    odir = Path(args.out_dir) if args.out_dir else run_dir()
    rows = [r for r in read_jsonl(args.base_judge) if has_trace(r)]
    if not rows:
        raise SystemExit(f"no trace-bearing base rows found: {args.base_judge}")
    random.Random(31).shuffle(rows)
    rows = rows[:args.limit]
    out_jsonl = odir / "102_corrected_v3_placebo_trace_surgery.jsonl"
    report = odir / "102_corrected_v3_placebo_report.md"
    decision_path = odir / "102_corrected_v3_placebo_decision.json"
    print(f"RESULT e1_plan rows={len(rows)} k={args.k} workers={args.workers}", flush=True)
    done = 0
    t0 = time.time()
    out = []
    with cf.ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futs = [pool.submit(judge_row, r, args.k) for r in rows]
        for fut in cf.as_completed(futs):
            out.append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(rows):
                print(f"PROGRESS e1_placebo {done}/{len(rows)} rate={done/max(time.time()-t0,1e-3):.2f}/s", flush=True)
    out.sort(key=lambda r: r["qid"])
    decision = {"input": file_fingerprint(args.base_judge), "k": args.k, "threshold_delta_h": args.threshold_delta_h}
    write_jsonl(out_jsonl, out)
    report.write_text(make_report(out, decision), encoding="utf-8")
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT e1_complete status={decision['status']} delta_h={decision['mean_delta_h']:+.3f} report={report}", flush=True)


if __name__ == "__main__":
    main()
