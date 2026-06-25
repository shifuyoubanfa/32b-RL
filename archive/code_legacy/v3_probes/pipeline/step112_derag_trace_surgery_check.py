"""corrected-v3.1 E2: mechanical trace surgery sanity check.

This checks whether the narrowed derag judge can read the simple thing we care
about: removing visible RAG traces while keeping answer/grounding stable.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_CALL_WORKERS
from pipeline.judge_common import aggregate_derag_judges, judge_text_derag
from pipeline.v3_utils import bootstrap_ci, mean, read_jsonl, trace_surgery, write_jsonl


def choose_rows(rows: list[dict], limit: int) -> list[dict]:
    base = [r for r in rows if r.get("model") == "base"]
    # Prefer rows where the current derag judge saw traces; then fill with lower trace_free rows.
    base.sort(key=lambda r: (
        -len((r.get("agg") or {}).get("rag_traces") or []),
        (r.get("agg") or {}).get("trace_free") or 0.0,
        r.get("qid") or "",
    ))
    picked = base[:limit]
    random.Random(37).shuffle(picked)
    return picked


def process(row: dict, k: int) -> dict:
    edited = trace_surgery(row.get("gen_text") or "")
    judges = []
    for _ in range(max(1, k)):
        try:
            judges.append(judge_text_derag(row["query"], row["user_prompt"], row["gold_answer"], edited))
        except Exception as exc:
            judges.append({"error": repr(exc)})
    old = row.get("agg") or {}
    new = aggregate_derag_judges(judges)
    return {
        "model": "base_trace_surgery",
        "qid": row.get("qid"),
        "query": row.get("query") or "",
        "user_prompt": row.get("user_prompt") or "",
        "gold_answer": row.get("gold_answer") or "",
        "original_text": row.get("gen_text") or "",
        "edited_text": edited,
        "original_agg": old,
        "edited_judges": judges,
        "edited_agg": new,
        "delta_trace_free": (new.get("trace_free") or 0.0) - (old.get("trace_free") or 0.0),
        "delta_grounded": (new.get("grounded") or 0.0) - (old.get("grounded") or 0.0),
        "delta_accuracy_score": (new.get("accuracy_score") or 0.0) - (old.get("accuracy_score") or 0.0),
        "original_trace_count": len(old.get("rag_traces") or []),
        "edited_trace_count": len(new.get("rag_traces") or []),
    }


def make_report(rows: list[dict], decision: dict) -> str:
    dt = [r["delta_trace_free"] for r in rows]
    lo, hi = bootstrap_ci(dt)
    decision.update({
        "rows": len(rows),
        "mean_delta_trace_free": mean(dt),
        "delta_trace_free_ci95": [lo, hi],
        "mean_delta_grounded": mean([r["delta_grounded"] for r in rows]),
        "mean_delta_accuracy_score": mean([r["delta_accuracy_score"] for r in rows]),
        "mean_original_trace_count": mean([r["original_trace_count"] for r in rows]),
        "mean_edited_trace_count": mean([r["edited_trace_count"] for r in rows]),
    })
    readable = lo > 0 and decision["mean_delta_grounded"] >= -0.02 and decision["mean_delta_accuracy_score"] >= -0.02
    decision["status"] = "PASS_DERAG_READABLE" if readable else "NO-GO_DERAG_READABLE"
    lines = [
        "# corrected-v3.1 E2 trace surgery sanity check",
        "",
        f"- status: **{decision['status']}**",
        f"- rows: {len(rows)}",
        f"- mean Δtrace_free after mechanical trace deletion: {mean(dt):+.3f}",
        f"- bootstrap 95% CI: [{lo:+.3f}, {hi:+.3f}]",
        f"- mean Δgrounded: {decision['mean_delta_grounded']:+.3f}",
        f"- mean Δaccuracy_score: {decision['mean_delta_accuracy_score']:+.3f}",
        f"- mean trace count: {decision['mean_original_trace_count']:.2f} -> {decision['mean_edited_trace_count']:.2f}",
        "",
        "This probe only tests whether the judge can read visible de-RAG changes; it does not test abstract human reasoning.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=JUDGE_CALL_WORKERS)
    args = ap.parse_args()

    odir = Path(args.out_dir)
    rows = choose_rows(read_jsonl(args.calibration), args.limit)
    if not rows:
        raise SystemExit(f"no base rows in calibration: {args.calibration}")
    out_jsonl = odir / "112_corrected_v31_trace_surgery_check.jsonl"
    report = odir / "112_corrected_v31_trace_surgery_check.md"
    decision_path = odir / "112_corrected_v31_trace_surgery_check_decision.json"

    existing = {r.get("qid"): r for r in read_jsonl(out_jsonl)}
    todo = [r for r in rows if r.get("qid") not in existing]
    print(f"RESULT v31_e2_plan rows={len(rows)} todo={len(todo)} k={args.k} workers={args.workers}", flush=True)
    lock = threading.Lock()
    done = 0
    t0 = time.time()
    with out_jsonl.open("a", encoding="utf-8") as fout:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = [pool.submit(process, r, args.k) for r in todo]
            for fut in cf.as_completed(futs):
                row = fut.result()
                existing[row["qid"]] = row
                with lock:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                done += 1
                if done % 10 == 0 or done == len(todo):
                    print(f"PROGRESS v31_e2_trace_surgery {done}/{len(todo)} rate={done/max(time.time()-t0,1e-3):.2f}/s", flush=True)

    out = sorted(existing.values(), key=lambda r: r.get("qid") or "")
    write_jsonl(out_jsonl, out)
    decision = {"status": "complete", "k": args.k, "calibration": args.calibration}
    report.write_text(make_report(out, decision), encoding="utf-8")
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT v31_e2_complete status={decision['status']} delta_trace_free={decision['mean_delta_trace_free']:+.3f} report={report}", flush=True)


if __name__ == "__main__":
    main()
