"""corrected-v3.1 E0: repeated Kimi derag judge on fixed outputs.

This is the narrowed rubric:
- trace_free: no visible RAG/retrieval/reference traces in think.
- grounded: facts, numbers and conclusions remain supported by references.
- accuracy: answer stays consistent with the V1 gold answer.

It intentionally does not ask Kimi to judge abstract "human reasoning".
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_CALL_WORKERS, OUTPUT_DIR
from pipeline.judge_common import aggregate_derag_judges, judge_text_derag
from pipeline.v3_utils import bootstrap_ci, file_fingerprint, mean, pearson, qid_for, read_jsonl, trace_counts, write_jsonl


def build_rows(infer_rows: list[dict], model_name: str) -> list[dict]:
    rows = []
    for r in infer_rows:
        rows.append({
            "model": model_name,
            "qid": qid_for(r),
            "query": r.get("query") or "",
            "user_prompt": r.get("user_prompt") or "",
            "gold_answer": r.get("gold_answer") or "",
            "gen_text": r.get("gen_text") or "",
        })
    return rows


def key(row: dict) -> tuple[str, str]:
    return (row.get("model") or "", row.get("qid") or "")


def judge_row(row: dict, k: int) -> dict:
    judges = []
    for _ in range(max(1, k)):
        try:
            judges.append(judge_text_derag(row["query"], row["user_prompt"], row["gold_answer"], row["gen_text"]))
        except Exception as exc:
            judges.append({"error": repr(exc)})
    return {**row, "judges": judges, "agg": aggregate_derag_judges(judges)}


def noise_summary(rows: list[dict], model: str) -> dict:
    mrows = [r for r in rows if r.get("model") == model]
    t0, t1, g0, g1, a0, a1, abs_dt = [], [], [], [], [], [], []
    acc_flip = acc_total = 0
    for r in mrows:
        js = [j for j in r.get("judges") or [] if not j.get("error")]
        if len(js) >= 2:
            t0.append(js[0].get("trace_free"))
            t1.append(js[1].get("trace_free"))
            g0.append(js[0].get("grounded"))
            g1.append(js[1].get("grounded"))
            a0.append(js[0].get("accuracy_score"))
            a1.append(js[1].get("accuracy_score"))
            abs_dt.append(abs((js[0].get("trace_free") or 0.0) - (js[1].get("trace_free") or 0.0)))
            acc_total += 1
            if js[0].get("accuracy") != js[1].get("accuracy"):
                acc_flip += 1
    return {
        "model": model,
        "rows": len(mrows),
        "mean_trace_free": mean([r.get("agg", {}).get("trace_free") for r in mrows]),
        "mean_grounded": mean([r.get("agg", {}).get("grounded") for r in mrows]),
        "mean_accuracy_score": mean([r.get("agg", {}).get("accuracy_score") for r in mrows]),
        "mean_trace_free_sd": mean([r.get("agg", {}).get("trace_free_sd") for r in mrows]),
        "test_retest_r_trace_free": pearson(t0, t1),
        "test_retest_r_grounded": pearson(g0, g1),
        "test_retest_r_accuracy": pearson(a0, a1),
        "mean_abs_delta_trace_first2": mean(abs_dt),
        "acc_tier_flip_rate_first2": acc_flip / max(1, acc_total),
    }


def paired_preview(rows: list[dict]) -> dict:
    b = {r["qid"]: r for r in rows if r.get("model") == "base"}
    c = {r["qid"]: r for r in rows if r.get("model") == "candidate"}
    qs = sorted(set(b) & set(c))
    dt = [(c[q]["agg"].get("trace_free") or 0.0) - (b[q]["agg"].get("trace_free") or 0.0) for q in qs]
    dg = [(c[q]["agg"].get("grounded") or 0.0) - (b[q]["agg"].get("grounded") or 0.0) for q in qs]
    da = [(c[q]["agg"].get("accuracy_score") or 0.0) - (b[q]["agg"].get("accuracy_score") or 0.0) for q in qs]
    lo, hi = bootstrap_ci(dt)
    return {
        "paired_n": len(qs),
        "delta_trace_free": mean(dt),
        "delta_trace_free_ci95": [lo, hi],
        "delta_grounded": mean(dg),
        "delta_accuracy_score": mean(da),
    }


def make_report(rows: list[dict], decision: dict, out_jsonl: Path) -> str:
    models = sorted({r["model"] for r in rows})
    summaries = [noise_summary(rows, m) for m in models]
    lines = [
        "# corrected-v3.1 E0 derag judge calibration",
        "",
        f"- status: **{decision['status']}**",
        f"- output jsonl: `{out_jsonl}`",
        f"- target k: {decision['k']}",
        f"- total rows: {len(rows)}",
        "",
        "## Noise By Model",
        "",
        "| model | rows | trace_free | grounded | acc | trace_sd | trace_r12 | mean |Δtrace| first2 | acc_flip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        rt = "NA" if s["test_retest_r_trace_free"] is None else f"{s['test_retest_r_trace_free']:.3f}"
        lines.append(
            f"| {s['model']} | {s['rows']} | {s['mean_trace_free']:.3f} | {s['mean_grounded']:.3f} | "
            f"{s['mean_accuracy_score']:.3f} | {s['mean_trace_free_sd']:.3f} | {rt} | "
            f"{s['mean_abs_delta_trace_first2']:.3f} | {100*s['acc_tier_flip_rate_first2']:.1f}% |"
        )
    if {"base", "candidate"}.issubset(set(models)):
        pv = paired_preview(rows)
        decision.update(pv)
        lines += [
            "",
            "## Paired Base vs Candidate Preview",
            "",
            f"- paired n: {pv['paired_n']}",
            f"- Δtrace_free: {pv['delta_trace_free']:+.3f}, bootstrap 95% CI [{pv['delta_trace_free_ci95'][0]:+.3f}, {pv['delta_trace_free_ci95'][1]:+.3f}]",
            f"- Δgrounded: {pv['delta_grounded']:+.3f}",
            f"- Δaccuracy_score: {pv['delta_accuracy_score']:+.3f}",
            "",
            "## Trace Counts",
        ]
        for m in models:
            tc = trace_counts([r for r in rows if r["model"] == m])
            lines.append(f"- {m}: " + ", ".join(f"{k}={tc.get(k,0)}" for k in ("explicit_ref","verbatim_copy","ref_enumeration","policy_source")))
    lines += [
        "",
        "## Inputs",
        "",
        "```json",
        json.dumps(decision.get("inputs", {}), ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-infer", default=str(Path(OUTPUT_DIR) / "80_corrected_v1_rft_merged_base_infer.jsonl"))
    ap.add_argument("--candidate-infer", default=str(Path(OUTPUT_DIR) / "96_corrected_v2_mini_dpo_infer.jsonl"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=int(os.environ.get("V31_JUDGE_K", "3")))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("V31_E0_LIMIT", "0")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("V31_KIMI_WORKERS", str(JUDGE_CALL_WORKERS))))
    args = ap.parse_args()

    odir = Path(args.out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    out_jsonl = odir / "110_corrected_v31_derag_calibration.jsonl"
    report = odir / "110_corrected_v31_derag_calibration.md"
    decision_path = odir / "110_corrected_v31_derag_calibration_decision.json"

    base = read_jsonl(args.base_infer)
    cand = read_jsonl(args.candidate_infer)
    if not base:
        raise SystemExit(f"missing base infer: {args.base_infer}")
    rows = build_rows(base, "base") + build_rows(cand, "candidate")
    if args.limit > 0:
        rng = random.Random(31)
        by_model = {}
        for r in rows:
            by_model.setdefault(r["model"], []).append(r)
        rows = []
        for rs in by_model.values():
            rng.shuffle(rs)
            rows.extend(rs[:args.limit])

    existing = {key(r): r for r in read_jsonl(out_jsonl) if r.get("agg")}
    todo = [r for r in rows if key(r) not in existing]
    print(f"RESULT v31_e0_plan total={len(rows)} todo={len(todo)} k={args.k} workers={args.workers} out={out_jsonl}", flush=True)

    lock = threading.Lock()
    done = 0
    t0 = time.time()
    with out_jsonl.open("a", encoding="utf-8") as fout:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = [pool.submit(judge_row, r, args.k) for r in todo]
            for fut in cf.as_completed(futs):
                row = fut.result()
                existing[key(row)] = row
                with lock:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                done += 1
                if done % 20 == 0 or done == len(todo):
                    print(f"PROGRESS v31_e0_judge {done}/{len(todo)} rate={done/max(time.time()-t0,1e-3):.2f}/s", flush=True)

    out_rows = sorted(existing.values(), key=lambda r: (r["model"], r["qid"]))
    write_jsonl(out_jsonl, out_rows)
    decision = {
        "status": "complete",
        "rubric": "trace_free_grounded_accuracy",
        "k": args.k,
        "rows": len(out_rows),
        "inputs": {
            "base_infer": file_fingerprint(args.base_infer),
            "candidate_infer": file_fingerprint(args.candidate_infer),
        },
    }
    report.write_text(make_report(out_rows, decision, out_jsonl), encoding="utf-8")
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT v31_e0_complete report={report}", flush=True)


if __name__ == "__main__":
    main()
