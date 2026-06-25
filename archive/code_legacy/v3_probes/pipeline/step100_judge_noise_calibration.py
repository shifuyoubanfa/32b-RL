"""corrected-v3 E0: repeated Kimi judge calibration on fixed outputs.

This step does not generate model outputs or train. It repeatedly judges fixed
base/candidate outputs, then writes raw repeated judges plus aggregates.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_CALL_WORKERS, OUTPUT_DIR
from pipeline.judge_common import aggregate_judges, judge_text
from pipeline.v3_utils import (
    bootstrap_ci,
    file_fingerprint,
    mean,
    pearson,
    qid_for,
    read_jsonl,
    run_dir,
    sd,
    trace_counts,
    write_jsonl,
)


def by_qid(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for r in rows:
        out[qid_for(r)] = r
    return out


def merge_infer_and_existing(infer_rows: list[dict], judge_rows: list[dict], model_name: str, use_existing: bool) -> list[dict]:
    existing = by_qid(judge_rows)
    merged = []
    for r in infer_rows:
        qid = qid_for(r)
        judges = []
        if use_existing and qid in existing and existing[qid].get("judge"):
            judges.append(existing[qid]["judge"])
        merged.append({
            "model": model_name,
            "qid": qid,
            "query": r.get("query") or "",
            "user_prompt": r.get("user_prompt") or "",
            "gold_answer": r.get("gold_answer") or "",
            "gen_text": r.get("gen_text") or "",
            "initial_judges": judges,
        })
    return merged


def judge_missing(row: dict, target_k: int) -> dict:
    judges = list(row.get("initial_judges") or [])
    need = max(0, target_k - len([j for j in judges if not j.get("error")]))
    for _ in range(need):
        try:
            judges.append(judge_text(row["query"], row["user_prompt"], row["gold_answer"], row["gen_text"]))
        except Exception as exc:
            judges.append({"error": repr(exc)})
    out = dict(row)
    out.pop("initial_judges", None)
    out["judges"] = judges
    out["agg"] = aggregate_judges(judges)
    return out


def noise_summary(rows: list[dict], model: str) -> dict:
    mrows = [r for r in rows if r.get("model") == model]
    h_sds = [r.get("agg", {}).get("humanness_sd", 0.0) for r in mrows]
    g_sds = [r.get("agg", {}).get("grounded_sd", 0.0) for r in mrows]
    a_sds = [r.get("agg", {}).get("accuracy_score_sd", 0.0) for r in mrows]
    h0, h1, g0, g1, a0, a1 = [], [], [], [], [], []
    acc_flip = 0
    acc_total = 0
    abs_dh = []
    for r in mrows:
        js = [j for j in r.get("judges") or [] if not j.get("error")]
        if len(js) >= 2:
            h0.append(js[0].get("humanness"))
            h1.append(js[1].get("humanness"))
            g0.append(js[0].get("grounded"))
            g1.append(js[1].get("grounded"))
            a0.append(js[0].get("accuracy_score"))
            a1.append(js[1].get("accuracy_score"))
            abs_dh.append(abs((js[0].get("humanness") or 0.0) - (js[1].get("humanness") or 0.0)))
            acc_total += 1
            if js[0].get("accuracy") != js[1].get("accuracy"):
                acc_flip += 1
    return {
        "model": model,
        "rows": len(mrows),
        "mean_h": mean([r.get("agg", {}).get("humanness") for r in mrows]),
        "mean_g": mean([r.get("agg", {}).get("grounded") for r in mrows]),
        "mean_acc_score": mean([r.get("agg", {}).get("accuracy_score") for r in mrows]),
        "mean_h_within_sd": mean(h_sds),
        "mean_g_within_sd": mean(g_sds),
        "mean_acc_within_sd": mean(a_sds),
        "test_retest_r_h": pearson(h0, h1),
        "test_retest_r_g": pearson(g0, g1),
        "test_retest_r_acc": pearson(a0, a1),
        "mean_abs_dh_first2": mean(abs_dh),
        "acc_tier_flip_rate_first2": acc_flip / max(1, acc_total),
    }


def make_report(rows: list[dict], decision: dict, out_jsonl: Path) -> str:
    models = sorted({r["model"] for r in rows})
    summaries = [noise_summary(rows, m) for m in models]
    lines = [
        "# corrected-v3 E0 judge noise calibration",
        "",
        f"- status: **{decision['status']}**",
        f"- output jsonl: `{out_jsonl}`",
        f"- target k: {decision['k']}",
        f"- total rows: {len(rows)}",
        "",
        "## Noise By Model",
        "",
        "| model | rows | h | g | acc | h_sd | h_r12 | mean |Δh| first2 | acc_flip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        rh = "NA" if s["test_retest_r_h"] is None else f"{s['test_retest_r_h']:.3f}"
        lines.append(
            f"| {s['model']} | {s['rows']} | {s['mean_h']:.3f} | {s['mean_g']:.3f} | "
            f"{s['mean_acc_score']:.3f} | {s['mean_h_within_sd']:.3f} | {rh} | "
            f"{s['mean_abs_dh_first2']:.3f} | {100*s['acc_tier_flip_rate_first2']:.1f}% |"
        )
    if {"base", "candidate"}.issubset(set(models)):
        b = {r["qid"]: r for r in rows if r["model"] == "base"}
        c = {r["qid"]: r for r in rows if r["model"] == "candidate"}
        qs = sorted(set(b) & set(c))
        dh = [(c[q]["agg"].get("humanness") or 0.0) - (b[q]["agg"].get("humanness") or 0.0) for q in qs]
        dg = [(c[q]["agg"].get("grounded") or 0.0) - (b[q]["agg"].get("grounded") or 0.0) for q in qs]
        da = [(c[q]["agg"].get("accuracy_score") or 0.0) - (b[q]["agg"].get("accuracy_score") or 0.0) for q in qs]
        lo, hi = bootstrap_ci(dh)
        decision.update({
            "paired_n": len(qs),
            "delta_h": mean(dh),
            "delta_h_ci95": [lo, hi],
            "delta_g": mean(dg),
            "delta_acc_score": mean(da),
        })
        lines += [
            "",
            "## Paired Base vs Candidate Preview",
            "",
            f"- paired n: {len(qs)}",
            f"- Δh: {mean(dh):+.3f}, bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]",
            f"- Δgrounded: {mean(dg):+.3f}",
            f"- Δacc_score: {mean(da):+.3f}",
        ]
        lines += ["", "## Trace Counts"]
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
    ap.add_argument("--base-judge", default=str(Path(OUTPUT_DIR) / "80_corrected_v1_rft_merged_base_judge.jsonl"))
    ap.add_argument("--candidate-infer", default=str(Path(OUTPUT_DIR) / "96_corrected_v2_mini_dpo_infer.jsonl"))
    ap.add_argument("--candidate-judge", default=str(Path(OUTPUT_DIR) / "96_corrected_v2_mini_dpo_judge.jsonl"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--k", type=int, default=int(os.environ.get("V3_JUDGE_K", "3")))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("V3_E0_LIMIT", "0")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("V3_KIMI_WORKERS", str(JUDGE_CALL_WORKERS))))
    ap.add_argument("--no-use-existing", action="store_true")
    args = ap.parse_args()

    odir = Path(args.out_dir) if args.out_dir else run_dir()
    out_jsonl = odir / "100_corrected_v3_noise_calibration.jsonl"
    report = odir / "100_corrected_v3_noise_calibration.md"
    decision_path = odir / "100_corrected_v3_noise_calibration_decision.json"

    base_infer = read_jsonl(args.base_infer)
    cand_infer = read_jsonl(args.candidate_infer)
    if not base_infer:
        raise SystemExit(f"missing base infer: {args.base_infer}")
    rows = merge_infer_and_existing(base_infer, read_jsonl(args.base_judge), "base", not args.no_use_existing)
    if cand_infer:
        rows += merge_infer_and_existing(cand_infer, read_jsonl(args.candidate_judge), "candidate", not args.no_use_existing)
    if args.limit > 0:
        rng = random.Random(19)
        by_model = {}
        for r in rows:
            by_model.setdefault(r["model"], []).append(r)
        rows = []
        for rs in by_model.values():
            rng.shuffle(rs)
            rows.extend(rs[:args.limit])

    print(f"RESULT e0_plan rows={len(rows)} k={args.k} workers={args.workers} out={out_jsonl}", flush=True)
    done = 0
    t0 = time.time()
    out_rows = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(judge_missing, r, args.k) for r in rows]
        for fut in cf.as_completed(futs):
            out_rows.append(fut.result())
            done += 1
            if done % 20 == 0 or done == len(rows):
                print(f"PROGRESS e0_judge {done}/{len(rows)} rate={done/max(time.time()-t0,1e-3):.2f}/s", flush=True)
    out_rows.sort(key=lambda r: (r["model"], r["qid"]))
    write_jsonl(out_jsonl, out_rows)
    decision = {
        "status": "complete",
        "k": args.k,
        "rows": len(out_rows),
        "inputs": {
            "base_infer": file_fingerprint(args.base_infer),
            "base_judge": file_fingerprint(args.base_judge),
            "candidate_infer": file_fingerprint(args.candidate_infer),
            "candidate_judge": file_fingerprint(args.candidate_judge),
        },
    }
    text = make_report(out_rows, decision, out_jsonl)
    report.write_text(text, encoding="utf-8")
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT e0_complete report={report}", flush=True)


if __name__ == "__main__":
    main()
