"""corrected-v3.1 E1: paired readout under the narrowed derag rubric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.v3_utils import bootstrap_ci, mean, mcnemar_counts, read_jsonl, row_agg, trace_counts


def make_report(base_rows: list[dict], cand_rows: list[dict], decision: dict) -> str:
    b = {r["qid"]: r for r in base_rows}
    c = {r["qid"]: r for r in cand_rows}
    qs = sorted(set(b) & set(c))
    pairs = [(b[q], c[q]) for q in qs]
    dt = [(row_agg(n).get("trace_free") or 0.0) - (row_agg(o).get("trace_free") or 0.0) for o, n in pairs]
    dg = [(row_agg(n).get("grounded") or 0.0) - (row_agg(o).get("grounded") or 0.0) for o, n in pairs]
    da = [(row_agg(n).get("accuracy_score") or 0.0) - (row_agg(o).get("accuracy_score") or 0.0) for o, n in pairs]
    lo, hi = bootstrap_ci(dt, n_boot=decision["bootstrap"])
    mc = mcnemar_counts([p[0] for p in pairs], [p[1] for p in pairs])
    b_tr = trace_counts([p[0] for p in pairs])
    c_tr = trace_counts([p[1] for p in pairs])
    trace_reduction = sum(c_tr.values()) - sum(b_tr.values())
    verdict = "GO" if lo > 0 and mean(dg) >= -0.01 and mean(da) >= -0.01 else "NO-GO"
    decision.update({
        "paired_n": len(pairs),
        "delta_trace_free": mean(dt),
        "delta_trace_free_ci95": [lo, hi],
        "delta_grounded": mean(dg),
        "delta_accuracy_score": mean(da),
        "mcnemar": mc,
        "base_trace_counts": dict(b_tr),
        "candidate_trace_counts": dict(c_tr),
        "trace_count_delta_total": trace_reduction,
        "verdict": verdict,
    })
    trace_lines = []
    for k in ("explicit_ref", "verbatim_copy", "ref_enumeration", "policy_source"):
        trace_lines.append(f"| {k} | {b_tr.get(k,0)} | {c_tr.get(k,0)} | {c_tr.get(k,0)-b_tr.get(k,0):+d} |")
    lines = [
        "# corrected-v3.1 E1 derag paired readout",
        "",
        f"- verdict: **{verdict}**",
        f"- paired n: {len(pairs)}",
        f"- Δtrace_free: {mean(dt):+.3f}, bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]",
        f"- Δgrounded: {mean(dg):+.3f}",
        f"- Δaccuracy_score: {mean(da):+.3f}",
        f"- trace count total Δ(candidate-base): {trace_reduction:+d}",
        f"- McNemar b(base correct -> candidate not)={mc['base_correct_new_not']}, c(candidate correct -> base not)={mc['new_correct_base_not']}, chi2_cc={mc['mcnemar_chi2_cc']:.3f}",
        "",
        "## Trace Counts",
        "",
        "| trace | base | candidate | Δ |",
        "|---|---:|---:|---:|",
        *trace_lines,
        "",
        "## Interpretation",
        "",
        "- GO means candidate has statistically readable trace-free improvement while grounded/accuracy guardrails hold.",
        "- NO-GO means do not train from this candidate signal yet; either the model did not improve trace removal, or the readout is still noisy.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    odir = Path(args.out_dir)
    rows = read_jsonl(args.calibration)
    if not rows:
        raise SystemExit(f"missing calibration rows: {args.calibration}")
    base = [r for r in rows if r.get("model") == "base"]
    cand = [r for r in rows if r.get("model") == "candidate"]
    if not base or not cand:
        raise SystemExit("paired readout requires base and candidate rows")

    decision = {"status": "complete", "calibration": args.calibration, "bootstrap": args.bootstrap}
    report = odir / "111_corrected_v31_derag_paired_readout.md"
    decision_path = odir / "111_corrected_v31_derag_paired_readout_decision.json"
    report.write_text(make_report(base, cand, decision), encoding="utf-8")
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT v31_e1_readout verdict={decision['verdict']} delta_trace_free={decision['delta_trace_free']:+.3f} report={report}", flush=True)


if __name__ == "__main__":
    main()
