"""corrected-v3 E0 readout: paired delta, bootstrap CI, McNemar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR
from pipeline.v3_utils import (
    acc_tier,
    bootstrap_ci,
    mean,
    mcnemar_counts,
    read_jsonl,
    row_agg,
    run_dir,
    trace_counts,
)


def make_report(base_rows: list[dict], cand_rows: list[dict], decision: dict) -> str:
    b = {r["qid"]: r for r in base_rows}
    c = {r["qid"]: r for r in cand_rows}
    qs = sorted(set(b) & set(c))
    pairs = [(b[q], c[q]) for q in qs]
    dh = [(row_agg(n).get("humanness") or 0.0) - (row_agg(o).get("humanness") or 0.0) for o, n in pairs]
    dg = [(row_agg(n).get("grounded") or 0.0) - (row_agg(o).get("grounded") or 0.0) for o, n in pairs]
    da = [(row_agg(n).get("accuracy_score") or 0.0) - (row_agg(o).get("accuracy_score") or 0.0) for o, n in pairs]
    lo, hi = bootstrap_ci(dh, n_boot=decision["bootstrap"])
    mc = mcnemar_counts([p[0] for p in pairs], [p[1] for p in pairs])
    b_tr = trace_counts([p[0] for p in pairs])
    c_tr = trace_counts([p[1] for p in pairs])
    decision.update({
        "paired_n": len(pairs),
        "delta_h": mean(dh),
        "delta_h_ci95": [lo, hi],
        "delta_grounded": mean(dg),
        "delta_acc_score": mean(da),
        "mcnemar": mc,
        "base_trace_counts": dict(b_tr),
        "candidate_trace_counts": dict(c_tr),
        "verdict": "GO" if lo > 0 and mean(dg) >= -0.01 and mean(da) >= -0.01 else "NO-GO",
    })
    trace_lines = []
    for k in ("explicit_ref", "verbatim_copy", "ref_enumeration", "policy_source"):
        trace_lines.append(f"| {k} | {b_tr.get(k,0)} | {c_tr.get(k,0)} | {c_tr.get(k,0)-b_tr.get(k,0):+d} |")
    lines = [
        "# corrected-v3 E0 paired eval readout",
        "",
        f"- verdict: **{decision['verdict']}**",
        f"- paired n: {len(pairs)}",
        f"- Δhumanness: {mean(dh):+.3f}, bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]",
        f"- Δgrounded: {mean(dg):+.3f}",
        f"- Δaccuracy_score: {mean(da):+.3f}",
        f"- McNemar b(base correct -> new not)={mc['base_correct_new_not']}, c(new correct -> base not)={mc['new_correct_base_not']}, chi2_cc={mc['mcnemar_chi2_cc']:.3f}",
        "",
        "## Trace Counts",
        "",
        "| trace | base | candidate | Δ |",
        "|---|---:|---:|---:|",
        *trace_lines,
        "",
        "## Interpretation",
        "",
        "- GO requires the lower bound of Δhumanness CI to be above 0 and guardrails not to drop.",
        "- If CI crosses 0, report no-evidence rather than improvement/regression.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    odir = Path(args.out_dir) if args.out_dir else run_dir()
    cal = Path(args.calibration) if args.calibration else odir / "100_corrected_v3_noise_calibration.jsonl"
    rows = read_jsonl(cal)
    if not rows:
        raise SystemExit(f"missing calibration rows: {cal}")
    base = [r for r in rows if r.get("model") == "base"]
    cand = [r for r in rows if r.get("model") == "candidate"]
    if not base or not cand:
        raise SystemExit("paired readout requires base and candidate rows")
    decision = {"status": "complete", "calibration": str(cal), "bootstrap": args.bootstrap}
    report = odir / "101_corrected_v3_mini_paired_readout.md"
    decision_path = odir / "101_corrected_v3_mini_paired_readout_decision.json"
    report.write_text(make_report(base, cand, decision), encoding="utf-8")
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT e0_readout verdict={decision['verdict']} delta_h={decision['delta_h']:+.3f} report={report}", flush=True)


if __name__ == "__main__":
    main()
