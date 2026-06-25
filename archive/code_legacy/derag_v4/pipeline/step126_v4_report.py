"""derag_v4 deterministic stage report and accounting."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline import reward_v3  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402

log = get_logger("step126_v4_report")


def load_infer(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def features(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        text = r.get("gen_text") or f"<think>{r.get('think','')}</think><answer>{r.get('answer','')}</answer>"
        out.append(reward_v3.candidate_features(text, r.get("user_prompt", ""), r.get("gold_answer", ""), r.get("query", "")))
    return out


def summarize(fs: list[dict]) -> dict:
    n = len(fs)
    counts = Counter()
    for f in fs:
        counts.update(f.get("frozen_trace_counts") or {})
    clean = sum(1 for f in fs if f.get("clean"))
    return {
        "n": n,
        "clean_rate": round(clean / max(1, n), 4),
        "trace_total": sum(int(f.get("frozen_trace_total", 0)) for f in fs),
        "trace_counter_version": reward_v3.TRACE_RE_V3_VERSION,
        "trace_counts": {k: counts.get(k, 0) for k in reward_v3.TRACE_KEYS},
        "copy_ratio_mean": round(statistics.mean([float(f.get("copy_ratio", 0.0)) for f in fs]) if fs else 0.0, 4),
        "masked_copy_mean": round(statistics.mean([float(f.get("masked_copy", 0.0)) for f in fs]) if fs else 0.0, 4),
        "fact_recall_mean": round(statistics.mean([float(f.get("fact_recall", 0.0)) for f in fs]) if fs else 0.0, 4),
        "answer_score_mean": round(statistics.mean([float(f.get("answer_score", 0.0)) for f in fs]) if fs else 0.0, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", action="append", default=[], help="label=path")
    ap.add_argument("--stage", action="append", default=[], help="label=path")
    args = ap.parse_args()

    entries = []
    all_items = args.base + args.stage
    prev_label = prev_feats = None
    for item in all_items:
        label, path = item.split("=", 1)
        fs = features(load_infer(path))
        s = summarize(fs)
        row = {"label": label, "path": path, **s}
        if prev_feats is not None:
            row["mcnemar_vs_prev"] = reward_v3.mcnemar_net(prev_feats, fs)
            row["delta_clean_rate_vs_prev"] = round(s["clean_rate"] - entries[-1]["clean_rate"], 4)
            row["delta_trace_total_vs_prev"] = s["trace_total"] - entries[-1]["trace_total"]
        entries.append(row)
        prev_label, prev_feats = label, fs

    lines = ["# derag_v4 deterministic report", ""]
    lines += ["| model | clean_rate | trace_total | explicit | verbatim | enum | policy | fact_recall | answer_score | Δclean | Δtrace | McNemar net |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in entries:
        c = e["trace_counts"]
        mc = e.get("mcnemar_vs_prev", {})
        lines.append(
            f"| {e['label']} | {e['clean_rate']:.3f} | {e['trace_total']} | "
            f"{c.get('explicit_ref',0)} | {c.get('verbatim_copy',0)} | {c.get('ref_enumeration',0)} | {c.get('policy_source',0)} | "
            f"{e['fact_recall_mean']:.3f} | {e['answer_score_mean']:.3f} | "
            f"{e.get('delta_clean_rate_vs_prev','')} | {e.get('delta_trace_total_vs_prev','')} | {mc.get('net','')} |"
        )
    lines += ["", "## JSON", "", "```json", json.dumps(entries, ensure_ascii=False, indent=2), "```", ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    log.warning("RESULT v4_report -> %s", out)
    print(json.dumps({"status": "PASS", "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
