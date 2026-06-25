"""G1-1 pool density probe for derag_v4.

The probe is deliberately Kimi-free.  It measures whether the current policy
distribution contains enough clean candidates and same-query contrast pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline import reward_v3  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402

log = get_logger("step121_pool_density_probe")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="s1")
    ap.add_argument("--p_clean_min", type=float, default=0.60)
    ap.add_argument("--p_pair_min", type=float, default=0.40)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.rollout).open(encoding="utf-8") if l.strip()]
    eval_rows = []
    n_clean_q = n_pair_q = n_trace_q = 0
    total_cands = clean_cands = trace_cands = 0
    for r in rows:
        feats = []
        up = r.get("user_prompt") or ""
        gold = r.get("gold_answer") or r.get("answer") or ""
        q = r.get("query") or ""
        for i, cand in enumerate(r.get("candidates") or []):
            f = reward_v3.candidate_features(cand, up, gold, q)
            f = {k: v for k, v in f.items() if k != "text"}
            f["cand_id"] = i
            feats.append(f)
        has_clean = any(f.get("clean") for f in feats)
        has_trace = any(f.get("trace_heavy") for f in feats)
        total_cands += len(feats)
        clean_cands += sum(1 for f in feats if f.get("clean"))
        trace_cands += sum(1 for f in feats if f.get("trace_heavy"))
        n_clean_q += int(has_clean)
        n_trace_q += int(has_trace)
        n_pair_q += int(has_clean and has_trace)
        eval_rows.append({
            "query": q,
            "n_candidates": len(feats),
            "has_clean": has_clean,
            "has_trace_heavy": has_trace,
            "has_pair": has_clean and has_trace,
            "clean_count": sum(1 for f in feats if f.get("clean")),
            "trace_heavy_count": sum(1 for f in feats if f.get("trace_heavy")),
            "features": feats,
        })
    n = len(rows)
    p_clean = n_clean_q / max(1, n)
    p_pair = n_pair_q / max(1, n)
    status = "GO" if p_clean >= args.p_clean_min and p_pair >= args.p_pair_min else "NO-GO"
    decision = {
        "status": status,
        "mode": args.mode,
        "rollout": args.rollout,
        "queries": n,
        "total_candidates": total_cands,
        "clean_candidates": clean_cands,
        "trace_heavy_candidates": trace_cands,
        "p_clean": round(p_clean, 4),
        "p_pair": round(p_pair, 4),
        "p_trace": round(n_trace_q / max(1, n), 4),
        "thresholds": {"p_clean_min": args.p_clean_min, "p_pair_min": args.p_pair_min},
        "n_kimi_calls": 0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    rows_path = out.with_suffix(".rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.warning("RESULT pool_density status=%s p_clean=%.3f p_pair=%.3f clean_cands=%d/%d -> %s",
                status, p_clean, p_pair, clean_cands, total_cands, out)
    print(json.dumps(decision, ensure_ascii=False))
    if status != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
