"""Zero-Kimi offline replay of the v4 deterministic gate and cached decisions."""

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

log = get_logger("step125b_replay_gate_dryrun")


def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()] if p.exists() else []


def assistant(think: str, answer: str) -> str:
    return f"<think>{think}</think><answer>{answer}</answer>"


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    vals = sorted(values)
    return {
        f"p{p}": round(vals[round((len(vals) - 1) * p / 100)], 4)
        for p in (10, 25, 50, 75, 90, 99)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites", required=True)
    ap.add_argument("--gate_rows", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rewrites = load_jsonl(args.rewrites)
    gate_rows = load_jsonl(args.gate_rows) if args.gate_rows else []
    features = []
    kills = Counter()
    for row in rewrites:
        f = reward_v3.candidate_features(
            assistant(row.get("natural_think", ""), row.get("answer", "")),
            row.get("user_prompt", ""),
            row.get("answer", ""),
            row.get("query", ""),
        )
        features.append(f)
        kills.update(f["l0_reasons"])
    entrants = [f for f in features if f["l0_pass"]]
    killed = [f for f in features if not f["l0_pass"]]
    paths = Counter(row.get("l3_path") for row in gate_rows)
    report = {
        "status": "PASS",
        "trace_version": reward_v3.TRACE_RE_V4_VERSION,
        "l0": {
            "input": len(features),
            "entrants": len(entrants),
            "killed": len(killed),
            "kills_by_reason": dict(kills),
            "img_kills": sum(f["img_trace"] for f in killed),
        },
        "l1_quantiles": {
            "entrant_masked_copy": quantiles([f["masked_copy"] for f in entrants]),
            "killed_masked_copy": quantiles([f["masked_copy"] for f in killed]),
            "entrant_citation_density": quantiles([f["citation_density"] for f in entrants]),
            "entrant_burden": quantiles([float(f["burden"]) for f in entrants]),
        },
        "cached_l3": {"rows": len(gate_rows), "paths": dict(paths)},
        "dpo_preview": {
            "source_burden_distribution": dict(Counter(f["burden"] for f in features)),
            "fast_chosen": sum(reward_v3.gate_decision(f, "dpo_chosen")["pass"] for f in features),
        },
        "means": {
            "raw_copy": round(statistics.mean(f["copy_ratio"] for f in features), 4) if features else 0.0,
            "masked_copy": round(statistics.mean(f["masked_copy"] for f in features), 4) if features else 0.0,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.warning("RESULT gate_replay entrants=%d/%d -> %s", len(entrants), len(features), out)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
