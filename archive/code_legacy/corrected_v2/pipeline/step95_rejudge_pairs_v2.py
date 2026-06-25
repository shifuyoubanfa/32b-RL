"""corrected-v2 Phase 1: second-judge a sample of v2 DPO pairs."""

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR
from pipeline.judge_common import judge_text


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


def acc_tier(label: str | None) -> int:
    return {"incorrect": 0, "partial": 1, "correct": 2}.get(label or "incorrect", 0)


def rejudge(row: dict) -> dict:
    out = dict(row)
    try:
        cj = judge_text(row["query"], row["user_prompt"], row["gold_answer"], row["chosen_text"])
        rj = judge_text(row["query"], row["user_prompt"], row["gold_answer"], row["rejected_text"])
    except Exception as exc:
        out.update({
            "rejudge_error": repr(exc),
            "rejudge_direction_ok": False,
            "rejudge_h_margin": 0.0,
            "rejudge_g_margin": 0.0,
            "rejudge_acc_tier_margin": -1,
        })
        return out
    h_margin = (cj.get("humanness") or 0.0) - (rj.get("humanness") or 0.0)
    g_margin = (cj.get("grounded") or 0.0) - (rj.get("grounded") or 0.0)
    acc_margin = acc_tier(cj.get("accuracy")) - acc_tier(rj.get("accuracy"))
    out.update({
        "rejudge_chosen": cj,
        "rejudge_rejected": rj,
        "rejudge_h_margin": h_margin,
        "rejudge_g_margin": g_margin,
        "rejudge_acc_tier_margin": acc_margin,
        "rejudge_direction_ok": h_margin > 0 and acc_margin >= 0 and g_margin >= -0.05,
    })
    return out


def make_report(rows: list[dict], threshold: float) -> tuple[str, dict]:
    n = len(rows)
    ok = sum(1 for r in rows if r.get("rejudge_direction_ok"))
    rate = ok / max(1, n)
    h_mean = sum(r.get("rejudge_h_margin", 0.0) for r in rows) / max(1, n)
    acc_loss = sum(1 for r in rows if r.get("rejudge_acc_tier_margin", 0) < 0)
    decision = {
        "status": "GO" if rate >= threshold else "NO-GO",
        "sampled_pairs": n,
        "rejudged_pairs": n,
        "direction_ok": ok,
        "direction_rate": rate,
        "threshold": threshold,
        "mean_h_margin": h_mean,
        "acc_rank_losses": acc_loss,
    }
    lines = [
        "# corrected-v2 Phase 1 / step95 pair rejudge report",
        "",
        f"- status: **{decision['status']}**",
        f"- rejudged pairs: {n}",
        f"- direction kept: {ok}/{n} ({100*rate:.1f}%)",
        f"- threshold: {100*threshold:.1f}%",
        f"- mean rejudge humanness margin: {h_mean:+.3f}",
        f"- accuracy-rank losses: {acc_loss}",
        "",
        "Direction is counted as OK when chosen keeps higher humanness, accuracy tier does not get worse, and grounded does not drop by more than 0.05.",
    ]
    return "\n".join(lines) + "\n", decision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=str(Path(OUTPUT_DIR) / "94_corrected_v2_dpo_pairs_meta.jsonl"))
    ap.add_argument("--out", default=str(Path(OUTPUT_DIR) / "95_corrected_v2_pair_rejudge.jsonl"))
    ap.add_argument("--report", default=str(Path(OUTPUT_DIR) / "95_corrected_v2_pair_rejudge_report.md"))
    ap.add_argument("--decision", default=str(Path(OUTPUT_DIR) / "95_corrected_v2_pair_rejudge_decision.json"))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("V2_REJUDGE_PAIRS", "0")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("V2_KIMI_WORKERS", "3")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("V2_REJUDGE_SEED", "17")))
    ap.add_argument("--threshold", type=float, default=float(os.environ.get("V2_REJUDGE_THRESHOLD", "0.70")))
    args = ap.parse_args()

    rows = read_jsonl(Path(args.meta))
    if not rows:
        raise SystemExit(f"missing pair meta: {args.meta}")
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit] if args.limit > 0 else rows
    print(f"RESULT rejudge_plan pairs={len(rows)} workers={args.workers}", flush=True)
    done = 0
    t0 = time.time()
    out_rows = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(rejudge, r) for r in rows]
        for fut in cf.as_completed(futs):
            out_rows.append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(rows):
                rate = done / max(time.time() - t0, 1e-3)
                print(f"PROGRESS rejudge_pairs {done}/{len(rows)} rate={rate:.2f}/s", flush=True)
    report, decision = make_report(out_rows, args.threshold)
    write_jsonl(Path(args.out), out_rows)
    Path(args.report).write_text(report, encoding="utf-8")
    Path(args.decision).write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"RESULT rejudge_complete status={decision['status']} direction={100*decision['direction_rate']:.1f}% report={args.report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
