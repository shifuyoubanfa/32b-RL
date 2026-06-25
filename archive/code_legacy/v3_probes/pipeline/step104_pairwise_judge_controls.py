"""corrected-v3 E3: pairwise judge negative controls.

Before using pairwise Kimi preference as a training/eval signal, measure whether
it has position bias on identical texts and length bias on same-answer variants.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_CALL_WORKERS, OUTPUT_DIR
from pipeline.judge_common import pairwise_judge
from pipeline.reward import parse_think_answer
from pipeline.v3_utils import compact_len, file_fingerprint, qid_for, read_jsonl, run_dir, write_jsonl


def shorten_text(text: str) -> str:
    think, answer = parse_think_answer(text)
    parts = re.split(r"(?<=[。！？!?])", think)
    keep = "".join(parts[: max(1, len(parts) // 2)]).strip() or think[: max(80, len(think)//2)]
    return f"<think>\n{keep}\n</think>\n\n<answer>\n{answer}\n</answer>"


def judge_identical(row: dict) -> dict:
    text = row.get("gen_text") or ""
    ab = pairwise_judge(row.get("query",""), row.get("user_prompt",""), row.get("gold_answer",""), text, text)
    ba = pairwise_judge(row.get("query",""), row.get("user_prompt",""), row.get("gold_answer",""), text, text)
    return {"control": "identical", "qid": qid_for(row), "query": row.get("query"), "ab": ab, "ba": ba}


def judge_length(row: dict) -> dict:
    long = row.get("gen_text") or ""
    short = shorten_text(long)
    if random.Random(qid_for(row)).random() < 0.5:
        a, b, short_pos = short, long, "A"
    else:
        a, b, short_pos = long, short, "B"
    res = pairwise_judge(row.get("query",""), row.get("user_prompt",""), row.get("gold_answer",""), a, b)
    winner = res.get("winner")
    short_wins = winner == short_pos
    return {
        "control": "length",
        "qid": qid_for(row),
        "query": row.get("query"),
        "short_pos": short_pos,
        "winner": winner,
        "short_wins": short_wins,
        "long_len": compact_len(long),
        "short_len": compact_len(short),
        "judge": res,
    }


def make_report(rows: list[dict], decision: dict) -> str:
    ident = [r for r in rows if r["control"] == "identical"]
    length = [r for r in rows if r["control"] == "length"]
    ident_winners = Counter()
    for r in ident:
        ident_winners[r["ab"].get("winner")] += 1
        ident_winners[r["ba"].get("winner")] += 1
    length_winners = Counter(r.get("winner") for r in length)
    short_win = sum(1 for r in length if r.get("short_wins"))
    short_rate = short_win / max(1, len(length))
    non_tie_ident = ident_winners.get("A", 0) + ident_winners.get("B", 0)
    ident_bias = non_tie_ident / max(1, sum(ident_winners.values()))
    decision.update({
        "identical_pairs": len(ident),
        "identical_winners": dict(ident_winners),
        "identical_non_tie_rate": ident_bias,
        "length_pairs": len(length),
        "length_winners": dict(length_winners),
        "short_win_rate": short_rate,
        "status": "PASS" if ident_bias <= 0.20 and 0.40 <= short_rate <= 0.60 else "BIAS-SUSPECTED",
    })
    lines = [
        "# corrected-v3 E3 pairwise judge controls",
        "",
        f"- status: **{decision['status']}**",
        f"- identical non-tie rate: {100*ident_bias:.1f}% (should be low)",
        f"- identical winners: {dict(ident_winners)}",
        f"- short win rate in length control: {100*short_rate:.1f}% (healthy around 50%)",
        f"- length winners: {dict(length_winners)}",
        "",
        "If this reports BIAS-SUSPECTED, pairwise judge must be calibrated or avoided for hard gates.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-judge", default=str(Path(OUTPUT_DIR) / "80_corrected_v1_rft_merged_base_judge.jsonl"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--identical-n", type=int, default=50)
    ap.add_argument("--length-n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=JUDGE_CALL_WORKERS)
    args = ap.parse_args()
    odir = Path(args.out_dir) if args.out_dir else run_dir()
    rows = read_jsonl(args.base_judge)
    if not rows:
        raise SystemExit(f"missing base judge: {args.base_judge}")
    random.Random(53).shuffle(rows)
    ident_rows = rows[:args.identical_n]
    len_rows = rows[args.identical_n:args.identical_n + args.length_n]
    tasks = [(judge_identical, r) for r in ident_rows] + [(judge_length, r) for r in len_rows]
    out_jsonl = odir / "104_corrected_v3_pairwise_controls.jsonl"
    report = odir / "104_corrected_v3_pairwise_controls.md"
    decision_path = odir / "104_corrected_v3_pairwise_controls_decision.json"
    print(f"RESULT e3_plan tasks={len(tasks)} workers={args.workers}", flush=True)
    done = 0
    t0 = time.time()
    out = []
    with cf.ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futs = [pool.submit(fn, r) for fn, r in tasks]
        for fut in cf.as_completed(futs):
            out.append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"PROGRESS e3_pairwise {done}/{len(tasks)} rate={done/max(time.time()-t0,1e-3):.2f}/s", flush=True)
    decision = {"input": file_fingerprint(args.base_judge)}
    write_jsonl(out_jsonl, out)
    report.write_text(make_report(out, decision), encoding="utf-8")
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT e3_complete status={decision['status']} report={report}", flush=True)


if __name__ == "__main__":
    main()
