"""Kimi audit for old DPO pairs.

This is corrected-v2 Phase A. It samples existing DPO pairs, asks the same Kimi
judge to score chosen and rejected responses, and reports whether the pair
direction is actually aligned with Kimi humanness.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    DPO_PAIRS,
    DPO_ROLLOUT,
    KIMI_API_KEY_ENV,
    OUTPUT_DIR,
    REWARD_TAU_ACC,
    THINK_MAX_CHARS,
    THINK_MIN_CHARS,
)
from pipeline import reward
from pipeline.judge_common import judge_text
from pipeline.logger import get_logger
from pipeline.vllm_client import map_concurrent

log = get_logger("step91_audit_dpo_pairs_kimi")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(xs: list[float | None]) -> float:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return sum(vals) / len(vals) if vals else 0.0


def pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xbar = sum(x for x, _ in pairs) / len(pairs)
    ybar = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - xbar) * (y - ybar) for x, y in pairs)
    denx = math.sqrt(sum((x - xbar) ** 2 for x, _ in pairs))
    deny = math.sqrt(sum((y - ybar) ** 2 for _, y in pairs))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def ranks(vals: list[float]) -> list[float]:
    order = sorted(enumerate(vals), key=lambda p: p[1])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[order[k][0]] = rank
        i = j
    return out


def spearman(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def fmt(x: float | None, digits: int = 3) -> str:
    return "NA" if x is None else f"{x:.{digits}f}"


def sfmt(x: float | None, digits: int = 3) -> str:
    return "NA" if x is None else f"{x:+.{digits}f}"


def short(text: str | None, n: int = 180) -> str:
    s = " ".join((text or "").split())
    return s[:n] + ("..." if len(s) > n else "")


def acc_rank(label: str | None) -> int:
    return {"incorrect": 0, "partial": 1, "correct": 2}.get(label or "incorrect", 0)


def assistant_text(pair: dict) -> str:
    for msg in reversed(pair.get("messages") or []):
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return ""


def user_prompt(pair: dict) -> str:
    for msg in pair.get("messages") or []:
        if msg.get("role") == "user":
            return msg.get("content") or ""
    return ""


def score_local(text: str, up: str, gold: str) -> dict:
    return reward.score_rollout(
        text,
        up,
        gold,
        tau_acc=REWARD_TAU_ACC,
        think_min=THINK_MIN_CHARS,
        think_max=THINK_MAX_CHARS,
        s_pmi=None,
    )


def rollout_gold_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for rec in read_jsonl(path):
        q = rec.get("query")
        if q and q not in out:
            out[q] = {
                "gold_answer": rec.get("gold_answer") or "",
                "user_prompt": rec.get("user_prompt") or "",
            }
    return out


def build_entry(pair_id: int, pair: dict, gold_by_query: dict[str, dict]) -> dict:
    q = pair.get("query") or ""
    meta = gold_by_query.get(q, {})
    up = user_prompt(pair) or meta.get("user_prompt") or ""
    gold = meta.get("gold_answer") or ""
    chosen = assistant_text(pair)
    rejected = pair.get("rejected_response") or ""
    chosen_local = score_local(chosen, up, gold)
    rejected_local = score_local(rejected, up, gold)
    return {
        "pair_id": pair_id,
        "query": q,
        "user_prompt": up,
        "gold_answer": gold,
        "chosen_text": chosen,
        "rejected_text": rejected,
        "chosen_local": chosen_local,
        "rejected_local": rejected_local,
        "local_margin_reward": (chosen_local.get("reward") or 0.0) - (rejected_local.get("reward") or 0.0),
        "local_margin_R_human": (chosen_local.get("R_human") or 0.0) - (rejected_local.get("R_human") or 0.0),
        "local_margin_R_acc": (chosen_local.get("R_acc") or 0.0) - (rejected_local.get("R_acc") or 0.0),
    }


def audit_entry(entry: dict, local_only: bool) -> dict:
    if local_only:
        return entry
    out = dict(entry)
    try:
        cj = judge_text(entry["query"], entry["user_prompt"], entry["gold_answer"], entry["chosen_text"])
    except Exception as e:
        cj = {"accuracy": "incorrect", "accuracy_score": 0.0, "humanness": 0.0, "grounded": 0.0,
              "rag_traces": [], "comment": f"judge_error:{e}"}
    try:
        rj = judge_text(entry["query"], entry["user_prompt"], entry["gold_answer"], entry["rejected_text"])
    except Exception as e:
        rj = {"accuracy": "incorrect", "accuracy_score": 0.0, "humanness": 0.0, "grounded": 0.0,
              "rag_traces": [], "comment": f"judge_error:{e}"}
    out["chosen_judge"] = cj
    out["rejected_judge"] = rj
    out["kimi_margin_humanness"] = (cj.get("humanness") or 0.0) - (rj.get("humanness") or 0.0)
    out["kimi_margin_grounded"] = (cj.get("grounded") or 0.0) - (rj.get("grounded") or 0.0)
    out["kimi_margin_acc_score"] = (cj.get("accuracy_score") or 0.0) - (rj.get("accuracy_score") or 0.0)
    out["kimi_margin_acc_rank"] = acc_rank(cj.get("accuracy")) - acc_rank(rj.get("accuracy"))
    return out


def make_report(rows: list[dict], local_only: bool, margin: float) -> str:
    n = len(rows)
    lines = [
        "# corrected-v2 Phase A / step91 DPO pair Kimi audit",
        "",
        f"- audited pairs: {n}",
        f"- local_only: {local_only}",
        "",
        "## Local Pair Margins",
        "",
        "| metric | mean margin chosen-rejected |",
        "|---|---:|",
        f"| local reward | {mean([r.get('local_margin_reward') for r in rows]):.3f} |",
        f"| local R_human | {mean([r.get('local_margin_R_human') for r in rows]):.3f} |",
        f"| local R_acc | {mean([r.get('local_margin_R_acc') for r in rows]):.3f} |",
    ]
    if local_only:
        lines += [
            "",
            "Kimi audit was skipped by --local-only. Re-run without --local-only on the server.",
        ]
        return "\n".join(lines) + "\n"

    h_m = [r.get("kimi_margin_humanness") for r in rows]
    g_m = [r.get("kimi_margin_grounded") for r in rows]
    a_m = [r.get("kimi_margin_acc_score") for r in rows]
    rank_m = [r.get("kimi_margin_acc_rank") for r in rows]
    chosen_j = [r.get("chosen_judge") or {} for r in rows]
    rejected_j = [r.get("rejected_judge") or {} for r in rows]
    h_win = sum(1 for x in h_m if x is not None and x > 0)
    h_strong = sum(1 for x in h_m if x is not None and x >= margin)
    h_loss = sum(1 for x in h_m if x is not None and x < 0)
    acc_loss = sum(1 for x in rank_m if x is not None and x < 0)
    sp, pe = spearman([r.get("local_margin_reward") for r in rows], h_m), pearson([r.get("local_margin_reward") for r in rows], h_m)

    lines += [
        "",
        "## Kimi Pair Direction",
        "",
        "| metric | chosen mean | rejected mean | mean margin | win rate |",
        "|---|---:|---:|---:|---:|",
        f"| humanness | {mean([j.get('humanness') for j in chosen_j]):.3f} | {mean([j.get('humanness') for j in rejected_j]):.3f} | {mean(h_m):+.3f} | {100*h_win/max(1,n):.1f}% |",
        f"| grounded | {mean([j.get('grounded') for j in chosen_j]):.3f} | {mean([j.get('grounded') for j in rejected_j]):.3f} | {mean(g_m):+.3f} | {100*sum(1 for x in g_m if x is not None and x > 0)/max(1,n):.1f}% |",
        f"| accuracy_score | {mean([j.get('accuracy_score') for j in chosen_j]):.3f} | {mean([j.get('accuracy_score') for j in rejected_j]):.3f} | {mean(a_m):+.3f} | {100*sum(1 for x in a_m if x is not None and x > 0)/max(1,n):.1f}% |",
        "",
        "## Readout",
        "",
        f"- Strong Kimi humanness-aligned pairs (chosen - rejected >= {margin:.2f}): {h_strong}/{n} ({100*h_strong/max(1,n):.1f}%).",
        f"- Kimi humanness losses (chosen worse than rejected): {h_loss}/{n} ({100*h_loss/max(1,n):.1f}%).",
        f"- Kimi accuracy-rank losses: {acc_loss}/{n} ({100*acc_loss/max(1,n):.1f}%).",
        f"- Local reward margin vs Kimi humanness margin: Spearman/Pearson = {fmt(sp)}/{fmt(pe)}.",
        "- If strong aligned pairs are scarce, the old DPO data is a weak or noisy preference signal.",
        "",
    ]

    worst = sorted(rows, key=lambda r: r.get("kimi_margin_humanness") or 0.0)[:12]
    lines += ["## Worst Pair Direction Examples", ""]
    for r in worst:
        cj = r.get("chosen_judge") or {}
        rj = r.get("rejected_judge") or {}
        lines += [
            f"### pair_id={r['pair_id']}",
            f"- query: {short(r.get('query'), 260)}",
            f"- margins: Kimi_h={sfmt(r.get('kimi_margin_humanness'))} Kimi_acc={sfmt(r.get('kimi_margin_acc_score'))} local_reward={sfmt(r.get('local_margin_reward'))} local_h={sfmt(r.get('local_margin_R_human'))} local_acc={sfmt(r.get('local_margin_R_acc'))}",
            f"- chosen judge: h={cj.get('humanness')} g={cj.get('grounded')} acc={cj.get('accuracy')}({cj.get('accuracy_score')}) comment={short(cj.get('comment'), 220)}",
            f"- rejected judge: h={rj.get('humanness')} g={rj.get('grounded')} acc={rj.get('accuracy')}({rj.get('accuracy_score')}) comment={short(rj.get('comment'), 220)}",
            f"- chosen think: {short((r.get('chosen_local') or {}).get('think'), 320)}",
            f"- rejected think: {short((r.get('rejected_local') or {}).get('think'), 320)}",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=DPO_PAIRS)
    ap.add_argument("--rollout", default=DPO_ROLLOUT, help="Used to recover gold_answer by query.")
    ap.add_argument("--out-jsonl", default=str(Path(OUTPUT_DIR) / "91_corrected_v2_dpo_pair_kimi_audit.jsonl"))
    ap.add_argument("--out-md", default=str(Path(OUTPUT_DIR) / "91_corrected_v2_dpo_pair_kimi_audit.md"))
    ap.add_argument("--limit", type=int, default=150, help="0 means all pairs.")
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("DPO_PAIR_AUDIT_WORKERS", "1")))
    ap.add_argument("--margin", type=float, default=0.25, help="Kimi humanness margin treated as a strong aligned pair.")
    ap.add_argument("--local-only", action="store_true", help="Do not call Kimi; useful for local smoke tests.")
    args = ap.parse_args()

    pair_path = Path(args.pairs)
    rollout_path = Path(args.rollout)
    out_jsonl = Path(args.out_jsonl)
    out_md = Path(args.out_md)
    if not pair_path.is_absolute():
        pair_path = Path(OUTPUT_DIR) / pair_path
    if not rollout_path.is_absolute():
        rollout_path = Path(OUTPUT_DIR) / rollout_path

    if not args.local_only and not os.environ.get(KIMI_API_KEY_ENV, "").strip():
        raise RuntimeError(f"Missing {KIMI_API_KEY_ENV}. Use --local-only for a smoke test, or export the key.")

    pairs = read_jsonl(pair_path)
    ids = list(range(len(pairs)))
    random.Random(args.seed).shuffle(ids)
    if args.limit and args.limit > 0:
        ids = ids[: args.limit]
    ids = sorted(ids)
    gold_by_query = rollout_gold_map(rollout_path)
    if not gold_by_query:
        log.warning("rollout gold map is empty: %s", rollout_path)
    entries = [build_entry(i, pairs[i], gold_by_query) for i in ids]
    log.info("audit DPO pairs: %d sampled from %d, local_only=%s", len(entries), len(pairs), args.local_only)

    rows = map_concurrent(entries, lambda e: audit_entry(e, args.local_only), workers=max(1, args.workers), desc="dpo_pair_audit")
    write_jsonl(out_jsonl, rows)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(make_report(rows, args.local_only, args.margin), encoding="utf-8")
    log.info("wrote rows: %s", out_jsonl)
    log.info("wrote report: %s", out_md)


if __name__ == "__main__":
    main()
