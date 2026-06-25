"""Offline audit: local reward components vs Kimi judge scores.

This is corrected-v2 Phase A. It does not call any model and does not train.
It reads existing *_judge.jsonl files, recomputes the local reward components,
and measures whether they align with the independent Kimi judge dimensions.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR, REWARD_TAU_ACC, THINK_MIN_CHARS, THINK_MAX_CHARS
from pipeline import reward
from pipeline.logger import get_logger

log = get_logger("step90_audit_reward_alignment")


DEFAULT_JUDGES = [
    ("RFT merged base", "80_corrected_v1_rft_merged_base_judge.jsonl"),
    ("DPO on merged base", "81_corrected_v1_dpo_on_rft_merged_judge.jsonl"),
    ("GRPO from RFT merged", "82_corrected_v1_grpo_from_rft_merged_judge.jsonl"),
    ("GRPO from DPO merged", "83_corrected_v1_grpo_from_dpo_merged_judge.jsonl"),
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(xs: list[float | None]) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(float(x))]
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


def acc_rank(label: str | None) -> int:
    return {"incorrect": 0, "partial": 1, "correct": 2}.get(label or "incorrect", 0)


def full_text(rec: dict) -> str:
    if rec.get("gen_text"):
        return rec["gen_text"]
    return (
        "<think>\n"
        + (rec.get("think") or "").strip()
        + "\n</think>\n\n<answer>\n"
        + (rec.get("answer") or "").strip()
        + "\n</answer>"
    )


def short(text: str | None, n: int = 180) -> str:
    s = " ".join((text or "").split())
    return s[:n] + ("..." if len(s) > n else "")


def parse_judge_arg(item: str) -> tuple[str, Path]:
    if "=" in item:
        label, path = item.split("=", 1)
    else:
        path = item
        label = Path(path).stem.replace("_judge", "")
    p = Path(path)
    if not p.is_absolute():
        p = Path(OUTPUT_DIR) / p
    return label, p


def row_from_record(label: str, idx: int, rec: dict) -> dict:
    j = rec.get("judge") or {}
    sc = reward.score_rollout(
        full_text(rec),
        rec.get("user_prompt") or "",
        rec.get("gold_answer") or "",
        tau_acc=REWARD_TAU_ACC,
        think_min=THINK_MIN_CHARS,
        think_max=THINK_MAX_CHARS,
        s_pmi=None,
    )
    return {
        "model": label,
        "idx": idx,
        "query": rec.get("query"),
        "local_reward": sc.get("reward"),
        "local_R_human": sc.get("R_human"),
        "local_R_acc": sc.get("R_acc"),
        "local_gate": sc.get("gate"),
        "local_format_ok": sc.get("format_ok"),
        "local_trace_hits": sc.get("h_trace_hits"),
        "local_copy_ratio": sc.get("h_copy_ratio"),
        "local_answer_sim": sc.get("a_sim"),
        "local_fact_recall": sc.get("a_fact_recall"),
        "kimi_humanness": j.get("humanness"),
        "kimi_grounded": j.get("grounded"),
        "kimi_accuracy_score": j.get("accuracy_score"),
        "kimi_accuracy": j.get("accuracy"),
        "kimi_acc_rank": acc_rank(j.get("accuracy")),
        "kimi_rag_traces": j.get("rag_traces") or [],
        "kimi_comment": j.get("comment"),
        "think": sc.get("think") or rec.get("think") or "",
        "answer": sc.get("answer") or rec.get("answer") or "",
    }


def corr_block(rows: list[dict], xkey: str, ykey: str) -> tuple[str, str]:
    xs = [r.get(xkey) for r in rows]
    ys = [r.get(ykey) for r in rows]
    return fmt(spearman(xs, ys)), fmt(pearson(xs, ys))


def make_report(rows: list[dict], judges: list[tuple[str, Path]], regression_n: int) -> str:
    labels = [label for label, _ in judges]
    lines = [
        "# corrected-v2 Phase A / step90 reward alignment audit",
        "",
        "Purpose: verify whether the local training reward agrees with independent Kimi judging.",
        "",
        "## Inputs",
        "",
        "| label | file |",
        "|---|---|",
    ]
    for label, path in judges:
        lines.append(f"| {label} | `{path}` |")

    lines += [
        "",
        "## Means",
        "",
        "| model | n | local_reward | local_R_human | local_R_acc | Kimi_h | Kimi_grounded | Kimi_acc | correct% | c+p% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        rs = [r for r in rows if r["model"] == label]
        n = len(rs)
        correct = sum(1 for r in rs if r["kimi_accuracy"] == "correct")
        partial = sum(1 for r in rs if r["kimi_accuracy"] == "partial")
        lines.append(
            f"| {label} | {n} | {mean([r['local_reward'] for r in rs]):.3f} | "
            f"{mean([r['local_R_human'] for r in rs]):.3f} | {mean([r['local_R_acc'] for r in rs]):.3f} | "
            f"{mean([r['kimi_humanness'] for r in rs]):.3f} | {mean([r['kimi_grounded'] for r in rs]):.3f} | "
            f"{mean([r['kimi_accuracy_score'] for r in rs]):.3f} | "
            f"{100*correct/max(1,n):.1f}% | {100*(correct+partial)/max(1,n):.1f}% |"
        )

    lines += [
        "",
        "## Spearman / Pearson Correlations",
        "",
        "| model | local_reward~Kimi_h | R_human~Kimi_h | R_acc~Kimi_acc | R_acc~Kimi_h | copy_ratio~Kimi_h | trace_hits~Kimi_h |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        rs = [r for r in rows if r["model"] == label]
        vals = []
        for x, y in [
            ("local_reward", "kimi_humanness"),
            ("local_R_human", "kimi_humanness"),
            ("local_R_acc", "kimi_accuracy_score"),
            ("local_R_acc", "kimi_humanness"),
            ("local_copy_ratio", "kimi_humanness"),
            ("local_trace_hits", "kimi_humanness"),
        ]:
            sp, pe = corr_block(rs, x, y)
            vals.append(f"{sp}/{pe}")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    all_reward_h, all_reward_h_p = corr_block(rows, "local_reward", "kimi_humanness")
    all_rhuman_h, all_rhuman_h_p = corr_block(rows, "local_R_human", "kimi_humanness")
    all_racc_acc, all_racc_acc_p = corr_block(rows, "local_R_acc", "kimi_accuracy_score")
    lines += [
        "",
        "## Readout",
        "",
        f"- Combined local_reward vs Kimi humanness: Spearman/Pearson = {all_reward_h}/{all_reward_h_p}.",
        f"- Combined local_R_human vs Kimi humanness: Spearman/Pearson = {all_rhuman_h}/{all_rhuman_h_p}.",
        f"- Combined local_R_acc vs Kimi accuracy_score: Spearman/Pearson = {all_racc_acc}/{all_racc_acc_p}.",
        "- If reward/humanness correlation is weak while R_acc dominates, DPO/GRPO can run correctly but optimize the wrong target.",
        "",
    ]

    bad = sorted(
        rows,
        key=lambda r: ((r.get("local_reward") or 0.0) - (r.get("kimi_humanness") or 0.0),
                       (r.get("local_R_human") or 0.0) - (r.get("kimi_humanness") or 0.0)),
        reverse=True,
    )[:regression_n]
    lines += [
        "## High Local Reward But Low Kimi Humanness",
        "",
    ]
    for r in bad:
        lines += [
            f"### {r['model']} / idx={r['idx']}",
            f"- query: {short(r.get('query'), 260)}",
            f"- local: reward={r['local_reward']} R_human={r['local_R_human']} R_acc={r['local_R_acc']} gate={r['local_gate']} trace={r['local_trace_hits']} copy={r['local_copy_ratio']}",
            f"- Kimi: h={r['kimi_humanness']} grounded={r['kimi_grounded']} acc={r['kimi_accuracy']}({r['kimi_accuracy_score']}) traces={r['kimi_rag_traces']} comment={short(r.get('kimi_comment'), 220)}",
            f"- think: {short(r.get('think'), 360)}",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="append", help="label=path. Relative paths are under OUTPUT_DIR.")
    ap.add_argument("--out-md", default=str(Path(OUTPUT_DIR) / "90_corrected_v2_reward_alignment_audit.md"))
    ap.add_argument("--out-jsonl", default=str(Path(OUTPUT_DIR) / "90_corrected_v2_reward_alignment_rows.jsonl"))
    ap.add_argument("--regression-n", type=int, default=12)
    args = ap.parse_args()

    judges = [parse_judge_arg(x) for x in args.judge] if args.judge else [
        (label, Path(OUTPUT_DIR) / rel) for label, rel in DEFAULT_JUDGES
    ]
    existing = [(label, path) for label, path in judges if path.exists()]
    missing = [(label, path) for label, path in judges if not path.exists()]
    for label, path in missing:
        log.warning("skip missing judge file: %s -> %s", label, path)
    if not existing:
        raise FileNotFoundError("No judge files found for reward alignment audit.")

    rows: list[dict] = []
    for label, path in existing:
        recs = read_jsonl(path)
        log.info("read %s: %d rows", path, len(recs))
        rows.extend(row_from_record(label, i, rec) for i, rec in enumerate(recs))

    out_jsonl = Path(args.out_jsonl)
    out_md = Path(args.out_md)
    write_jsonl(out_jsonl, rows)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(make_report(rows, existing, args.regression_n), encoding="utf-8")
    log.info("wrote rows: %s", out_jsonl)
    log.info("wrote report: %s", out_md)


if __name__ == "__main__":
    main()
