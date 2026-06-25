"""corrected-v3 E2: controlled rewrite ceiling probe.

Use Kimi to rewrite base think into question-driven reasoning while preserving
the answer. Then judge k times. This tests whether the current rubric can
recognize genuinely more human reasoning when it is explicitly supplied.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_CALL_WORKERS, OUTPUT_DIR
from pipeline import kimi_client, reward
from pipeline.judge_common import aggregate_judges, judge_text
from pipeline.v3_utils import bootstrap_ci, file_fingerprint, mean, qid_for, read_jsonl, run_dir, write_jsonl

REWRITE_SYSTEM = "你是一名严谨的中文税务推理改写专家。只输出改写后的完整 <think>...</think><answer>...</answer>。"

REWRITE_TEMPLATE = """请把下面税务模型输出改写成更像真人从问题出发推导的版本。

硬约束：
1. 最终 <answer> 必须保持原答案的事实口径、数字、结论，不要新增未经参考资料支持的内容；
2. <think> 要从用户问题出发，一步步解释为什么得到答案；
3. 不要出现“参考资料/检索结果/现有回答/资料显示/问答对/文件链接/图片链接”等 RAG 口吻；
4. 不要机械罗列政策原文，不要为了自然牺牲准确性；
5. 输出必须是完整标签：
<think>
...
</think>

<answer>
...
</answer>

【问题】
{query}

【参考资料】
{reference}

【原始输出】
{text}
"""


def rewrite_one(row: dict, k: int) -> dict:
    prompt = REWRITE_TEMPLATE.format(
        query=row.get("query") or "",
        reference=reward.extract_references(row.get("user_prompt") or "")[:3000],
        text=(row.get("gen_text") or "")[:6000],
    )
    rewritten = kimi_client.chat(
        [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2048,
    )
    judges = []
    for _ in range(max(1, k)):
        try:
            judges.append(judge_text(row.get("query",""), row.get("user_prompt",""), row.get("gold_answer",""), rewritten))
        except Exception as exc:
            judges.append({"error": repr(exc)})
    out = {
        "qid": qid_for(row),
        "query": row.get("query") or "",
        "user_prompt": row.get("user_prompt") or "",
        "gold_answer": row.get("gold_answer") or "",
        "original_text": row.get("gen_text") or "",
        "original_judge": row.get("judge") or {},
        "rewritten_text": rewritten,
        "rewrite_judges": judges,
        "rewrite_agg": aggregate_judges(judges),
    }
    out["delta_h"] = (out["rewrite_agg"].get("humanness") or 0.0) - (out["original_judge"].get("humanness") or 0.0)
    out["delta_g"] = (out["rewrite_agg"].get("grounded") or 0.0) - (out["original_judge"].get("grounded") or 0.0)
    out["delta_acc"] = (out["rewrite_agg"].get("accuracy_score") or 0.0) - (out["original_judge"].get("accuracy_score") or 0.0)
    return out


def make_report(rows: list[dict], decision: dict) -> str:
    dh = [r["delta_h"] for r in rows]
    lo, hi = bootstrap_ci(dh)
    h_new = mean([r["rewrite_agg"].get("humanness") for r in rows])
    decision.update({
        "rows": len(rows),
        "rewrite_mean_h": h_new,
        "mean_delta_h": mean(dh),
        "delta_h_ci95": [lo, hi],
        "mean_delta_g": mean([r["delta_g"] for r in rows]),
        "mean_delta_acc": mean([r["delta_acc"] for r in rows]),
        "accuracy_guard_ok": mean([r["delta_acc"] for r in rows]) >= -0.02,
        "grounded_guard_ok": mean([r["delta_g"] for r in rows]) >= -0.02,
    })
    go = h_new >= decision["target_mean_h"] and decision["accuracy_guard_ok"] and decision["grounded_guard_ok"]
    decision["status"] = "GO_REWRITE_CEILING" if go else "NO-GO_REWRITE_CEILING"
    lines = [
        "# corrected-v3 E2 controlled rewrite ceiling probe",
        "",
        f"- status: **{decision['status']}**",
        f"- rows: {len(rows)}",
        f"- rewritten mean h: {h_new:.3f} (target >= {decision['target_mean_h']:.3f})",
        f"- mean Δh: {mean(dh):+.3f}, bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]",
        f"- mean Δgrounded: {decision['mean_delta_g']:+.3f}",
        f"- mean Δacc_score: {decision['mean_delta_acc']:+.3f}",
        "",
        "If this is GO, a rewrite-based DPO dataset is plausible. If it is NO-GO, the current rubric may not see the desired structural improvement.",
    ]
    return "\n".join(lines) + "\n"


def write_blind_csv(path: Path, rows: list[dict], n: int) -> None:
    sample = rows[:n]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "query", "A", "B", "which_is_better(A/B/tie)", "notes"])
        for i, r in enumerate(sample):
            if i % 2 == 0:
                a, b = r["original_text"], r["rewritten_text"]
            else:
                a, b = r["rewritten_text"], r["original_text"]
            w.writerow([r["qid"], r["query"], a, b, "", ""])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-judge", default=str(Path(OUTPUT_DIR) / "80_corrected_v1_rft_merged_base_judge.jsonl"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=JUDGE_CALL_WORKERS)
    ap.add_argument("--target-mean-h", type=float, default=0.80)
    ap.add_argument("--blind-n", type=int, default=20)
    args = ap.parse_args()
    odir = Path(args.out_dir) if args.out_dir else run_dir()
    rows = read_jsonl(args.base_judge)
    if not rows:
        raise SystemExit(f"missing base judge: {args.base_judge}")
    random.Random(41).shuffle(rows)
    rows = rows[:args.limit]
    out_jsonl = odir / "103_corrected_v3_rewrite_probe.jsonl"
    report = odir / "103_corrected_v3_rewrite_probe.md"
    decision_path = odir / "103_corrected_v3_rewrite_probe_decision.json"
    blind_csv = odir / "103_corrected_v3_rewrite_blind_review.csv"
    print(f"RESULT e2_plan rows={len(rows)} k={args.k} workers={args.workers}", flush=True)
    done = 0
    t0 = time.time()
    out = []
    with cf.ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futs = [pool.submit(rewrite_one, r, args.k) for r in rows]
        for fut in cf.as_completed(futs):
            out.append(fut.result())
            done += 1
            if done % 5 == 0 or done == len(rows):
                print(f"PROGRESS e2_rewrite {done}/{len(rows)} rate={done/max(time.time()-t0,1e-3):.2f}/s", flush=True)
    out.sort(key=lambda r: r["qid"])
    decision = {"input": file_fingerprint(args.base_judge), "k": args.k, "target_mean_h": args.target_mean_h}
    write_jsonl(out_jsonl, out)
    write_blind_csv(blind_csv, out, args.blind_n)
    report.write_text(make_report(out, decision), encoding="utf-8")
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT e2_complete status={decision['status']} mean_h={decision['rewrite_mean_h']:.3f} report={report}", flush=True)


if __name__ == "__main__":
    main()
