"""corrected-v2 Phase 1: Kimi-score candidates from the existing K=8 rollout pool.

Reads output/60_dpo_rollout.jsonl and writes one row per candidate. This step is
append-only and resumable; the done-key is (qid, cand_idx), not just query text.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DPO_ROLLOUT, OUTPUT_DIR, REWARD_TAU_ACC, THINK_MAX_CHARS, THINK_MIN_CHARS
from pipeline import reward
from pipeline.judge_common import judge_text
from pipeline.logger import get_logger

log = get_logger("step93_kimi_score_rollouts")


def sha1(text: str, n: int = 12) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:n]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_existing(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("judge") is not None:
                rows[(r.get("qid"), int(r.get("cand_idx", -1)))] = r
    return rows


def pick_queries(rows: list[dict], n: int, seed: int) -> list[tuple[int, dict]]:
    indexed = list(enumerate(rows))
    random.Random(seed).shuffle(indexed)
    return indexed[:n] if n and n > 0 else indexed


def make_tasks(rows: list[tuple[int, dict]], existing: dict[tuple[str, int], dict]) -> list[dict]:
    tasks = []
    for query_index, rec in rows:
        q = rec.get("query") or ""
        qid = sha1(q, 12)
        cands = rec.get("candidates") or []
        for cand_idx, text in enumerate(cands):
            key = (qid, cand_idx)
            if key in existing:
                continue
            tasks.append({
                "qid": qid,
                "query_index": query_index,
                "cand_idx": cand_idx,
                "query": q,
                "user_prompt": rec.get("user_prompt") or "",
                "gold_answer": rec.get("gold_answer") or "",
                "text": text or "",
                "text_sha1": sha1(text or "", 16),
            })
    return tasks


def score_one(task: dict, local_only: bool) -> dict:
    local = reward.score_rollout(
        task["text"],
        task["user_prompt"],
        task["gold_answer"],
        tau_acc=REWARD_TAU_ACC,
        think_min=THINK_MIN_CHARS,
        think_max=THINK_MAX_CHARS,
        s_pmi=None,
    )
    out = {**task, "local": local}
    if local_only:
        out["judge"] = None
        return out
    try:
        out["judge"] = judge_text(task["query"], task["user_prompt"], task["gold_answer"], task["text"])
    except Exception as exc:
        # Do not write fake zero-score rows; failed candidates can be retried by
        # deleting this row or by rerunning with a fresh output path.
        out["judge"] = None
        out["judge_error"] = repr(exc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", default=DPO_ROLLOUT)
    ap.add_argument("--out", default=str(Path(OUTPUT_DIR) / "93_corrected_v2_rollout_scores.jsonl"))
    ap.add_argument("--queries", type=int, default=int(os.environ.get("V2_SCORE_QUERIES", "120")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("V2_SCORE_SEED", "7")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("V2_KIMI_WORKERS", "3")))
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    rollout = Path(args.rollout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rolls = read_jsonl(rollout)
    if not rolls:
        raise SystemExit(f"missing or empty rollout: {rollout}")
    selected = pick_queries(rolls, args.queries, args.seed)
    existing = iter_existing(out_path)
    tasks = make_tasks(selected, existing)
    expected = sum(len((r.get("candidates") or [])) for _, r in selected)
    print(f"RESULT score_plan queries={len(selected)} expected_candidates={expected} remaining={len(tasks)} out={out_path}", flush=True)
    log.info("score rollout candidates: queries=%d expected=%d remaining=%d out=%s",
             len(selected), expected, len(tasks), out_path)
    if not tasks:
        print(f"RESULT score_done existing={len(existing)} out={out_path}", flush=True)
        return

    done = 0
    ok = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f, cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(score_one, task, args.local_only) for task in tasks]
        for fut in cf.as_completed(futs):
            row = fut.result()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            ok += 1 if row.get("judge") is not None or args.local_only else 0
            if done % 50 == 0 or done == len(tasks):
                rate = done / max(time.time() - t0, 1e-3)
                print(f"PROGRESS score_candidates {done}/{len(tasks)} ok={ok} rate={rate:.2f}/s", flush=True)
    print(f"RESULT score_complete wrote={len(tasks)} judged_ok={ok} out={out_path}", flush=True)


if __name__ == "__main__":
    main()
