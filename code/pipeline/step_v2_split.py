"""V2 step：把 V1 自产 (think, answer) 重切成 1739 训练 + 500 冻结验证，并建 2239 建池题集。

读 step01 的 output/00_v1_outputs.jsonl（query/user_prompt/raw/reasoning/answer）。
- answer  = V1 贪心金标准（answer-lock 锚，全程不训练）。
- reasoning = V1 原 think（DPO rejected 源、冷启动/RFT σ 选择的"脏"对照）。
切法照搬 step02（random.Random(seed).shuffle），但用【固定 500 验证】而非比例（不碰 V1 旧 2014/224 口径）。
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import V1_OUTPUTS
from pipeline.v2_paths import (V2_TRAIN, V2_EVAL, V2_PROBLEMS, V2_PROBLEMS_TRAIN, V2_N_EVAL,
                               V2_SPLIT_SEED, qid_of, read_jsonl, write_jsonl)
from pipeline.logger import get_logger

log = get_logger("step_v2_split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(V1_OUTPUTS))
    ap.add_argument("--n_eval", type=int, default=V2_N_EVAL)
    ap.add_argument("--seed", type=int, default=V2_SPLIT_SEED)
    args = ap.parse_args()

    recs = read_jsonl(args.inp)
    rows, seen = [], set()
    for r in recs:
        q = r.get("query")
        up = (r.get("user_prompt") or "").strip()
        ans = (r.get("answer") or "").strip()
        think = (r.get("reasoning") or "").strip()
        if not up or not ans:            # 无参考/无答案：与 step02.to_sample 同口径直接弃
            continue
        qid = qid_of(q)
        if qid in seen:                  # 同题去重（qid=sha1(query)）
            continue
        seen.add(qid)
        rows.append({"qid": qid, "query": q, "user_prompt": up, "answer": ans, "reasoning": think})
    log.info("有效样本 %d / %d", len(rows), len(recs))
    if len(rows) <= args.n_eval:
        raise SystemExit(f"有效样本 {len(rows)} <= 验证集 {args.n_eval}，无法切分")

    random.Random(args.seed).shuffle(rows)
    eval_rows, train_rows = rows[: args.n_eval], rows[args.n_eval:]
    for r in eval_rows:
        r["split"] = "eval"
    for r in train_rows:
        r["split"] = "train"

    write_jsonl(V2_EVAL, eval_rows)
    write_jsonl(V2_TRAIN, train_rows)
    # 建池题集（全 2239，step152 口径字段；含 eval 题，answer_drift 对 500 验证也要池）
    problems = [{"qid": r["qid"], "split": r["split"], "query": r["query"],
                 "user_prompt": r["user_prompt"], "gold_answer": r["answer"]}
                for r in (eval_rows + train_rows)]
    write_jsonl(V2_PROBLEMS, problems)
    # 仅 train（1739）的池题集：RFT 自采样 step151 用（含 gold_answer，且不泄漏 500 冻结 eval）
    write_jsonl(V2_PROBLEMS_TRAIN, [p for p in problems if p["split"] == "train"])

    log.info("train=%d -> %s", len(train_rows), V2_TRAIN)
    log.info("eval =%d（冻结）-> %s", len(eval_rows), V2_EVAL)
    log.info("problems(全)=%d -> %s ; problems(train)=%d -> %s",
             len(problems), V2_PROBLEMS, len(train_rows), V2_PROBLEMS_TRAIN)
    if len(rows) < 2000:   # 2239 原始，正常丢空答案后仍应近 2239；显著偏少=上游 step01 可能被截断
        log.warning("有效样本 %d < 2000：疑似上游 00_v1_outputs.jsonl 不完整（step01 被中断？），请核对再继续。", len(rows))


if __name__ == "__main__":
    main()
