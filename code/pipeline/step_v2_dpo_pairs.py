"""V2 step：构 DPO 偏好对（answer-lock）→ 2σ/3σ 两桶。

读 step10 rollout {query,user_prompt,gold_answer,candidates:[K]}（来自 RFT-merged 模型）
+ V2_TRAIN（取 v1_think=rejected 源、v1_answer=共用 answer）+ 152_v1_support（答案门）。
每题：候选过 答案门+规则门 → k=2 粗筛取最干净 1 条 → k=16 确认它比 V1 原 think 干净 2σ/3σ。
  chosen   = full(洗净 think, V1 原 answer)
  rejected = full(V1 原 think,  V1 原 answer)      ← 只 think 不同（answer-lock）；天然 3σ 可分、零额外数据
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_CALL_WORKERS
from pipeline import reward, vllm_client
from pipeline.reward import parse_think_answer
from pipeline.rules_v6 import detect_rag_style, answer_in_v1_pool
from pipeline.v2_common import score_think_kimi, score_think_select, confident_cleaner
from pipeline.v2_paths import (V2_TRAIN, dpo_pairs, dpo_progress, dpo_row, read_jsonl, write_jsonl,
                               index_by_qid, load_support_index, qid_of,
                               gather_until, V2_DPO_TARGET, V2_GATHER_CHUNK)
from pipeline.logger import get_logger

log = get_logger("step_v2_dpo_pairs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True, help="step10 rollout 输出 jsonl（RFT-merged 采样）")
    ap.add_argument("--lineage", required=True, help="该 RFT 父线，如 2s-3s")
    args = ap.parse_args()

    rollouts = read_jsonl(args.rollout)
    for r in rollouts:                                    # rollout 只有 query，补上 qid 供早停/续跑按 qid 去重
        r["qid"] = qid_of(r.get("query"))
    train_idx = index_by_qid(read_jsonl(V2_TRAIN))
    support = load_support_index()
    log.info("DPO 构对：%d 题（lineage=%s, workers=%d）", len(rollouts), args.lineage, JUDGE_CALL_WORKERS)

    def _pair(rec: dict) -> dict:
        qid = qid_of(rec.get("query"))
        base = train_idx.get(qid) or {}
        v1_think = (base.get("reasoning") or "").strip()
        v1_answer = (base.get("answer") or "").strip()
        refs = reward.extract_references(rec.get("user_prompt") or "")
        pool_answers = (support.get(qid) or {}).get("v1_answers") or []
        out = {"qid": qid, "query": rec.get("query"), "user_prompt": rec.get("user_prompt"),
               "v1_think": v1_think, "answer": v1_answer, "chosen_think": None,
               "pass2": False, "pass3": False}
        if not v1_think or not v1_answer:
            return out
        survivors = []
        for text in rec.get("candidates") or []:
            think, answer = parse_think_answer(text)
            if not think.strip():
                continue
            if not answer_in_v1_pool(answer, pool_answers)["in_pool"]:
                continue
            if detect_rag_style(think)["has_rag_style"]:
                continue
            survivors.append(think)
        if not survivors:
            return out
        # k=2 粗筛【挑最干净 1 条】（argmax，不与 dirty 比；语义异于 cleaner_scores 的"严格>dirty"），再 k=16 确认比 V1 干净
        scored = [(t, score_think_kimi(refs, t, k=2)["clean_score"]) for t in survivors]
        scored = [(t, s) for t, s in scored if s is not None]
        if not scored:
            return out
        best_think = max(scored, key=lambda x: x[1])[0]
        s_clean = score_think_select(refs, best_think, k=16)["clean_score"]
        s_dirty = score_think_select(refs, v1_think, k=16)["clean_score"]
        if s_clean is None or s_dirty is None:
            return out
        out["chosen_think"] = best_think
        out["s_clean"], out["s_dirty"] = s_clean, s_dirty
        out["pass2"] = confident_cleaner(s_clean, s_dirty, 2.0)
        out["pass3"] = confident_cleaner(s_clean, s_dirty, 3.0)
        return out

    # 凑够就停 + 边攒边落盘续跑：2σ 偏好对攒到 V2_DPO_TARGET 即停（每题的门/k=16 不变）
    results = gather_until(
        rollouts, _pair, workers=JUDGE_CALL_WORKERS, chunk=V2_GATHER_CHUNK, desc=f"dpo:{args.lineage}",
        progress_path=dpo_progress(args.lineage),
        enough=lambda rs: V2_DPO_TARGET > 0
        and sum(1 for r in rs if r and r.get("pass2")) >= V2_DPO_TARGET)

    def _rows(flag):
        return [dpo_row(r["user_prompt"], r["chosen_think"], r["v1_think"], r["answer"],
                        query=r.get("query"), meta={"s_clean": r.get("s_clean"), "s_dirty": r.get("s_dirty")})
                for r in results if r.get(flag)]

    pairs2, pairs3 = _rows("pass2"), _rows("pass3")
    write_jsonl(dpo_pairs(args.lineage, 2), pairs2)
    write_jsonl(dpo_pairs(args.lineage, 3), pairs3)
    log.info("DPO 偏好对 2σ=%d -> %s", len(pairs2), dpo_pairs(args.lineage, 2))
    log.info("DPO 偏好对 3σ=%d -> %s", len(pairs3), dpo_pairs(args.lineage, 3))
    if len(pairs2) < 50:
        log.warning("DPO 2σ 对 %d < 50：偏好对偏少，DPO 信号弱；可放宽到 2σ 或检查上游 rollout 是否够干净。", len(pairs2))


if __name__ == "__main__":
    main()
