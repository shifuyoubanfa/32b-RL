"""V2 step：RFT 自采样选样 → 三门（答案门→规则门→σ门）→ 2σ/3σ 两训练桶（answer-lock）。

读 step151 自采样 {qid,query,user_prompt,gold_answer,samples:[K段text]} + V2_TRAIN（取 v1_think 当 σ 脏对照）
+ 152_v1_support（答案门用 v1_answers）。每题：
  ① 答案门 answer_in_v1_pool(answer, v1_answers).in_pool   —— 扔掉 answer 漂出 V1 池的样本（其 think 朝错结论推）
  ② 规则门 detect_rag_style(think).has_rag_style==False     —— 扔掉带检索腔的
  ③ σ 门：存活候选先 k=2 粗筛取最干净 1 条，再 k=16 确认它比 V1 原 think 干净 2σ/3σ（confident_cleaner）
选中 think + 该题 V1 原 answer → SFT 行（只 think 不同）。3σ 桶 ⊆ 2σ 桶。
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
from pipeline.v2_paths import (V2_TRAIN, rft_train, rft_progress, sft_row, read_jsonl, write_jsonl,
                               index_by_qid, load_support_index,
                               gather_until, V2_RFT_TARGET, V2_GATHER_CHUNK)
from pipeline.logger import get_logger

log = get_logger("step_v2_rft_select")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfsample", required=True, help="step151 自采样输出 jsonl")
    ap.add_argument("--lineage", required=True, help="该 SFT 父线，如 2s / 3s")
    args = ap.parse_args()

    samples = read_jsonl(args.selfsample)
    train_idx = index_by_qid(read_jsonl(V2_TRAIN))     # qid -> {reasoning(v1_think), answer, ...}
    support = load_support_index()                      # qid -> {v1_answers, ...}
    log.info("RFT 选样：%d 题（lineage=%s, workers=%d）", len(samples), args.lineage, JUDGE_CALL_WORKERS)

    def _select(rec: dict) -> dict:
        qid = rec.get("qid")
        base = train_idx.get(qid) or {}
        v1_think = (base.get("reasoning") or "").strip()
        v1_answer = (base.get("answer") or "").strip()
        refs = reward.extract_references(rec.get("user_prompt") or "")
        pool_answers = (support.get(qid) or {}).get("v1_answers") or []
        out = {"qid": qid, "query": rec.get("query"), "user_prompt": rec.get("user_prompt"),
               "answer": v1_answer, "best_think": None, "pass2": False, "pass3": False}
        if not v1_think or not v1_answer:
            return out
        # ①②门（确定性、零 Kimi）
        survivors = []
        for text in rec.get("samples") or []:
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
        # ③σ门：k=2 粗筛【挑最干净 1 条】（注意：此处 argmax 选 best，不与 dirty 比；语义异于 v2_common.cleaner_scores 的"严格>dirty 才放行"）
        scored = [(t, score_think_kimi(refs, t, k=2)["clean_score"]) for t in survivors]
        scored = [(t, s) for t, s in scored if s is not None]
        if not scored:
            return out
        best_think = max(scored, key=lambda x: x[1])[0]
        s_clean = score_think_select(refs, best_think, k=16)["clean_score"]
        s_dirty = score_think_select(refs, v1_think, k=16)["clean_score"]
        if s_clean is None or s_dirty is None:
            return out
        out["best_think"] = best_think
        out["pass2"] = confident_cleaner(s_clean, s_dirty, 2.0)
        out["pass3"] = confident_cleaner(s_clean, s_dirty, 3.0)
        return out

    # 凑够就停：2σ 桶攒到 V2_RFT_TARGET 即停（每题的三道门/k=16 不变）
    results = gather_until(
        samples, _select, workers=JUDGE_CALL_WORKERS, chunk=V2_GATHER_CHUNK, desc=f"rft:{args.lineage}",
        progress_path=rft_progress(args.lineage),         # 边攒边落盘 + 续跑跳过已选过的题
        enough=lambda rs: V2_RFT_TARGET > 0
        and sum(1 for r in rs if r and r.get("pass2")) >= V2_RFT_TARGET)

    train2 = [sft_row(r["user_prompt"], r["best_think"], r["answer"], query=r.get("query"))
              for r in results if r.get("pass2")]
    train3 = [sft_row(r["user_prompt"], r["best_think"], r["answer"], query=r.get("query"))
              for r in results if r.get("pass3")]
    write_jsonl(rft_train(args.lineage, 2), train2)
    write_jsonl(rft_train(args.lineage, 3), train3)
    log.info("RFT 选中 2σ=%d -> %s", len(train2), rft_train(args.lineage, 2))
    log.info("RFT 选中 3σ=%d -> %s", len(train3), rft_train(args.lineage, 3))
    if len(train2) < 100:
        log.warning("RFT 2σ 桶 %d < 100：洗干净后自采都挤干净端、Kimi 分不开（分辨率天花板，符合预期）。", len(train2))


if __name__ == "__main__":
    main()
