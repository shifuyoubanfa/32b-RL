"""V2 step：三分评测（复用 step03 infer 产物，调 v2_common 三函数，绝不调旧 step04/05）。

读 step03_eval_infer 输出 {query,user_prompt,gold_answer,gen_text,think,answer}（500 验证，贪心）：
  think-Kimi : score_think_eval(reference, think)  —— k=3 整段干净分（N=500 时 SE≈0.05；reference=extract_references 对齐 σ 标定口径）
  think-规则 : score_think_rule(think)             —— detect_rag_style 检索腔（确定性）
  答案漂移   : answer_drift(answer, v1_answers[qid]) —— 极性+数字在不在 V1 池（基线 100%）
聚合：干净分先按 n>0 过滤再求均值（全 k 失败=None≠0.0=最脏档，避免污染）；规则通过率；答案在池率。
两阶段平均差 > ~3×eval_se(N,3)≈0.15 才算真涨。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_CALL_WORKERS
from pipeline import reward, vllm_client
from pipeline.v2_common import score_think_eval, score_think_rule, answer_drift, eval_se
from pipeline.v2_paths import (V2_V1_SUPPORT, read_jsonl, write_jsonl, qid_of, load_support_index,
                               gather_until, eval_progress, V2_GATHER_CHUNK)
from pipeline.logger import get_logger

log = get_logger("step_v2_eval")

# report_complete 校验用的固定标记（run_v2 据此判报告是否写全）；不含 k，免改 k 又要同步 marker
REPORT_MARKERS = ("Kimi干净分", "规则去检索腔通过率", "答案在池率")


def score_record(rec: dict, support: dict) -> dict:
    """Score one frozen-eval row, including an explicitly accounted format failure."""
    think = rec.get("think") or ""
    answer = rec.get("answer") or ""
    refs = reward.extract_references(rec.get("user_prompt") or "")
    qid = qid_of(rec.get("query"))
    pool_answers = (support.get(qid) or {}).get("v1_answers") or []
    se = score_think_eval(refs, think)
    sr = score_think_rule(think)
    format_ok = bool(rec.get("format_ok", bool(think.strip() and answer.strip())))
    format_reason = str(rec.get("format_reason") or ("ok" if format_ok else "legacy_unmarked_empty"))
    rule_forced_failure = (not format_ok) or (not think.strip())
    rule_pass = (not rule_forced_failure) and (not sr["has_rag_style"])
    # Missing-pool qids are kept separate.  An empty model answer with a real
    # pool is deliberately comparable and fails in_pool via rules_v6.
    if pool_answers:
        ad = answer_drift(answer, pool_answers)
        in_pool, comparable = ad["in_pool"], ad["comparable"]
        drift, answer_reason, no_pool = ad["drift_facts"], ad["reason"], False
    else:
        in_pool, comparable, drift, answer_reason, no_pool = None, False, [], "no_pool", True
    return {"query": rec.get("query"), "qid": qid,
            "clean_score": se["clean_score"], "clean_n": se["n"],
            "has_rag_style": sr["has_rag_style"], "n_traces": sr["n_traces"],
            "rule_pass": rule_pass, "rule_forced_failure": rule_forced_failure,
            "format_ok": format_ok, "format_reason": format_reason,
            "empty_think": not bool(think.strip()), "empty_answer": not bool(answer.strip()),
            "in_pool": in_pool, "answer_comparable": comparable, "answer_reason": answer_reason,
            "drift_facts": drift, "no_pool": no_pool}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", required=True, help="step03_eval_infer 输出 jsonl")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--support", default=str(V2_V1_SUPPORT))
    ap.add_argument("--summary", default="", help="机读摘要 json（剪枝用三分数值）；空=由 report 派生")
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()
    summary_path = args.summary or str(
        Path(args.report).with_name(Path(args.report).stem.replace("_report", "") + "_summary.json"))

    rows = read_jsonl(args.infer)
    for r in rows:                                        # 补 qid 供增量落盘按 qid 续跑（infer 只有 query）
        r["qid"] = qid_of(r.get("query"))
    support = load_support_index(args.support)
    log.info("三分评测：%d 条（tag=%s, workers=%d）", len(rows), args.tag, JUDGE_CALL_WORKERS)

    def _score(rec: dict) -> dict:
        return score_record(rec, support)

    # 增量落盘 + 续跑：k=3 评测分不进无损缓存，靠 progress 中断不重烧（enough 恒 False=全评、不早停）
    scored = gather_until(rows, _score, enough=lambda rs: False, chunk=V2_GATHER_CHUNK,
                          workers=JUDGE_CALL_WORKERS, desc=f"eval:{args.tag}",
                          progress_path=eval_progress(args.tag), key="qid")
    write_jsonl(args.scores, scored)

    valid = [r["clean_score"] for r in scored if r["clean_n"] > 0 and r["clean_score"] is not None]
    n = len(scored)
    rule_pass = sum(1 for r in scored if r["rule_pass"])
    n_pool = sum(1 for r in scored if not r["no_pool"])  # 主指标分母沿用既有口径：有 V1 池的全部题
    in_pool = sum(1 for r in scored if not r["no_pool"] and r["in_pool"])
    n_comparable = sum(1 for r in scored if not r["no_pool"] and r.get("answer_comparable", True))
    in_pool_comparable = sum(1 for r in scored if not r["no_pool"]
                             and r.get("answer_comparable", True) and r["in_pool"])
    n_no_pool = sum(1 for r in scored if r["no_pool"])
    n_uncomparable = n_pool - n_comparable
    n_empty_answer = sum(1 for r in scored if r.get("answer_reason") == "empty_answer")
    n_format_failure = sum(1 for r in scored if not r.get("format_ok"))
    n_rule_forced_failure = sum(1 for r in scored if r.get("rule_forced_failure"))
    n_unaccounted_empty = sum(1 for r in scored if r.get("format_ok")
                              and (r.get("empty_think") or r.get("empty_answer")))
    format_failure_qids = [r["qid"] for r in scored if not r.get("format_ok")]
    format_failure_reasons = dict(Counter(
        r.get("format_reason") or "unknown" for r in scored if not r.get("format_ok")))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    if n == 0 or not valid:
        # infer 空 / Kimi 判分全挂：写 FAILED（不含 REPORT_MARKERS）→ report_complete 判 False、续跑会重评，
        # 不把"外部 API 全挂"伪装成"已评测 clean=nan"。
        Path(args.report).write_text(
            f"# V2 三分评测 FAILED · {args.tag}\nN={n} 有效Kimi打分={len(valid)}；"
            f"infer 为空或 Kimi 判分全部失败，未产出有效分。请检查 DASHSCOPE/vLLM 后重评。\n",
            encoding="utf-8")
        log.error("评测无有效分（N=%d valid=%d）→ 写 FAILED 报告，续跑会重评：%s", n, len(valid), args.report)
        raise SystemExit(f"eval {args.tag}: 无有效 Kimi 打分")

    clean_mean = sum(valid) / len(valid)
    se_eval = eval_se(n, 3)
    in_pool_rate = in_pool / max(1, n_pool)
    in_pool_comparable_rate = in_pool_comparable / max(1, n_comparable)
    rule_pass_rate = rule_pass / max(1, n)

    # 先写机读摘要（剪枝用）：在 report 之前落盘 → report_complete 为真时 summary 必在（续跑可读）
    summary = {"tag": args.tag, "n": n, "n_valid": len(valid), "n_pool": n_pool,
               "n_answer_comparable": n_comparable, "n_answer_uncomparable": n_uncomparable,
               "n_empty_answer": n_empty_answer, "n_format_failure": n_format_failure,
               "format_failure_rate": round(n_format_failure / max(1, n), 4),
               "format_pass_rate": round((n - n_format_failure) / max(1, n), 4),
               "n_rule_forced_failure": n_rule_forced_failure,
               "n_unaccounted_empty": n_unaccounted_empty,
               "format_failure_qids": format_failure_qids,
               "format_failure_reasons": format_failure_reasons,
               "clean_mean": round(clean_mean, 4), "rule_pass_rate": round(rule_pass_rate, 4),
               "in_pool_rate": round(in_pool_rate, 4),
               "in_pool_comparable_rate": round(in_pool_comparable_rate, 4), "se": round(se_eval, 4)}
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# V2 三分评测报告 · {args.tag}",
        "",
        f"- 样本数 N = {n}（其中 Kimi 有效打分 {len(valid)}；缺 V1 池 {n_no_pool} 题；空答案 {n_empty_answer} 题）",
        f"- **生成格式完整率** = {n-n_format_failure}/{n} = {(n-n_format_failure)/max(1,n):.1%}"
        f"（格式失败 {n_format_failure} 题；Kimi评原始残缺think，规则think强制失败，空answer判不在池）",
        f"- **Kimi干净分(k=3)均值** = {clean_mean:.3f}（0-10，越高越干净；SE≈{se_eval:.3f}，两阶段差 >~{3*se_eval:.2f} 才算真涨）",
        f"- **规则去检索腔通过率** = {rule_pass}/{n} = {rule_pass_rate:.1%}（detect_rag_style 无痕迹）",
        f"- **答案在池率** = {in_pool}/{n_pool} = {in_pool_rate:.1%}（漂移率 {1-in_pool_rate:.1%}；越接近 V1 自评基线越好，见 v2-baseline-v1 报告）",
        f"- 答案可比较审计 = {n_comparable}/{n_pool}；可比较题在池率 {in_pool_comparable}/{n_comparable} = {in_pool_comparable_rate:.1%}（无极性/数字/日期的非空回答不改主指标，只单列披露）",
        "",
        "口径：reference=extract_references；干净分复用 judgecal 标定提示词(k=3评测)；答案锁 V1 原版只训 think。",
    ]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("干净分均值=%.3f | 规则通过=%.1f%% | 在池率=%.1f%%(分母%d) -> %s",
             clean_mean, 100*rule_pass/max(1, n), 100*in_pool/max(1, n_pool), n_pool, args.report)


if __name__ == "__main__":
    main()
