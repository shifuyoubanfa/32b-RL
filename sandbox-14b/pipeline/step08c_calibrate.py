"""Step 08c（RL 阶段0）：校准本地打分器是否和 Kimi 裁判同向——零新增 Kimi 调用。

复用已有产物：
- 10_sft_infer.jsonl：student_think / student_answer / teacher_answer
- 10_sft_judge.jsonl：Kimi 的 student_reasoning_humanness / student_accuracy_label
- 00_data_sft_eval.jsonl：取 user_prompt 还原参考资料（算 copy_ratio 要用）

按 query join 后：
- 本地 R_human(student_think, references) vs Kimi humanness -> Spearman 相关
- 本地 GATE_acc(student_answer vs teacher_answer >= τ) vs Kimi (correct|partial) -> 一致率

开训门槛建议：Spearman ≥ 0.6 且一致率 ≥ 0.85，否则别急着上 RL。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    STUDENT_OUTPUTS, JUDGE_RESULTS, SFT_EVAL, CALIB_REPORT,
    REWARD_TAU_ACC,
)
from pipeline.logger import get_logger
from pipeline import reward as R

log = get_logger("step08c_calib")


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _safe_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def _ranks(xs):
    """平均秩（处理并列）。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((a[i] - ma) ** 2 for i in range(n)) ** 0.5
    vb = sum((b[i] - mb) ** 2 for i in range(n)) ** 0.5
    if va == 0 or vb == 0:
        return 0.0
    return cov / (va * vb)


def spearman(a, b):
    return _pearson(_ranks(a), _ranks(b))


def _load_aligned():
    stu = {r.get("query"): r for r in load_jsonl(STUDENT_OUTPUTS)}
    jud = {r.get("query"): r for r in load_jsonl(JUDGE_RESULTS)}
    refs = {}
    for s in load_jsonl(SFT_EVAL):
        msgs = s.get("messages") or []
        if len(msgs) >= 2 and isinstance(msgs[1], dict):
            refs[s.get("query")] = msgs[1].get("content", "")
    common = [q for q in stu if q in jud]
    return stu, jud, refs, common


def tune():
    """网格搜索 reward 参数，最大化 humanness Spearman 与 准确率门控一致率。零 Kimi 调用。"""
    stu, jud, refs, common = _load_aligned()
    if not common:
        log.error("没有可对齐样本；请先跑完 step05/step06。")
        sys.exit(1)
    log.info("调参：可对齐样本 %d", len(common))

    # 预取每条样本的特征
    feats = []
    for q in common:
        so, jo = stu[q], jud[q]
        kh = _safe_float(jo.get("student_reasoning_humanness"))
        kimi_ok = jo.get("student_accuracy_label") in ("correct", "partial")
        racc, _ = R.answer_drift(so.get("student_answer", ""), so.get("teacher_answer", ""))
        feats.append({"think": so.get("student_think", ""), "ref": refs.get(q, ""),
                      "kh": kh, "kimi_ok": kimi_ok, "racc": racc})

    # humanness：搜 c_trace × c_copy 最大化 Spearman
    best_h = (-2, None)
    for ct in [0.25, 0.34, 0.5, 0.7, 1.0]:
        for cc in [0.3, 0.6, 0.8, 1.0]:
            lh = [R.humanness(f["think"], f["ref"], c_trace=ct, c_copy=cc)[0] for f in feats]
            rho = spearman(lh, [f["kh"] for f in feats])
            if rho > best_h[0]:
                best_h = (rho, (ct, cc, sum(lh) / len(lh)))
    # 准确率：搜 tau 最大化一致率
    best_t = (-1, None)
    for tau in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        agree = sum(1 for f in feats if (f["racc"] >= tau) == f["kimi_ok"]) / len(feats)
        if agree > best_t[0]:
            best_t = (agree, tau)

    rho, (ct, cc, mean_lh) = best_h
    log.info("=== 调参结果 ===")
    log.info("humanness 最优: c_trace=%.2f c_copy=%.2f -> Spearman=%.3f (本地均值%.3f, Kimi均值%.3f)",
             ct, cc, rho, mean_lh, sum(f["kh"] for f in feats) / len(feats))
    log.info("准确率 最优: tau=%.2f -> 一致率=%.1f%%", best_t[1], best_t[0] * 100)
    log.info(">>> 把这三个值写回 config：REWARD_C_TRACE=%.2f  REWARD_C_COPY=%.2f  REWARD_TAU_ACC=%.2f",
             ct, cc, best_t[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=REWARD_TAU_ACC)
    parser.add_argument("--tune", action="store_true", help="网格搜索最优 reward 参数（零 Kimi）")
    args = parser.parse_args()

    if args.tune:
        tune()
        return

    stu, jud, refs, common = _load_aligned()
    if not common:
        log.error("05_student_outputs 与 06_judge_results 没有可对齐的 query；请先跑完 step05/step06。")
        sys.exit(1)
    log.info("可对齐样本：%d", len(common))

    local_h, kimi_h = [], []
    acc_agree = 0
    acc_total = 0
    rows = []
    for q in common:
        so = stu[q]
        jo = jud[q]
        references = refs.get(q, "")
        lh, _ = R.humanness(so.get("student_think", ""), references)
        kh = _safe_float(jo.get("student_reasoning_humanness"))
        local_h.append(lh)
        kimi_h.append(kh)

        racc, _ = R.answer_drift(so.get("student_answer", ""), so.get("teacher_answer", ""))
        gate_pass = racc >= args.tau
        kimi_ok = jo.get("student_accuracy_label") in ("correct", "partial")
        acc_total += 1
        if gate_pass == kimi_ok:
            acc_agree += 1
        rows.append((q, round(lh, 3), round(kh, 3), round(racc, 3), gate_pass, kimi_ok))

    rho = spearman(local_h, kimi_h)
    agree_rate = acc_agree / acc_total if acc_total else 0.0
    avg_local_h = sum(local_h) / len(local_h)
    avg_kimi_h = sum(kimi_h) / len(kimi_h)

    L = []
    L.append("# 打分器校准报告（本地 reward vs Kimi 裁判）\n")
    L.append(f"- 对齐样本数: **{len(common)}**")
    L.append(f"- humanness 相关性 Spearman(本地 R_human, Kimi humanness): **{rho:.3f}**（门槛建议 ≥0.6）")
    L.append(f"- 本地 R_human 均值 {avg_local_h:.3f} / Kimi humanness 均值 {avg_kimi_h:.3f}")
    L.append(f"- 准确率门控一致率 GATE_acc(τ={args.tau}) vs Kimi(correct|partial): **{agree_rate:.1%}**（门槛建议 ≥85%）")
    L.append("")
    verdict = "✅ 达标，可以开训" if (rho >= 0.6 and agree_rate >= 0.85) else "⚠️ 未达标：reward 代理与 Kimi 偏离，先调阈值/正则或谨慎对待 RL 结果"
    L.append(f"## 结论：{verdict}")
    L.append("")
    L.append("## 抽样明细（前 20）")
    L.append("| query | 本地R_human | KimiHuman | 本地R_acc | 门控过 | Kimi对/部分 |")
    L.append("| --- | ---: | ---: | ---: | :-: | :-: |")
    for r in rows[:20]:
        q = (r[0] or "").replace("\n", " ")[:30]
        L.append(f"| {q} | {r[1]} | {r[2]} | {r[3]} | {'✓' if r[4] else '✗'} | {'✓' if r[5] else '✗'} |")

    report = "\n".join(L) + "\n"
    Path(CALIB_REPORT).write_text(report, encoding="utf-8")
    log.info("\n%s", report)
    log.info("已写入 %s", CALIB_REPORT)
    if rho < 0.6 or agree_rate < 0.85:
        log.warning("校准未达标：Spearman=%.3f 一致率=%.1f%%。建议先调 reward 再上 RL。", rho, agree_rate * 100)


if __name__ == "__main__":
    main()
