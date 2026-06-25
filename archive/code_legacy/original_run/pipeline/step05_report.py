"""阶段报告（所有阶段复用）：聚合 Kimi 判分 → humanness / 准确率 / RAG 痕迹 / 交叉表。

输出 report.md + 在 stdout/日志打印关键数字，便于实时监督。
阶段1（基线）可加 --baseline：把实测 humanness0 写进 output/baseline_consts.json 供后续读取。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR
from pipeline.logger import get_logger

log = get_logger("step05_report")

_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
_TRACE_KEYS = ["explicit_ref", "verbatim_copy", "ref_enumeration", "policy_source"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="step04 判分产物")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="", help="阶段标签(报告标题)")
    ap.add_argument("--baseline", action="store_true", help="写回 humanness0/verbatim_base 供后续阶段读取")
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    n = len(recs)
    js = [r.get("judge", {}) for r in recs]

    humanness = [j.get("humanness") for j in js]
    h_mean = _mean(humanness)
    g_mean = _mean([j.get("grounded") for j in js])   # grounding 忠实度（本次新增，看 think 是否扣参考）
    acc_labels = Counter(j.get("accuracy", "incorrect") for j in js)
    acc_score_mean = _mean([j.get("accuracy_score") for j in js])
    correct = acc_labels.get("correct", 0)
    partial = acc_labels.get("partial", 0)
    incorrect = acc_labels.get("incorrect", 0)

    # humanness 分布
    dist = []
    for lo, hi in _BINS:
        c = sum(1 for h in humanness if h is not None and lo <= h < hi)
        dist.append((lo, hi, c, round(100 * c / max(1, n), 1)))

    # RAG 痕迹计数
    trace_counts = Counter()
    for j in js:
        for t in (j.get("rag_traces") or []):
            trace_counts[t] += 1

    # accuracy × humanness 交叉
    cross = {}
    for lab in ("correct", "partial", "incorrect"):
        hs = [j.get("humanness") for j in js if j.get("accuracy") == lab]
        cross[lab] = round(_mean(hs), 3)
    verbatim = trace_counts.get("verbatim_copy", 0)

    lines = [
        f"# {args.tag or '评测'} 报告（验收集 {n} 条，Kimi 裁判）", "",
        f"- 推理 humanness 均值：**{h_mean:.3f}**",
        f"- 推理 grounded(忠于参考) 均值：**{g_mean:.3f}**",
        f"- 准确率(平均分,漂移)：**{acc_score_mean:.3f}**",
        f"- correct / partial / incorrect：{correct} / {partial} / {incorrect}",
        f"- correct%：**{100*correct/max(1,n):.1f}%**  ｜ correct+partial%：**{100*(correct+partial)/max(1,n):.1f}%**", "",
        "## humanness 分布",
        "| 区间 | 条数 | 占比 |", "|---|---|---|",
        *[f"| {lo:.1f}-{hi if hi<=1 else 1.0:.1f} | {c} | {p}% |" for lo, hi, c, p in dist], "",
        "## RAG 痕迹计数（学生）",
        "| 类型 | 次数 |", "|---|---|",
        *[f"| {k} | {trace_counts.get(k,0)} |" for k in _TRACE_KEYS], "",
        "## accuracy × humanness 交叉（健康收敛应 correct ≥ incorrect）",
        f"- correct={cross['correct']} ｜ partial={cross['partial']} ｜ incorrect={cross['incorrect']}", "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")

    log.info("===== %s 报告 =====", args.tag or "评测")
    log.info("humanness=%.3f  grounded=%.3f  acc=%.3f  correct%%=%.1f  c+p%%=%.1f  verbatim_copy=%d",
             h_mean, g_mean, acc_score_mean, 100*correct/max(1, n), 100*(correct+partial)/max(1, n), verbatim)
    log.info("交叉 correct=%.3f partial=%.3f incorrect=%.3f -> %s",
             cross["correct"], cross["partial"], cross["incorrect"], args.out)

    if args.baseline:
        consts = {"humanness0": round(h_mean, 4), "grounded0": round(g_mean, 4), "verbatim_base": verbatim,
                  "acc0_drift": round(acc_score_mean, 4)}
        cpath = Path(OUTPUT_DIR) / "baseline_consts.json"
        cpath.write_text(json.dumps(consts, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("基线常量写回 %s : %s（humanness0/verbatim_base；acc0≈1.0 见技术方案）", cpath, consts)


if __name__ == "__main__":
    main()
