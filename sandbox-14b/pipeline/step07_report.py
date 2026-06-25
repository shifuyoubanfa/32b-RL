"""Step 7: 汇总评测结果，输出对比报告。

核心指标：
1. 学生答案准确率（以 teacher 为绝对正确）
2. 推理过程的 humanness（端到端 CoT 感）：teacher vs student
3. RAG 痕迹类型分布：teacher vs student
4. accuracy × humanness 交叉表
5. humanness 最低的样本（RL 阶段的重点优化目标）
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_RESULTS, REPORT_MD, JUDGE_MODEL
from pipeline.logger import get_logger

log = get_logger("step07_report")


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def safe_list(v):
    return v if isinstance(v, list) else []


BUCKETS = [
    (0.0, 0.2, "0.0-0.2 (强 RAG 痕迹)"),
    (0.2, 0.4, "0.2-0.4"),
    (0.4, 0.6, "0.4-0.6"),
    (0.6, 0.8, "0.6-0.8"),
    (0.8, 1.001, "0.8-1.0 (像端到端 CoT)"),
]


def bucket_of(x: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= x < hi:
            return name
    return BUCKETS[-1][2]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def format_dist(values: list[float]) -> str:
    counts = Counter(bucket_of(v) for v in values)
    n = len(values) or 1
    lines = []
    for _, _, name in BUCKETS:
        c = counts.get(name, 0)
        bar = "█" * int(round(20 * c / n))
        lines.append(f"  {name:<28} {c:>4}  {c/n:>6.1%}  {bar}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=JUDGE_RESULTS, help="评测结果 jsonl")
    parser.add_argument("--out", default=REPORT_MD, help="报告 md 输出")
    args = parser.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    n = len(recs)
    if n == 0:
        log.warning("没有评测结果")
        return

    # ---------- 准确率 ----------
    label_cnt = Counter(r.get("student_accuracy_label", "unknown") for r in recs)
    acc_scores = [safe_float(r.get("student_accuracy_score")) for r in recs]
    avg_acc = mean(acc_scores)
    correct_rate = label_cnt.get("correct", 0) / n
    not_incorrect_rate = (label_cnt.get("correct", 0) + label_cnt.get("partial", 0)) / n

    # ---------- humanness ----------
    teacher_h = [safe_float(r.get("teacher_reasoning_humanness")) for r in recs]
    student_h = [safe_float(r.get("student_reasoning_humanness")) for r in recs]
    avg_teacher_h = mean(teacher_h)
    avg_student_h = mean(student_h)

    # ---------- RAG 痕迹类型分布 ----------
    teacher_traces = Counter()
    student_traces = Counter()
    for r in recs:
        for t in safe_list(r.get("teacher_rag_trace_types")):
            teacher_traces[t] += 1
        for t in safe_list(r.get("student_rag_trace_types")):
            student_traces[t] += 1

    # ---------- accuracy × humanness 交叉 ----------
    by_label = defaultdict(list)
    for r in recs:
        lbl = r.get("student_accuracy_label", "unknown")
        by_label[lbl].append(safe_float(r.get("student_reasoning_humanness")))

    # ---------- humanness 最低样本（RL 重点）----------
    lowest = sorted(
        recs,
        key=lambda r: safe_float(r.get("student_reasoning_humanness"), 1.0),
    )[:10]

    # ---------- 写报告 ----------
    L = []
    L.append("# 学生模型 DeepSeek-R1-Distill-Qwen-14B vs 公司微调模型(V1) 评测报告\n")
    L.append(f"- 评测样本数: **{n}**")
    L.append(f"- 裁判模型: {JUDGE_MODEL}")
    L.append("")

    L.append("## 1) 学生答案准确率（以 teacher 为绝对正确）")
    L.append(f"- 平均得分 (0~1): **{avg_acc:.3f}**")
    L.append(f"- correct           占比: **{correct_rate:.1%}**")
    L.append(f"- correct + partial 占比: **{not_incorrect_rate:.1%}**")
    L.append(f"- 标签分布: `{dict(label_cnt)}`")
    L.append("")

    L.append("## 2) 推理过程 humanness（端到端 CoT 感，0~1，越高越像人/越不像 RAG）")
    L.append(f"- teacher 平均: **{avg_teacher_h:.3f}**")
    L.append(f"- student 平均: **{avg_student_h:.3f}**")
    L.append(f"- 差异 (student - teacher): **{(avg_student_h - avg_teacher_h):+.3f}**")
    L.append("")
    L.append("### teacher humanness 分布")
    L.append("```")
    L.append(format_dist(teacher_h))
    L.append("```")
    L.append("### student humanness 分布")
    L.append("```")
    L.append(format_dist(student_h))
    L.append("```")
    L.append("")

    L.append("## 3) RAG 痕迹类型频次（多标签累计）")
    all_types = sorted(set(teacher_traces) | set(student_traces))
    L.append("| 类型 | teacher | student |")
    L.append("| --- | ---: | ---: |")
    for t in all_types:
        L.append(f"| `{t}` | {teacher_traces.get(t, 0)} | {student_traces.get(t, 0)} |")
    L.append("")
    L.append(
        "_类型说明：`explicit_ref`=显式引用语；`verbatim_copy`=大段照搬政策；"
        "`ref_enumeration`=罗列参考；`policy_source`=显式标注政策依据。_"
    )
    L.append("")

    L.append("## 4) accuracy × humanness 交叉")
    L.append("不同准确率档位下，student 的平均 humanness：")
    L.append("| accuracy_label | 样本数 | student humanness 均值 |")
    L.append("| --- | ---: | ---: |")
    for lbl in ["correct", "partial", "incorrect", "unknown"]:
        vals = by_label.get(lbl, [])
        if not vals:
            continue
        L.append(f"| {lbl} | {len(vals)} | {mean(vals):.3f} |")
    L.append("")

    L.append("## 5) student humanness 最低 Top-10（RL 重点优化目标）")
    for i, r in enumerate(lowest, 1):
        q = (r.get("query") or "").replace("\n", " ")[:80]
        h = safe_float(r.get("student_reasoning_humanness"))
        reason = (r.get("student_humanness_reason") or "").replace("\n", " ")[:120]
        types = safe_list(r.get("student_rag_trace_types"))
        L.append(f"{i}. humanness=**{h:.2f}** types={types}")
        L.append(f"   - Q: {q}")
        L.append(f"   - 理由: {reason}")
    L.append("")

    L.append("## 解读提示")
    L.append("- SFT 阶段 student humanness ≈ teacher humanness 是预期的（蒸馏对齐结果）。")
    L.append("- RL 阶段目标：在保持 accuracy 的前提下把 student humanness 拉高。")
    L.append("- 上面 Top-10 humanness 最低的样本，是 RL 收益最大的训练源。")

    report = "\n".join(L) + "\n"
    Path(args.out).write_text(report, encoding="utf-8")
    log.info("\n%s", report)
    log.info("已写入 %s", args.out)


if __name__ == "__main__":
    main()
