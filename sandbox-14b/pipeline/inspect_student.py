"""诊断工具：把 step05 学生模型的原始输出打出来，定位"空 think / 乱码"问题。

用法：
    python pipeline/inspect_student.py          # 看前 3 条 + 全量统计
    python pipeline/inspect_student.py --n 10    # 看前 10 条
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import STUDENT_OUTPUTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    with open(STUDENT_OUTPUTS, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]

    print(f"总样本: {len(recs)}\n" + "=" * 80)

    # 全量统计
    empty_think = sum(1 for r in recs if not (r.get("student_think") or "").strip())
    empty_ans = sum(1 for r in recs if not (r.get("student_answer") or "").strip())
    raw_lens = [len((r.get("student_raw") or "")) for r in recs]
    has_think_tag = sum(1 for r in recs if "<think>" in (r.get("student_raw") or ""))
    has_answer_tag = sum(1 for r in recs if "<answer>" in (r.get("student_raw") or ""))
    avg_raw = sum(raw_lens) / len(raw_lens) if raw_lens else 0

    print("【全量统计】")
    print(f"  student_raw 平均长度:      {avg_raw:.0f} 字符  (min={min(raw_lens)}, max={max(raw_lens)})")
    print(f"  含 <think> 标签的:         {has_think_tag}/{len(recs)}")
    print(f"  含 <answer> 标签的:        {has_answer_tag}/{len(recs)}")
    print(f"  解析后 think 为空的:       {empty_think}/{len(recs)}")
    print(f"  解析后 answer 为空的:      {empty_ans}/{len(recs)}")
    print("=" * 80)

    # 逐条原文
    for i, r in enumerate(recs[: args.n], 1):
        print(f"\n########## 样本 {i} ##########")
        print(f"[query]\n{r.get('query')}")
        print(f"\n[student_raw 原始输出] (repr，能看到换行/特殊字符)")
        print(repr(r.get("student_raw")))
        print(f"\n[解析 student_think]\n{r.get('student_think')!r}")
        print(f"\n[解析 student_answer]\n{r.get('student_answer')!r}")
        print(f"\n[teacher_answer 对照]\n{(r.get('teacher_answer') or '')[:200]}")
        print("-" * 80)


if __name__ == "__main__":
    main()
