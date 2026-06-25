"""Compare the reports produced by the corrected merged-base DPO/GRPO chain."""

import argparse
import re
from pathlib import Path


PATTERNS = {
    "humanness": r"humanness 均值：\*\*([0-9.]+)\*\*",
    "grounded": r"grounded\(忠于参考\) 均值：\*\*([0-9.]+)\*\*",
    "accuracy": r"准确率\(平均分,漂移\)：\*\*([0-9.]+)\*\*",
    "correct": r"correct%：\*\*([0-9.]+)%",
    "correct_partial": r"correct\+partial%：\*\*([0-9.]+)%",
}


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    row = {"report": str(path)}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"cannot parse {key} from {path}")
        row[key] = float(match.group(1))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", required=True, help="label=/path/to/report.md")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for item in args.report:
        label, path = item.split("=", 1)
        rows.append((label, parse(Path(path))))
    base = rows[0][1]
    lines = [
        "# 合并基座 DPO→双 GRPO 最终对比报告",
        "",
        "| 模型 | humanness | grounded | acc | correct% | correct+partial% | Δh vs RFT基座 | Δacc vs RFT基座 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in rows:
        lines.append(
            f"| {label} | {row['humanness']:.3f} | {row['grounded']:.3f} | {row['accuracy']:.3f} | "
            f"{row['correct']:.1f}% | {row['correct_partial']:.1f}% | "
            f"{row['humanness'] - base['humanness']:+.3f} | {row['accuracy'] - base['accuracy']:+.3f} |")
    lines += [
        "",
        "## 实验语义",
        "- RFT merged base：原始 V1 与最终 RFT LoRA 合并后的完整模型。",
        "- DPO on merged base：在 RFT merged base 上训练新 LoRA，πref 为冻结 RFT merged base。",
        "- GRPO from RFT merged：从 RFT merged base 新建 LoRA，πref 为冻结 RFT merged base。",
        "- GRPO from DPO merged：先把 DPO LoRA 合并进 RFT merged base，再新建 GRPO LoRA，πref 为冻结 DPO merged base。",
        "",
        "所有 πref 均通过“禁用当前新 LoRA后回到对应完整基座”实现，不依赖隐式双 adapter。",
    ]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
