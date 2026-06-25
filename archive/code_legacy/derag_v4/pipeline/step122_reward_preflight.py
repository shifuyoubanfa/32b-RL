"""Lightweight derag_v4 reward preflight.

This is a cheap guard that catches obvious regressions before GPU training:
format failures score negative, trace-heavy text scores below clean text, and
simple surgery-like trace removal raises reward without touching answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline import reward_v3  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402

log = get_logger("step122_reward_preflight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    toy_prompt = "【参考问答对】\n企业所得税按规定处理，税率25%。\n【问题】\n企业所得税税率是多少？"
    gold = "企业所得税税率为25%。"
    clean = (
        "<think>用户询问企业所得税适用税率。题目明确给出该税种按照25%的税率处理，"
        "并且没有提供需要适用特殊优惠税率的其他条件，因此应当采用一般税率，结论为25%。"
        "</think><answer>企业所得税税率为25%。</answer>"
    )
    dirty = "<think>根据参考问答对1，参考资料显示企业所得税按规定处理，税率25%。参考文件如下。</think><answer>企业所得税税率为25%。</answer>"
    bad = "没有标签"
    rc, ic = reward_v3.derag_reward(clean, toy_prompt, gold, "企业所得税税率是多少？")
    rd, id_ = reward_v3.derag_reward(dirty, toy_prompt, gold, "企业所得税税率是多少？")
    rb, ib = reward_v3.derag_reward(bad, toy_prompt, gold, "企业所得税税率是多少？")
    status = "PASS" if rc > rd and rb < 0 and ic["clean"] else "FAIL"
    decision = {
        "status": status,
        "clean_reward": round(rc, 4),
        "dirty_reward": round(rd, 4),
        "bad_reward": round(rb, 4),
        "clean_features": {k: ic.get(k) for k in (
            "clean", "format_ok", "degenerate", "think_len", "trace_total",
            "trace_types", "copy_ratio", "enum_density", "fact_recall",
            "answer_score", "introduced_nums", "grounding_floor_ok", "reward_gate")},
        "dirty_features": {k: id_.get(k) for k in (
            "clean", "format_ok", "degenerate", "think_len", "trace_total",
            "trace_types", "copy_ratio", "enum_density", "fact_recall",
            "answer_score", "introduced_nums", "grounding_floor_ok", "reward_gate")},
        "bad_gate": ib.get("reward_gate"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    log.warning("RESULT reward_preflight status=%s clean=%.3f dirty=%.3f bad=%.3f -> %s",
                status, rc, rd, rb, out)
    print(json.dumps(decision, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
