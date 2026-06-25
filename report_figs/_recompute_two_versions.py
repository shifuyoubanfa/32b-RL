# -*- coding: utf-8 -*-
"""从 5 个 scores.jsonl 复算两版三件套：① 原版(格式失败计入分母) ② 剔除格式失败版。
口径严格对齐 step_v2_eval.py 的聚合逻辑。只读，输出写 UTF-8 文件。"""
import json

D = "32b强化学习/derag2"
STAGES = [("V1 基线", "v2-baseline-v1"), ("SFT", "v2-sft-2s"), ("RFT", "v2-rft-2s-2s"),
          ("DPO", "v2-2s-2s-2s"), ("GRPO", "v2-2s-2s-2s-grpo")]

def load(tag):
    rows = []
    with open(f"{D}/{tag}_scores.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def rule_ok(r):
    # 新版 scores 有 rule_pass；老版(baseline/sft/rft)只有 has_rag_style → rule_pass = 无表面痕迹∧未格式失败
    if r.get("rule_pass") is not None:
        return bool(r["rule_pass"])
    return (not r.get("has_rag_style")) and r.get("format_ok", True)

def agg(rows):
    valid = [r["clean_score"] for r in rows if r.get("clean_n", 0) > 0 and r.get("clean_score") is not None]
    n = len(rows)
    rule_pass = sum(1 for r in rows if rule_ok(r))
    n_pool = sum(1 for r in rows if not r.get("no_pool"))
    in_pool = sum(1 for r in rows if (not r.get("no_pool")) and r.get("in_pool"))
    clean = sum(valid) / len(valid) if valid else float("nan")
    return {"n": n, "n_clean_valid": len(valid), "clean": clean,
            "rule_pass": rule_pass, "rule_rate": rule_pass / max(1, n),
            "n_pool": n_pool, "in_pool": in_pool, "in_pool_rate": in_pool / max(1, n_pool)}

out = open("32b强化学习/report_figs/_two_versions_out.txt", "w", encoding="utf-8")
def P(*a): print(*a, file=out)

P("阶段 | 版本 | N | kimi干净分 | 规则think通过 | 规则answer在池 | 格式失败")
P("-"*92)
results = {}
for name, tag in STAGES:
    rows = load(tag)
    n_fmt = sum(1 for r in rows if not r.get("format_ok", True))
    n_empty_ans = sum(1 for r in rows if r.get("answer_reason") == "empty_answer")
    raw = agg(rows)
    excl = agg([r for r in rows if r.get("format_ok", True)])  # 剔除格式失败
    results[tag] = {"raw": raw, "excl": excl, "n_fmt": n_fmt, "n_empty_ans": n_empty_ans}
    P(f"{name:8s} | 原版    | {raw['n']:3d} | {raw['clean']:.3f} | "
      f"{raw['rule_pass']}/{raw['n']}={raw['rule_rate']*100:.1f}% | "
      f"{raw['in_pool']}/{raw['n_pool']}={raw['in_pool_rate']*100:.1f}% | {n_fmt}")
    P(f"{'':8s} | 剔格式失败| {excl['n']:3d} | {excl['clean']:.3f} | "
      f"{excl['rule_pass']}/{excl['n']}={excl['rule_rate']*100:.1f}% | "
      f"{excl['in_pool']}/{excl['n_pool']}={excl['in_pool_rate']*100:.1f}% | (空answer={n_empty_ans})")
    P("-"*92)

P("\n# 机读（给画图脚本用）:")
P(json.dumps({tag: {"raw": {k: results[tag]["raw"][k] for k in ("clean","rule_rate","in_pool_rate")},
                    "excl": {k: results[tag]["excl"][k] for k in ("clean","rule_rate","in_pool_rate")},
                    "n_fmt": results[tag]["n_fmt"]} for _, tag in STAGES}, ensure_ascii=False, indent=2))
out.close()
print("written -> _two_versions_out.txt")
