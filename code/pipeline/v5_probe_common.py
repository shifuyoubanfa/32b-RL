"""derag_v5 探针共用逻辑 —— 全确定性，复用 reward_v3 的探测器与事实抽取。

三件事，全链唯一来源（step150/151/152/153/159 共用，防各处定义漂移）：
1. real_trace：真痕迹计数 = explicit_ref + ref_enumeration + verbatim_copy，【排除 policy_source】
   （102 号已证 policy_source ~60% 是合法税法引用假阳；排除后才是该优化的真目标）。
2. is_clean：think 没有真痕迹 且 copy_ratio≤0.30。
3. answer_in_support：答案的【结论极性/关键数字】落在 V1 的 N 次采样集合里 = 没漂移 V1。
   —— 用户点1：不追求"答案对"，只追求"V1 自己也会这么答"。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline import reward_v3 as R3

# 复制率上限：与冷启动/门控同口径（grounding≠照抄）
CLEAN_COPY_MAX = 0.30
# 照抄阈值：copy_ratio≥此值即记一笔 verbatim_copy 真痕迹。
# 注意：reward_v3.trace_counts（v4 版）把 verbatim_copy 硬编码为 0（只有 frozen 版才返回），
# 所以 real_trace 这里自己按 copy_ratio 判，保证"纯照抄但无触发词"也计入 count（否则 rule 模式漏检）。
VERBATIM_COPY_MIN = 0.40

# 结论极性反义对（answer 说 A，V1 多数说 ¬A = 跑偏）。键值双向。
_FLIP_PAIRS = [
    ("免征", "应缴"), ("免税", "应缴"), ("不得", "可以"), ("不得", "允许"),
    ("禁止", "可以"), ("禁止", "允许"), ("无需", "需要"), ("不需要", "需要"),
    ("不超过", "超过"),
]
FLIP: dict[str, set[str]] = {}
for _a, _b in _FLIP_PAIRS:
    FLIP.setdefault(_a, set()).add(_b)
    FLIP.setdefault(_b, set()).add(_a)


def parse(text: str) -> tuple[str, str]:
    return R3.parse_think_answer(text)


def refs_of(user_prompt: str) -> str:
    return R3.extract_references(user_prompt or "")


def real_trace(think: str, refs: str = "") -> dict:
    """真痕迹（排除 policy_source）。返回 {count, breakdown, copy_ratio, types}。"""
    tc = R3.trace_counts(think or "", refs or "")
    c = tc["counts"]
    copy_ratio = tc["copy_ratio"]
    # verbatim_copy 自己按 copy_ratio 判（trace_counts 的 v4 版恒返回 0，不可信，见模块顶注）
    breakdown = {
        "explicit_ref": int(c.get("explicit_ref", 0)),
        "ref_enumeration": int(c.get("ref_enumeration", 0)),
        "verbatim_copy": int(copy_ratio >= VERBATIM_COPY_MIN),
    }
    count = sum(breakdown.values())
    return {
        "count": count,
        "breakdown": breakdown,
        "policy_source": int(c.get("policy_source", 0)),  # 仅监控，不计入 count
        "copy_ratio": copy_ratio,
        "types": [k for k, v in breakdown.items() if v > 0],
    }


def is_problem(think: str, refs: str = "") -> bool:
    """这道题的 think 还有真念手册痕迹 = 是病题（需要去 RAG）。"""
    return real_trace(think, refs)["count"] >= 1


def is_clean(think: str, refs: str = "") -> bool:
    """think 干净 = 无真痕迹 且 不照抄。"""
    rt = real_trace(think, refs)
    return rt["count"] == 0 and rt["copy_ratio"] <= CLEAN_COPY_MAX


def conclusion_slots(answer: str) -> set[str]:
    """答案里的结论极性词集合（免征/不得/应缴…）。"""
    a = answer or ""
    return {p for p in R3._POLARITY if p in a}


def value_slots(answer: str) -> set[str]:
    """答案里的关键数字（税率/金额/期限…）。"""
    return set(R3.nums(answer or ""))


def build_support(v1_answers: list[str]) -> dict:
    """从 V1 的 N 个答案建"V1 认可答案范围"。"""
    sample_concls = [conclusion_slots(a) for a in v1_answers]
    value_union: set[str] = set()
    for a in v1_answers:
        value_union |= value_slots(a)
    # 多数结论极性：在 ≥半数 V1 采样里出现的极性词
    from collections import Counter
    cnt = Counter()
    for s in sample_concls:
        for w in s:
            cnt[w] += 1
    half = max(1, len(v1_answers) // 2 + 1)
    majority_concl = {w for w, n in cnt.items() if n >= half}
    return {
        "n": len(v1_answers),
        "sample_concls": [sorted(s) for s in sample_concls],
        "value_union": sorted(value_union),
        "majority_concl": sorted(majority_concl),
    }


def answer_in_support(answer: str, support: dict) -> dict:
    """答案是否落在 V1 范围里（没漂移）。返回 {in_support, reason, ...}。

    主判：结论极性。新答案的结论极性须被 ≥1 个 V1 采样覆盖，且不与 V1 多数结论矛盾。
    结论极性为空（纯数字型答案）时回退到数字：不得引入 V1 从未出现过的关键数字。
    """
    concl = conclusion_slots(answer)
    vals = value_slots(answer)
    sample_concls = [set(s) for s in support.get("sample_concls", [])]
    majority = set(support.get("majority_concl", []))
    value_union = set(support.get("value_union", []))

    # 反义矛盾：答案断言某极性，而 V1 多数持相反极性
    contradict = any((FLIP.get(c, set()) & majority) for c in concl)
    value_ok = vals <= value_union  # 没引入 V1 没说过的数字

    if concl:
        covered = any(concl <= s for s in sample_concls)
        in_support = covered and not contradict
        reason = ("ok" if in_support else
                  ("contradict_majority" if contradict else "concl_not_covered"))
    else:
        # 纯数字型答案：靠数字守门
        in_support = value_ok
        reason = "ok" if in_support else "introduced_new_number"
    return {
        "in_support": bool(in_support),
        "reason": reason,
        "concl": sorted(concl),
        "values": sorted(vals),
        "contradict": bool(contradict),
        "value_ok": bool(value_ok),
    }


def check_sample(text: str, user_prompt: str, support: dict) -> dict:
    """对一条模型输出（含 think+answer）做完整判定：think 干净 ∧ 答案在 V1 范围。"""
    think, answer = parse(text)
    refs = refs_of(user_prompt)
    rt = real_trace(think, refs)
    clean = rt["count"] == 0 and rt["copy_ratio"] <= CLEAN_COPY_MAX
    sup = answer_in_support(answer, support)
    return {
        "think_clean": bool(clean),
        "answer_in_support": sup["in_support"],
        "pass": bool(clean and sup["in_support"]),
        "real_trace": rt["count"],
        "trace_types": rt["types"],
        "copy_ratio": rt["copy_ratio"],
        "support_reason": sup["reason"],
        "answer_concl": sup["concl"],
        "answer_values": sup["values"],
    }
