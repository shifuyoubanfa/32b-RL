"""derag_v4 deterministic features, gates, margins, and GRPO reward.

The v4 gate intentionally separates three concerns:

* L0 hard failures are high-precision and deterministic.
* L1 tax-aware features describe suspicious text but never reject by themselves.
* Kimi handles the remaining semantic judgement in step125.

The old v3 trace counter is frozen for longitudinal reports.  Do not use it to
make new training decisions.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from pipeline import reward as legacy_reward

TRACE_RE_V3_VERSION = "trace_re_v3_frozen_20260612"
TRACE_RE_V4_VERSION = "trace_re_v4_gate_judge_v4.0"
TRACE_KEYS = ("explicit_ref", "verbatim_copy", "ref_enumeration", "policy_source")

_EXPLICIT_RE = [
    re.compile(p)
    for p in (
        r"参考问答对",
        r"参考资料",
        r"参考内容",
        r"检索结果",
        r"资料显示",
        r"资料表明",
        r"原始资料",
        r"原始问答",
        r"根据(检索|参考|提供的|资料|原文)",
        r"如(上|前)(参考|资料|所示|文|述)",
        r"对照(参考|资料|问答)",
    )
]
_ENUM_RE = [
    re.compile(p)
    for p in (
        r"问答对\s*[0-9一二三四五六七八九十]",
        r"第\s*[0-9一二三四五六七八九十]\s*个参考",
        r"(逐条|依次|分别)(对照|参考|分析|归纳)",
    )
]
_DOC_TOKEN_RE = re.compile(
    r"(?:《[^》]{2,100}》(?:第[一二三四五六七八九十百千万\d]+条)?|"
    r"[^\s，。；、]{0,16}[财税会发函公告令]\s*[〔\[\(（]?\d{4}[〕\]\)）]?\s*\d+\s*号)"
)
_POLICY_LABEL_LINE_RE = re.compile(
    r"(?m)^\s*(?:政策依据|参考文件|参考法规|文件依据)\s*[:：]"
)
_CONTENT_WORD_RE = re.compile(
    r"(规定|按照|明确|适用|应当|可以|不得|免征|征收|处理|确认|扣除|申报|"
    r"判断|区分|属于|不属于|满足|导致|计算|缴纳|计入|确认|结转|抵扣|"
    r"依据|由于|若|如果|因此|所以|意味着|对应|决定|需要|无需|涉及|"
    r"享受|选择|采用|执行|发生|取得|提供|销售|转让|出租|支付|收到)"
)
_IMG_TRACE_RE = re.compile(
    r"<img\b|https?://\S*(?:aliyuncs|oss-|servu)|\b\S+\.(?:png|jpe?g|xlsx?|pdf)\b",
    re.I,
)
_OFFICIAL_URL_RE = re.compile(r"https?://\S*(?:gov\.cn|chinatax\.gov\.cn)\S*", re.I)
_CUSTOMER_TRACE_RE = re.compile(
    r"小贴士|温馨提(?:示|醒)|参考下图|如下图|哦[~～]|哒[。~～]|亲[，~～]"
)
_QA_TRACE_RE = re.compile(r"问题\s*\d+\s*[:：]|回答\s*[:：]")
_SENT_SPLIT = re.compile(r"[。！？!?\n;；]+")
_NUM_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|个?(?:日|天|个月|月|年|倍|周|季度|元|万元|亿元|万|亿)|号)?"
)
_BARE_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_POLARITY = [
    "免征", "免税", "不得", "不可以", "无需", "不需要", "不超过",
    "应缴", "需要", "可以", "超过", "禁止", "允许", "应当",
]
_ENTRY_LINE_RE = re.compile(r"(?m)^\s*(?:借|贷)\s*[:：记].*$")
_ACCOUNTING_HEADER_RE = re.compile(r"(?m)^\s*【?(?:财务|预算)会计】?\s*$")
_DATE_RE = re.compile(
    r"(?:自)?\d{4}年\d{1,2}月\d{1,2}日起|\d{4}年\d{1,2}月|\d{1,3}(?:日|天|个月|月|年|季度)"
)
_RATE_MONEY_RE = re.compile(
    r"\d+(?:\.\d+)?%|万分之[零一二三四五六七八九十百点]+|千分之[零一二三四五六七八九十百点]+|"
    r"\d+(?:\.\d+)?(?:万元|亿元|元|万|亿|个月|倍)"
)
_FORM_PATH_RE = re.compile(
    r"A\d{5,6}表?|第\d+行(?:第\d+列|栏)?|【[^】]{1,40}】\s*[→>-]+\s*【[^】]{1,40}】"
)
_FORM_NAME_RE = re.compile(r"《[^》]{1,80}(?:表|单|凭证)》")
_ACCOUNTING_TERM_RE = re.compile(
    r"(?:应交税费|应付职工薪酬|银行存款|管理费用|销售费用|财务费用|主营业务收入|"
    r"其他业务收入|固定资产|无形资产|累计折旧|递延所得税资产|递延所得税负债)"
    r"(?:[—\-－][^\s，。；]{1,30})?"
)
_LEGAL_TERMS = (
    "非正常损失", "汇算清缴", "应纳税所得额", "增值税专用发票", "进项税额",
    "销项税额", "留抵税额", "一般纳税人", "小规模纳税人", "企业所得税",
    "个人所得税", "印花税", "土地增值税", "契税", "房产税", "城镇土地使用税",
    "消费税", "附加税费", "研发费用加计扣除", "税前扣除", "视同销售",
    "纳税义务发生时间", "计税依据", "应纳税额", "应税所得率", "税收优惠",
    "免税收入", "不征税收入", "递延纳税", "纳税调整", "关联交易", "特别纳税调整",
    "完税凭证", "应税凭证", "产权转移书据", "营业账簿", "财产租赁合同",
    "融资租赁合同", "建设工程合同", "技术合同", "买卖合同", "运输合同",
    "财产保险合同", "证券交易", "固定资产", "无形资产", "长期待摊费用",
    "公允价值", "账面价值", "计税基础", "会计利润", "税务处理", "会计处理",
    "纳税申报", "代扣代缴", "自行申报", "主管税务机关",
)
_LEGAL_TERMS_RE = re.compile("|".join(map(re.escape, sorted(_LEGAL_TERMS, key=len, reverse=True))))


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def parse_think_answer(text: str) -> tuple[str, str]:
    return legacy_reward.parse_think_answer(text)


def extract_references(user_prompt: str) -> str:
    return legacy_reward.extract_references(user_prompt)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def ngrams(text: str, n: int = 8) -> Counter:
    s = re.sub(r"\s+", "", text or "")
    return Counter(s[i:i + n] for i in range(max(0, len(s) - n + 1)))


def distinct2(text: str) -> float:
    s = re.sub(r"\s+", "", text or "")
    if len(s) < 2:
        return 0.0
    grams = [s[i:i + 2] for i in range(len(s) - 1)]
    return len(set(grams)) / max(1, len(grams))


def nums(text: str) -> set[str]:
    out: set[str] = set()
    t = nfkc(text)
    for match in _NUM_RE.findall(t):
        item = re.sub(r"\s+", "", match.replace(",", ""))
        if item:
            out.add(item)
    for match in _BARE_NUM_RE.findall(t):
        item = match.replace(",", "")
        if len(re.sub(r"\D", "", item)) >= 2:
            out.add(item)
    return out


def facts(text: str) -> set[str]:
    out = set(nums(text))
    for polarity in _POLARITY:
        if polarity in (text or ""):
            out.add(polarity)
    return out


def mask_tax_facts(text: str) -> str:
    """Mask tax facts that often have only one correct surface form.

    The mask is intentionally broad because masked_copy is an L1 soft feature.
    Raw copy still has a 0.55 L0 cap.
    """
    s = nfkc(text)
    s = _OFFICIAL_URL_RE.sub("□", s)
    for regex in (
        _ENTRY_LINE_RE,
        _DATE_RE,
        _RATE_MONEY_RE,
        _DOC_TOKEN_RE,
        _FORM_PATH_RE,
        _FORM_NAME_RE,
        _ACCOUNTING_TERM_RE,
        _LEGAL_TERMS_RE,
    ):
        s = regex.sub("□", s)
    return re.sub(r"□+", "□", s)


def _policy_hits_v3(think: str) -> tuple[int, list[str]]:
    hits: list[str] = []
    for sent in split_sentences(think):
        docs = _DOC_TOKEN_RE.findall(sent)
        if not docs:
            continue
        if len(docs) == 1 and len(sent) >= 15 and _CONTENT_WORD_RE.search(sent):
            continue
        if len(docs) >= 2 or not _CONTENT_WORD_RE.search(sent):
            hits.extend(docs)
    list_like = len(_DOC_TOKEN_RE.findall(think or "")) >= 3
    label = bool(_POLICY_LABEL_LINE_RE.search(think or ""))
    return len(hits) + int(list_like) + int(label), hits[:8]


def trace_counts_v3_frozen(think: str, references: str = "") -> dict[str, Any]:
    """The original counter used by prior reports. Never change its semantics."""
    think = think or ""
    explicit = sum(len(r.findall(think)) for r in _EXPLICIT_RE)
    enum = sum(len(r.findall(think)) for r in _ENUM_RE)
    policy, policy_examples = _policy_hits_v3(think)
    copy_ratio = legacy_reward.copy_signal(think, references or "")
    verbatim = int(copy_ratio >= 0.40)
    counts = {
        "explicit_ref": explicit,
        "ref_enumeration": enum,
        "policy_source": policy,
        "verbatim_copy": verbatim,
    }
    return {
        "version": TRACE_RE_V3_VERSION,
        "counts": counts,
        "total": sum(counts.values()),
        "types": [k for k in TRACE_KEYS if counts.get(k, 0) > 0],
        "copy_ratio": round(copy_ratio, 4),
        "enum_density": round(enum / max(1, len(split_sentences(think))), 4),
        "policy_examples": policy_examples,
    }


def citation_metrics(think: str) -> dict[str, Any]:
    standalone = 0
    examples: list[str] = []
    for sentence in split_sentences(think):
        reduced = _FORM_NAME_RE.sub("", sentence)
        docs = _DOC_TOKEN_RE.findall(reduced)
        if not docs:
            continue
        # A citation supports reasoning when content/reasoning words are in the
        # same sentence. Multiple documents are allowed for comparisons.
        if not _CONTENT_WORD_RE.search(reduced) or len(docs) >= 3:
            standalone += 1
            examples.append(sentence[:180])
    n_sent = max(1, len(split_sentences(think)))
    return {
        "standalone_units": standalone,
        "citation_density": round(standalone / n_sent, 4),
        "citation_examples": examples[:8],
    }


def trace_counts(think: str, references: str = "") -> dict[str, Any]:
    """v4 operational traces. Necessary citations and tax facts are not traces."""
    think = think or ""
    explicit = sum(len(r.findall(think)) for r in _EXPLICIT_RE)
    enum = sum(len(r.findall(think)) for r in _ENUM_RE)
    citations = citation_metrics(think)
    raw_copy = legacy_reward.copy_signal(think, references or "")
    masked_copy = legacy_reward.copy_signal(mask_tax_facts(think), mask_tax_facts(references or ""))
    counts = {
        "explicit_ref": explicit,
        "ref_enumeration": enum,
        "policy_source": citations["standalone_units"],
        # v4 does not convert copy into a binary trace. It remains a continuous
        # feature and burden term.
        "verbatim_copy": 0,
    }
    return {
        "version": TRACE_RE_V4_VERSION,
        "counts": counts,
        "total": sum(counts.values()),
        "types": [k for k in TRACE_KEYS if counts.get(k, 0) > 0],
        "copy_ratio": round(raw_copy, 4),
        "masked_copy": round(masked_copy, 4),
        "enum_density": round(enum / max(1, len(split_sentences(think))), 4),
        **citations,
    }


def answer_score(answer: str, gold: str) -> tuple[float, dict[str, Any]]:
    score, info = legacy_reward.answer_drift(answer, gold)
    gold_facts = facts(gold)
    answer_facts = facts(answer)
    info["fact_recall"] = round(
        (len(gold_facts & answer_facts) / len(gold_facts)) if gold_facts else 1.0,
        4,
    )
    return score, info


def grounding_metrics(
    think: str,
    answer: str,
    user_prompt: str,
    gold: str,
    query: str = "",
) -> dict[str, Any]:
    refs = extract_references(user_prompt)
    allowed_text = "\n".join([refs, gold or "", query or ""])
    allowed_nums = nums(allowed_text)
    used_nums = nums("\n".join([think or "", answer or ""]))
    introduced = sorted(used_nums - allowed_nums)
    gold_facts = facts(gold)
    think_facts = facts(think)
    grounding_floor_ok = not gold_facts or bool(think_facts & (facts(refs) | gold_facts))
    _, info = answer_score(answer, gold)
    return {
        "introduced_nums": introduced,
        "introduced_nums_ok": len(introduced) == 0,
        "grounding_floor_ok": grounding_floor_ok,
        "fact_recall": float(info.get("fact_recall", 0.0)),
    }


def _normalise_degen(text: str) -> str:
    s = nfkc(text)
    s = _ENTRY_LINE_RE.sub("", s)
    s = _ACCOUNTING_HEADER_RE.sub("", s)
    s = re.sub(r"[“”\"'‘’]", "", s)
    s = re.sub(r"[—–－]+", "-", s)
    s = re.sub(r"A\d{5,6}", "⟨F⟩", s)
    s = _DATE_RE.sub("⟨D⟩", s)
    s = _DOC_TOKEN_RE.sub("⟨C⟩", s)
    s = _BARE_NUM_RE.sub("⟨N⟩", s)
    return s


def degeneration_metrics(think: str) -> dict[str, Any]:
    norm = _normalise_degen(think)
    sentences = split_sentences(norm)
    eligible_short = [
        s for s in sentences
        if not _ENTRY_LINE_RE.match(s) and not re.search(r"\d", s)
    ]
    short_ratio = sum(1 for s in eligible_short if len(s) <= 6) / max(1, len(eligible_short))
    eight = ngrams(norm, 8)
    max_repeat = max(eight.values()) if eight else 0
    d2 = distinct2(norm)
    sentence_counts = Counter(re.sub(r"\s+", "", s) for s in sentences if len(re.sub(r"\s+", "", s)) >= 10)
    repeated_sentence = max(sentence_counts.values()) if sentence_counts else 0
    # A repeated tax/accounting term is not degeneration. High n-gram repeat
    # becomes a hard failure only when the whole text is also abnormally low
    # diversity; otherwise Kimi receives it as an L1 soft warning.
    extreme = d2 < 0.10 or repeated_sentence >= 3 or (max_repeat >= 5 and d2 < 0.35)
    soft = short_ratio > 0.25 or max_repeat >= 3
    return {
        "short_sentence_ratio": round(short_ratio, 4),
        "max_8gram_repeat": max_repeat,
        "distinct2": round(d2, 4),
        "max_sentence_repeat": repeated_sentence,
        "degen_soft": soft,
        "extreme_degen": extreme,
        # Compatibility alias. v4 only hard-rejects extreme degeneration.
        "degenerate": extreme,
    }


def _l0_reasons(features: dict[str, Any]) -> list[str]:
    reasons = []
    if not features.get("format_ok"):
        reasons.append("format")
    if features.get("introduced_nums"):
        reasons.append("introduced_nums")
    if not features.get("grounding_floor_ok"):
        reasons.append("grounding_floor")
    counts = features.get("trace_counts") or {}
    if counts.get("explicit_ref", 0):
        reasons.append("explicit_ref")
    if counts.get("ref_enumeration", 0):
        reasons.append("ref_enumeration")
    if features.get("label_line"):
        reasons.append("label_line")
    if float(features.get("copy_ratio", 0.0)) > 0.55:
        reasons.append("raw_copy_cap")
    if features.get("img_trace"):
        reasons.append("img_trace")
    if features.get("extreme_degen"):
        reasons.append("extreme_degen")
    return reasons


def burden(features: dict[str, Any]) -> int:
    counts = features.get("trace_counts") or {}
    masked = float(features.get("masked_copy", 0.0))
    copy_units = 0 if masked <= 0.35 else 1 if masked <= 0.45 else 2 if masked <= 0.55 else 3
    return int(counts.get("explicit_ref", 0)) + int(counts.get("ref_enumeration", 0)) + int(
        features.get("standalone_citation_units", 0)
    ) + copy_units


def gate_decision(features: dict[str, Any], mode: str = "s1_keep") -> dict[str, Any]:
    """Single deterministic entry point for Stage1 and on-policy DPO choices."""
    reasons = list(features.get("l0_reasons") or _l0_reasons(features))
    l0_pass = not reasons
    if mode == "s1_keep":
        return {"pass": l0_pass, "route": "judge" if l0_pass else "hard_fail", "reasons": reasons}
    if mode == "dpo_chosen":
        chosen = (
            l0_pass
            and burden(features) == 0
            and float(features.get("masked_copy", 1.0)) <= 0.30
            and not features.get("degen_soft")
            and float(features.get("fact_recall", 0.0)) >= 0.75
            and float(features.get("answer_score", 0.0)) >= 0.55
        )
        return {"pass": chosen, "route": "fast" if chosen else "reject", "reasons": reasons}
    raise ValueError(f"unknown gate mode: {mode}")


def candidate_features(
    text: str,
    user_prompt: str,
    gold_answer: str,
    query: str = "",
    *,
    min_think: int = 40,
    max_think: int = 2200,
) -> dict[str, Any]:
    think, answer = parse_think_answer(text)
    refs = extract_references(user_prompt)
    tc = trace_counts(think, refs)
    frozen = trace_counts_v3_frozen(think, refs)
    answer_value, answer_info = answer_score(answer, gold_answer)
    grounding = grounding_metrics(think, answer, user_prompt, gold_answer, query)
    degen = degeneration_metrics(think)
    format_ok = bool(answer.strip()) and min_think <= len(think.strip()) <= max_think
    features: dict[str, Any] = {
        "think": think,
        "answer": answer,
        "think_len": len(think.strip()),
        "format_ok": format_ok,
        "trace_version": tc["version"],
        "trace_total": tc["total"],
        "trace_types": tc["types"],
        "trace_counts": tc["counts"],
        "copy_ratio": tc["copy_ratio"],
        "masked_copy": tc["masked_copy"],
        "enum_density": tc["enum_density"],
        "citation_density": tc["citation_density"],
        "standalone_citation_units": tc["standalone_units"],
        "citation_examples": tc["citation_examples"],
        "answer_score": round(float(answer_value), 4),
        "answer_sim": answer_info.get("sim"),
        "fact_recall": grounding["fact_recall"],
        "introduced_nums": grounding["introduced_nums"],
        "grounding_floor_ok": grounding["grounding_floor_ok"],
        "label_line": bool(_POLICY_LABEL_LINE_RE.search(think or "")),
        "img_trace": bool(_IMG_TRACE_RE.search(_OFFICIAL_URL_RE.sub("", think or ""))),
        "customer_trace": bool(_CUSTOMER_TRACE_RE.search(think or "")),
        "qa_trace": bool(_QA_TRACE_RE.search(think or "")),
        **degen,
        "frozen_trace_version": frozen["version"],
        "frozen_trace_total": frozen["total"],
        "frozen_trace_counts": frozen["counts"],
        "frozen_trace_types": frozen["types"],
    }
    features["l0_reasons"] = _l0_reasons(features)
    features["l0_pass"] = not features["l0_reasons"]
    features["burden"] = burden(features)
    features["clean"] = gate_decision(features, "dpo_chosen")["pass"]
    features["trace_heavy"] = features["burden"] >= 1
    features["repair_route"] = (
        "trace" if features["customer_trace"] or features["qa_trace"]
        else "paraphrase" if features["masked_copy"] > 0.35
        else "citation" if features["standalone_citation_units"] > 0
        else "none"
    )
    features["text"] = text
    return features


def derag_reward(text: str, user_prompt: str, gold_answer: str, query: str = "") -> tuple[float, dict[str, Any]]:
    features = candidate_features(text, user_prompt, gold_answer, query)
    if not features["format_ok"] or features["extreme_degen"]:
        return -1.0, {**features, "reward_gate": "format_or_extreme_degen"}
    if features["fact_recall"] < 0.80 or features["introduced_nums"] or not features["grounding_floor_ok"]:
        value = 0.1 * max(0.0, min(1.0, features["fact_recall"]))
        return value, {**features, "reward_gate": "fact_guard"}
    if features["copy_ratio"] > 0.55 or features["img_trace"]:
        return 0.1, {**features, "reward_gate": "hard_trace_guard"}
    counts = features["trace_counts"]
    masked = features["masked_copy"]
    penalty = (
        0.34 * counts.get("explicit_ref", 0)
        + 1.00 * counts.get("ref_enumeration", 0)
        + 0.20 * features["standalone_citation_units"]
        + 1.20 * max(0.0, features["citation_density"] - 0.15)
        + 2.00 * max(0.0, masked - 0.25)
        + 8.00 * max(0.0, masked - 0.40)
    )
    trace_score = 1.0 / (1.0 + penalty)
    value = 0.3 + 0.7 * trace_score
    return float(max(-1.0, min(1.0, value))), {
        **features,
        "reward_gate": "ok",
        "trace_score": round(trace_score, 4),
    }


def margin_ok(chosen: dict[str, Any], rejected: dict[str, Any]) -> bool:
    delta_b = burden(rejected) - burden(chosen)
    delta_masked = float(rejected.get("masked_copy", 0.0)) - float(chosen.get("masked_copy", 0.0))
    return delta_b >= 2 or (delta_b >= 1 and delta_masked >= 0.12)


def mcnemar_net(base_feats: list[dict[str, Any]], cand_feats: list[dict[str, Any]]) -> dict[str, Any]:
    improved = worsened = tied = 0
    for base, cand in zip(base_feats, cand_feats):
        if base.get("frozen_trace_version") != cand.get("frozen_trace_version"):
            raise AssertionError("frozen trace counter versions differ")
        bt = int(base.get("frozen_trace_total", 0))
        ct = int(cand.get("frozen_trace_total", 0))
        if ct < bt:
            improved += 1
        elif ct > bt:
            worsened += 1
        else:
            tied += 1
    return {"improved": improved, "worsened": worsened, "tied": tied, "net": improved - worsened}
