"""Shared Kimi judge rubric for corrected-v2 and legacy evaluation steps."""

import json
import re
import statistics

from pipeline import kimi_client, reward

JUDGE_SYSTEM = "你是严格、客观的中文税务问答评审专家。只输出 JSON，不要任何额外文字。"

JUDGE_TEMPLATE = """请评审一个税务模型的「推理过程(think)」和「答案(answer)」。

【问题】
{query}

【参考资料（模型作答时可见的权威依据）】
{reference}

【标准答案（以此为绝对正确基准）】
{gold}

【待评审 think】
{think}

【待评审 answer】
{answer}

评审四件事，严格按 JSON 输出：
1. accuracy：待评审 answer 与标准答案是否一致。取值 "correct"(一致)/"partial"(部分一致)/"incorrect"(不一致或矛盾)；accuracy_score 给 0~1（1=完全一致）。
   注意：是否与标准答案一致，不是判它客观对错。
2. humanness：think 像不像「人在端到端推理」、越没有 RAG 检索痕迹越高，取 0~1。
   锚点：0.9~1.0 完全像人在从问题推导；0.0~0.2 出现"参考问答对1/2/3""根据检索结果"或大段照搬政策原文。
   即使没有这些词，若 think 是"从资料向答案归纳"而非"从问题向答案推导"，也应偏低。
3. grounded：think 的推理依据/口径/数字/结论是否【忠于参考资料、不与之矛盾】，取 0~1。
   1.0 完全扣参考、无矛盾；0.0 凭记忆臆测或与参考资料相矛盾（即使措辞自然）。这一项与 humanness 正交：自然但脱离/违背参考也应低分。
4. rag_traces：think 里出现的 RAG 痕迹类型列表，取值范围 ["explicit_ref","verbatim_copy","ref_enumeration","policy_source"]（没有则空列表）。

只输出如下 JSON：
{{"accuracy":"correct|partial|incorrect","accuracy_score":0.x,"humanness":0.x,"grounded":0.x,"rag_traces":[...],"comment":"一句话理由"}}"""


DERAG_JUDGE_SYSTEM = "你是严格、客观的中文税务问答质检专家。只输出 JSON，不要任何额外文字。"

DERAG_JUDGE_TEMPLATE = """请评审一个税务模型输出。注意：本任务不要求你判断“是否真的像人类推理”，只评审三个可操作目标。

【问题】
{query}

【参考资料（模型作答时可见的权威依据）】
{reference}

【标准答案（以此为答案一致性基准）】
{gold}

【待评审 think】
{think}

【待评审 answer】
{answer}

请严格按下面三项打分：

1. accuracy：待评审 answer 与标准答案是否一致。取值 "correct" / "partial" / "incorrect"；accuracy_score 取 0~1。
   注意：这里判断 answer 是否与标准答案一致，不判断标准答案本身是否客观正确。

2. trace_free：think 是否没有 RAG/检索/参考资料暴露痕迹，取 0~1。
   高分标准：自然说明依据和结论，但不暴露“参考问答对/参考资料/检索结果/资料显示/上述资料/问题1/原文提到/图片链接/文件链接”等系统检索痕迹。
   低分标准：出现上述字眼、机械编号引用、显式说自己在看参考资料，或大段照搬政策/参考原文。
   重要：不要因为它没有“像真人一样从问题出发推理”就扣分；只看是否去掉 RAG 痕迹、是否不机械照抄。

3. grounded：think 和 answer 中的事实、数字、政策口径、结论是否被参考资料支持且不与参考资料矛盾，取 0~1。
   高分标准：关键事实落点来自参考资料，未新增无依据数字/口径，answer 没有为了去 RAG 痕迹而变错。
   低分标准：凭常识发挥、遗漏关键限制、数字/税率/期限/结论与参考资料或标准答案冲突。

4. rag_traces：think 中出现的痕迹类型列表，取值范围 ["explicit_ref","verbatim_copy","ref_enumeration","policy_source"]。
   explicit_ref=显式“参考资料/检索/资料显示”等；verbatim_copy=明显照抄参考原文；ref_enumeration=问题1/参考问答对1等编号引用；policy_source=机械罗列文件号/政策依据。

只输出如下 JSON：
{{"accuracy":"correct|partial|incorrect","accuracy_score":0.x,"trace_free":0.x,"grounded":0.x,"rag_traces":[...],"comment":"一句话理由"}}"""


def parse_judge_json(text: str) -> dict:
    t = (text or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    acc = re.search(r'"accuracy"\s*:\s*"(correct|partial|incorrect)"', t)
    if not acc:
        raise ValueError(f"裁判 JSON 无法解析且抽不到 accuracy: {t[:160]}")

    def _num(key: str):
        mm = re.search(rf'"{key}"\s*:\s*([0-9.]+)', t)
        return float(mm.group(1)) if mm else None

    traces = list(dict.fromkeys(re.findall(r'"(explicit_ref|verbatim_copy|ref_enumeration|policy_source)"', t)))
    return {
        "accuracy": acc.group(1),
        "accuracy_score": _num("accuracy_score") or 0.0,
        "humanness": _num("humanness") or 0.0,
        "grounded": _num("grounded"),
        "rag_traces": traces,
        "comment": "salvaged_from_malformed_json",
    }


def parse_derag_judge_json(text: str) -> dict:
    t = (text or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "trace_free" in obj:
                return obj
        except Exception:
            pass
    acc = re.search(r'"accuracy"\s*:\s*"(correct|partial|incorrect)"', t)
    if not acc:
        raise ValueError(f"去RAG裁判 JSON 无法解析且抽不到 accuracy: {t[:160]}")

    def _num(key: str):
        mm = re.search(rf'"{key}"\s*:\s*([0-9.]+)', t)
        return float(mm.group(1)) if mm else None

    traces = list(dict.fromkeys(re.findall(r'"(explicit_ref|verbatim_copy|ref_enumeration|policy_source)"', t)))
    return {
        "accuracy": acc.group(1),
        "accuracy_score": _num("accuracy_score") or 0.0,
        "trace_free": _num("trace_free") or 0.0,
        "grounded": _num("grounded"),
        "rag_traces": traces,
        "comment": "salvaged_from_malformed_json",
    }


def judge_text(query: str, user_prompt: str, gold_answer: str, text: str) -> dict:
    think, answer = reward.parse_think_answer(text)
    prompt = JUDGE_TEMPLATE.format(
        query=query or "",
        reference=reward.extract_references(user_prompt or "")[:3000],
        gold=gold_answer or "",
        think=think[:4000],
        answer=answer[:2000],
    )
    raw = kimi_client.chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    parsed = parse_judge_json(raw)
    parsed.setdefault("grounded", None)
    return parsed


def judge_text_derag(query: str, user_prompt: str, gold_answer: str, text: str) -> dict:
    """Judge only the operational target: trace-free think + grounded facts + answer consistency."""
    think, answer = reward.parse_think_answer(text)
    prompt = DERAG_JUDGE_TEMPLATE.format(
        query=query or "",
        reference=reward.extract_references(user_prompt or "")[:3000],
        gold=gold_answer or "",
        think=think[:4000],
        answer=answer[:2000],
    )
    raw = kimi_client.chat(
        [{"role": "system", "content": DERAG_JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    parsed = parse_derag_judge_json(raw)
    parsed.setdefault("grounded", None)
    parsed.setdefault("trace_free", 0.0)
    parsed.setdefault("rag_traces", [])
    # Backward-compatible alias for generic aggregators; semantics are trace_free, not abstract humanness.
    parsed.setdefault("humanness", parsed.get("trace_free"))
    return parsed


PAIRWISE_SYSTEM = "你是严格、客观的中文税务问答偏好评审专家。只输出 JSON，不要任何额外文字。"

PAIRWISE_TEMPLATE = """请比较同一个税务问题下两段模型推理/答案，判断哪一个更符合目标。

【问题】
{query}

【参考资料（模型作答时可见的权威依据）】
{reference}

【标准答案】
{gold}

【候选 A】
{a}

【候选 B】
{b}

目标优先级：
1. answer 不得比另一方更不准确；
2. think 必须忠于参考资料、不臆测；
3. 在满足 1/2 的前提下，更像人在从问题出发自然推导，而不是从参考资料/检索结果/政策原文机械归纳；
4. 不奖励单纯更短；若两者都正确且自然度接近，应判 tie。

只输出如下 JSON：
{{"winner":"A|B|tie","confidence":0.x,"accuracy_risk":"A|B|none","reason":"一句话理由"}}"""


def _mean(nums: list[float]) -> float:
    vals = [float(x) for x in nums if x is not None]
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_judges(judges: list[dict]) -> dict:
    """Aggregate repeated noisy Kimi judges for one fixed output."""
    valid = [j for j in judges if isinstance(j, dict) and not j.get("error")]
    if not valid:
        return {
            "accuracy": "incorrect",
            "accuracy_score": 0.0,
            "humanness": 0.0,
            "grounded": 0.0,
            "rag_traces": [],
            "n": 0,
        }
    acc_rank = {"incorrect": 0, "partial": 1, "correct": 2}
    inv_acc = {0: "incorrect", 1: "partial", 2: "correct"}
    acc_votes = [acc_rank.get(j.get("accuracy") or "incorrect", 0) for j in valid]
    traces = []
    for j in valid:
        traces.extend(j.get("rag_traces") or [])
    trace_vote = []
    for t in ("explicit_ref", "verbatim_copy", "ref_enumeration", "policy_source"):
        if sum(1 for x in traces if x == t) >= max(1, len(valid) // 2 + 1):
            trace_vote.append(t)
    return {
        "accuracy": inv_acc.get(round(_mean(acc_votes)), "incorrect"),
        "accuracy_score": _mean([j.get("accuracy_score") for j in valid]),
        "humanness": _mean([j.get("humanness") for j in valid]),
        "grounded": _mean([j.get("grounded") for j in valid]),
        "rag_traces": trace_vote,
        "n": len(valid),
        "humanness_sd": statistics.stdev([float(j.get("humanness") or 0.0) for j in valid]) if len(valid) >= 2 else 0.0,
        "grounded_sd": statistics.stdev([float(j.get("grounded") or 0.0) for j in valid]) if len(valid) >= 2 else 0.0,
        "accuracy_score_sd": statistics.stdev([float(j.get("accuracy_score") or 0.0) for j in valid]) if len(valid) >= 2 else 0.0,
    }


def aggregate_derag_judges(judges: list[dict]) -> dict:
    """Aggregate repeated derag judges for one fixed output."""
    agg = aggregate_judges(judges)
    valid = [j for j in judges if isinstance(j, dict) and not j.get("error")]
    trace_vals = [j.get("trace_free") for j in valid if j.get("trace_free") is not None]
    agg["trace_free"] = _mean(trace_vals) if trace_vals else agg.get("humanness", 0.0)
    agg["trace_free_sd"] = statistics.stdev([float(x or 0.0) for x in trace_vals]) if len(trace_vals) >= 2 else 0.0
    # Keep this alias inside v3.1 outputs only so old table utilities still work.
    agg["humanness"] = agg["trace_free"]
    return agg


def judge_text_k(query: str, user_prompt: str, gold_answer: str, text: str, k: int = 3) -> dict:
    """Run k independent judge calls and return raw judges plus mean/majority aggregate."""
    judges = []
    for _ in range(max(1, int(k))):
        try:
            judges.append(judge_text(query, user_prompt, gold_answer, text))
        except Exception as exc:
            judges.append({"error": repr(exc)})
    return {"judges": judges, "agg": aggregate_judges(judges)}


def parse_pairwise_json(text: str) -> dict:
    t = (text or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if obj.get("winner") in ("A", "B", "tie"):
                return obj
        except Exception:
            pass
    win = re.search(r'"winner"\s*:\s*"(A|B|tie)"', t)
    if not win:
        raise ValueError(f"成对裁判 JSON 无法解析: {t[:160]}")
    return {"winner": win.group(1), "confidence": 0.0, "accuracy_risk": "none", "reason": "salvaged_from_malformed_json"}


def pairwise_judge(query: str, user_prompt: str, gold_answer: str, a_text: str, b_text: str) -> dict:
    prompt = PAIRWISE_TEMPLATE.format(
        query=query or "",
        reference=reward.extract_references(user_prompt or "")[:3000],
        gold=gold_answer or "",
        a=(a_text or "")[:5000],
        b=(b_text or "")[:5000],
    )
    raw = kimi_client.chat(
        [{"role": "system", "content": PAIRWISE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    return parse_pairwise_json(raw)


# ====================== derag_v4 Stage1 evidence-first judges ======================

JUDGE_V4_VERSION = "judge_v4.0"
JUDGE_V4_BIN_VERSION = "judge_v4.1_bin"

J_TRACE_V4_SYSTEM = "你是税务领域 RAG 痕迹审查员。必须先找证据，再打分。只输出 JSON。"
J_TRACE_V4_TEMPLATE = """输入包含税务问题、参考材料、待审 think 和机器软特征。
你只判断 think 是否机械搬运参考资料，不判断答案对错。

【问题】
{query}

【参考材料】
{reference}

【待审 think】
{think}

【辅助特征】
masked_copy={masked_copy}
citation_density={citation_density}
残余匹配提示={copy_hints}

先逐条列出 trace_spans，每个 span 必须逐字来自待审 think，长度至少 6 个字；然后给 trace_free。
痕迹类型：
A 提及检索装置，例如“参考问答对/参考资料/资料显示/根据提供的/检索结果/问题1/回答：”；
B 客服回答话术，例如“小贴士/温馨提示/您可以参考下图/图片链接”；
C 答案体复刻，把多个参考回答或无关内容按原版式机械串联；
D 清单式甩文号，政策文件独立成行或不服务任何推理步骤；
E 无消化照搬，必要事实之外的建议语、举例或叙述成段复制。

以下永远不是痕迹：必要税率、日期、金额、法定枚举、法定术语、会计分录、表单栏次、
操作路径、官方税务网址，以及嵌入真实推理句中的法规名或文号。问题本身要求税率表、
清单或政策依据时，列出必要项目本身不是痕迹。

长度和抽象“像不像人”都不是评分维度。分数必须完全由 trace_spans 解释。
无 span=0.95~1.0；仅一处轻微 B/E=0.85；出现 A/D≤0.6；C 成段复刻≤0.4。
无 span 不得低于 0.9；有 A/C/D 不得高于 0.7。

只输出：
{{"trace_spans":[{{"span":"逐字片段","type":"A-E"}}],"trace_free":0.x}}
"""

J_FACT_V4_SYSTEM = "你是税务事实保持审查员。必须先列事实问题，再打分。只输出 JSON。"
J_FACT_V4_TEMPLATE = """判断改写 think 是否保持原 think 中答案成立所依赖的税务事实。

【问题】
{query}

【参考材料】
{reference}

【标准答案】
{gold}

【原 think】
{original}

【改写 think】
{rewrite}

先输出 fact_issues。type 只能是 数字丢失/口径改变/极性翻转/新增无据/依据链断裂。
quote 必须是原 think 或改写 think 中逐字摘录的不超过 30 字片段。
允许精简冗余；“乘以9%”与“×9%”等价。关键数字、适用条件、主体、极性和依据链不可改变。
无 issue=0.95~1.0；一处次要弱化=0.85；关键数字/口径/极性问题≤0.5；主体臆测=0.2。

只输出：
{{"fact_issues":[{{"type":"口径改变","quote":"逐字片段"}}],"grounded":0.x}}
"""

J_TRACE_BIN_SYSTEM = "你是税务领域 RAG 痕迹审查员。只做二值判断，必须先指认证据，只输出 JSON。"
J_TRACE_BIN_TEMPLATE = """输入包含税务问题、参考材料、待审 think 和机器软特征。
你只回答一个问题：待审 think 是否残留机械 RAG 痕迹？不判断答案对错，不打连续分。

【问题】
{query}

【参考材料】
{reference}

【待审 think】
{think}

【辅助特征】
masked_copy={masked_copy}
citation_density={citation_density}
残余匹配提示={copy_hints}

痕迹类型：
A 提及检索装置，例如“参考问答对/参考资料/资料显示/根据提供的/检索结果/问题1/回答：”；
B 客服回答话术，例如“小贴士/温馨提示/您可以参考下图/图片链接”；
C 答案体复刻，把多个参考回答或无关内容按原版式机械串联；
D 清单式甩文号，政策文件独立成行或不服务任何推理步骤；
E 无消化照搬，必要事实之外的建议语、举例或叙述成段复制。

以下永远不是痕迹：
1. 必要税率、日期、金额、法定枚举和法定术语；
2. 会计分录、表单栏次和操作路径；
3. 官方税务网址；
4. 嵌入真实推理句并支撑结论的法规名或文号；
5. 问题本身要求税率表、清单或政策依据时列出的必要项目；
6. 为保证答案正确而必须保留的事实条件、主体和适用口径；
7. 单纯篇幅较长、表达正式或包含专业术语。

若有机械 RAG 痕迹：列出 trace_spans，span 必须逐字来自待审 think、长度至少 6 个字，verdict=traced。
若无机械 RAG 痕迹：trace_spans 必须为空，verdict=clean。
不评文风和长短；拿不准时如实填写 confidence=low。

只输出：
{{"trace_spans":[{{"span":"逐字片段","type":"A-E"}}],"verdict":"clean|traced","confidence":"high|low"}}
"""

J_FACT_BIN_SYSTEM = "你是税务事实保持审查员。只做二值判断，必须先指认证据，只输出 JSON。"
J_FACT_BIN_TEMPLATE = """判断改写 think 是否丢失或改变了原 think 中答案成立所依赖的税务事实落点。

【问题】
{query}

【参考材料】
{reference}

【标准答案】
{gold}

【原 think】
{original}

【改写 think】
{rewrite}

issue type 只能是：数字丢失、口径改变、极性翻转、新增无据、依据链断裂。
quote 必须是原 think 或改写 think 中逐字摘录的不超过 30 字片段。
允许精简冗余；“乘以9%”与“×9%”等价。关键数字、适用条件、主体、极性和依据链不可改变。

若事实保持完整：fact_issues 为空，fact_ok=true。
若存在事实问题：至少列出一条 fact_issues，fact_ok=false。

只输出：
{{"fact_issues":[{{"type":"口径改变","quote":"逐字片段"}}],"fact_ok":true|false}}
"""

J_ARBITER_V4_SYSTEM = "你是税务去 RAG 门禁的终审裁判。只按二值清单裁决，只输出 JSON。"
J_ARBITER_V4_TEMPLATE = """前序裁判对待审 think 存在分歧。请只按 T1-T4 检查。

【问题】
{query}

【参考材料】
{reference}

【待审 think】
{think}

【机器特征】
masked_copy={masked_copy}; raw_copy={raw_copy}; citation_density={citation_density}

【J-trace 有效票】
{trace_votes}

【J-fact 有效票】
{fact_votes}

T1 显式指代检索材料；
T2 与参考连续重合至少 40 字，且不是税率表/法定枚举/会计分录等唯一正确表述；
T3 think 中出现 URL、<img> 或附件名；
T4 文号或《文件名》清单式罗列，不服务于推理。

不评文风、不评长度，不因必要税法名、税率、日期扣分。每个证据 span 必须逐字来自待审 think。
任一 Tx 存在时 verdict=fail；若可通过局部修复消除，fix_type 填 trace/paraphrase/citation/fact。
若 J-fact 有经过验真的事实问题，也应 verdict=fail 且 fix_type=fact；不要用 T1-T4 的 clean 覆盖事实问题。

只输出：
{{"t1":false,"t2":false,"t3":false,"t4":false,"evidence_spans":[],
"verdict":"pass|fail","fix_type":"none|trace|paraphrase|citation|fact","reason":"一句话"}}
"""

R_FIX_V4_SYSTEM = (
    "你是税务推理的局部修复员。只修指定问题，不改变答案成立所依赖的任何事实。"
    "数字、税率、日期、文号零改动；禁用“综上/综上所述/因此可知/需要注意的是”；"
    "不输出 URL/<img>，只输出修复后的 think 正文。"
)
R_FIX_V4_TEMPLATE = """请对 think 做一次最小局部修复。

【修复类型】{fix_type}
【问题】{query}
【参考材料】{reference}
【标准答案】{gold}
【原 think】{think}
【需修 spans/问题】{issues}

修复规则：
- trace：删除检索装置、客服话术和问答体痕迹，保留事实；
- paraphrase：改写非必要的连续照搬，法定枚举、税率、期限、政策名和数字必须保留；
- citation：将孤立政策清单融入真实推理句，同一文号可去重但至少保留一处支撑结论；
- fact：把丢失或改变的事实用最小改动恢复。
只输出修复后的 think 正文。
"""


def _extract_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object: {text[:180]}")
    return json.loads(match.group(0))


def _norm_quote(text: str) -> str:
    import unicodedata
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))


def verify_quote(quote: str, target: str, *, min_len: int = 6) -> bool:
    q = _norm_quote(quote)
    return len(q) >= min_len and q in _norm_quote(target)


def _call_evidence_json(
    messages: list[dict],
    *,
    evidence_key: str,
    score_key: str,
    target: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Call Kimi with bounded format/evidence consistency retries."""
    last_error = None
    for attempt in range(3):
        raw = kimi_client.chat(
            messages,
            temperature=temperature,
            top_p=0.7,
            max_tokens=max_tokens,
        )
        try:
            parsed = _extract_json_object(raw)
            if raw.find(f'"{evidence_key}"') > raw.find(f'"{score_key}"') >= 0:
                raise ValueError("score appeared before evidence")
            evidence = parsed.get(evidence_key) or []
            if not isinstance(evidence, list):
                raise ValueError(f"{evidence_key} is not a list")
            invalid = 0
            checked = []
            for item in evidence:
                if not isinstance(item, dict):
                    invalid += 1
                    continue
                item = dict(item)
                item["verified"] = verify_quote(str(item.get("span") or item.get("quote") or ""), target)
                invalid += int(not item["verified"])
                checked.append(item)
            parsed[evidence_key] = checked
            parsed["invalid_span_count"] = invalid
            parsed["raw"] = raw
            parsed["attempt"] = attempt + 1
            score = float(parsed.get(score_key))
            valid_spans = sum(1 for item in checked if item.get("verified"))
            if checked and invalid / len(checked) > 0.50:
                raise ValueError("more than half evidence spans are invalid")
            if score >= 0.90 and valid_spans:
                raise ValueError("high score conflicts with verified issue spans")
            if score < 0.60 and valid_spans == 0:
                raise ValueError("low score has no verified evidence")
            return parsed
        except Exception as exc:
            last_error = exc
    return {"error": repr(last_error), "raw": raw if "raw" in locals() else "", "attempt": 3}


def _call_binary_evidence_json(
    messages: list[dict],
    *,
    evidence_key: str,
    target: str,
    temperature: float,
    max_tokens: int,
    verdict_key: str,
    positive_value,
    negative_value,
) -> dict:
    """Call a binary Kimi judge and locally enforce evidence/verdict consistency."""
    last_error = None
    for attempt in range(3):
        raw = kimi_client.chat(
            messages,
            temperature=temperature,
            top_p=0.7,
            max_tokens=max_tokens,
        )
        try:
            parsed = _extract_json_object(raw)
            if raw.find(f'"{evidence_key}"') > raw.find(f'"{verdict_key}"') >= 0:
                raise ValueError("verdict appeared before evidence")
            if parsed.get(verdict_key) not in (positive_value, negative_value):
                raise ValueError(f"invalid {verdict_key}")
            evidence = parsed.get(evidence_key) or []
            if not isinstance(evidence, list):
                raise ValueError(f"{evidence_key} is not a list")
            invalid = 0
            checked = []
            for item in evidence:
                if not isinstance(item, dict):
                    invalid += 1
                    continue
                item = dict(item)
                quote = str(item.get("span") or item.get("quote") or "")
                item["verified"] = verify_quote(quote, target)
                invalid += int(not item["verified"])
                checked.append(item)
            valid = [item for item in checked if item.get("verified")]
            parsed[evidence_key] = checked
            parsed["invalid_span_count"] = invalid
            parsed["raw"] = raw
            parsed["attempt"] = attempt + 1

            if parsed.get(verdict_key) == positive_value and not valid:
                # A traced/failing vote without locally verifiable evidence is
                # deliberately weakened instead of being trusted.
                parsed["original_verdict"] = parsed.get(verdict_key)
                parsed[verdict_key] = negative_value
                parsed["confidence"] = "low"
                parsed["evidence_validation"] = "positive_without_verified_evidence"
                return parsed
            if parsed.get(verdict_key) == negative_value and valid:
                raise ValueError("negative verdict conflicts with verified evidence")
            parsed.setdefault("confidence", "high")
            return parsed
        except Exception as exc:
            last_error = exc
    return {
        "error": repr(last_error),
        "raw": raw if "raw" in locals() else "",
        "attempt": 3,
        verdict_key: negative_value,
        "confidence": "low",
        evidence_key: [],
    }


def judge_trace_bin_v4(
    query: str,
    user_prompt: str,
    think: str,
    features: dict,
    *,
    temperature: float,
) -> dict:
    prompt = J_TRACE_BIN_TEMPLATE.format(
        query=(query or "")[:800],
        reference=reward.extract_references(user_prompt or "")[:3000],
        think=(think or "")[:4000],
        masked_copy=features.get("masked_copy"),
        citation_density=features.get("citation_density"),
        copy_hints=features.get("citation_examples") or [],
    )
    result = _call_binary_evidence_json(
        [{"role": "system", "content": J_TRACE_BIN_SYSTEM}, {"role": "user", "content": prompt}],
        evidence_key="trace_spans",
        target=think,
        temperature=temperature,
        max_tokens=700,
        verdict_key="verdict",
        positive_value="traced",
        negative_value="clean",
    )
    result["judge_version"] = JUDGE_V4_BIN_VERSION
    result["temperature"] = temperature
    return result


def judge_fact_bin_v4(
    query: str,
    user_prompt: str,
    gold: str,
    original_think: str,
    rewrite_think: str,
    *,
    temperature: float = 0.0,
) -> dict:
    prompt = J_FACT_BIN_TEMPLATE.format(
        query=(query or "")[:800],
        reference=reward.extract_references(user_prompt or "")[:3000],
        gold=(gold or "")[:2500],
        original=(original_think or "")[:4000],
        rewrite=(rewrite_think or "")[:4000],
    )
    result = _call_binary_evidence_json(
        [{"role": "system", "content": J_FACT_BIN_SYSTEM}, {"role": "user", "content": prompt}],
        evidence_key="fact_issues",
        target="\n".join([original_think or "", rewrite_think or ""]),
        temperature=temperature,
        max_tokens=700,
        verdict_key="fact_ok",
        positive_value=False,
        negative_value=True,
    )
    result["judge_version"] = JUDGE_V4_BIN_VERSION
    result["temperature"] = temperature
    return result


def judge_trace_v4(query: str, user_prompt: str, think: str, features: dict, *, temperature: float) -> dict:
    prompt = J_TRACE_V4_TEMPLATE.format(
        query=(query or "")[:800],
        reference=reward.extract_references(user_prompt or "")[:3000],
        think=(think or "")[:4000],
        masked_copy=features.get("masked_copy"),
        citation_density=features.get("citation_density"),
        copy_hints=features.get("citation_examples") or [],
    )
    result = _call_evidence_json(
        [{"role": "system", "content": J_TRACE_V4_SYSTEM}, {"role": "user", "content": prompt}],
        evidence_key="trace_spans",
        score_key="trace_free",
        target=think,
        temperature=temperature,
        max_tokens=900,
    )
    result["judge_version"] = JUDGE_V4_VERSION
    result["temperature"] = temperature
    return result


def judge_fact_v4(
    query: str,
    user_prompt: str,
    gold: str,
    original_think: str,
    rewrite_think: str,
    *,
    temperature: float = 0.0,
) -> dict:
    prompt = J_FACT_V4_TEMPLATE.format(
        query=(query or "")[:800],
        reference=reward.extract_references(user_prompt or "")[:3000],
        gold=(gold or "")[:2500],
        original=(original_think or "")[:4000],
        rewrite=(rewrite_think or "")[:4000],
    )
    result = _call_evidence_json(
        [{"role": "system", "content": J_FACT_V4_SYSTEM}, {"role": "user", "content": prompt}],
        evidence_key="fact_issues",
        score_key="grounded",
        target="\n".join([original_think or "", rewrite_think or ""]),
        temperature=temperature,
        max_tokens=900,
    )
    result["judge_version"] = JUDGE_V4_VERSION
    result["temperature"] = temperature
    return result


def judge_arbiter_v4(
    query: str,
    user_prompt: str,
    think: str,
    features: dict,
    trace_votes: list[dict],
    fact_votes: list[dict],
) -> dict:
    prompt = J_ARBITER_V4_TEMPLATE.format(
        query=(query or "")[:800],
        reference=reward.extract_references(user_prompt or "")[:3000],
        think=(think or "")[:4000],
        masked_copy=features.get("masked_copy"),
        raw_copy=features.get("copy_ratio"),
        citation_density=features.get("citation_density"),
        trace_votes=json.dumps(trace_votes, ensure_ascii=False)[:5000],
        fact_votes=json.dumps(fact_votes, ensure_ascii=False)[:3500],
    )
    last_error = None
    for attempt in range(2):
        raw = kimi_client.chat(
            [{"role": "system", "content": J_ARBITER_V4_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=0.7,
            max_tokens=1200,
        )
        try:
            parsed = _extract_json_object(raw)
            if parsed.get("verdict") not in ("pass", "fail"):
                raise ValueError("invalid verdict")
            checked = []
            for item in parsed.get("evidence_spans") or []:
                if isinstance(item, dict):
                    item = dict(item)
                    item["verified"] = verify_quote(str(item.get("span") or ""), think)
                    checked.append(item)
            parsed["evidence_spans"] = checked
            parsed["raw"] = raw
            parsed["attempt"] = attempt + 1
            parsed["judge_version"] = JUDGE_V4_VERSION
            return parsed
        except Exception as exc:
            last_error = exc
    return {"error": repr(last_error), "verdict": "fail", "fix_type": "none", "judge_version": JUDGE_V4_VERSION}


def repair_think_v4(
    query: str,
    user_prompt: str,
    gold: str,
    think: str,
    fix_type: str,
    issues: list,
) -> str:
    prompt = R_FIX_V4_TEMPLATE.format(
        fix_type=fix_type,
        query=(query or "")[:800],
        reference=reward.extract_references(user_prompt or "")[:4000],
        gold=(gold or "")[:2500],
        think=(think or "")[:5000],
        issues=json.dumps(issues, ensure_ascii=False)[:4000],
    )
    return kimi_client.chat(
        [{"role": "system", "content": R_FIX_V4_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.3,
        top_p=0.9,
        max_tokens=2048,
    ).strip()
