"""judgecal（判官标定）· 逐句"换词复述"识别裁判。

新版实验设计模块1 的核心裁判：把 Kimi 的活儿钉死成——
**逐句判断 think 里哪些句子是把参考资料某段"换了说法复述了一遍"**（换词复述照抄）。

分工（见 docs/journal/109_judgecal_module_design.md）：
- 原样照抄 → 规则字符匹配就能抓（这里 verbatim 类只当对照）；
- **换词复述**（词换了、规则瞎了，但仍是逐条复述参考）→ **这才是 Kimi 唯一要干的活**；
- 把参考里的事实正常用进推理（legit_use）→ 不该标，是误伤陷阱（阶段7 规则栽的那种）；
- 完全从问题自己推（original）→ 不该标。

只调 Kimi（DashScope），CPU，不碰 GPU、不 serve vLLM。
"""

from __future__ import annotations

import json
import re
import unicodedata

from pipeline import kimi_client

JUDGECAL_SYSTEM = (
    "你是税务领域『换词复述』识别员。你只做一件事：逐句判断一段推理(think)里，"
    "哪些句子是把【参考资料里的某一段】换了措辞、重新复述了一遍。"
    "不判答案对错，不评文风长短，必须先指认参考出处，只输出 JSON。"
)

JUDGECAL_TEMPLATE = """给你一道税务题的【参考资料】，和一段【已编号 think】（推理过程，每句前有句号）。
请逐句判断：哪些句子是把参考资料里某一段【换个说法、重新复述了一遍】（即"换词复述照抄"）。

判定标准（重要，先读再判）：
- 算照抄(标记它)：这句话的主要内容就是参考资料某一段的复述/改写——哪怕换了词、调了语序、做了精简，只要你能在参考资料里指认出"它在复述哪一段"。
- 不算照抄(不要标)：把参考里的某个事实（税率/金额/期限/适用条件/结论）自然地用在自己的推理里。例如"这档免税线是10万，本月9万还在线内""他8月买的就从8月起按月算"——整句是在推进自己的判断、得出本题结论，而不是在逐条复述参考。
- 不算照抄(不要标)：完全从题目本身出发、自己推导的句子。
- 只看"是不是把参考某段复述搬运了一遍"，绝不要因为它"不像真人说话/篇幅长/用词正式/含专业术语"而标记。

【参考资料】
{reference}

【已编号 think】
{numbered_think}

先对每一个你认定为"换词复述照抄"的句子，逐条给出：sid(句号数字)、ref_span(参考资料里被它复述的那一段【逐字摘录】，至少6个字)、type(verbatim=近乎原样照搬 / reworded=换了词改写复述)。
然后给出 copied_sids = 所有被你判为照抄的句号列表。没有任何照抄就给空列表。

只输出如下 JSON：
{{"flags":[{{"sid":1,"ref_span":"参考里被复述的逐字片段","type":"verbatim|reworded"}}],"copied_sids":[1,3]}}"""


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))


def verify_span(span: str, reference: str, *, min_len: int = 6) -> bool:
    """ref_span 必须真是参考资料里的逐字片段（归一化后子串、且≥min_len 字）。"""
    q = _norm(span)
    return len(q) >= min_len and q in _norm(reference)


def number_think(sentences: list[dict]) -> str:
    """把句子列表拼成 [sid] 句 的已编号 think，给 Kimi 看上下文。"""
    return "\n".join(f"[{s['sid']}] {s['text']}" for s in sentences)


def _parse(raw: str, valid_sids: set[int], reference: str) -> dict:
    """抽 JSON → copied_sids（整数集合）+ 证据校验后的 flags。解析不出抛异常（上层记一次失败）。"""
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        raise ValueError(f"judgecal JSON 解析失败: {text[:160]}")

    raw_flags = obj.get("flags") or []
    copied = set()
    flags = []
    for it in raw_flags if isinstance(raw_flags, list) else []:
        if not isinstance(it, dict):
            continue
        try:
            sid = int(it.get("sid"))
        except Exception:
            continue
        if sid not in valid_sids:
            continue
        span = str(it.get("ref_span") or "")
        verified = verify_span(span, reference)
        flags.append({"sid": sid, "ref_span": span, "type": it.get("type"), "verified": verified})
        copied.add(sid)
    # copied_sids 字段兜底：模型可能只填了 copied_sids 没填 flags
    for s in obj.get("copied_sids") or []:
        try:
            si = int(s)
        except Exception:
            continue
        if si in valid_sids:
            copied.add(si)
    return {"copied_sids": sorted(copied), "flags": flags}


CLEAN_SCORE_SYSTEM = (
    "你是税务 think 的『换词复述照抄』程度评分员。给整段 think 打一个 0-10 的『干净分』："
    "完全没有换词复述照抄=10 分，整段几乎都是把参考逐条换词复述=0 分。"
    "不评文风长短、不判答案对错，只看『有多少是把参考某段换个说法复述了一遍』。只输出 JSON。"
)

CLEAN_SCORE_TEMPLATE = """给你一道税务题的【参考资料】和一段【think】。请给这段 think 打一个 0-10 的『干净分』——
分数越高 = 越没有"把参考资料某段换个说法、重新复述一遍"(换词复述照抄)；分数越低 = 越多句子是在逐条复述参考。

评分锚点(共 6 档，照抄越多分越低)：
- 10 分：完全没照抄——全是从问题自己一步步推，或只是把参考里的事实(税率/金额/期限)自然用进推理(这【不算】照抄)。
- 8 分：只有 1 句是把参考某段换词复述。
- 6 分：有 2 句换词复述。
- 4 分：有 3 句换词复述。
- 2 分：有 4 句或更多换词复述。
- 0 分：整段几乎每句都是把参考逐条换词复述(完全照抄)。

关键：把参考里的某个事实自然用在自己的推理里【不算照抄】(例如"这档线是10万，本月9万还在线内")；只有"整句在复述参考某段"才算。
不要因为 think 篇幅长、用词正式、含专业术语就扣分。

【参考资料】
{reference}

【think】
{think}

只输出如下 JSON：{{"clean_score": 0到10之间的数字, "n_copied_est": 你估计的换词复述句数, "reason":"一句话理由"}}"""


def _parse_clean_score(raw: str) -> dict:
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None
    if not isinstance(obj, dict) or obj.get("clean_score") is None:
        mm = re.search(r'"clean_score"\s*:\s*([0-9.]+)', text)
        if not mm:
            raise ValueError(f"clean_score JSON 解析失败: {text[:160]}")
        obj = {"clean_score": float(mm.group(1))}
    score = max(0.0, min(10.0, float(obj.get("clean_score"))))
    return {"clean_score": score, "n_copied_est": obj.get("n_copied_est"),
            "reason": obj.get("reason", "")}


def judge_clean_score(reference: str, think: str, *, temperature: float = 0.0) -> dict:
    """整段 think 的 0-10 干净分（10=完全没换词复述照抄, 0=整段都是照抄）。返回 {clean_score, ...}。"""
    prompt = CLEAN_SCORE_TEMPLATE.format(reference=(reference or "")[:3500], think=(think or "")[:4000])
    raw = kimi_client.chat(
        [{"role": "system", "content": CLEAN_SCORE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=0.7,
        max_tokens=400,
    )
    out = _parse_clean_score(raw)
    out["raw"] = raw
    return out


def think_text(sentences: list[dict]) -> str:
    """把句子列表拼回一整段 think（给整段打分用，不带句号）。"""
    return "".join(s.get("text", "") for s in sentences)


def judge_sentences(reference: str, sentences: list[dict], *, temperature: float = 0.0) -> dict:
    """让 Kimi 对一段 think 逐句判"换词复述"，返回 {copied_sids:[...], flags:[...]}。

    copied_sids = Kimi 认定为照抄的句号（这是核心判定）；flags 里 verified=ref_span 是否真在参考里
    （证据是否成立，给人工核对/质量旁证用，不改变 copied_sids）。
    """
    valid_sids = {int(s["sid"]) for s in sentences}
    prompt = JUDGECAL_TEMPLATE.format(
        reference=(reference or "")[:3500],
        numbered_think=number_think(sentences)[:4000],
    )
    raw = kimi_client.chat(
        [{"role": "system", "content": JUDGECAL_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=0.7,
        max_tokens=900,
    )
    out = _parse(raw, valid_sids, reference or "")
    out["raw"] = raw
    return out
