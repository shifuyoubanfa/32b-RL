"""V2(derag_v6) 确定性规则层 —— 只有两个职责，零噪声、可逐条核对。

设计原则（用户 2026-06-15 定）：
  规则【不染指"是否照抄/换词复述"】——那是语义判断、规则做不可靠（阶段7 实测把 96/96 合法引用
  误判成痕迹），全交 Kimi。规则只干两件确定性的活：
    1) detect_rag_style    —— 判 think 有没有"检索腔/念手册的表面标记"（检索装置腔/编号引用/客服话术/图床附件）。
    2) answer_in_v1_pool   —— 判模型答案的结论极性+关键数字，在不在 V1 多次采样形成的认可池里（漂没漂）。

刻意不做的事（阶段7/8 的教训）：
  - 不做任何字面照抄率/复述判断（reward.py 的 copy_signal / reward_v3 的 masked_copy 不在 V2 规则职责内）。
  - 不把"法规名/单个文号"当痕迹（阶段7：嵌进推理的合法引用反而让答案更准）；只有"政策依据："清单式
    甩文号（≥3 个、不服务推理）才算痕迹。
"""

from __future__ import annotations

import re
import unicodedata

# ====================== 规则函数 1：检索腔表面标记 ======================
# 来源：reward.py _TRACE_PATTERNS + reward_v3 的 explicit/enum/customer/qa/img 正则，按 A-D 归类、
# 并剔除"单个法规/文号"（阶段7 误判源）。每类命中都是逐字、确定性的。

_RAG_PATTERNS = {
    "A_检索装置腔": [
        r"参考问答对", r"参考资料", r"参考内容", r"检索结果", r"资料(显示|表明|指出)",
        r"原始(问答对|资料|回答)", r"根据(检索|参考|提供的资料|上述资料|以上资料)",
        r"如(上|前)(文|述|参考|资料|所示)", r"对照(参考|资料|问答)",
    ],
    "B_编号引用": [
        r"问答对\s*[0-9一二三四五六七八九十]", r"问题\s*\d+\s*[:：]", r"回答\s*[:：]",
        r"第\s*[0-9一二三四五六七八九十]\s*个?参考", r"(逐条|依次|分别)(对照|参考|归纳)",
    ],
    "C_客服话术": [
        r"小贴士", r"温馨提(示|醒)", r"参考下图", r"如下图", r"您可以参考", r"哦[~～]", r"亲[，,~～]",
    ],
    "D_图床附件": [
        r"<img\b", r"https?://\S*(?:aliyuncs|oss-|servu)\S*", r"\b\S+\.(?:png|jpe?g|xlsx?|pdf)(?![a-z0-9])",
    ],
}
# D 类（图床/附件/链接）大小写无关：截图文件名常见 .PNG/.JPG、<IMG>；尾部用负向预查而非 \b，
# 让中文紧贴的文件名（…table1.png的数据）也命中（对齐 V1 reward_v3._IMG_TRACE_RE 的 re.I）。
_RAG_RE = {
    t: [re.compile(p, re.I) if t == "D_图床附件" else re.compile(p) for p in ps]
    for t, ps in _RAG_PATTERNS.items()
}

# 清单式甩文号（唯一被算作痕迹的"政策"情形）：政策依据:起头的清单，或一段里 ≥3 个文号且不服务推理。
_DOC_TOKEN_RE = re.compile(
    r"(?:《[^》]{2,80}》(?:第[一二三四五六七八九十百千万\d]+条)?|"
    r"[^\s，。；、]{0,12}[财税会发函公告令]\s*[〔\[\(（]?\d{4}[〕\]\)）]?\s*\d+\s*号|"
    r"[^\s，。；、]{0,16}(?:公告|令)\s*(?:\d{4}\s*年)?\s*第?\s*\d+\s*号)"  # 公告YYYY年第N号 / 令第N号（现行主流公告体，旧分支抓不到）
)
_POLICY_LABEL_RE = re.compile(r"(?m)^\s*(?:政策依据|参考文件|参考法规|文件依据)\s*[:：]")

# 按句豁免所需（移植 V1 reward_v3）：合法引用"服务推理"就放过，只把"清单式甩文号"算痕迹。
_FORM_NAME_RE = re.compile(r"《[^》]{1,80}(?:表|单|凭证)》")
_CONTENT_WORD_RE = re.compile(
    r"(规定|按照|明确|适用|应当|可以|不得|免征|征收|处理|确认|扣除|申报|"
    r"判断|区分|属于|不属于|满足|导致|计算|缴纳|计入|结转|抵扣|"
    r"依据|由于|若|如果|因此|所以|意味着|对应|决定|需要|无需|涉及|"
    r"享受|选择|采用|执行|发生|取得|提供|销售|转让|出租|支付|收到)"
)
_SENT_SPLIT_RE = re.compile(r"[。！？!?\n;；]+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]


def detect_rag_style(think: str) -> dict:
    """判 think 有没有检索腔的【表面标记】。不判是否照抄/复述（那交 Kimi）。

    返回 {has_rag_style, spans:[{text, type}], n_by_type}。确定性、零噪声、可逐字核对。
    """
    t = think or ""
    spans: list[dict] = []
    n_by_type: dict[str, int] = {}

    # A/B/C/D-img：表面标记逐字命中。收 (start,end,text,type) 以便去重（OSS 链接会与 .png 后缀重叠）。
    raw: list[tuple[int, int, str, str]] = []
    for typ, regs in _RAG_RE.items():
        for r in regs:
            for m in r.finditer(t):
                raw.append((m.start(), m.end(), m.group(0), typ))
    seen: set = set()
    uniq: list[tuple[int, int, str, str]] = []
    for s, e, txt, typ in sorted(raw, key=lambda x: (x[0], x[1])):
        if (s, e, typ) in seen:
            continue
        seen.add((s, e, typ))
        uniq.append((s, e, txt, typ))
    for i, (s, e, txt, typ) in enumerate(uniq):
        # 被同类更大 span 完全包含的丢掉（如 OSS 图片链接已含 .png 后缀，避免重复计数）
        if any(j != i and ks <= s and e <= ke and (ke - ks) > (e - s) and ktyp == typ
               for j, (ks, ke, _kt, ktyp) in enumerate(uniq)):
            continue
        spans.append({"text": txt, "type": typ})
        n_by_type[typ] = n_by_type.get(typ, 0) + 1

    # D 类·清单式甩文号：移植 V1 citation_metrics 的按句豁免——本句"服务推理"（含内容/推理词）且文号<3
    # 视为合法引用、放过（阶段7 误判源）；只数"不服务推理 / 一句≥3 文号"的甩号。
    # ≥3 个甩号 或 "政策依据:"清单 才算痕迹（单个嵌进推理的法规/文号不算）。
    standalone_docs: list[str] = []
    for sent in _split_sentences(t):
        reduced = _FORM_NAME_RE.sub("", sent)            # 《XX表/单/凭证》名不当文号
        sdocs = _DOC_TOKEN_RE.findall(reduced)
        if not sdocs:
            continue
        if not _CONTENT_WORD_RE.search(reduced) or len(sdocs) >= 3:
            standalone_docs.extend(sdocs)
    label_m = _POLICY_LABEL_RE.search(t)
    if len(standalone_docs) >= 3 or label_m:
        for d in standalone_docs[:8]:
            spans.append({"text": d, "type": "D_清单式甩文号"})
        if label_m and not standalone_docs:
            # "政策依据:"标签本身即充分痕迹：公告体盲点/纯标签行没抽到文号时也落一条（对齐 V1 label 独立计分）。
            spans.append({"text": label_m.group(0).strip(), "type": "D_清单式甩文号"})
        n_by_type["D_清单式甩文号"] = max(len(standalone_docs), 1 if label_m else 0)

    return {"has_rag_style": bool(spans), "spans": spans, "n_by_type": n_by_type, "n": len(spans)}


# ====================== 规则函数 2：答案在不在 V1 认可池 ======================
# 来源：reward.answer_drift（极性 + 关键数字抽事实）+ derag_v5 answer_in_support（确定性比对）。
# 只判"漂没漂出 V1"，不判对错（V1 错它也跟着认）。


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


# 关键数字事实（分类抽取：日期单独按"粒度前缀"判覆盖，其余按集合比对）。
_PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")                                    # 13% / 5%
_FRAC_RE = re.compile(r"万分之[零一二三四五六七八九十百点\d]+|千分之[零一二三四五六七八九十百点\d]+")  # 万分之三 / 千分之五
_MONEY_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:万元|亿元|万|亿|元)")          # 10万元 / 10万
_DATE_RE = re.compile(r"\d{4}\s*年(?:\s*\d{1,2}\s*月)?(?:\s*\d{1,2}\s*日)?")   # 2023年 / 2023年1月1日
_DUR_RE = re.compile(r"\d+\s*(?:个)?(?:日|天|个月|月|年|季度)")                # 30天 / 6年（日期已先消费，不会再抽出"1月""1日"）

# 结论极性词（答案立场，相反极性=漂）。长词优先匹配并消位，避免"不超过"里又抽出"超过"。
_POLARITY = ["免征", "免税", "不征税", "应缴", "应纳税", "需缴纳", "不得", "不可以", "无需", "不需要",
             "可以", "允许", "禁止", "超过", "不超过", "属于", "不属于"]
_POLARITY_SORTED = sorted(_POLARITY, key=len, reverse=True)


def _norm_money(item: str) -> str:
    """金额单位归一：万元→万、亿元→亿（消掉 '10万' vs '10万元' 假漂）。"""
    return item.replace("万元", "万").replace("亿元", "亿")


def _polarity_set(t: str) -> set[str]:
    """抽极性：长词优先命中后从文本消位，使短子串（超过/属于/可以）不再被误抽（防 S-7 子串污染）。"""
    s = t
    found: set[str] = set()
    for p in _POLARITY_SORTED:
        if p in s:
            found.add(p)
            s = s.replace(p, "　")
    return found


def _extract_facts(text: str) -> dict:
    """抽【极性 / 数字 / 日期】三类事实并归一化。日期先消费，避免期限正则在日期里重复抽 '1月''1日'。"""
    t = nfkc(text)
    pol = _polarity_set(t)
    dates: set[str] = set()
    consumed = t
    for m in _DATE_RE.findall(t):
        dates.add(re.sub(r"\s+", "", m))
        consumed = consumed.replace(m, " ", 1)
    vals: set[str] = set()
    for r in (_PCT_RE, _FRAC_RE, _MONEY_RE, _DUR_RE):
        for m in r.findall(consumed):
            vals.add(_norm_money(re.sub(r"\s+", "", m).replace(",", "")))
    return {"pol": pol, "val": vals, "date": dates}


def _date_covered(d: str, pool_dates: set[str]) -> bool:
    """日期按粒度前缀判覆盖：同一日期写粗/写细（'2023年' ↔ '2023年1月1日'）视为没漂。"""
    return any(d == p or d.startswith(p) or p.startswith(d) for p in pool_dates)


def answer_in_v1_pool(model_answer: str, v1_answers: list[str]) -> dict:
    """判模型答案的极性+关键数字是否都被 V1 认可池覆盖（漂没漂）。

    v1_answers：原始 V1 对该题贪心1次+采样8次得到的答案列表（step152 的 v1_support）。
    判据：模型答案抽出的极性/数字/日期都被 V1 池覆盖 → 没漂(in_pool=True)；
          出现池里没有的极性/数字、或粒度对不上的日期 → 漂了(False)，列在 drift_facts。
    归一化（用户铁律口径1）：金额万/万元同键、极性长词消位防子串污染、日期按粒度前缀判覆盖。
    确定性事实比对、零噪声（阶段8 已认定可信）。只判"漂没漂出 V1"，不判对错（V1 错它也跟着认）。
    """
    samples = [_extract_facts(a) for a in (v1_answers or []) if (a or "").strip()]
    pool = {"pol": set(), "val": set(), "date": set()}
    for e in samples:
        pool["pol"] |= e["pol"]
        pool["val"] |= e["val"]
        pool["date"] |= e["date"]
    m = _extract_facts(model_answer)
    model_has_facts = bool(m["pol"] or m["val"] or m["date"])
    pool_has_facts = bool(pool["pol"] or pool["val"] or pool["date"])

    # 空输出不是“没漂移”；此前空集合天然是任意池的子集，会把空答案/泛化空话虚判为通过。
    empty_answer = not (model_answer or "").strip()
    if empty_answer:
        comparable, reason = True, "empty_answer"
    elif not model_has_facts:
        # 非空回答没有可抽取槽位：规则无法判断是否漂移，明确退出评测分母，不能再按空集子集虚算通过。
        comparable = False
        reason = "no_comparable_facts" if pool_has_facts else "no_facts_in_either"
    elif not pool_has_facts:
        comparable, reason = False, "v1_pool_has_no_facts"
    else:
        comparable, reason = True, "ok"

    drift = sorted(m["pol"] - pool["pol"]) + sorted(m["val"] - pool["val"])
    drift += [d for d in sorted(m["date"]) if not _date_covered(d, pool["date"])]
    # 主指标保持既有“模型事实是 V1 池事实子集”口径，保证与已产出的 baseline/SFT/RFT summary 可比。
    # 唯一硬修：真正的空输出不能再因空集子集而通过。无槽位的非空回答仍保留旧判定，但 comparable=False
    # 单列审计，防止把规则测不到误写成零噪声证据。
    in_pool = not empty_answer and not drift
    if comparable and reason == "ok" and not in_pool:
        reason = "introduced_new_fact"
    model_all = sorted(m["pol"] | m["val"] | m["date"])
    pool_all = sorted(pool["pol"] | pool["val"] | pool["date"])
    return {
        "in_pool": bool(in_pool),
        "comparable": bool(comparable),
        "reason": reason,
        "drift_facts": sorted(drift),     # 模型多说了/相反/对不上的极性·数字·日期
        "model_facts": model_all,
        "pool_facts": pool_all,
        "pool_size": len(pool_all),
    }


if __name__ == "__main__":  # 最小自测：直观看两个函数的判别
    clean = "这题先看销售额，本月9万没过10万这条线，所以可以免征增值税。"
    traced = "根据参考问答对1，资料显示月销售额未超过10万元的免征增值税。参考下图。"
    print("clean :", detect_rag_style(clean))
    print("traced:", detect_rag_style(traced))
    v1 = ["本月销售额9万元，未超过10万元，免征增值税。", "9万元未达起征线，可享受免征增值税。"]
    print("in    :", answer_in_v1_pool("本月9万元，免征增值税。", v1))
    print("drift :", answer_in_v1_pool("本月应缴纳增值税，税率13%。", v1))
