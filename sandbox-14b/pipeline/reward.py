"""本地打分器（RL 各阶段共用）。

设计原则（见项目记忆 rl-method-grpo-not-ppo）：
- think 自然度：纯规则——RAG 痕迹词密度 + 与参考资料的照抄率。这部分容易做扎实。
- answer 正确性：不判对错（那是 85% 天花板的难题），改判"答案有没有漂移"——
  与已知标准答案的相似度 + 关键事实(数字/税率/金额/期限/极性)召回。配合训练时
  对 answer 段的 KL 锚定，准确率主要靠"按住答案"保住，本打分器只做廉价兜底探针。
- 零重依赖：只用标准库 re / difflib，不加载任何模型、不调 Kimi。离线在线都能高频调用。

reward 组合（乘法门控）：
- 格式不合法            -> reward = -1.0
- 答案漂移(R_acc<τ)    -> reward = 0.1 * R_acc        （掐掉自然度增益，防"为自然牺牲准确"）
- 否则                 -> reward = R_acc * (w_acc + w_human * R_human)   （答对才解锁自然度增益）
"""

import re
from difflib import SequenceMatcher


# ---- RAG 痕迹正则（对应 step06 judge 的 explicit_ref / ref_enumeration / policy_source）----
# 经 step08c 校准发现：要匹配学生实际写法（"参考问答对中""根据原始问题""问题1的回答指出"等），
# 不能要求"参考问答对"后面紧跟数字。下面放宽覆盖。
_TRACE_PATTERNS = [
    r"参考问答对",                 # 任意"参考问答对"提及（最强信号，不要求后接数字）
    r"问答对\s*[0-9一二三四五六七八九十]",
    r"问题\s*[0-9一二三四五六七八九十]",
    r"根据(检索|参考|提供的|以上|上述|原始|题目)",
    r"检索(结果|到的?|内容)",
    r"参考(资料|内容|文本|问答|回答)",
    r"原始(问答对|回答|资料|问题|答案)",
    r"政策依据[:：]?",
    r"参考文件",
    r"〔\d+〕\s*号",
    r"如(上|前)(参考|资料|所示|文|述)",
    r"资料(显示|表明|指出|中|里)",
    r"(逐条|依次|分别)(对照|参考|分析|归纳)",
    r"对照(参考|资料|问答)",
]
_TRACE_RE = [re.compile(p) for p in _TRACE_PATTERNS]

# humanness 惩罚系数（可被 config 覆盖；用 step08c --tune 网格搜索最优值后写回 config）
try:
    from config import REWARD_C_TRACE as _C_TRACE, REWARD_C_COPY as _C_COPY
except Exception:
    _C_TRACE, _C_COPY = 0.5, 0.8
try:
    from config import REWARD_W_PMI as _W_PMI
except Exception:
    _W_PMI = 0.5

_SENT_SPLIT = re.compile(r"[。！？!?\n;；]+")

# ---- 硬事实槽：百分比 / 金额 / 期限 / 极性词 ----
_FACT_RES = [
    re.compile(r"\d+(?:\.\d+)?\s*%"),
    re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:万元|亿元|万|亿|元)"),
    re.compile(r"\d{4}\s*年(?:\d{1,2}\s*月)?(?:\d{1,2}\s*日)?"),
    re.compile(r"\d+\s*(?:个)?(?:日|天|月|年)"),
]
_POLARITY = ["免征", "免税", "不得", "不可以", "无需", "不需要", "不超过",
             "应缴", "需要", "可以", "超过", "禁止"]


def parse_think_answer(text: str) -> tuple[str, str]:
    """R1 生成文本：模板已注入 <think>，生成从推理正文起、以 </think> 收束，随后 <answer>...</answer>。"""
    text = text or ""
    close = text.find("</think>")
    if close != -1:
        think = text[:close].replace("<think>", "").strip()
        tail = text[close + len("</think>"):]
    else:
        think, tail = "", text
    i = tail.find("<answer>")
    if i != -1:
        j = tail.rfind("</answer>")
        inner = tail[i + len("<answer>"): j] if (j != -1 and j > i) else tail[i + len("<answer>"):]
        answer = inner.replace("<answer>", "").replace("</answer>", "").strip()
    else:
        answer = tail.replace("<answer>", "").replace("</answer>", "").strip()
    return think, answer


def extract_references(user_prompt: str) -> str:
    """从 user_prompt 抽 【参考问答对】 与 【问题】 之间的参考文本。"""
    up = user_prompt or ""
    a = up.find("【参考问答对】")
    if a == -1:
        return ""
    start = a + len("【参考问答对】")
    b = up.find("【问题】", start)
    return up[start:b].strip() if (b != -1 and b > start) else up[start:].strip()


def _facts(text: str) -> set:
    out = set()
    t = text or ""
    for r in _FACT_RES:
        for m in r.findall(t):
            out.add(re.sub(r"\s+", "", m))
    for p in _POLARITY:
        if p in t:
            out.add(p)
    return out


def _ngram_set(s: str, n: int = 5) -> set:
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()


def copy_signal(think: str, references: str) -> float:
    """照抄强度 = max(最长公共子串占比, 5-gram 重叠率)。
    LCS 抓连续照搬；n-gram 抓打散/改写式照搬（治 RFT 后 verbatim_copy 飙升的 Goodhart）。
    """
    t = (think or "")[:3000]
    ref = (references or "")[:6000]
    if not t or not ref:
        return 0.0
    # 1) 最长连续公共子串占比
    m = SequenceMatcher(None, t, ref, autojunk=False).find_longest_match(0, len(t), 0, len(ref))
    lcs = m.size / max(1, len(t))
    # 2) 5-gram 重叠率（think 的 5-gram 有多少出现在 references 里）
    tg = _ngram_set(t, 5)
    ng = (len(tg & _ngram_set(ref, 5)) / len(tg)) if tg else 0.0
    return max(lcs, ng)


def humanness(think: str, references: str, c_trace: float = None, c_copy: float = None,
              s_pmi: float = None, w_pmi: float = None) -> tuple[float, dict]:
    """think 越像端到端 CoT、越没 RAG 痕迹 -> 分越高。

    基础项(纯CPU)：关键词命中 + 字符照抄率 -> base = 1/(1+c_trace·hits+c_copy·copy)。
    可选 PMI(结构信号,需外部用 14B 算好传入 s_pmi=-PMI)：score = w_pmi·sigmoid(2·s_pmi) + (1-w_pmi)·base。
    证伪：s_trace(关键词)AUC0.74、PMI AUC0.73、嵌入弱已弃。系数取自 config。
    """
    c_trace = _C_TRACE if c_trace is None else c_trace
    c_copy = _C_COPY if c_copy is None else c_copy
    think = (think or "").strip()
    if not think:
        return 0.0, {"trace_hits": 0, "copy_ratio": 0.0, "empty": True}
    hits = sum(len(r.findall(think)) for r in _TRACE_RE)
    copy_ratio = copy_signal(think, references)   # LCS + 5-gram，抓连续/打散/改写式照抄
    # 平滑、不饱和：引用越多分越低但不归零并列。引用数与照抄率共同决定 RAG 强度。
    signal = c_trace * hits + c_copy * copy_ratio
    base = 1.0 / (1.0 + signal)                   # 关键词+字符照抄(AUC~0.74，纯CPU)
    if s_pmi is None:
        score = base
        s_pmi_norm = None
    else:
        # 加入 PMI 结构信号(AUC~0.73，抗关键词规避)。s_pmi=-PMI，越大越自然；sigmoid 归一。
        import math
        wp = _W_PMI if w_pmi is None else w_pmi
        s_pmi_norm = 1.0 / (1.0 + math.exp(-2.0 * s_pmi))
        score = wp * s_pmi_norm + (1.0 - wp) * base
    return score, {
        "trace_hits": hits,
        "copy_ratio": round(copy_ratio, 3),
        "signal": round(signal, 3),
        "base_human": round(base, 3),
        "s_pmi_norm": (round(s_pmi_norm, 3) if s_pmi_norm is not None else None),
        "empty": False,
    }


def answer_drift(gen_answer: str, gold_answer: str) -> tuple[float, dict]:
    """答案没漂移=高分：与标准答案的相似度 + 关键事实召回。不是判对错，是判'变没变'。"""
    g = (gen_answer or "").strip()
    gold = (gold_answer or "").strip()
    if not g:
        return 0.0, {"sim": 0.0, "fact_recall": 0.0}
    if not gold:
        return 1.0, {"sim": 1.0, "fact_recall": 1.0}
    sim = SequenceMatcher(None, gold[:3000], g[:3000], autojunk=False).ratio()
    gf = _facts(gold)
    recall = (len(gf & _facts(g)) / len(gf)) if gf else 1.0
    return 0.5 * sim + 0.5 * recall, {"sim": round(sim, 3), "fact_recall": round(recall, 3)}


def format_ok(think: str, answer: str, min_chars: int = 40, max_chars: int = 2000) -> bool:
    if not answer or not answer.strip():
        return False
    n = len((think or "").strip())
    return min_chars <= n <= max_chars


def pmi_cond(model, tok, system_prompt: str, qr_user: str, qa_user: str,
             think: str, max_len: int = 4096, think_cap: int = 1024) -> float:
    """条件化 PMI(结构信号)：logP(think|Q+参考资料) − logP(think|Q+标准答案)，返回 -PMI(越大越自然)。
    只前向打分、不生成；同一 think_ids 在两上下文各算一次。需调用方传入已加载的 base 模型(冻结)。
    """
    import torch
    think = (think or "").strip()
    if not think:
        return 0.0

    def _prompt_ids(user):
        ids = tok.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False,
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        elif hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return ids.to(model.device)

    def _think_logp(p_ids, t_ids):
        max_ctx = max_len - t_ids.shape[1] - 8
        if max_ctx > 0 and p_ids.shape[1] > max_ctx:
            p_ids = p_ids[:, -max_ctx:]
        full = torch.cat([p_ids, t_ids], dim=1)
        with torch.no_grad():
            logits = model(full).logits
        Lc, Lt = p_ids.shape[1], t_ids.shape[1]
        if Lt == 0:
            return 0.0
        logp = torch.log_softmax(logits[0, Lc - 1:Lc + Lt - 1, :].float(), dim=-1)
        return float(logp[torch.arange(Lt, device=logp.device), t_ids[0]].mean().item())

    t_ids = tok(think, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    if t_ids.shape[1] > think_cap:
        t_ids = t_ids[:, :think_cap]
    if t_ids.shape[1] == 0:
        return 0.0
    pmi = _think_logp(_prompt_ids(qr_user), t_ids) - _think_logp(_prompt_ids(qa_user), t_ids)
    return -pmi


def score_rollout(gen_text: str, user_prompt: str, gold_answer: str, *,
                  tau_acc: float = 0.5, w_human: float = 0.5, w_acc: float = 0.5,
                  think_min: int = 40, think_max: int = 2000, s_pmi: float = None) -> dict:
    """对一条生成结果打分，返回 think/answer/各分项/总 reward。s_pmi 给定则把 PMI 结构信号并入 humanness。"""
    think, answer = parse_think_answer(gen_text)
    refs = extract_references(user_prompt)
    fmt = format_ok(think, answer, think_min, think_max)
    r_human, hinfo = humanness(think, refs, s_pmi=s_pmi)
    r_acc, ainfo = answer_drift(answer, gold_answer)

    if not fmt:
        reward = -1.0
        gate = "format_fail"
    elif r_acc < tau_acc:
        reward = 0.1 * r_acc
        gate = "acc_drift"
    else:
        reward = r_acc * (w_acc + w_human * r_human)
        gate = "ok"

    return {
        "think": think,
        "answer": answer,
        "reward": round(float(reward), 4),
        "R_human": round(float(r_human), 4),
        "R_acc": round(float(r_acc), 4),
        "s_pmi": (round(float(s_pmi), 4) if s_pmi is not None else None),
        "format_ok": fmt,
        "gate": gate,
        **{f"h_{k}": v for k, v in hinfo.items()},
        **{f"a_{k}": v for k, v in ainfo.items()},
    }
