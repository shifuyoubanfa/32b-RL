"""Step 13（Phase 2 · 信号证伪闸门）：验证新奖励的语义信号能不能把"自然 vs RAG"分开。

这是"先证伪再烧 GPU"的关键一步：在【自然种子(Kimi humanness≈0.64) + RAG 腔样本(≈0.21)】两端上，
算嵌入语义信号(ΔRAG / 语义照抄)与显式引用正则，报：
  - 各信号与 Kimi humanness 的 Spearman；
  - 各信号区分"自然 vs RAG"的 AUC；
两数据源：
  RAG 端：05_student_outputs(student_think/teacher_answer) ∩ 06_judge_results(humanness) ∩ SFT_EVAL(references)
  自然端：12_seeds_scored(natural_think + kimi_humanness)
GO/NO-GO：最佳信号 AUC ≥ 0.70 → 嵌入信号够用，可重写 reward 进 RFT；否则再加 PMI(--need-pmi 提示)。
全程只调嵌入 API，不加载 14B、不训练。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import STUDENT_OUTPUTS, JUDGE_RESULTS, SFT_EVAL, SEEDS_SCORED
from pipeline.logger import get_logger
from pipeline import reward as R
from pipeline import embed as E

log = get_logger("step13_probe")

THETA_COPY = 0.86          # 语义照抄判定阈值（句对参考相似度）
NAT_THRESHOLD = 0.45       # Kimi humanness ≥ 此值算"自然"(正例)，否则"RAG"(负例)


def load_jsonl(p):
    path = Path(p)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _safe_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def split_sents(text):
    return [s.strip() for s in R._SENT_SPLIT.split(text or "") if s.strip()]


def build_samples():
    """汇总两端样本：每条 = {think, q, a, references, kimi_h, group}。"""
    samples = []
    # RAG 端
    stu = {r.get("query"): r for r in load_jsonl(STUDENT_OUTPUTS)}
    jud = {r.get("query"): r for r in load_jsonl(JUDGE_RESULTS)}
    refs = {}
    up_full = {}
    for s in load_jsonl(SFT_EVAL):
        m = s.get("messages") or []
        if len(m) >= 2 and isinstance(m[1], dict):
            up = m[1].get("content", "")
            refs[s.get("query")] = R.extract_references(up)
            up_full[s.get("query")] = up
    for q, so in stu.items():
        if q not in jud:
            continue
        samples.append({
            "think": so.get("student_think", ""),
            "q": q or "",
            "a": so.get("teacher_answer", ""),
            "references": refs.get(q, ""),
            "user_prompt_full": up_full.get(q, ""),   # 含参考资料的完整 user(PMI 的 Q+R 上下文)
            "kimi_h": _safe_float(jud[q].get("student_reasoning_humanness")),
            "group": "RAG",
        })
    # 自然端
    for s in load_jsonl(SEEDS_SCORED):
        samples.append({
            "think": s.get("natural_think", ""),
            "q": s.get("query") or "",
            "a": s.get("answer", ""),
            "references": R.extract_references(s.get("user_prompt", "")),
            "user_prompt_full": s.get("user_prompt", ""),
            "kimi_h": _safe_float(s.get("kimi_humanness")),
            "group": "NAT",
        })
    return [s for s in samples if s["think"].strip()]


def compute_signals(samples):
    """对每条样本算 s_drag / s_copy / s_trace（都已定向：越大越自然）。"""
    # 预编码所有需要的文本（encode 内部去重+批量+缓存）
    pool = set()
    for s in samples:
        pool.update(split_sents(s["think"]))
        pool.update(split_sents(s["references"]))
        if s["q"]:
            pool.add(s["q"])
        if s["a"]:
            pool.add(s["a"])
    pool = [t for t in pool if t]
    log.info("预编码 %d 条唯一文本（批量调 bge-m3）...", len(pool))
    E.encode(pool)
    log.info("编码完成，逐样本算信号...")

    for s in samples:
        T = split_sents(s["think"])
        Rs = split_sents(s["references"])
        if not T:
            s["s_drag"] = s["s_copy"] = s["s_trace"] = 0.0
            continue
        tv = E.encode(T)
        qv = E.encode([s["q"]]) if s["q"] else np.zeros((1, tv.shape[1]), np.float32)
        av = E.encode([s["a"]]) if s["a"] else np.zeros((1, tv.shape[1]), np.float32)
        rv = E.encode(Rs) if Rs else np.zeros((1, tv.shape[1]), np.float32)

        sim_ref = E.cos_matrix(tv, rv).max(axis=1) if Rs else np.zeros(len(T), np.float32)
        sim_q = E.cos_matrix(tv, qv).max(axis=1)
        sim_a = E.cos_matrix(tv, av).max(axis=1)
        sim_qa = np.maximum(sim_q, sim_a)

        lens = np.array([len(t) for t in T], np.float32)
        w = lens / max(lens.sum(), 1.0)
        align_ref = float((w * sim_ref).sum())
        align_qa = float((w * sim_qa).sum())
        drag = align_ref - align_qa                          # 越大越像 RAG
        copy = float((lens[(sim_ref >= THETA_COPY) & (sim_ref > sim_qa)].sum()) / max(lens.sum(), 1.0))
        hits = sum(len(r.findall(s["think"])) for r in R._TRACE_RE)

        s["s_drag"] = -drag            # 定向：越大越自然
        s["s_copy"] = -copy
        s["s_trace"] = 1.0 / (1.0 + hits)
    return samples


def compute_pmi(samples):
    """满血主信号·条件化 PMI：logP(think|Q+参考资料) − logP(think|Q+标准答案)，用 14B base 算。
    越大=think 的呈现结构越依赖资料排布(RAG)；定向 s_pmi=-PMI(越大越自然)。
    逐样本 try/except，单条失败不影响其余；失败的用均值补。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from config import STUDENT_LOCAL_DIR, SYSTEM_PROMPT, MAX_LEN

    log.info("加载 14B base 算 PMI（仅前向、no_grad）：%s", STUDENT_LOCAL_DIR)
    tok = AutoTokenizer.from_pretrained(STUDENT_LOCAL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_LOCAL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    dev = model.device

    def prompt_ids(user):
        ids = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False,
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        elif hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return ids.to(dev)

    def think_logp(p_ids, t_ids):
        # 上下文超长则从左截断，留出 think
        max_ctx = MAX_LEN - t_ids.shape[1] - 8
        if max_ctx > 0 and p_ids.shape[1] > max_ctx:
            p_ids = p_ids[:, -max_ctx:]
        full = torch.cat([p_ids, t_ids], dim=1)
        with torch.no_grad():
            logits = model(full).logits
        Lc, Lt = p_ids.shape[1], t_ids.shape[1]
        if Lt == 0:
            return 0.0
        logp = torch.log_softmax(logits[0, Lc - 1:Lc + Lt - 1, :].float(), dim=-1)
        tok_lp = logp[torch.arange(Lt, device=logp.device), t_ids[0]]
        return float(tok_lp.mean().item())

    ok = 0
    for i, s in enumerate(samples, 1):
        s["s_pmi"] = None
        try:
            t_ids = tok(s["think"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            if t_ids.shape[1] > 1024:
                t_ids = t_ids[:, :1024]
            qr = s.get("user_prompt_full") or s.get("references") or s["q"]
            qa = "【问题】\n%s\n\n【标准答案】\n%s" % (s["q"], s["a"])
            pmi = think_logp(prompt_ids(qr), t_ids) - think_logp(prompt_ids(qa), t_ids)
            s["s_pmi"] = -pmi
            ok += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log.error("PMI 失败 query=%s...: %r", (s.get("q") or "")[:24], e)
        if i % 50 == 0 or i == len(samples):
            log.info("PMI [%d/%d] ok=%d", i, len(samples), ok)

    vals = [s["s_pmi"] for s in samples if s["s_pmi"] is not None]
    mean = sum(vals) / len(vals) if vals else 0.0
    for s in samples:
        if s["s_pmi"] is None:
            s["s_pmi"] = mean
    log.info("PMI 计算完成：成功 %d/%d，缺失用均值 %.3f 补", ok, len(samples), mean)


def spearman(a, b):
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rk = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def auc(scores, labels):
    """labels: 1=自然(正), 0=RAG(负)。返回 score 区分正负的 AUC（Mann-Whitney）。"""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    # 平均秩
    allv = sorted(zip(scores, labels), key=lambda x: x[0])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(rk for rk, (_, l) in zip(ranks, allv) if l == 1)
    npos, nneg = len(pos), len(neg)
    return (rank_sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nat_threshold", type=float, default=NAT_THRESHOLD)
    parser.add_argument("--with-pmi", dest="with_pmi", action="store_true",
                        help="加上 PMI 主信号(需加载 14B、~30-60min)。默认关，先只测嵌入信号。")
    args = parser.parse_args()

    samples = build_samples()
    nat = [s for s in samples if s["group"] == "NAT"]
    rag = [s for s in samples if s["group"] == "RAG"]
    if not nat or not rag:
        log.error("两端样本不全：NAT=%d RAG=%d。需先跑完 step12(自然种子打分) 与 step05/06(RAG基线)。", len(nat), len(rag))
        sys.exit(1)
    log.info("样本：自然端 %d(Kimi均值%.3f) + RAG端 %d(Kimi均值%.3f)",
             len(nat), sum(s["kimi_h"] for s in nat) / len(nat),
             len(rag), sum(s["kimi_h"] for s in rag) / len(rag))

    # 嵌入信号（默认，纯 bge-m3 API，不用 14B）
    compute_signals(samples)
    # PMI 主信号（可选开关，需要 14B）
    if args.with_pmi:
        log.info("--with-pmi 已开：加载 14B 算 PMI 主信号...")
        compute_pmi(samples)

    kimi = [s["kimi_h"] for s in samples]
    label = [1 if s["kimi_h"] >= args.nat_threshold else 0 for s in samples]

    sig_names = ["s_drag", "s_copy", "s_trace"] + (["s_pmi"] if args.with_pmi else [])
    log.info("======== 信号证伪结果（%s）========", "嵌入+PMI" if args.with_pmi else "仅嵌入")
    results = {}
    for name in sig_names:
        sig = [s[name] for s in samples]
        rho, a = spearman(sig, kimi), auc(sig, label)
        results[name] = (rho, a)
        log.info("%-8s  Spearman(对Kimi)=%+.3f   分离AUC(自然vsRAG)=%.3f", name, rho, a)

    # 满血组合：开 PMI 用 0.5/0.25/0.15/0.1；仅嵌入用 0.5/0.35/0.15
    if args.with_pmi:
        comb = [0.5 * s["s_pmi"] + 0.25 * s["s_drag"] + 0.15 * s["s_copy"] + 0.1 * s["s_trace"] for s in samples]
    else:
        comb = [0.5 * s["s_drag"] + 0.35 * s["s_copy"] + 0.15 * s["s_trace"] for s in samples]
    rho_c, auc_c = spearman(comb, kimi), auc(comb, label)
    results["combo(满血奖励)"] = (rho_c, auc_c)
    log.info("%-12s  Spearman(对Kimi)=%+.3f   分离AUC(自然vsRAG)=%.3f", "combo(满血)", rho_c, auc_c)

    best_auc = max(a for _, a in results.values())
    log.info("===============================")
    log.info("旧版正则奖励 Spearman 参考≈0.226；本次最佳分离 AUC=%.3f", best_auc)
    if best_auc >= 0.70:
        log.info("✅ 过闸门(AUC≥0.70)：新奖励能分开自然 vs RAG，可重写 reward.py 进 RFT。")
    elif not args.with_pmi:
        log.warning("⚠️ 仅嵌入未过闸门(AUC=%.3f<0.70)：加 PMI 主信号再试 → 重跑加 --with-pmi。", best_auc)
    else:
        log.warning("⚠️ 含 PMI 仍未过闸门(AUC=%.3f<0.70)：检查分句/embed 质量，或奖励范式需再调。", best_auc)


if __name__ == "__main__":
    main()
