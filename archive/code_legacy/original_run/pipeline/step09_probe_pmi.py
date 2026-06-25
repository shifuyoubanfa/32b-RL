"""阶段3：奖励校准与证伪 —— 用探针为 PMI 选"尺子模型"，并确认 AUC≥0.7 才放行后续训练。

数据：高分对照 c（facts_ok ∧ humanness≥门槛 的自然改写 think，label=1）
      低分对照（原始机器腔 think，label=0）。
做法：对每个候选尺子模型（clean_base=Qwen2.5-32B-Instruct / pi_ref=冷启动模型 / current=同冷启动模型），
      用 reward.pmi_cond 算每条 think 的 -PMI，sklearn 算 AUC；同时报表面项 s_trace 的 AUC 作对照。
      选 AUC 最高且≥0.7 的尺子 -> 写回 output/pmi_ruler.json 供后续阶段读取。
      尺子排除原始 V1（RAG 腔偏，最差）。先证伪再烧卡：AUC<0.7 阻塞。

资源：一次加载一个尺子(device_map=auto 摊在训练卡 2-7)，算完释放再下一个。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    SEEDS_SCORED, PROBE_LOW, PROBE_REPORT, OUTPUT_DIR,
    SYSTEM_PROMPT, V1_DIR, CLEAN_BASE_DIR, COLDSTART_LORA_DIR,
    PMI_RULER_CANDIDATES, PMI_THINK_CAP, PMI_MAX_LEN, resolve_adapter, seed_is_chosen,
)
from pipeline import reward
from pipeline.logger import get_logger

log = get_logger("step09_probe_pmi")


def _qa_user(query: str, answer: str) -> str:
    """Q + 标准答案 作为 PMI 分母上下文（把答案放到参考位）。"""
    return f"【参考问答对】\n[已知正确答案：{answer}]\n【问题】\n{query}"


def _load_samples(max_each: int):
    high, low = [], []
    for r in (json.loads(l) for l in open(SEEDS_SCORED, encoding="utf-8") if l.strip()):
        # 高分对照 = 真冷启动训练集：与 step08 共用 config.seed_is_chosen（含 copy 闸），定义不漂移
        if seed_is_chosen(r):
            high.append({"think": r.get("natural_think", ""), "user_prompt": r.get("user_prompt", ""),
                         "query": r.get("query", ""), "answer": r.get("answer", "")})
    for r in (json.loads(l) for l in open(PROBE_LOW, encoding="utf-8") if l.strip()):
        low.append({"think": r.get("think", ""), "user_prompt": r.get("user_prompt", ""),
                    "query": r.get("query", ""), "answer": r.get("answer", "")})
    return high[:max_each], low[:max_each]


def _ruler_path(name: str):
    """候选尺子 -> (base_path, adapter_path or None)。探针时 stage_model = 冷启动模型。"""
    if name == "clean_base":
        return CLEAN_BASE_DIR, None
    if name in ("stage_model", "pi_ref", "current"):
        return V1_DIR, resolve_adapter(COLDSTART_LORA_DIR)   # swift 嵌套路径，解析到真正的 checkpoint 目录
    raise ValueError(f"未知尺子候选: {name}")


def _load_model(base_path: str, adapter_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).eval()
    return model, tok


def _auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels, scores))
    except Exception as e:
        log.warning("AUC 计算失败: %r", e)
        return 0.0


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_each", type=int, default=400, help="高/低各取多少条做探针(控时长)")
    args = ap.parse_args()

    high, low = _load_samples(args.max_each)
    samples = [(s, 1) for s in high] + [(s, 0) for s in low]
    labels = [lab for _, lab in samples]
    log.info("探针样本：高分 %d + 低分 %d = %d", len(high), len(low), len(samples))

    # 表面项 s_trace 作对照（关键词命中越多越机器 -> 用 -hits 当"自然分"）
    s_trace = []
    for s, _ in samples:
        _, hinfo = reward.humanness(s["think"], reward.extract_references(s["user_prompt"]))
        s_trace.append(-hinfo.get("trace_hits", 0))
    auc_trace = _auc(labels, s_trace)
    log.info("对照 s_trace(表面关键词) AUC=%.3f", auc_trace)

    results = {"s_trace": round(auc_trace, 4)}
    best_name, best_auc = None, -1.0
    for name in PMI_RULER_CANDIDATES:
        try:
            base_p, adp = _ruler_path(name)
            log.info("加载尺子 [%s] base=%s adapter=%s", name, base_p, adp)
            model, tok = _load_model(base_p, adp)
            pmi_scores = []
            for i, (s, _) in enumerate(samples):
                val = reward.pmi_cond(model, tok, SYSTEM_PROMPT, s["user_prompt"],
                                      _qa_user(s["query"], s["answer"]), s["think"],
                                      max_len=PMI_MAX_LEN, think_cap=PMI_THINK_CAP)
                pmi_scores.append(val)
                if (i + 1) % 100 == 0:
                    log.info("[%s] PMI %d/%d", name, i + 1, len(samples))
            auc = _auc(labels, pmi_scores)
            results[f"pmi_{name}"] = round(auc, 4)
            hi_mean = _mean([sc for sc, lab in zip(pmi_scores, labels) if lab == 1])
            lo_mean = _mean([sc for sc, lab in zip(pmi_scores, labels) if lab == 0])
            log.info("尺子 [%s] PMI AUC=%.3f ｜ -PMI high均值 %.3f vs low均值 %.3f（high>low 方向才对）",
                     name, auc, hi_mean, lo_mean)
            if auc > best_auc:
                best_name, best_auc = name, auc
            del model, tok
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            log.error("尺子 [%s] 失败: %r", name, e)
            results[f"pmi_{name}"] = None

    # 写报告 + 选优
    chosen = best_name if best_auc >= 0.7 else None
    lines = ["# 阶段3 奖励证伪 / PMI 尺子选优报告", "",
             f"- 探针样本：高 {len(high)} / 低 {len(low)}",
             f"- 表面项 s_trace AUC = **{auc_trace:.3f}**", "",
             "## 候选尺子 PMI AUC（排除原始 V1）",
             "| 尺子 | AUC |", "|---|---|",
             *[f"| {k.replace('pmi_','')} | {v} |" for k, v in results.items() if k.startswith("pmi_")], "",
             f"## 选优结果：**{chosen or '无候选达 0.7（阻塞，需在 V1 种子上重 tune τ/C/W）'}**（best AUC={best_auc:.3f}）", ""]
    Path(PROBE_REPORT).write_text("\n".join(lines), encoding="utf-8")

    out = {"pmi_ruler": chosen, "best_auc": round(best_auc, 4), "all": results}
    (Path(OUTPUT_DIR) / "pmi_ruler.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("===== PMI 尺子选优：%s（AUC=%.3f）=====", chosen, best_auc)
    if chosen is None:
        log.error("⛔ 无尺子达 AUC≥0.7：先证伪未过，按技术方案应在 V1 自然种子上重 tune，不进 RFT。")
        raise SystemExit(3)
    log.info("已写 output/pmi_ruler.json；后续 RFT/GRPO 用尺子 = %s", chosen)


if __name__ == "__main__":
    main()
