"""derag_v5 探针 · 补丁 153b：用 Kimi 重判 think 干净，得到真自救率 X_kimi，与规则 X 对照。

为什么要这一步：原 s153 的 think_clean 用确定性规则(real_trace)，对结构性念手册全瞎
（489 病题里 98% det_trace=0），把"删掉字面词、换皮搬运"也判成干净 → X=0.851 系统性虚高
（5 裁判复核真干净仅 ~0.44，自救失败 73 题里 72 题是答案漂移、仅 1 题是 think 脏 = clean 闸几乎不卡人）。

本步只换"判 think 干净"那一把尺子，其余不动：
- 答案是否漂移 V1：仍用确定性 answer_in_support（这把可信，不动）。
- think 是否干净：改用 Kimi judge_text_derag（与 s150 挑题同一把尺子，apples-to-apples）：
  trace_free≥TF_CLEAN 且 无结构性 rag_traces(explicit_ref/verbatim_copy/ref_enumeration，policy_source 合法不算) = 干净。
  k 次取均值降噪（噪声在比例上会抵消，这正是用 Kimi 而非规则测"一批题干净率"的理由）。

只读已在盘的 151_rft_samples + 152_v1_support，纯 Kimi API + CPU，不动 GPU、不再生成任何采样。
省调用：先用确定性 in-support 过滤（漂移样本不可能自救，跳过不判），逐样本判到首个干净即早停（对"自救率"这个二值无偏）。
输出 153b_kimi_headroom.jsonl（逐题）+ .json（汇总，含 X_kimi vs X_rule 对照）。
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SEED_WORKERS
from pipeline import vllm_client  # map_concurrent 是纯线程池，不依赖 vLLM
from pipeline.judge_common import judge_text_derag
from pipeline.v5_probe_common import parse, answer_in_support
from pipeline.logger import get_logger

log = get_logger("step153b_kimi_clean_rescore")

# 结构性真痕迹（policy_source=合法税法引用，永不计入）
STRUCT_TRACE = {"explicit_ref", "verbatim_copy", "ref_enumeration"}


def kimi_think_clean(query, user_prompt, gold, sample_text, k, tf_clean):
    """k 次 Kimi DERAG 判 think。返回 (clean, tf_mean, struct_traces, n_valid)。
    clean = tf_mean≥tf_clean 且 无结构性痕迹。"""
    tfs, traces = [], set()
    for _ in range(max(1, k)):
        try:
            j = judge_text_derag(query, user_prompt, gold, sample_text)
        except Exception as e:
            log.warning("Kimi 判失败: %r", e)
            continue
        tf = j.get("trace_free")
        if tf is not None:
            tfs.append(float(tf))
        for t in (j.get("rag_traces") or []):
            if t in STRUCT_TRACE:
                traces.add(t)
    tf_mean = sum(tfs) / len(tfs) if tfs else None
    clean = bool(tf_mean is not None and tf_mean >= tf_clean and not traces)
    return clean, (round(tf_mean, 4) if tf_mean is not None else None), sorted(traces), len(tfs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--rule_json", default=None, help="原 153_rft_headroom.json，做 X_rule 对照")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--k", type=int, default=int(os.environ.get("V5X_K", "2")))
    ap.add_argument("--tf_clean", type=float, default=float(os.environ.get("V5X_TF_CLEAN", "0.70")))
    ap.add_argument("--cap", type=int, default=int(os.environ.get("V5X_CAP", "16")),
                    help="每题最多判几条 in-support 样本（早停于首个干净；cap 仅在全脏时生效）")
    args = ap.parse_args()

    sup_by_qid = {r["qid"]: r for r in (json.loads(l) for l in open(args.support, encoding="utf-8") if l.strip())}
    rows = [json.loads(l) for l in open(args.samples, encoding="utf-8") if l.strip()]
    log.info("PROGRESS Kimi 重判 think 干净：%d 题，仅判 in-support 样本(k=%d, tf_clean=%.2f, cap=%d)，早停于首个干净",
             len(rows), args.k, args.tf_clean, args.cap)

    def _score(r):
        sup = sup_by_qid.get(r["qid"])
        if not sup:
            return None
        support = sup["support"]
        q, up, gold = r.get("query", ""), r.get("user_prompt", ""), r.get("gold_answer", "")
        # 1) 确定性 in-support 过滤（这把尺子可信，不换）
        insup = []
        for i, s in enumerate(r["samples"]):
            _, ans = parse(s)
            if answer_in_support(ans, support)["in_support"]:
                insup.append((i, s))
        # 2) 仅对 in-support 样本用 Kimi 判 think 干净，早停于首个干净
        judged = []
        kimi_rescuable, pass_idx = False, None
        for i, s in insup[:args.cap]:
            clean, tf, traces, nv = kimi_think_clean(q, up, gold, s, args.k, args.tf_clean)
            judged.append({"i": i, "kimi_clean": clean, "tf": tf, "traces": traces, "n_valid": nv})
            if clean:
                kimi_rescuable, pass_idx = True, i
                break
        return {"qid": r["qid"], "split": r["split"], "query": r["query"][:80],
                "n_insupport": len(insup), "n_judged": len(judged),
                "kimi_rescuable": kimi_rescuable, "pass_idx": pass_idx, "judged": judged}

    per_q = [r for r in vllm_client.map_concurrent(rows, _score, workers=SEED_WORKERS, desc="kimi_clean_rescore") if r]

    with open(args.out, "w", encoding="utf-8") as f:
        for r in per_q:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rule = {}
    if args.rule_json and Path(args.rule_json).exists():
        rule = json.loads(Path(args.rule_json).read_text(encoding="utf-8"))

    def agg(split):
        sub = [r for r in per_q if split is None or r["split"] == split]
        n = len(sub)
        resc = sum(1 for r in sub if r["kimi_rescuable"])
        x_rule = (rule.get(split or "all", {}) or {}).get("rescue_rate")
        return {"n_problems": n, "n_rescuable_kimi": resc,
                "X_kimi": round(resc / max(1, n), 4),
                "X_rule": x_rule,
                "samples_judged": sum(r["n_judged"] for r in sub)}

    summary = {"all": agg(None), "eval": agg("eval"), "train": agg("train"),
               "params": {"k": args.k, "tf_clean": args.tf_clean, "cap": args.cap}}
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    a = summary["all"]
    log.info("RESULT X_kimi all=%d/%d=%.3f (规则 X_rule=%s) | eval=%.3f train=%.3f | 判样本≈%d×k=%d次 -> %s",
             a["n_rescuable_kimi"], a["n_problems"], a["X_kimi"], a["X_rule"],
             summary["eval"]["X_kimi"], summary["train"]["X_kimi"],
             a["samples_judged"], args.k, args.out_json)


if __name__ == "__main__":
    main()
