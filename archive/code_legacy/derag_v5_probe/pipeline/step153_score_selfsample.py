"""derag_v5 探针 · 第四步（CPU）：评 RFT 16 遍自采样有没有"干净 + 答案在 V1 范围"的。

这是核心读数：一道题 16 遍里只要 ≥1 遍同时(think 干净 ∧ 答案没漂 V1) = 这道题"RFT 自己已能展示 RL 信号"
（DPO/GRPO 有正样本可学）。无需 GPU/Kimi，纯本地确定性。
输入：151_rft_samples.jsonl + 152_v1_support.jsonl。
输出 153_rft_headroom.jsonl（逐题）+ 153_rft_headroom.json（汇总）。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.v5_probe_common import check_sample
from pipeline.logger import get_logger

log = get_logger("step153_score_selfsample")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    sup_by_qid = {r["qid"]: r for r in (json.loads(l) for l in open(args.support, encoding="utf-8") if l.strip())}
    rows = [json.loads(l) for l in open(args.samples, encoding="utf-8") if l.strip()]

    per_q = []
    for r in rows:
        sup = sup_by_qid.get(r["qid"])
        if not sup:
            continue
        checks = [check_sample(s, r["user_prompt"], sup["support"]) for s in r["samples"]]
        n_pass = sum(1 for c in checks if c["pass"])
        n_clean = sum(1 for c in checks if c["think_clean"])
        n_insupport = sum(1 for c in checks if c["answer_in_support"])
        per_q.append({
            "qid": r["qid"], "split": r["split"], "query": r["query"][:80],
            "k": len(checks), "n_pass": n_pass, "n_clean": n_clean, "n_in_support": n_insupport,
            "rescuable": n_pass >= 1,
            "checks": checks,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in per_q:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def agg(split):
        sub = [r for r in per_q if split is None or r["split"] == split]
        n = len(sub)
        resc = sum(1 for r in sub if r["rescuable"])
        return {"n_problems": n, "n_rescuable": resc,
                "rescue_rate": round(resc / max(1, n), 4),
                "mean_pass_per16": round(sum(r["n_pass"] for r in sub) / max(1, n), 3),
                "mean_clean_per16": round(sum(r["n_clean"] for r in sub) / max(1, n), 3),
                "mean_insupport_per16": round(sum(r["n_in_support"] for r in sub) / max(1, n), 3)}

    summary = {"all": agg(None), "eval": agg("eval"), "train": agg("train")}
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    a = summary["all"]
    log.info("RESULT RFT 自救率 all=%d/%d=%.1f%% (eval=%.1f%% train=%.1f%%) -> %s",
             a["n_rescuable"], a["n_problems"], 100 * a["rescue_rate"],
             100 * summary["eval"]["rescue_rate"], 100 * summary["train"]["rescue_rate"], args.out_json)


if __name__ == "__main__":
    main()
