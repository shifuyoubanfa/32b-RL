"""derag_v5 探针 · 第五步：Kimi 把 V1 的 think 改写干净，看改得动改不动（上界 / 兜底信号）。

含义：就算 RFT 自己 16 遍蒙不出来(step153=0)，只要 Kimi 能把 think 改干净、答案不动(=V1 答案，天然在 V1 范围)，
我们就能用 SFT 把模型搬到干净区，RL 信号随后回来。复用 step06 的改写 prompt（兼容现有链路）。
输入：152_v1_support.jsonl（要 v1_canonical_think/answer + user_prompt）。
输出 154_rewrite_headroom.jsonl + 154_rewrite_headroom.json。
"""

import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SEED_WORKERS
from pipeline import kimi_client
from pipeline.step06_rewrite_seeds import REWRITE_SYSTEM, REWRITE_TEMPLATE
from pipeline.v5_probe_common import refs_of, real_trace, is_clean
from pipeline.logger import get_logger

log = get_logger("step154_kimi_rewrite_check")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.support, encoding="utf-8") if l.strip()]
    log.info("PROGRESS Kimi 改写 think：%d 道病题（只洗思路、答案不动）", len(rows))

    lock = threading.Lock()
    fout = open(args.out, "w", encoding="utf-8")
    out_rows = []

    def _rewrite(r):
        refs = refs_of(r["user_prompt"])
        think = r.get("v1_canonical_think") or ""
        try:
            new_think = kimi_client.chat(
                [{"role": "system", "content": REWRITE_SYSTEM},
                 {"role": "user", "content": REWRITE_TEMPLATE.format(
                     reference=refs[:6000], query=(r.get("query") or "")[:500], think=think[:4000])}],
                temperature=0.3, top_p=0.9, max_tokens=2048).strip()
        except Exception as e:
            log.warning("Kimi 改写失败 qid=%s: %r", r["qid"], e)
            return None
        before = real_trace(think, refs)
        after = real_trace(new_think, refs)
        rec = {"qid": r["qid"], "split": r["split"], "query": r["query"][:80],
               "before_trace": before["count"], "after_trace": after["count"],
               "rewrite_clean": is_clean(new_think, refs), "new_think": new_think}
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            out_rows.append(rec)
        return rec

    from pipeline.vllm_client import map_concurrent  # 复用并发器(纯线程池，不依赖 vLLM)
    map_concurrent(rows, _rewrite, workers=SEED_WORKERS, desc="kimi_rewrite")
    fout.close()

    def agg(split):
        sub = [r for r in out_rows if split is None or r["split"] == split]
        n = len(sub)
        ok = sum(1 for r in sub if r["rewrite_clean"])
        return {"n_problems": n, "n_rewrite_clean": ok, "rewrite_clean_rate": round(ok / max(1, n), 4)}

    summary = {"all": agg(None), "eval": agg("eval"), "train": agg("train")}
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    a = summary["all"]
    log.info("RESULT Kimi 改写成功率 all=%d/%d=%.1f%% (eval=%.1f%% train=%.1f%%) -> %s",
             a["n_rewrite_clean"], a["n_problems"], 100 * a["rewrite_clean_rate"],
             100 * summary["eval"]["rewrite_clean_rate"], 100 * summary["train"]["rewrite_clean_rate"], args.out_json)


if __name__ == "__main__":
    main()
