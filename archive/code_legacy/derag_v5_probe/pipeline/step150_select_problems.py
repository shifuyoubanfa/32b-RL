"""derag_v5 探针 · 第一步：找出"病题"（默认用 Kimi 判，能看见结构性念手册）。

为什么默认 Kimi 而不是规则：确定性规则只抓字面词"参考问答对"+重照抄(copy≥0.40)，
而 RFT 早把字面词去了、真正的"从资料向答案归纳"是结构性的、没有字面词 → 规则几乎全漏
（实测同一 RFT base：Kimi 标 explicit38/verbatim60/enum16，规则只抓到 5）。挑题阶段容噪声
（多挑错点无所谓，只是要个有代表性的病题盘子量 headroom），不像门控错放会污染训练，故用 Kimi。

流程：RFT-merged 贪心答全题集 → (kimi 模式)对每题 think 跑 k 次 DERAG 判 → 低 trace_free / 有结构痕迹的判病题。
需 RFT-merged base 在 vLLM（COLDSTART 系统提示=RFT 训练口径）。
输出 150_problems.jsonl（病题）+ 150_problems.all.jsonl（全部题的分，便于看分布/复核漏检）。
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SFT_TRAIN, SFT_EVAL, COLDSTART_SYSTEM_PROMPT, GEN_MAX_NEW_TOKENS, SEED_WORKERS
from pipeline import vllm_client
from pipeline.v5_probe_common import real_trace, refs_of, parse
from pipeline.logger import get_logger

log = get_logger("step150_select_problems")

# 结构性真痕迹类型（policy_source=合法税法引用假阳，永不计入）
STRUCT_TRACE = {"explicit_ref", "verbatim_copy", "ref_enumeration"}


def qid_of(q: str) -> str:
    return hashlib.sha1((q or "").encode("utf-8")).hexdigest()[:12]


def load_set(path: str, split: str, cap: int) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        log.warning("题集不存在，跳过：%s", path)
        return rows
    for line in p.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({"qid": qid_of(r.get("query", "")), "split": split,
                     "query": r.get("query", ""), "user_prompt": r.get("user_prompt", ""),
                     "gold_answer": r.get("answer", "")})
    if cap and cap > 0:
        rows = rows[:cap]
    return rows


def kimi_detect(rows: list[dict], k: int, tf_max: float) -> None:
    """对每题 think 跑 k 次 Kimi DERAG 判，写回 kimi_tf_mean / kimi_struct_traces / is_problem_kimi。"""
    from pipeline.judge_common import judge_text_derag

    def _judge(r):
        tfs, traces = [], set()
        for _ in range(k):
            try:
                j = judge_text_derag(r["query"], r["user_prompt"], r["gold_answer"], r["rft_text"])
            except Exception as e:
                log.warning("Kimi 判失败 qid=%s: %r", r["qid"], e)
                continue
            tf = j.get("trace_free")
            if tf is not None:
                tfs.append(float(tf))
            for t in (j.get("rag_traces") or []):
                if t in STRUCT_TRACE:
                    traces.add(t)
        tf_mean = sum(tfs) / len(tfs) if tfs else None
        r["kimi_tf_mean"] = round(tf_mean, 4) if tf_mean is not None else None
        r["kimi_struct_traces"] = sorted(traces)
        r["kimi_n_valid"] = len(tfs)
        low_tf = (tf_mean is not None and tf_mean < tf_max)
        r["is_problem_kimi"] = bool(low_tf or traces)
        return r

    vllm_client.map_concurrent(rows, _judge, workers=SEED_WORKERS, desc="kimi_detect")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_all", default=None, help="全部题打分明细（默认 <out 去后缀>.all.jsonl）")
    ap.add_argument("--detect", default=os.environ.get("V5_DETECT", "kimi"), choices=["kimi", "rule"])
    ap.add_argument("--k", type=int, default=int(os.environ.get("V5_DETECT_K", "2")))
    ap.add_argument("--tf_max", type=float, default=float(os.environ.get("V5_PROBLEM_TF", "0.70")))
    ap.add_argument("--train_cap", type=int, default=int(os.environ.get("V5_TRAIN_CAP", "1000")))
    ap.add_argument("--eval_cap", type=int, default=0)
    args = ap.parse_args()
    out_all = args.out_all or (args.out.rsplit(".", 1)[0] + ".all.jsonl")

    vllm_client.wait_ready()
    rows = load_set(SFT_EVAL, "eval", args.eval_cap) + load_set(SFT_TRAIN, "train", args.train_cap)
    log.info("PROGRESS 扫题集 %d 条（eval+train），RFT 贪心生成 think；挑题方式=%s", len(rows), args.detect)

    # 阶段1：RFT 贪心答全题（vLLM 高并发）
    def _gen(r):
        try:
            txt = vllm_client.gen_one(r["user_prompt"], system=COLDSTART_SYSTEM_PROMPT,
                                      temperature=0.0, max_tokens=GEN_MAX_NEW_TOKENS)
        except Exception as e:
            log.warning("RFT 生成失败 qid=%s: %r", r["qid"], e)
            return None
        think, _ = parse(txt)
        r.update({"rft_text": txt, "rft_think": think, "real_trace": real_trace(think, refs_of(r["user_prompt"]))})
        return r

    rows = [r for r in vllm_client.map_concurrent(rows, _gen, desc="rft_greedy") if r]
    n_det = sum(1 for r in rows if r["real_trace"]["count"] >= 1)
    log.info("PROGRESS RFT 贪心完成 %d 题；确定性规则命中(仅参考)=%d 题", len(rows), n_det)

    # 阶段2：判病题
    if args.detect == "kimi":
        log.info("PROGRESS 开始 Kimi DERAG 判病题：%d 题 × k=%d（trace_free<%.2f 或有结构痕迹=病题）",
                 len(rows), args.k, args.tf_max)
        kimi_detect(rows, args.k, args.tf_max)
        for r in rows:
            r["is_problem"] = r.get("is_problem_kimi", False)
    else:
        for r in rows:
            r["is_problem"] = r["real_trace"]["count"] >= 1

    problems = [r for r in rows if r.get("is_problem")]

    with open(out_all, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "qid": r["qid"], "split": r["split"], "query": r["query"][:80],
                "det_trace": r["real_trace"]["count"], "det_copy": r["real_trace"]["copy_ratio"],
                "kimi_tf_mean": r.get("kimi_tf_mean"), "kimi_struct_traces": r.get("kimi_struct_traces"),
                "is_problem": r.get("is_problem", False),
            }, ensure_ascii=False) + "\n")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in problems:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def rate(split):
        sub = [r for r in rows if r["split"] == split]
        return sum(1 for r in sub if r.get("is_problem")), len(sub)
    pe, te = rate("eval")
    pt, tt = rate("train")
    log.info("RESULT 病题(%s)：eval %d/%d (%.1f%%) | train %d/%d (%.1f%%) | 共 %d -> %s（全量分→%s）",
             args.detect, pe, te, 100 * pe / max(1, te), pt, tt, 100 * pt / max(1, tt),
             len(problems), args.out, out_all)


if __name__ == "__main__":
    main()
