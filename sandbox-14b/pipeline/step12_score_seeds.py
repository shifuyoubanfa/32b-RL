"""Step 12（冷启动种子，阶段②）：用 Kimi 给改写后的自然 think 打 humanness 分。

目的：给这批"自然样本"贴上 Kimi humanness 标签，使它们成为【该高分的对照】——
和现有 224 条"该低分的 RAG 腔样本"一起，才能在阶段③真正验证奖励"两头都判得对"
(而不是靠'全打低分'蒙混)。

读 30_seeds_rewritten.jsonl，对每条 natural_think 让 Kimi 打 humanness(0~1) + 顺带判 facts_kept。
断点续跑、并发、INFO 实时日志。产出 30_seeds_scored.jsonl。
"""

import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SEEDS_RAW, SEEDS_SCORED, SEED_WORKERS
from pipeline.logger import get_logger
from pipeline.kimi_client import call_kimi, extract_json

log = get_logger("step12_score")
_lock = threading.Lock()

SCORE_SYS = """你是评测官。给你一个税务问题、它的标准答案、以及一段"推理过程(think)"。只评两件事，输出严格 JSON：
{
  "humanness": 0~1 小数,        // think 像不像一个人从问题自然推导出来的(越像越高)；越像"对照资料归纳/复述/罗列参考"越低
  "humanness_reason": "<40字内>",
  "facts_kept": true/false,     // think 里的关键事实(数字/税率/期限/结论)是否与标准答案一致、无明显错误
  "rag_trace_types": [<可空，从 explicit_ref/ref_enumeration/verbatim_copy/policy_source 选>]
}
只输出 JSON，不要任何额外文字。"""


def load_done(path: str) -> set:
    p = Path(path)
    if not p.exists():
        return set()
    done = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["query"])
            except Exception:
                continue
    return done


def worker(seed: dict):
    query = seed.get("query")
    think = (seed.get("natural_think") or "").strip()
    answer = (seed.get("answer") or "").strip()
    if not think:
        return ("skip", {"query": query}, "natural_think 为空")
    user = (
        "【用户问题】\n" + (query or "") +
        "\n\n【标准答案】\n" + answer +
        "\n\n【待评推理过程 think】\n" + think +
        "\n\n请按要求输出 JSON。"
    )
    try:
        raw = call_kimi([
            {"role": "system", "content": SCORE_SYS},
            {"role": "user", "content": user},
        ])
        j = extract_json(raw)
    except Exception as e:
        return ("err", {"query": query}, repr(e))
    out = dict(seed)
    out["kimi_humanness"] = j.get("humanness")
    out["kimi_humanness_reason"] = j.get("humanness_reason")
    out["kimi_facts_kept"] = j.get("facts_kept")
    out["kimi_rag_trace_types"] = j.get("rag_trace_types")
    return ("ok", out, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=SEEDS_RAW)
    parser.add_argument("--out", default=SEEDS_SCORED)
    parser.add_argument("--workers", type=int, default=SEED_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not Path(args.in_path).exists():
        log.error("找不到 %s，请先跑 step11_rewrite_seeds.py", args.in_path)
        sys.exit(1)

    with open(args.in_path, "r", encoding="utf-8") as f:
        seeds = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        seeds = seeds[: args.limit]

    done = load_done(args.out)
    todo = [s for s in seeds if s.get("query") not in done]
    log.info("种子打分：待处理 %d / %d（已完成 %d）workers=%d", len(todo), len(seeds), len(done), args.workers)
    if not todo:
        log.info("无待处理，结束。")
        return

    fout = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    ok = err = 0
    hsum = 0.0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, s): s for s in todo}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            status, rec, msg = fut.result()
            if status == "ok":
                with _lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                ok += 1
                try:
                    hsum += float(rec.get("kimi_humanness") or 0)
                except Exception:
                    pass
            elif status == "err":
                err += 1
                log.error("query=%s... 打分失败：%s", (rec.get("query") or "")[:30], msg)
            if i % 10 == 0 or i == len(futures):
                rate = i / max(time.time() - t0, 1e-3)
                avg = hsum / ok if ok else 0.0
                log.info("[%d/%d] ok=%d err=%d 自然种子 Kimi humanness 均值=%.3f  %.2f q/s",
                         i, len(futures), ok, err, avg, rate)
    fout.close()
    avg = hsum / ok if ok else 0.0
    log.info("种子打分完成：ok=%d err=%d，自然种子 Kimi humanness 均值=%.3f -> %s", ok, err, avg, args.out)
    log.info("对照：现有 RAG 腔样本 Kimi humanness 均值约 0.21。若自然种子均值明显更高(如 ≥0.5)，"
             "说明改写有效、且为奖励验证提供了'该高分'的对照。")


if __name__ == "__main__":
    main()
