"""Step 2: 对每条可用 query 调用公司 RAG + 微调模型，保留 think / answer / 拼好的 user_prompt。

支持断点续跑：按 query 去重，已有 query 跳过。
"""

import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import ALL_QUERIES, COMPANY_OUTPUTS, COMPANY_CALL_WORKERS
from pipeline.rag_client import rag_answer
from pipeline.logger import get_logger

log = get_logger("step02")


_write_lock = threading.Lock()


def load_done(path: str) -> set[str]:
    done = set()
    p = Path(path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["query"])
            except Exception:
                continue
    return done


def worker(query: str, top_k: int):
    try:
        record = rag_answer(query, top_k=top_k)
        return ("ok", record, None)
    except Exception as e:
        return ("err", {"query": query}, repr(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条，便于联调")
    parser.add_argument("--workers", type=int, default=COMPANY_CALL_WORKERS)
    args = parser.parse_args()

    with open(ALL_QUERIES, "r", encoding="utf-8") as f:
        all_queries = [json.loads(l)["query"] for l in f if l.strip()]
    if args.limit:
        all_queries = all_queries[: args.limit]

    done = load_done(COMPANY_OUTPUTS)
    todo = [q for q in all_queries if q not in done]
    log.info("待处理 %d / %d (已完成 %d) workers=%d top_k=%d",
             len(todo), len(all_queries), len(done), args.workers, args.top_k)

    fout = open(COMPANY_OUTPUTS, "a", encoding="utf-8")
    t0 = time.time()
    ok = err = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, q, args.top_k): q for q in todo}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            status, record, msg = fut.result()
            if status == "ok":
                with _write_lock:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                ok += 1
            else:
                err += 1
                log.error("query=%s... -> %s", record["query"][:40], msg)
            if i % 10 == 0 or i == len(futures):
                rate = i / max(time.time() - t0, 1e-3)
                log.info("[%d/%d] ok=%d err=%d %.2f q/s", i, len(futures), ok, err, rate)
    fout.close()
    log.info("完成: ok=%d err=%d 写入=%s", ok, err, COMPANY_OUTPUTS)


if __name__ == "__main__":
    main()
