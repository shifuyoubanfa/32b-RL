"""阶段0：本地 V1 重产 (think, answer) —— 替代 14B 依赖内网的 step02。

输入：data/00_data_teacher_outputs.jsonl（14B 缓存，每行含 query + user_prompt(=参考资料)）。
做法：取缓存的 user_prompt（含【参考问答对】），喂本地 V1(vLLM 静态服务) 贪心生成 →
      得 V1 自产 think/answer。answer=准确率金标准，think=冷启动改写原料。
输出：output/00_v1_outputs.jsonl（query, user_prompt, raw, reasoning, answer）。
支持断点续跑（按 query 去重、追加写）。并发吃满 vLLM 连续批处理（资源利用）。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    CACHED_TEACHER_OUTPUTS, V1_OUTPUTS, GEN_TEMPERATURE, GEN_TOP_P, GEN_MAX_NEW_TOKENS,
)
from pipeline import vllm_client
from pipeline.reward import parse_think_answer
from pipeline.logger import get_logger

log = get_logger("step01_build_v1_data")


def _load_done(path: str) -> set:
    done = set()
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["query"])
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=CACHED_TEACHER_OUTPUTS)
    ap.add_argument("--out", default=V1_OUTPUTS)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（联调用），0=全量")
    args = ap.parse_args()

    src = Path(args.inp)
    if not src.exists():
        raise SystemExit(
            f"缺少缓存输入 {src}。请把 14B 的 output/00_data_teacher_outputs.jsonl 拷进 code/data/ "
            f"（含 query + user_prompt 参考资料）。"
        )

    vllm_client.wait_ready()  # 等 V1 vLLM 服务就绪（serve_v1_vllm.sh 起的）

    with src.open("r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        recs = recs[: args.limit]

    done = _load_done(args.out)
    todo = [r for r in recs if r.get("query") not in done]
    log.info("V1 重产：待处理 %d / %d（已完成 %d）", len(todo), len(recs), len(done))

    # 增量落盘：每完成一条立即追加写+flush（崩溃/被kill后 _load_done 才能真正跳过已完成，不从头重跑）
    import threading
    lock = threading.Lock()
    fout = open(args.out, "a", encoding="utf-8")

    def _gen(rec: dict) -> dict:
        up = rec.get("user_prompt") or ""
        raw = vllm_client.gen_one(
            up, temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, max_tokens=GEN_MAX_NEW_TOKENS,
        )
        think, answer = parse_think_answer(raw)
        r = {"query": rec.get("query"), "user_prompt": up,
             "raw": raw, "reasoning": think, "answer": answer}
        with lock:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
        return r

    results = vllm_client.map_concurrent(todo, _gen, desc="V1重产")
    fout.close()
    empties = sum(1 for r in results if not (r["answer"] or "").strip())
    log.info("完成：写入 %d 条 -> %s（空答案 %d，需关注）", len(results), args.out, empties)


if __name__ == "__main__":
    main()
