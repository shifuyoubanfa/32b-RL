"""Step 11（冷启动种子，阶段①）：把 teacher 的 think 改写成"自然推导"风格,产出自然样本。

这批数据【一份两用】：
  - 当"高质量自然样本"验证奖励函数(第2点：给奖励一批该高分的对照，避免它靠"全打低"蒙混)；
  - 当 cold-start SFT 种子(第3点：让 14B 学会偶尔产出自然推理)。

改写规则(写进 prompt)：删掉所有 RAG 检索痕迹(参考问答对N/根据检索/参考资料…)，改成从问题出发自然推导；
【所有数字/税率/金额/期限/结论一字不改】。改完做事实校验：改写引入了原文/答案里没有的数字 → 标记 facts_ok=False。

源：SFT_TRAIN(含 teacher 的 reasoning + answer + user_prompt)。断点续跑、并发、INFO 实时日志。
产出：30_seeds_rewritten.jsonl，每行 {query, original_think, natural_think, answer, user_prompt, facts_ok}
"""

import argparse
import concurrent.futures as cf
import json
import re
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SFT_TRAIN, SEEDS_RAW, SEED_MAX, SEED_WORKERS
from pipeline.logger import get_logger
from pipeline.kimi_client import call_kimi

log = get_logger("step11_seeds")
_lock = threading.Lock()

REWRITE_SYS = """你是税务文字编辑。把给你的"税务推理过程"改写成【像人自然推导】的口吻。严格遵守：
1. 删除所有"参考问答对1/2""根据检索结果""参考资料显示""问题1的回答"等检索/引用痕迹；
2. 改成从用户问题出发、一步步自然推导到结论的口吻（像一个懂行的人在脑子里想清楚后讲出来）；
3. 【所有事实必须一字不改】：数字、税率、百分比、金额、期限、文号、会计科目、最终结论都保持原样，不得增删改；
4. 只输出改写后的推理正文，不要任何解释、不要标签、不要"改写如下"之类的话。"""


def _nums(text: str) -> set:
    """抽数字类事实(用于校验改写没引入新数字)。"""
    out = set()
    for r in (re.compile(r"\d+(?:\.\d+)?\s*%"),
              re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:万元|亿元|万|亿|元)"),
              re.compile(r"\d{4}\s*年(?:\d{1,2}\s*月)?(?:\d{1,2}\s*日)?"),
              re.compile(r"\d+")):
        for m in r.findall(text or ""):
            out.add(re.sub(r"\s+", "", m))
    return out


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


def worker(sample: dict):
    query = sample.get("query")
    original = (sample.get("reasoning") or "").strip()
    answer = (sample.get("answer") or "").strip()
    user_prompt = sample["messages"][1]["content"] if sample.get("messages") else ""
    if not original or not answer:
        return ("skip", {"query": query}, "原始 think/answer 为空")
    user = (
        "【用户问题对应的最终答案（结论，供你保持事实一致）】\n" + answer +
        "\n\n【待改写的推理过程】\n" + original +
        "\n\n请按系统要求改写这段推理，只输出改写后的推理正文。"
    )
    try:
        natural = call_kimi([
            {"role": "system", "content": REWRITE_SYS},
            {"role": "user", "content": user},
        ]).strip()
    except Exception as e:
        return ("err", {"query": query}, repr(e))
    if not natural:
        return ("err", {"query": query}, "改写结果为空")

    # 事实校验：改写里的数字应来自 原始think∪答案（不得凭空引入新数字）
    allowed = _nums(original) | _nums(answer)
    new_nums = _nums(natural)
    introduced = {x for x in new_nums if x not in allowed and len(x) >= 2}  # 忽略 1 位数(序号噪声)
    facts_ok = len(introduced) == 0

    rec = {
        "query": query,
        "original_think": original,
        "natural_think": natural,
        "answer": answer,
        "user_prompt": user_prompt,
        "facts_ok": facts_ok,
        "introduced_nums": sorted(introduced)[:10],
    }
    return ("ok", rec, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=SEED_MAX, help="只改写前 N 条；0=全量")
    parser.add_argument("--workers", type=int, default=SEED_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="冒烟：只跑前 N 条")
    parser.add_argument("--out", default=SEEDS_RAW)
    args = parser.parse_args()

    with open(SFT_TRAIN, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if args.max and args.max > 0:
        samples = samples[: args.max]
    if args.limit:
        samples = samples[: args.limit]

    done = load_done(args.out)
    todo = [s for s in samples if s.get("query") not in done]
    log.info("种子改写：待处理 %d / %d（已完成 %d）workers=%d 模型=Kimi",
             len(todo), len(samples), len(done), args.workers)
    if not todo:
        log.info("无待处理，结束。")
        return

    fout = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    ok = err = skip = bad_fact = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, s): s for s in todo}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            status, rec, msg = fut.result()
            if status == "ok":
                with _lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                ok += 1
                if not rec["facts_ok"]:
                    bad_fact += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                log.error("query=%s... 改写失败：%s", (rec.get("query") or "")[:30], msg)
            if i % 10 == 0 or i == len(futures):
                rate = i / max(time.time() - t0, 1e-3)
                log.info("[%d/%d] ok=%d(其中事实存疑 %d) err=%d skip=%d  %.2f q/s",
                         i, len(futures), ok, bad_fact, err, skip, rate)
    fout.close()
    log.info("种子改写完成：ok=%d（事实存疑 %d，建议丢弃）err=%d skip=%d -> %s",
             ok, bad_fact, err, skip, args.out)


if __name__ == "__main__":
    main()
