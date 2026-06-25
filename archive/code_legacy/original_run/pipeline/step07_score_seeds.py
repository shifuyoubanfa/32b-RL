"""阶段2-②：Kimi 给改写后的自然 think 同时打 humanness + faithfulness(grounding)。

- humanness：像不像人端到端推导（去检索腔）——做冷启动筛选 + 探针"该高分"标签。
- faithfulness(写作 grounded)：think 的依据/口径/数字/结论是否【忠于参考资料、不与之矛盾】——
  这是本次 grounding 修复的【度量与闸】：自然但偏离/矛盾参考的样本，faithfulness 会低、被 step08 剔除。
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SEEDS_RAW, SEEDS_SCORED, SEED_WORKERS
from pipeline import kimi_client, reward, vllm_client
from pipeline.logger import get_logger

log = get_logger("step07_score_seeds")

SCORE_SYSTEM = "你是严格的中文税务文风+忠实度评审。只输出 JSON，不要任何额外文字。"
SCORE_TEMPLATE = """根据【参考资料】（权威依据），给下面这段税务推理(think)打两项分，各取 0~1：
1. humanness：像不像人在端到端自然推导（越没有"参考问答对/根据检索结果"检索痕迹、越像从问题一步步推到结论越高）。0.9~1.0 完全像人；0.0~0.2 满是检索腔或大段照搬。
2. faithfulness：推理的依据/口径/数字/结论是否【完全来自并忠于参考资料】、有无与参考资料矛盾或凭空臆造。1.0 完全扣参考且无矛盾；0.5 大体扣参考但有偏差；0.0 与参考矛盾或脱离参考自行发挥。

【参考资料】
{reference}

【推理】
{think}

只输出：{{"humanness":0.x,"faithfulness":0.x}}"""


def _parse(text: str) -> tuple:
    t = text or ""
    mh = re.search(r'"humanness"\s*:\s*([0-9.]+)', t)
    mf = re.search(r'"faithfulness"\s*:\s*([0-9.]+)', t)
    if not mh or not mf:   # 两项都要齐；缺任一 → 抛错 → _score 接住跳过 → 下次续跑重打(不静默写死 0.0)
        raise ValueError(f"打分未返回 humanness/faithfulness: {t[:120]}")
    return float(mh.group(1)), float(mf.group(1))


def _load_done(path: str) -> set:
    done = set()
    p = Path(path)
    if p.exists():
        for line in p.open("r", encoding="utf-8"):
            try:
                done.add(json.loads(line)["query"])
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=SEEDS_RAW)
    ap.add_argument("--out", default=SEEDS_SCORED)
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    done = _load_done(args.out)
    todo = [r for r in recs if r.get("query") not in done]
    log.info("打分：待处理 %d / %d", len(todo), len(recs))

    import threading
    lock = threading.Lock()
    fout = open(args.out, "a", encoding="utf-8")

    def _score(rec: dict):
        reference = reward.extract_references(rec.get("user_prompt") or "")   # 只取【参考问答对】段，口径与 reward/step04 一致
        try:
            out = kimi_client.chat(
                [{"role": "system", "content": SCORE_SYSTEM},
                 {"role": "user", "content": SCORE_TEMPLATE.format(
                     reference=reference[:3000], think=(rec.get("natural_think") or "")[:4000])}],
                temperature=0.0, max_tokens=64)
            h, f = _parse(out)
        except Exception as e:
            log.warning("打分失败(跳过，下次续跑重试) query=%s...: %r", (rec.get("query") or "")[:30], e)
            return None   # 失败不当 0 分写死(否则把抖动误判成"不自然/不忠实"永久剔除)，下次续跑重试
        r = {**rec, "kimi_humanness": round(h, 4), "grounded": round(f, 4)}
        with lock:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
        return r

    results = vllm_client.map_concurrent(todo, _score, workers=SEED_WORKERS, desc="打分")
    fout.close()
    done = [r for r in results if r]
    hmean = (sum(r["kimi_humanness"] for r in done) / len(done)) if done else 0.0
    gmean = (sum(r["grounded"] for r in done) / len(done)) if done else 0.0
    log.info("完成：本轮成功 %d/%d（humanness 均值 %.3f / grounded 均值 %.3f；失败的下次续跑补）-> %s",
             len(done), len(results), hmean, gmean, args.out)


if __name__ == "__main__":
    main()
