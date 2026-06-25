"""阶段2-①：Kimi 把 V1 机器腔 think 改写成自然推导（一份两用：冷启动 SFT 种子 + 探针"该高分"对照）。

源 = SFT_TRAIN（全部 2014，SEED_MAX=0）。改写红线：删检索腔、grounding 于参考资料、所有数字/结论一字不改。
facts_ok 只保证"未臆造 原think∪答案∪参考资料 之外的新数字"，【不】保证"扣对参考、不与参考矛盾"——
后者由 step07 的 Kimi faithfulness 打分度量、step08 入选门强制（见两步）。
另落 trace_hits(确定性去检索腔自检) 与 copy_ratio(照抄率) 供 step08 闸用。
输出 SEEDS_RAW：{query, user_prompt, original_think, natural_think, answer, facts_ok, introduced_nums, trace_hits, copy_ratio}。
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SFT_TRAIN, SEEDS_RAW, SEED_MAX, SEED_WORKERS
from pipeline import kimi_client, reward, vllm_client
from pipeline.logger import get_logger

log = get_logger("step06_rewrite_seeds")

REWRITE_SYSTEM = (
    "你是中文税务写作专家。你要把'查资料式'的机器推理，改写成'资深税务老师从问题出发自然讲解'的口吻——"
    "但讲解的【依据、口径、数字、结论必须完全来自给定的参考资料】，绝不能凭通用知识自行发挥或与参考资料矛盾。"
)

REWRITE_TEMPLATE = """下面是一道税务题的【参考资料】（权威问答，是答案的唯一依据）、【用户问题】和一段【原始推理】。
请把【原始推理】改写得更像一个人在自然地思考这个问题，严格满足：

1.【去检索腔】删掉"根据参考问答对/检索结果/参考资料/问题1的回答"之类字眼，改成从用户问题出发、一步步自然推导。
2.【必须扣依据·最重要】改写后推理的依据、政策口径、数字、结论【必须来自上面的参考资料】，把参考资料里的关键事实
   自然地融进推导过程，而不是凭你自己的通用税务知识去推；【绝不能得出与参考资料相矛盾的结论】。
3.【事实零改动】所有数字、税率、百分比、金额、日期、期限、政策结论一个字都不能改、不能新增臆造、不能删除关键点。
4. 只输出改写后的推理过程，不要输出答案，不要任何解释或标签。

【参考资料】
{reference}

【用户问题】
{query}

【原始推理】
{think}"""

# 数值事实校验：① 带单位/%的数（即使 1 位也算，如 3%/1%/5日/2倍/100万元——税务结论性事实）；
#                ② 裸数字仅 ≥2 位才纳入（避开"问题1/问答对2"序号噪声）。
_FACT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|个?(?:日|天|个月|月|年|倍|周|季度|元|万元|亿元|万|亿))")
_BARE_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _nums(text: str) -> set:
    out = set()
    t = text or ""
    for m in _FACT_RE.findall(t):
        out.add(re.sub(r"\s+", "", m))
    for m in _BARE_RE.findall(t):
        s = m.replace(",", "")
        if len(re.sub(r"\D", "", s)) >= 2:
            out.add(s)
    return out


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
    ap.add_argument("--in", dest="inp", default=SFT_TRAIN)
    ap.add_argument("--out", default=SEEDS_RAW)
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if SEED_MAX and SEED_MAX > 0:
        recs = recs[:SEED_MAX]
    done = _load_done(args.out)
    todo = [r for r in recs if r.get("query") not in done]
    log.info("改写：待处理 %d / %d（已完成 %d，全池=%s）", len(todo), len(recs), len(done), SEED_MAX == 0)

    import threading
    lock = threading.Lock()
    fout = open(args.out, "a", encoding="utf-8")   # 增量落盘：成功一条写一条，断点续跑安全

    def _rewrite(rec: dict):
        original = (rec.get("reasoning") or "").strip()
        answer = (rec.get("answer") or "").strip()
        full_prompt = (rec.get("user_prompt") or "").strip()
        refs = reward.extract_references(full_prompt)   # 只取【参考问答对】段：喂改写做 grounding + 算 copy_ratio（与 step07/09/评测同口径）
        try:
            natural = kimi_client.chat(
                [{"role": "system", "content": REWRITE_SYSTEM},
                 {"role": "user", "content": REWRITE_TEMPLATE.format(
                     reference=refs[:6000], query=(rec.get("query") or "")[:500], think=original[:4000])}],
                temperature=0.3, top_p=0.9, max_tokens=2048).strip()
        except Exception as e:
            log.warning("改写失败(跳过，下次续跑重试) query=%s...: %r", (rec.get("query") or "")[:30], e)
            return None   # 失败不写、不崩；_load_done 下次会重试这条
        # facts 校验用全量 user_prompt（题干里的数字也算合法依据）；copy_ratio 用 refs（对齐下游 copy 闸口径，避免被题干文本抬高而偏严误剔）。
        allowed = _nums(original) | _nums(answer) | _nums(full_prompt)
        introduced = sorted(_nums(natural) - allowed)
        _, hinfo = reward.humanness(natural, refs)
        r = {"query": rec.get("query"), "user_prompt": rec.get("user_prompt", ""),
             "original_think": original, "natural_think": natural, "answer": answer,
             "facts_ok": len(introduced) == 0, "introduced_nums": introduced,
             "trace_hits": hinfo.get("trace_hits", 0), "copy_ratio": round(hinfo.get("copy_ratio", 0.0), 4)}
        with lock:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
        return r

    results = vllm_client.map_concurrent(todo, _rewrite, workers=SEED_WORKERS, desc="改写")
    fout.close()
    done = [r for r in results if r]
    ok = sum(1 for r in done if r["facts_ok"])
    log.info("完成：本轮成功 %d/%d（facts_ok=%d / 引入新数字 %d；失败的下次续跑补）-> %s",
             len(done), len(results), ok, len(done) - ok, args.out)


if __name__ == "__main__":
    main()
