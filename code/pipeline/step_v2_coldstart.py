"""V2 step：冷启动数据 = Kimi【从问题+依据+池锚点从头改写】成自然推导 → 规则门 + facts闸 + σ选择门（2σ/3σ两桶）。

一次跑产三份：coldstart_2sigma_train / coldstart_3sigma_train / coldstart_eval（自然腔留出）。
- 改写（治本版 rewrite_v2，2026-06-18）：【不喂 V1 think】(防换词复述锚定)，给 问题+事实依据+池锚点(V1答案池抽出的
  极性/数字=answer_in_v1_pool靶子)让 Kimi 从头自推；探针实测干净分 3.13→6.6、破 3.44 天花板。t=0.6/top_p=0.9/max2048。
- 规则门：detect_rag_style(natural).has_rag_style==False（确定性、零噪声）。
- facts闸（治本配套护 grounding，多agent审核P0修正）：只拦 think 推向【V1池没有的结论极性】(相反/越界结论)，口径同
  answer_in_v1_pool(_extract_facts 归一)；合法算术派生数字【不拦】(禁派生会误杀'代入数字一步步推'最像人的样本)。规则门后、σ门前。
- σ 门：cleaner_scores(refs, natural, v1_think) 先 k=2 粗筛再 k=16 双评；confident_cleaner(.,.,2/3) 分桶。
  打分【复用 judgecal 同一套提示词】（score_think_select 内 assert k>=16），否则标定 σ 表不作数。改写到 6.6 后 σ门
  对 v1_think(~1.9) 间距够大、产出率健康（不再 9% 饿死，退役 broad/selective 权宜之计）。
- answer-lock：选中的 natural + 该题 V1 原 answer 拼 messages（system=去检索腔）。
3σ 桶 ⊆ 2σ 桶。eval 从 2σ-eligible 里确定性留出 ~10%，绝不进任何训练桶（避免越练越自然 eval_loss 反升选错 best）。
★ k 口径铁律未动：σ选样 k=2粗筛→k=16双评（score_think_select assert k>=16），评测另在 step_v2_eval 用 k=3。
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SEED_WORKERS, COLDSTART_EVAL_FRAC
from pipeline import kimi_client, reward, vllm_client
from pipeline.rewrite_v2 import REWRITE_SYSTEM, REWRITE_TEMPLATE_POOL, REWRITE_TEMPERATURE, build_pool_anchor
from pipeline.rules_v6 import detect_rag_style, _extract_facts
from pipeline.v2_common import cleaner_scores, confident_cleaner
from pipeline.v2_paths import (V2_TRAIN, coldstart_train, coldstart_eval, coldstart_progress,
                               sft_row, read_jsonl, write_jsonl, load_support_index,
                               gather_until, V2_COLDSTART_TARGET, V2_GATHER_CHUNK)
from pipeline.logger import get_logger

log = get_logger("step_v2_coldstart")

_SUPPORT: dict = {}   # qid -> V1答案池记录（main 里 load 一次；_process 只读，map_concurrent 并发安全）


def _facts_ok(natural: str, rec: dict, v1ans: list) -> bool:
    """grounding 硬冲突闸（多agent审核 P0 修正版）：只拦 think 推向【V1 认可池里没有的结论极性】(=相反/越界结论)。
    口径与池锚点/answer_in_v1_pool 完全一致(rules_v6._extract_facts 归一：万元↔万同键、极性长词消位防子串)。
    ★数字的合法算术派生【不拦】——禁派生值会系统性误杀'代入数字一步步推到结论'这种最像人的样本(与本轮破天花板目标冲突)；
     最终答案由 answer-lock 保、结论数字由池锚点锁进输入，think 只需结论极性不与 V1 认可池矛盾即可。"""
    pool_pol = set()
    for a in (rec.get("answer"), rec.get("reasoning"), *(v1ans or [])):
        pool_pol |= _extract_facts(a or "")["pol"]
    if not pool_pol:
        return True                       # V1 池没抽到极性词→无从判，放过（交给规则门/σ门）
    return not (_extract_facts(natural)["pol"] - pool_pol)


def _process(rec: dict) -> dict:
    refs = reward.extract_references(rec.get("user_prompt") or "")
    v1_think = (rec.get("reasoning") or "").strip()
    v1ans = (_SUPPORT.get(rec.get("qid")) or {}).get("v1_answers") or []
    anchor = build_pool_anchor(v1ans) or (rec.get("answer") or "").strip()[:1500]  # 缺池则回退原答案 prose
    out = {"qid": rec.get("qid"), "query": rec.get("query"),
           "user_prompt": rec.get("user_prompt"), "answer": rec.get("answer"),
           "natural": None, "rule_ok": False, "facts_ok": False, "s_clean": None, "s_dirty": None,
           "pass2": False, "pass3": False}
    try:
        natural = kimi_client.chat(
            [{"role": "system", "content": REWRITE_SYSTEM},
             {"role": "user", "content": REWRITE_TEMPLATE_POOL.format(  # 治本版：不喂 v1_think；池锚点锁结论/数字到 answer_in_v1_pool 靶子
                 reference=refs[:6000], query=(rec.get("query") or "")[:500], anchor=anchor)}],
            temperature=REWRITE_TEMPERATURE, top_p=0.9, max_tokens=2048).strip()
    except Exception as e:
        log.warning("改写失败 qid=%s: %r", rec.get("qid"), e)
        return out
    out["natural"] = natural
    out["rule_ok"] = not detect_rag_style(natural)["has_rag_style"]
    if not out["rule_ok"]:                      # 规则门没过：改写还带检索腔，不进任何桶
        return out
    out["facts_ok"] = _facts_ok(natural, rec, v1ans)
    if not out["facts_ok"]:                     # facts 闸：think 冒出 参考∪答案∪池∪原think∪题干 之外的新数字=臆造，丢（在 σ 前、省 Kimi）
        return out
    s_clean, s_dirty = cleaner_scores(refs, natural, v1_think)
    out["s_clean"], out["s_dirty"] = s_clean, s_dirty
    if s_clean is None:                         # k=2 粗筛即被弃（不比 V1 更干净）
        return out
    out["pass2"] = confident_cleaner(s_clean, s_dirty, 2.0)
    out["pass3"] = confident_cleaner(s_clean, s_dirty, 3.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(V2_TRAIN))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（联调），0=全量")
    args = ap.parse_args()

    rows = read_jsonl(args.inp)
    if args.limit:
        rows = rows[: args.limit]
    global _SUPPORT
    _SUPPORT = load_support_index()    # qid -> V1 答案池（池锚点 + facts 闸用）；缺池的题自动回退原答案 prose
    log.info("冷启动改写+评分：%d 题（workers=%d）；V1 答案池 %d 题有池", len(rows), SEED_WORKERS, len(_SUPPORT))

    # 凑够就停：2σ 桶攒到 V2_COLDSTART_TARGET 即停、不再调 Kimi（每题的改写/规则门/σ门一概不变）
    results = gather_until(
        rows, _process, workers=SEED_WORKERS, chunk=V2_GATHER_CHUNK, desc="v2_coldstart",
        progress_path=coldstart_progress(),               # 边攒边落盘 + 续跑跳过已改写打分过的题
        enough=lambda rs: V2_COLDSTART_TARGET > 0
        and sum(1 for r in rs if r and r.get("pass2")) >= V2_COLDSTART_TARGET)

    eligible = [r for r in results if r.get("pass2")]    # 规则门+σ(≥2σ) 都过 → 可用自然 think
    eligible.sort(key=lambda r: r.get("qid") or "")      # 按 qid 稳定排序：eval 留出确定可复现（续跑/跑满同一批 qid → 同一 eval 划分；不动 gather_until 的洗牌处理序）
    n_eval = max(1, round(len(eligible) * COLDSTART_EVAL_FRAC)) if eligible else 0
    every = max(2, round(1 / COLDSTART_EVAL_FRAC))
    eval_rows, train2_rows, train3_rows = [], [], []
    for i, r in enumerate(eligible):
        row = sft_row(r["user_prompt"], r["natural"], r["answer"], query=r.get("query"))
        if i % every == 0:                                # 确定性留出自然腔 eval（不进训练桶）
            eval_rows.append(row)
            continue
        train2_rows.append(row)                           # 2σ 桶
        if r.get("pass3"):
            train3_rows.append(row)                       # 3σ 桶 ⊆ 2σ 桶

    # 防护栏②：找遍/早停后没凑够目标 → 用现有这些继续训，【不报错、只反馈】。
    # (i%every 留出保证：只要 2σ 桶非空，eval 必非空；故 swift 不会拿到空 val。空桶则 sft_node 自动跳过该线。)
    n_have = len(eligible)
    if 0 < n_have < V2_COLDSTART_TARGET and V2_COLDSTART_TARGET > 0:
        log.warning("冷启动：找遍/早停后只凑到 2σ=%d < 目标 %d —— 用现有这 %d 条继续训(train=%d/eval=%d)，不报错。",
                    n_have, V2_COLDSTART_TARGET, n_have, len(train2_rows), len(eval_rows))
    if not train2_rows:
        log.warning("冷启动：0 个 2σ 合格样本 —— 该桶空，SFT 会被编排器跳过该线(不崩不报错)。"
                    "请查改写质量/规则门/σ阈值，或放宽 confident_cleaner。")

    write_jsonl(coldstart_train(2), train2_rows)
    write_jsonl(coldstart_train(3), train3_rows)
    write_jsonl(coldstart_eval(), eval_rows)

    n_rule_fail = sum(1 for r in results if r.get("natural") and not r.get("rule_ok"))
    n_facts_fail = sum(1 for r in results if r.get("rule_ok") and not r.get("facts_ok"))
    n_screen = sum(1 for r in results if r.get("facts_ok") and r.get("s_clean") is None)
    log.info("漏斗：改写成功 %d / 规则挡 %d / facts闸挡 %d / k2筛掉 %d / 2σ可用 %d（eval留出 %d）",
             sum(1 for r in results if r.get("natural")), n_rule_fail, n_facts_fail, n_screen, len(eligible), len(eval_rows))
    log.info("train 2σ=%d -> %s", len(train2_rows), coldstart_train(2))
    log.info("train 3σ=%d -> %s", len(train3_rows), coldstart_train(3))
    log.info("eval(自然腔)=%d -> %s", len(eval_rows), coldstart_eval())
    if len(train2_rows) < 200:
        log.warning("2σ 训练桶 %d < 200：σ 分辨率把可用样本卡得很少，考虑放宽到 2σ-only 或多采改写。", len(train2_rows))


if __name__ == "__main__":
    main()
