"""探针：新改写提示词 A/B（先证伪、再决定要不要全量重烧）。零 GPU、纯 Kimi。

抽 N 道训练题，三臂对打，每臂产 think 后用 judgecal 判官打干净分(k 次取均值) + 规则去检索腔 + grounding：
  OLD        = 旧版 step06（喂 V1 think 让"改写"）—— 基线，实测均值 ~3.44
  FINAL      = 新主方案（不喂 think，锚点 = V1 原答案 prose）
  FINAL_POOL = 新方案 + 池锚点（不喂 think，锚点 = V1 答案池抽出的 极性+数字 结构化事实 = answer_in_v1_pool 靶子）
判据：干净分要明显 > OLD 且冲到 6+ 才算破天花板；FINAL_POOL 的 grounding 应最高（直接锁到评测靶子）。

用法：python probe_rewrite_v2.py --n 40 --k 3      （约 ¥12-18、几分钟）
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline import kimi_client, reward, vllm_client
from pipeline.rules_v6 import detect_rag_style
from pipeline.judgecal_common import judge_clean_score
from pipeline.step06_rewrite_seeds import REWRITE_SYSTEM as OLD_SYS, REWRITE_TEMPLATE as OLD_TPL
from pipeline.rewrite_v2 import (REWRITE_SYSTEM, REWRITE_TEMPLATE, REWRITE_TEMPLATE_POOL,
                                 REWRITE_TEMPERATURE, build_pool_anchor)
from pipeline.step_v2_coldstart import _facts_ok          # 与生产同口径：只拦极性硬冲突、放过算术派生值
from pipeline.v2_paths import V2_TRAIN, V2_V1_SUPPORT, read_jsonl, load_support_index


def _gen(sys_p: str, tpl: str, kw: dict, temp: float) -> str:
    msg = tpl.format(**{k: v for k, v in kw.items() if "{" + k + "}" in tpl})  # 只填模板含的占位符
    return kimi_client.chat(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": msg}],
        temperature=temp, top_p=0.9, max_tokens=2048).strip()


def run_arm(name: str, sys_p: str, tpl: str, temp: float, rows: list, k: int, support: dict) -> dict:
    def one(rec):
        refs = reward.extract_references(rec.get("user_prompt") or "")
        v1ans = (support.get(rec.get("qid")) or {}).get("v1_answers") or []
        anchor = build_pool_anchor(v1ans) or (rec.get("answer") or "").strip()[:1500]  # 无池则回退原答案
        kw = {"reference": refs[:6000], "query": (rec.get("query") or "")[:500],
              "think": (rec.get("reasoning") or "").strip()[:4000],
              "answer": (rec.get("answer") or "").strip()[:1500], "anchor": anchor}
        try:
            think = _gen(sys_p, tpl, kw, temp)
        except Exception:
            return None
        scores = []
        for _ in range(k):
            try:
                scores.append(judge_clean_score(refs, think)["clean_score"])
            except Exception:
                pass
        return {"clean": (sum(scores) / len(scores)) if scores else None,
                "rule_ok": not detect_rag_style(think)["has_rag_style"],
                "facts_ok": _facts_ok(think, rec, v1ans)}      # 生产同口径：只拦结论极性硬冲突
    res = [r for r in vllm_client.map_concurrent(rows, one, workers=4, desc=f"probe:{name}") if r]
    cl = [r["clean"] for r in res if r["clean"] is not None]
    n = max(1, len(res))
    mean = sum(cl) / len(cl) if cl else 0.0
    print(f"\n=== {name}  (temp={temp}, n={len(res)}) ===")
    print(f"  干净分均值        : {mean:.2f}" + ("" if cl else "（全失败）"))
    print(f"  规则去检索腔通过率: {sum(r['rule_ok'] for r in res) / n:.0%}")
    print(f"  grounding(极性不冲突): {sum(r['facts_ok'] for r in res) / n:.0%}")
    return {"clean": mean, "rule": sum(r['rule_ok'] for r in res) / n, "facts": sum(r['facts_ok'] for r in res) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="抽样题数")
    ap.add_argument("--k", type=int, default=3, help="每条 think 干净分打几次取均值（压判官噪声）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_jsonl(V2_TRAIN)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    support = load_support_index(V2_V1_SUPPORT)
    n_anch = sum(1 for r in rows if build_pool_anchor((support.get(r.get("qid")) or {}).get("v1_answers")))
    print(f"探针：{len(rows)} 题（{n_anch} 题有池锚点），每条干净分打 {args.k} 次取均值；旧版实测基线 ~3.44。")

    old = run_arm("OLD 旧·喂think改写", OLD_SYS, OLD_TPL, 0.3, rows, args.k, support)
    fin = run_arm("FINAL 新·不喂think·答案锚", REWRITE_SYSTEM, REWRITE_TEMPLATE, REWRITE_TEMPERATURE, rows, args.k, support)
    pool = run_arm("FINAL_POOL 新·不喂think·池锚点", REWRITE_SYSTEM, REWRITE_TEMPLATE_POOL, REWRITE_TEMPERATURE, rows, args.k, support)

    print("\n###################### 结论 ######################")
    print(f"干净分:  OLD {old['clean']:.2f}  →  FINAL {fin['clean']:.2f}  /  FINAL_POOL {pool['clean']:.2f}")
    bn, bv = max([("FINAL", fin), ("FINAL_POOL", pool)], key=lambda x: x[1]["clean"])
    print(f"最佳新方案: {bn}  干净 {bv['clean']:.2f}（{bv['clean'] - old['clean']:+.2f} vs OLD）", end="  ")
    if bv["clean"] >= 6 and bv["clean"] > old["clean"] + 1:
        print("✅ 破天花板（≥6 且明显超旧版）→ 用它全量重烧")
    elif bv["clean"] > old["clean"] + 1:
        print("🟡 有提升但没到 6 → 提示词还需再打磨，先别全量")
    else:
        print("❌ 没有实质提升 → 方向要重想，别浪费 Kimi 全量")
    print(f"grounding(极性不冲突): OLD {old['facts']:.0%} / FINAL {fin['facts']:.0%} / FINAL_POOL {pool['facts']:.0%}  ← 池锚点应最高")
    print(f"规则去检索腔: OLD {old['rule']:.0%} / FINAL {fin['rule']:.0%} / FINAL_POOL {pool['rule']:.0%}")


if __name__ == "__main__":
    main()
