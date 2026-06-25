# -*- coding: utf-8 -*-
"""为 1.2 各阶段输出对比 挑一道好例子：基线 think 有检索腔、DPO/GRPO think 干净、五阶段答案都在池。
从 derag2 的 5 个 infer + scores + support 读，纯只读，打印选中例子全文 + 候选清单。"""
import json, sys, os
sys.path.insert(0, "32b强化学习/code")
from pipeline.rules_v6 import detect_rag_style, answer_in_v1_pool
from pipeline.v2_paths import qid_of

D = "32b强化学习/derag2"
STAGES = [("V1 基线", "v2-baseline-v1"), ("SFT", "v2-sft-2s"), ("RFT", "v2-rft-2s-2s"),
          ("DPO", "v2-2s-2s-2s"), ("GRPO", "v2-2s-2s-2s-grpo")]

def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line); out[r.get("query")] = r
    return out

infer = {tag: load(f"{D}/{tag}_infer.jsonl") for _, tag in STAGES}
scores = {}
for _, tag in STAGES:
    sc = {}
    with open(f"{D}/{tag}_scores.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line); sc[r.get("qid")] = r
    scores[tag] = sc
support = {}
with open(f"{D}/152_v1_support.v2.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            support[r.get("qid") or qid_of(r.get("query"))] = r.get("v1_answers") or []

common = set(infer[STAGES[0][1]])
for _, tag in STAGES[1:]:
    common &= set(infer[tag])

def stage_info(tag, q):
    r = infer[tag].get(q, {})
    think = r.get("think") or ""
    ans = r.get("answer") or ""
    qid = qid_of(q)
    rs = detect_rag_style(think)
    ap = answer_in_v1_pool(ans, support.get(qid, []))
    cs = scores[tag].get(qid, {}).get("clean_score")
    return think, ans, rs, ap, cs

cands = []
for q in common:
    if not (15 <= len(q) <= 46):
        continue
    binfo = stage_info("v2-baseline-v1", q)
    dinfo = stage_info("v2-2s-2s-2s", q)
    ginfo = stage_info("v2-2s-2s-2s-grpo", q)
    # 基线有痕、DPO 与 GRPO 都无痕、且五阶段答案都在池
    allpool = all(stage_info(tag, q)[3].get("in_pool") for _, tag in STAGES)
    if binfo[2]["has_rag_style"] and not dinfo[2]["has_rag_style"] and not ginfo[2]["has_rag_style"] and allpool:
        score = len(binfo[2].get("spans", [])) - len(dinfo[0])/500.0
        cands.append((score, q))

cands.sort(reverse=True)
out = open("32b强化学习/report_figs/_example_out.txt", "w", encoding="utf-8")
def P(*a):
    print(*("".join(str(x) for x in a),), file=out)
P(f"== 候选数 {len(cands)} / 共同题 {len(common)} ==")
for _, q in cands[:10]:
    P("  - ", q)

for rank in range(min(3, len(cands))):
    q = cands[rank][1]
    P("\n========== 候选 #%d ==========" % rank)
    P("【问题】", q)
    refs_src = infer["v2-2s-2s-2s"].get(q, {}).get("user_prompt", "")
    P("\n【user_prompt(截断1400)】\n", refs_src[:1400])
    for name, tag in STAGES:
        think, ans, rs, ap, cs = stage_info(tag, q)
        P(f"\n---------- {name} ({tag}) ----------")
        P(f"[规则think] {'有痕迹' if rs['has_rag_style'] else '无痕迹'}  痕迹spans=", [s.get('span') for s in rs.get('spans', [])][:8])
        P(f"[规则answer] {'在池' if ap.get('in_pool') else '漂移'}  reason={ap.get('reason')} drift=", ap.get('drift_facts'))
        P(f"[kimi think 干净分] {cs}")
        P("[think]\n", think)
        P("[answer]\n", ans)
out.close()
print("written -> _example_out.txt")
