"""judgecal step162 · 一次采集读两遍：实验一(分数稳不稳→该打几遍) + 实验二(分辨率曲线→饱和拐点=RL天花板)。

从 step161 的"每条 think × Kimi 0-10 干净分 16 遍"里读两件事：

实验一（分数跳不跳 → 该打几遍）：
  每条 think 的 16 个分数有多大抖动(σ)；取平均后噪声按 σ/√k 缩；找 σ/√k 小于半个档距(1.0)所需的 k。

实验二（分辨率曲线 → 饱和拐点 = RL 天花板）【用户的核心设计】：
  6 个真实档位(没抄=锚10 / 抄1=8 / 抄2=6 / 抄3=4 / 抄4=2 / 完全照抄=0)，看 Kimi 平均干净分随档位怎么走。
  ① 单调性(锚点↑ Kimi分↑ 的 Spearman)；② 相邻档位 Kimi 分分不分得开(均值差≥1 且 1倍标准差带不重叠)。
  **曲线在哪一段"趋于直线/压平"，Kimi 就在那儿分不开 → 即使模型真有那段强化学习信号也captured不了 = RL 天花板。**
  单独盯【没抄↔抄1句】拐点：分不开 = 把合法用事实误判成照抄(阶段7 的病)。

纯 CPU、不调 Kimi、不碰 GPU。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.logger import get_logger
from pipeline.v3_utils import mean, pearson, read_jsonl, sd

log = get_logger("step162_judgecal")

LEVEL_ORDER = ["没抄", "抄1句", "抄2句", "抄3句", "抄4句", "完全照抄"]  # 由干净到脏，anchor 10→0
ANCHOR_GAP = 2.0
SEP_MIN = float(os.environ.get("JUDGECAL_SEP_MIN", "1.0"))   # 相邻档位 Kimi 分至少差这么多才算分得开


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def collect_items(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        valid = [s["clean_score"] for s in r.get("scores") or [] if not s.get("error") and s.get("clean_score") is not None]
        if not valid:
            continue
        out.append({"item_id": r.get("item_id"), "true_level": r.get("true_level"),
                    "level_idx": r.get("level_idx"), "anchor": r.get("anchor"),
                    "scores": valid, "n_valid": len(valid),
                    "item_mean": mean(valid), "item_sd": sd(valid)})
    return out


def exp_one(items: list[dict]) -> dict:
    """分数稳不稳：单遍打分 σ，取平均后 σ/√k，找够稳的 k。"""
    sigma = mean([it["item_sd"] for it in items])   # 典型单遍打分标准差
    ks = [1, 2, 4, 8, 16]
    se = {k: (sigma / (k ** 0.5)) for k in ks}
    rec_k = next((k for k in ks if se[k] < SEP_MIN), None)  # 噪声 < 半个档距即够
    return {"sigma": sigma, "se_by_k": se, "rec_k": rec_k}


def exp_two(items: list[dict]) -> dict:
    """分辨率曲线 + 饱和拐点。"""
    by_level = defaultdict(list)
    for it in items:
        by_level[it["level_idx"]].append(it)
    present = sorted(by_level)   # 0..5
    level_stats = {}
    for li in present:
        ms = [it["item_mean"] for it in by_level[li]]
        level_stats[li] = {"true_level": LEVEL_ORDER[li] if li < len(LEVEL_ORDER) else str(li),
                           "anchor": 10 - 2 * li, "n_items": len(ms),
                           "kimi_mean": mean(ms), "kimi_sd": sd(ms),
                           "kimi_min": min(ms), "kimi_max": max(ms)}
    # 单调性：锚点 vs Kimi 平均分（按每条算，应正相关 ~1）
    anchors = [it["anchor"] for it in items]
    means = [it["item_mean"] for it in items]
    rho = pearson(_ranks(anchors), _ranks(means))

    # 相邻档位（idx i 干净 → i+1 更脏）分不分得开：Kimi 分应从高往低掉
    adjacent = []
    for lo, hi in zip(present, present[1:]):
        a, b = level_stats[lo], level_stats[hi]   # lo 更干净(分应更高)
        gap = a["kimi_mean"] - b["kimi_mean"]
        sep = gap >= SEP_MIN and (a["kimi_mean"] - a["kimi_sd"]) > (b["kimi_mean"] + b["kimi_sd"])
        adjacent.append({"lo_idx": lo, "hi_idx": hi, "lo": a["true_level"], "hi": b["true_level"],
                         "gap": gap, "sep": sep})

    # 饱和：连续"分不开"的相邻档位 = 压平区段
    flat_runs = []
    cur = []
    for d in adjacent:
        if not d["sep"]:
            cur.append(d)
        else:
            if cur:
                flat_runs.append(cur); cur = []
    if cur:
        flat_runs.append(cur)
    flat_spans = [f"{run[0]['lo']}~{run[-1]['hi']}" for run in flat_runs]

    all_sep = all(d["sep"] for d in adjacent)
    # 没抄↔抄1句 拐点（合法用事实会不会被误判成照抄）
    legit_boundary = next((d for d in adjacent if d["lo_idx"] == 0 and d["hi_idx"] == 1), None)

    return {"present": present, "level_stats": level_stats, "spearman": rho,
            "adjacent": adjacent, "flat_spans": flat_spans, "all_sep": all_sep,
            "legit_boundary": legit_boundary}


def outliers(items: list[dict], thresh: float = 3.0) -> list[dict]:
    """Kimi 平均分离它自己档位锚点太远(>thresh)的条目，给人工核对(可能标错或样本太难)。"""
    bad = []
    for it in items:
        d = abs(it["item_mean"] - it["anchor"])
        if d > thresh:
            bad.append({"item_id": it["item_id"], "true_level": it["true_level"],
                        "anchor": it["anchor"], "kimi_mean": round(it["item_mean"], 2), "off": round(d, 2)})
    return sorted(bad, key=lambda x: -x["off"])


def make_report(e1: dict, e2: dict, outs: list[dict], n_items: int) -> str:
    L = ["# judgecal · Kimi 换词复述照抄 分辨率标定报告", "",
         f"- 数据：{n_items} 条 think，6 个真实档位（没抄=锚10 / 抄1=8 / 抄2=6 / 抄3=4 / 抄4=2 / 完全照抄=0）",
         f"- 采集：每条 think × Kimi 打 0-10 干净分 16 遍（一次采集、读两遍）",
         f"- 核心：看 Kimi 干净分随真实档位的曲线在哪压平（饱和），那拐点=分辨率上限=RL 天花板", ""]

    # 实验一
    L += ["## 实验一 · 分数跳不跳 → 该打几遍", "",
          f"- 单遍打分典型标准差 σ = **{e1['sigma']:.2f}**（0-10 量程）", "",
          "| k(取平均遍数) | 取平均后噪声 σ/√k |", "|---|---:|"]
    for k, v in e1["se_by_k"].items():
        L.append(f"| {k} | {v:.3f} |")
    rk = e1["rec_k"]
    L += ["", f"**核心结论·该打几遍 K = {rk if rk is not None else '>16(σ 太大，16 遍仍不够，需加大 JUDGECAL_KMAX)'}"
          f"（噪声压到 < 半个档距 {SEP_MIN} 所需）。**", ""]

    # 实验二 · 分辨率曲线
    L += ["## 实验二 · 分辨率曲线 → 饱和拐点 = RL 天花板", "",
          "| 真实档位 | 客观锚点 | 条数 | Kimi干净分·均值 | 标准差 | min..max |",
          "|---|---:|---:|---:|---:|---:|"]
    for li in e2["present"]:
        s = e2["level_stats"][li]
        L.append(f"| {s['true_level']} | {s['anchor']} | {s['n_items']} | {s['kimi_mean']:.2f} "
                 f"| {s['kimi_sd']:.2f} | {s['kimi_min']:.1f}..{s['kimi_max']:.1f} |")
    rho = e2["spearman"]
    L += ["", f"- 单调性（客观锚点↑ → Kimi 分↑ 的 Spearman）：{'NA' if rho is None else f'{rho:.3f}'}（越接近 1 越好）", "",
          "### 相邻档位分得开吗（差1档，Kimi 分该往下掉≥1 且误差带不重叠）", "",
          "| 干净档→脏档 | Kimi 均值差 | 分得开? |", "|---|---:|---|"]
    for d in e2["adjacent"]:
        L.append(f"| {d['lo']}→{d['hi']} | {d['gap']:+.2f} | {'是' if d['sep'] else '否'} |")

    # 饱和拐点 + RL 天花板
    if e2["all_sep"]:
        verdict = "**Kimi 能把 6 档全部分开 → 不存在分辨率盲区，全程都有可用 RL 梯度。**"
    else:
        spans = "、".join(e2["flat_spans"]) or "（见上表）"
        verdict = (f"**曲线在 {spans} 压平（相邻档位分不开）→ Kimi 在这一段没有分辨力；"
                   f"即使模型在这段真有质量差，Kimi 也捕捉不到 = 强化学习在此到顶（天花板）。**")
    L += ["", f"### 核心结论·饱和拐点 / RL 天花板\n{verdict}", ""]
    lb = e2["legit_boundary"]
    if lb is not None:
        if lb["sep"]:
            L.append(f"- 没抄↔抄1句拐点：**分得开**（Kimi 没把合法用事实误判成照抄，阶段7 的病这里没犯）。")
        else:
            L.append("- 没抄↔抄1句拐点：**分不开（警告）**——Kimi 把『只用了事实』的干净 think 也压低分，"
                     "等于把合法用事实当成照抄（阶段7 老病复发）。")
    L.append("")

    # 离群清单
    L += ["## 离群清单（Kimi 均值离自己档位锚点 >3 分，逐条可人工核对：标错？还是样本太难？）", ""]
    if outs:
        L += ["| item | 真实档位 | 锚点 | Kimi均值 | 偏离 |", "|---|---|---:|---:|---:|"]
        for b in outs:
            L.append(f"| {b['item_id']} | {b['true_level']} | {b['anchor']} | {b['kimi_mean']} | {b['off']} |")
    else:
        L.append("（无明显离群）")
    L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", required=True, help="step161 的 161_sentence_judges.jsonl")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.judges)
    if not rows:
        raise SystemExit(f"空采集结果: {args.judges}")
    items = collect_items(rows)
    if not items:
        raise SystemExit("没有有效打分（全失败？检查 Kimi 调用）")

    e1 = exp_one(items)
    e2 = exp_two(items)
    outs = outliers(items)
    # 规则函数1 旁证：这些 think 是"照抄但不带检索腔词"，规则应一律判无检索腔（照抄全靠 Kimi）。
    rule_hits = sum(1 for r in rows if r.get("rule_has_rag_style"))

    md = make_report(e1, e2, outs, len(items))
    md += (f"\n## 规则 vs Kimi 分工旁证（规则函数1 `detect_rag_style`）\n\n"
           f"- 规则检索腔命中：**{rule_hits}/{len(rows)}** 条 think\n"
           f"- 预期 0：这些 think 都是『照抄但不带检索腔词』，规则看不见照抄 → **照抄全靠 Kimi**；"
           f"规则只管检索腔(此处无)和答案漂移(本实验不涉及答案)。\n")
    Path(args.out_md).write_text(md, encoding="utf-8")
    decision = {
        "sigma": e1["sigma"], "rec_k": e1["rec_k"],
        "spearman": e2["spearman"], "all_levels_separable": e2["all_sep"],
        "flat_spans": e2["flat_spans"],
        "legit_boundary_separable": (e2["legit_boundary"] or {}).get("sep"),
        "level_stats": {str(k): v for k, v in e2["level_stats"].items()},
        "adjacent": e2["adjacent"], "n_items": len(items), "n_outliers": len(outs),
        "rule_rag_hits": rule_hits,
    }
    Path(args.out_json).write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"RESULT judgecal_analysis rec_k={e1['rec_k']} spearman="
          f"{'NA' if e2['spearman'] is None else round(e2['spearman'],3)} "
          f"all_separable={e2['all_sep']} flat={e2['flat_spans']} "
          f"legit_boundary_sep={(e2['legit_boundary'] or {}).get('sep')} "
          f"rule_rag_hits={rule_hits}/{len(rows)} report={args.out_md}", flush=True)
    log.info("分析完成 -> %s", args.out_md)


if __name__ == "__main__":
    main()
