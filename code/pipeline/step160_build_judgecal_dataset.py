"""judgecal step160 · 校验并装配逐句标定数据集。

读 data/judgecal_sentences.jsonl（人工构造、每句带标签），做严格校验：
- 四类标签合法：verbatim / reworded / legit_use / original；
- 每条 think 句子非空、sid 连续；
- 报四类各多少句（reworded/legit_use 是头条指标，太少会警告）。
装配出规范化 160_judgecal_items.jsonl，并打印类别计数。

纯 CPU、不调 Kimi、不碰 GPU。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR
from pipeline.logger import get_logger
from pipeline.rules_v6 import detect_rag_style
from pipeline.v3_utils import read_jsonl, write_jsonl

log = get_logger("step160_judgecal")

LABELS = ("verbatim", "reworded", "legit_use", "original")
# 真值"该不该被标为照抄"：verbatim/reworded 该标，legit_use/original 不该标。
SHOULD_FLAG = {"verbatim": True, "reworded": True, "legit_use": False, "original": False}


def validate_item(it: dict) -> list[str]:
    errs = []
    iid = it.get("item_id") or "<no-id>"
    if not (it.get("reference") or "").strip():
        errs.append(f"{iid}: reference 为空")
    sents = it.get("sentences") or []
    if not sents:
        errs.append(f"{iid}: 没有句子")
    for i, s in enumerate(sents):
        if not (s.get("text") or "").strip():
            errs.append(f"{iid}[{i}]: text 为空")
        if s.get("label") not in LABELS:
            errs.append(f"{iid}[{i}]: 非法 label={s.get('label')!r}（须 {LABELS}）")
        if int(s.get("sid", -1)) != i:
            errs.append(f"{iid}[{i}]: sid={s.get('sid')} 与位置不连续（应={i}）")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(Path(DATA_DIR) / "judgecal_sentences.jsonl"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = read_jsonl(args.inp)
    if not items:
        raise SystemExit(f"空数据集或不存在: {args.inp}")

    all_errs = []
    cat = Counter()
    for it in items:
        all_errs += validate_item(it)
        for s in it.get("sentences") or []:
            if s.get("label") in LABELS:
                cat[s["label"]] += 1
    if all_errs:
        for e in all_errs:
            log.error("校验失败 | %s", e)
        raise SystemExit(f"数据集校验未通过，共 {len(all_errs)} 处，先修 {args.inp}")

    # 规范化输出：每条 think 留存【客观事实标准】——照抄句数 + 6 档真实档位 + 0-10 干净度锚点。
    # 档位(由干净到脏): 没抄=10 / 抄1句=8 / 抄2句=6 / 抄3句=4 / 抄4句=2 / 完全照抄=0。
    out_rows = []
    level_bins = Counter()
    for it in items:
        sents = [{"sid": int(s["sid"]), "text": s["text"], "label": s["label"],
                  "should_flag": SHOULD_FLAG[s["label"]]} for s in it["sentences"]]
        copy_count = sum(1 for s in sents if s["should_flag"])
        total = len(sents)
        if copy_count == 0:
            level_idx, true_level = 0, "没抄"
        elif copy_count >= total:          # 整段每句都是照抄
            level_idx, true_level = 5, "完全照抄"
        else:
            level_idx, true_level = min(copy_count, 4), f"抄{copy_count}句"
        anchor = 10 - 2 * level_idx        # 客观锚点(干净度), 没抄=10 ... 完全照抄=0
        level_bins[true_level] += 1
        # 规则函数1 旁证：这些 think 是"照抄但不带检索腔词"，规则应一律判无检索腔（照抄全靠 Kimi）。
        rag = detect_rag_style("".join(s["text"] for s in sents))
        out_rows.append({"item_id": it.get("item_id"), "topic": it.get("topic"),
                         "reference": it["reference"], "sentences": sents,
                         "true_copy_count": copy_count, "true_level": true_level,
                         "level_idx": level_idx, "anchor": anchor,
                         "rule_has_rag_style": rag["has_rag_style"], "rule_rag_spans": rag["spans"]})
    write_jsonl(args.out, out_rows)

    n_sent = sum(len(r["sentences"]) for r in out_rows)
    order = ["没抄", "抄1句", "抄2句", "抄3句", "抄4句", "完全照抄"]
    bins_str = " ".join(f"{k}={level_bins[k]}" for k in order if level_bins[k])
    print(f"RESULT judgecal_dataset items={len(out_rows)} sentences={n_sent} "
          f"verbatim={cat['verbatim']} reworded={cat['reworded']} "
          f"legit_use={cat['legit_use']} original={cat['original']} "
          f"level_bins=[{bins_str}] out={args.out}", flush=True)
    for label in LABELS:
        if SHOULD_FLAG[label] and cat[label] < 12:
            log.warning("类别 %s 仅 %d 句，偏少（头条指标精度会差，建议 ≥15）", label, cat[label])
    log.info("装配完成：%d 条 / %d 句 -> %s", len(out_rows), n_sent, args.out)


if __name__ == "__main__":
    main()
