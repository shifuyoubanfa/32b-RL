"""Step 1c: 把 00_data_queries_usable.jsonl 与 00_data_queries_summary.jsonl 合并去重。"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import USABLE_QUERIES, SUMMARY_QUERIES, ALL_QUERIES
from pipeline.logger import get_logger

log = get_logger("step01c")


def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    a = load(USABLE_QUERIES)
    b = load(SUMMARY_QUERIES)
    log.info("usable=%d summary=%d", len(a), len(b))

    seen: set[str] = set()
    merged: list[dict] = []
    for rec in a + b:
        q = (rec.get("query") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        merged.append(rec)

    with open(ALL_QUERIES, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("合并后去重: %d -> %s", len(merged), ALL_QUERIES)


if __name__ == "__main__":
    main()
