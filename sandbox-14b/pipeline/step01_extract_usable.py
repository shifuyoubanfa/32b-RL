"""Step 1: 从 xlsx 抽取标注为'可用'的 query，去重后落盘 jsonl。"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_XLSX, USABLE_QUERIES
from pipeline.logger import get_logger

log = get_logger("step01")


def main():
    log.info("读取 %s", RAW_XLSX)
    df = pd.read_excel(RAW_XLSX)
    log.info("原始样本数: %d", len(df))
    log.info("A是否可用 分布: %s", df["A是否可用"].value_counts(dropna=False).to_dict())

    usable = df[df["A是否可用"] == "可用"].copy()
    usable = usable.dropna(subset=["query"])
    usable["query"] = usable["query"].astype(str).str.strip()
    usable = usable[usable["query"].str.len() > 0]
    usable = usable.drop_duplicates(subset=["query"])

    log.info("可用且去重后的 query 数: %d", len(usable))

    with open(USABLE_QUERIES, "w", encoding="utf-8") as f:
        for _, row in usable.iterrows():
            record = {
                "dialogue_id": row.get("dialogue_id"),
                "query": row["query"],
                "company_answer_from_xlsx": row.get("A"),
                "问题分类": row.get("问题分类"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("已写入: %s", USABLE_QUERIES)


if __name__ == "__main__":
    main()
