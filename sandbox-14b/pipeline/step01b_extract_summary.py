"""Step 1b: 从 '一阶段模型输出xxx.xlsx' 抽取 '模型总结问题'，去重落盘。"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SUMMARY_XLSX, SUMMARY_QUERIES
from pipeline.logger import get_logger

log = get_logger("step01b")


def main():
    log.info("读取 %s", SUMMARY_XLSX)
    df = pd.read_excel(SUMMARY_XLSX)
    log.info("原始样本数: %d", len(df))
    log.info("列: %s", list(df.columns))

    col = "模型总结问题"
    assert col in df.columns, f"未找到列 {col}"

    df = df.dropna(subset=[col]).copy()
    df[col] = df[col].astype(str).str.strip()
    df = df[df[col].str.len() > 0]
    df = df.drop_duplicates(subset=[col])

    log.info("去空+去重后: %d", len(df))

    with open(SUMMARY_QUERIES, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "dialogue_id": row.get("机器人会话ID"),
                "query": row[col],
                "company_answer_from_xlsx": row.get("答案"),
                "地区": row.get("地区"),
                "用户原问题": row.get("用户问题"),
                "source": "summary_v2",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("已写入: %s", SUMMARY_QUERIES)


if __name__ == "__main__":
    main()
