"""Kimi 判分（所有阶段复用）：独立金标准裁判，与本地选样器物理分开（防 Goodhart）。

裁判 = 公网 DashScope Kimi（跨家规避同源偏好）。对每条评测输出严格 JSON 打两类分：
1. 准确率：以 V1 答案为绝对正确 → correct/partial/incorrect + 0~1 分（measure 的是"漂没漂"）。
2. think humanness(0~1)：越像端到端 CoT、越不像 RAG 越高；并标 RAG 痕迹类型。
判分温度显式锁定 0.0（确定性、口径可比）。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JUDGE_CALL_WORKERS
from pipeline import vllm_client
from pipeline.judge_common import JUDGE_SYSTEM, JUDGE_TEMPLATE, judge_text, parse_judge_json as _parse_json
from pipeline.logger import get_logger

log = get_logger("step04_judge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="step03 评测推理产物")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    log.info("Kimi 判分：%d 条", len(recs))

    def _judge(rec: dict) -> dict:
        try:
            j = judge_text(
                rec.get("query", ""),
                rec.get("user_prompt") or "",
                rec.get("gold_answer", ""),
                rec.get("gen_text") or "",
            )
        except Exception as e:
            log.warning("判分失败 query=%s...: %r", (rec.get("query") or "")[:30], e)
            j = {"accuracy": "incorrect", "accuracy_score": 0.0, "humanness": 0.0, "grounded": 0.0,
                 "rag_traces": [], "comment": f"judge_error:{e}"}
        return {**rec, "judge": j}

    results = vllm_client.map_concurrent(recs, _judge, workers=JUDGE_CALL_WORKERS, desc="judge")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("完成：%d -> %s", len(results), args.out)


if __name__ == "__main__":
    main()
