"""Create and optionally score a blind human spot-check sheet for Stage1."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


ENUM_QUERY_RE = re.compile(r"哪些|包括|清单|税率表|具体|分别|如何操作|怎么填|填写|分录")


def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()] if p.exists() else []


def pick_unique(groups: list[list[dict]], limits: list[int]) -> list[dict]:
    picked, seen = [], set()
    for rows, limit in zip(groups, limits):
        count = 0
        for row in rows:
            key = row.get("cand_id") or hashlib.sha256(
                json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]
            if key in seen:
                continue
            picked.append(row)
            seen.add(key)
            count += 1
            if count >= limit:
                break
    return picked


def blind_row(row: dict, index: int, source: str) -> dict:
    record = row.get("record") or {}
    return {
        "blind_id": f"S{index:03d}",
        "source": source,
        "query": record.get("query") or "",
        "think": record.get("natural_think") or record.get("think") or "",
        "human_mechanical_trace": None,
        "human_note": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_rows", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--pass_n", type=int, default=30)
    ap.add_argument("--fail_n", type=int, default=20)
    ap.add_argument("--decisions", default="")
    args = ap.parse_args()

    rows = load_jsonl(args.gate_rows)
    passed = [row for row in rows if row.get("pass")]
    failed = [row for row in rows if not row.get("pass")]
    rng = random.Random(20260612)
    random_pass = list(passed)
    random_fail = list(failed)
    rng.shuffle(random_pass)
    rng.shuffle(random_fail)
    top_copy = sorted(
        passed, key=lambda row: float((row.get("features") or {}).get("masked_copy") or 0.0), reverse=True
    )
    enum_rows = [row for row in passed if ENUM_QUERY_RE.search(str((row.get("record") or {}).get("query") or ""))]

    third = max(1, args.pass_n // 3)
    pass_sample = pick_unique(
        [top_copy, enum_rows, random_pass, random_pass],
        [third, third, args.pass_n - 2 * third, args.pass_n],
    )[: args.pass_n]
    fail_sample = random_fail[: args.fail_n]
    joined = [(row, "PASS") for row in pass_sample] + [(row, "FAIL") for row in fail_sample]
    rng.shuffle(joined)

    blind = [blind_row(row, index + 1, source) for index, (row, source) in enumerate(joined)]
    mapping = []
    for item, (row, source) in zip(blind, joined):
        mapping.append({
            "blind_id": item["blind_id"],
            "source": source,
            "cand_id": row.get("cand_id"),
            "gate_path": row.get("l3_path"),
            "decision_reason": row.get("decision_reason"),
        })

    lines = [
        "# Stage1 blind spot-check sheet",
        "",
        "判定字段：`human_mechanical_trace = true/false`。只判断是否仍有机械 RAG 痕迹，不判断答案文风。",
        "",
    ]
    for item in blind:
        lines += [
            f"## {item['blind_id']}",
            "",
            f"**问题：** {item['query']}",
            "",
            "**待审 think：**",
            "",
            item["think"],
            "",
            "- human_mechanical_trace:",
            "- human_note:",
            "",
        ]
    sheet = Path(args.sheet)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text("\n".join(lines), encoding="utf-8")
    Path(args.mapping).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "status": "PENDING",
        "pass_sample": len(pass_sample),
        "fail_sample": len(fail_sample),
        "sheet": str(sheet),
        "mapping": args.mapping,
        "requirements": {
            "normal_pass_mechanical_max": 3,
            "degraded_pass_mechanical_max": 6,
            "fail_false_positive": "record_only",
        },
    }
    decisions_path = Path(args.decisions) if args.decisions else None
    if decisions_path and decisions_path.exists():
        decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
        by_id = {item["blind_id"]: item for item in decisions}
        pass_ids = {item["blind_id"] for item in mapping if item["source"] == "PASS"}
        fail_ids = {item["blind_id"] for item in mapping if item["source"] == "FAIL"}
        pass_trace = sum(bool(by_id.get(key, {}).get("human_mechanical_trace")) for key in pass_ids)
        fail_clean = sum(by_id.get(key, {}).get("human_mechanical_trace") is False for key in fail_ids)
        threshold = 6 if args.pass_n >= 60 else 3
        report.update({
            "status": "PASS" if pass_trace <= threshold else "FAIL",
            "pass_mechanical_trace": pass_trace,
            "pass_threshold": threshold,
            "fail_human_clean": fail_clean,
            "decisions": str(decisions_path),
        })
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
