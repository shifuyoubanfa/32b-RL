"""Utilities for corrected-v3 diagnostic probes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from config import OUTPUT_DIR

TRACE_KEYS = ("explicit_ref", "verbatim_copy", "ref_enumeration", "policy_source")
ACC_RANK = {"incorrect": 0, "partial": 1, "correct": 2}
INV_ACC = {0: "incorrect", 1: "partial", 2: "correct"}


def run_dir(prefix: str = "corrected_v3") -> Path:
    rid = os.environ.get("V3_RUN_ID", "").strip()
    if not rid:
        from datetime import datetime
        rid = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(OUTPUT_DIR) / prefix / rid
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_fingerprint(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    h = hashlib.sha256()
    n = 0
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            n = sum(1 for line in f if line.strip())
    return {"path": str(p), "exists": True, "bytes": p.stat().st_size, "sha256": h.hexdigest(), "rows": n}


def qid_for(row: dict) -> str:
    q = row.get("query") or row.get("qid") or ""
    return hashlib.sha1(q.encode("utf-8")).hexdigest()[:12]


def mean(xs) -> float:
    vals = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if not math.isnan(v):
            vals.append(v)
    return sum(vals) / len(vals) if vals else 0.0


def sd(xs) -> float:
    vals = [float(x) for x in xs if x is not None]
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def acc_tier(label: str | None) -> int:
    return ACC_RANK.get(label or "incorrect", 0)


def judge_metric(j: dict, key: str, default=None):
    if not isinstance(j, dict):
        return default
    return j.get(key, default)


def row_agg(row: dict) -> dict:
    return row.get("agg") or row.get("judge") or {}


def trace_counts(rows: list[dict], judge_key: str = "agg") -> Counter:
    c = Counter()
    for r in rows:
        j = r.get(judge_key) or r.get("judge") or {}
        for t in j.get("rag_traces") or []:
            c[t] += 1
    return c


def bootstrap_ci(vals: list[float], n_boot: int = 2000, seed: int = 13, alpha: float = 0.05) -> tuple[float, float]:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return 0.0, 0.0
    rng = random.Random(seed)
    boots = []
    for _ in range(max(100, n_boot)):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        boots.append(sum(s) / len(s))
    boots.sort()
    lo = boots[int((alpha / 2) * (len(boots) - 1))]
    hi = boots[int((1 - alpha / 2) * (len(boots) - 1))]
    return lo, hi


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xbar = mean([x for x, _ in pairs])
    ybar = mean([y for _, y in pairs])
    num = sum((x - xbar) * (y - ybar) for x, y in pairs)
    denx = math.sqrt(sum((x - xbar) ** 2 for x, _ in pairs))
    deny = math.sqrt(sum((y - ybar) ** 2 for _, y in pairs))
    return num / (denx * deny) if denx and deny else None


def mcnemar_counts(base_rows: list[dict], new_rows: list[dict]) -> dict:
    b = c = 0
    for br, nr in zip(base_rows, new_rows):
        ba = acc_tier(row_agg(br).get("accuracy")) == 2
        na = acc_tier(row_agg(nr).get("accuracy")) == 2
        if ba and not na:
            b += 1
        elif (not ba) and na:
            c += 1
    stat = ((abs(b - c) - 1) ** 2 / (b + c)) if (b + c) else 0.0
    return {"base_correct_new_not": b, "new_correct_base_not": c, "mcnemar_chi2_cc": stat}


def compact_len(text: str) -> int:
    return len("".join((text or "").split()))


TRACE_SURGERY_PATTERNS = [
    r"根据(?:参考资料|上述资料|资料|检索结果|现有(?:的)?回答)[，,：:]?",
    r"从现有的回答来看[，,]?",
    r"现有的回答(?:显示|提到|中)[，,]?",
    r"参考(?:资料|文件|问答对|内容)[显示表明指出中里]*[，,：:]?",
    r"资料(?:显示|表明|指出|中|里)[，,：:]?",
    r"检索(?:结果|到的?内容)[显示表明指出]*[，,：:]?",
    r"这里提供(?:了)?",
    r"图片链接",
    r"文件链接",
    r"<img[^>]*>",
    r"https?://\S+",
]


def trace_surgery(text: str) -> str:
    out = text or ""
    for pat in TRACE_SURGERY_PATTERNS:
        out = re.sub(pat, "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def infer_text(row: dict) -> str:
    return row.get("gen_text") or row.get("chosen_text") or row.get("text") or ""


def format_markdown_table(headers: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines
