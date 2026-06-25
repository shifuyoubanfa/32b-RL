"""derag_v4 preflight and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CKPT_DIR, LOG_DIR, OUTPUT_DIR, SFT_EVAL, SFT_TRAIN, V1_DIR  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402

log = get_logger("step120_v4_precheck")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--rft_merged", default="/home/nvme01/zhjg/models/v1-32b-corrected-v1-rft-merged")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or Path(OUTPUT_DIR) / "derag_v4" / args.run_id)
    ckpt_dir = Path(CKPT_DIR) / "derag_v4"
    log_dir = Path(LOG_DIR) / "derag_v4" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    required = [Path(V1_DIR) / "config.json", Path(SFT_TRAIN), Path(SFT_EVAL)]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing required inputs:\n" + "\n".join(missing))
    if not (Path(args.rft_merged) / "config.json").exists():
        raise SystemExit(f"missing RFT merged base: {args.rft_merged}")
    free_gib = shutil.disk_usage("/home/nvme01" if Path("/home/nvme01").exists() else ".").free / 1024**3
    if free_gib < float(os.environ.get("DERAG_V4_MIN_FREE_GIB", "200")):
        raise SystemExit(f"free disk too low: {free_gib:.1f} GiB")

    manifest = {
        "run_id": args.run_id,
        "namespace": "derag_v4",
        "out_dir": str(out_dir),
        "ckpt_dir": str(ckpt_dir),
        "log_dir": str(log_dir),
        "rft_merged": args.rft_merged,
        "judge_version": "judge_v4.1_bin",
        "stage1_gate_policy": "binary_votes_with_explicit_deterministic_fallback",
        "trace_re_version": "trace_re_v4_gate_judge_v4.0",
        "frozen_trace_re_version": "trace_re_v3_frozen_20260612",
        "free_gib": round(free_gib, 1),
        "inputs": {
            str(Path(SFT_TRAIN)): {"sha256": sha256(Path(SFT_TRAIN)), "lines": count_lines(Path(SFT_TRAIN))},
            str(Path(SFT_EVAL)): {"sha256": sha256(Path(SFT_EVAL)), "lines": count_lines(Path(SFT_EVAL))},
        },
    }
    path = out_dir / "120_v4_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.warning("RESULT precheck PASS run_id=%s out=%s free=%.1fGiB", args.run_id, out_dir, free_gib)
    print(json.dumps({"status": "PASS", "run_id": args.run_id, "manifest": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
