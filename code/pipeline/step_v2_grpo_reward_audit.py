"""Preflight audit for V2 online GRPO reward.

This does not generate model rollouts.  It validates that the ms-swift dataset
contains the columns required by ``swift/grpo_reward_plugin.py`` and that the
V2 reward ordering is still lexical:

    format fail < answer out of V1 pool < rule-think fail < rule-think pass

Optionally it can make one online-Kimi reward call to verify API connectivity
before GPUs are occupied by GRPO training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from pipeline.reward import extract_references  # noqa: E402
from pipeline.rules_v6 import answer_in_v1_pool  # noqa: E402


def _load_reward_plugin():
    """Load our local reward plugin by file path.

    The training environment also has the ms-swift package named ``swift``.
    Importing ``swift.grpo_reward_plugin`` can therefore resolve to the
    installed swift package instead of ``code/swift/grpo_reward_plugin.py``.
    The actual GRPO launcher passes the plugin by path, so the audit should do
    the same thing and avoid changing package precedence.
    """
    plugin_path = CODE_ROOT / "swift" / "grpo_reward_plugin.py"
    spec = importlib.util.spec_from_file_location("zhjg_grpo_reward_plugin", plugin_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reward plugin: {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_v2_reward_one = _load_reward_plugin()._v2_reward_one


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wrap(think: str, answer: str) -> str:
    return f"<think>\n{think.strip()}\n</think>\n\n<answer>\n{answer.strip()}\n</answer>"


def parse_pool(row: dict) -> list[str]:
    raw = row.get("v1_answers_json")
    if raw is None:
        raw = row.get("v1_answers")
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x or "").strip()]
    try:
        obj = json.loads(str(raw or ""))
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x or "").strip()]
    except Exception:
        pass
    return []


def pick_pool_answer(pool: list[str]) -> str | None:
    """Pick a V1 pool answer that passes the online answer hard gate.

    The synthetic ``good``/``rule_bad`` cases must use an answer that
    ``answer_in_v1_pool`` rates as ``in_pool`` *and* ``comparable``; otherwise
    ``_v2_reward_one`` returns -2.0 (no-comparable-facts) for both, collapsing
    ``good`` == ``rule_bad`` and failing the audit on a perfectly healthy pool.
    ``pool[0]`` (the V1 canonical answer) is often a qualitative/procedural
    answer with no extractable polarity/number/date, so blindly using it is
    wrong.  The build filter (``pool_has_trainable_answer``) guarantees at least
    one such answer exists, and iterating in order keeps ``pool[0]`` preferred
    whenever it is itself comparable.
    """
    for ans in pool:
        ad = answer_in_v1_pool(ans, pool)
        if ad.get("in_pool") and ad.get("comparable", True):
            return ans
    return None


def synthetic_cases(answer: str) -> dict[str, str]:
    good_think = (
        "先从题目本身给出的条件出发，找出需要比较的金额、期限和结论方向；"
        "再把这些条件放到同一个判断口径里，确认没有引入新的税率、金额或日期；"
        "最后只根据这些已知条件得出与答案一致的结论。"
    )
    rule_bad_think = good_think + " 这里故意写入参考问答对和资料显示，用于检查规则think硬门。"
    drift_answer = "应缴税额为999999万元，不得免征，需要立即补缴。"
    return {
        "good": wrap(good_think, answer),
        "rule_bad": wrap(rule_bad_think, answer),
        "answer_bad": wrap(good_think, drift_answer),
        "format_bad": "<think>这段输出故意不闭合，也没有answer标签",
    }


def audit_rows(rows: list[dict], *, n: int, use_kimi: bool) -> dict:
    checked = []
    missing = []
    failures = []
    for idx, row in enumerate(rows[: max(0, n)]):
        qid = row.get("qid") or row.get("query") or f"row{idx}"
        user_prompt = row.get("user_prompt") or ""
        pool = parse_pool(row)
        if not user_prompt or not pool:
            missing.append({"idx": idx, "qid": qid, "has_user_prompt": bool(user_prompt), "pool_size": len(pool)})
            continue
        answer = pick_pool_answer(pool)
        if answer is None:
            # Should not happen for build output (the build keeps only pools with
            # a trainable answer), but never silently audit a row we cannot build
            # a valid in-pool good case for.
            missing.append({"idx": idx, "qid": qid, "reason": "no_comparable_pool_answer", "pool_size": len(pool)})
            continue
        cases = synthetic_cases(answer)
        rewards = {
            name: _v2_reward_one(text, user_prompt, pool, use_kimi=False)
            for name, text in cases.items()
        }
        ok = rewards["good"] > rewards["rule_bad"] > rewards["answer_bad"] > rewards["format_bad"]
        ok = ok and rewards["answer_bad"] < 0 and rewards["format_bad"] < rewards["answer_bad"]
        if not ok:
            failures.append({"idx": idx, "qid": qid, "rewards": rewards})
        checked.append({"idx": idx, "qid": qid, "pool_size": len(pool), "rewards": rewards})

    kimi_probe = None
    if use_kimi:
        # The probe must clear the answer hard gate and be rule-clean, otherwise
        # _v2_reward_one short-circuits before any Kimi call and the probe never
        # actually tests API connectivity.  So require a comparable pool answer.
        probe_row = None
        probe_answer = None
        for r in rows:
            if not (r.get("user_prompt") or ""):
                continue
            cand = pick_pool_answer(parse_pool(r))
            if cand is not None:
                probe_row, probe_answer = r, cand
                break
        if probe_row is None:
            failures.append({"error": "no row available for kimi probe"})
        else:
            pool = parse_pool(probe_row)
            text = synthetic_cases(probe_answer)["good"]
            prev_k = os.environ.get("GRPO_V2_KIMI_K")
            os.environ["GRPO_V2_KIMI_K"] = os.environ.get("V2_GRPO_REWARD_AUDIT_KIMI_K", "1")
            try:
                kimi_reward = _v2_reward_one(text, probe_row.get("user_prompt") or "", pool, use_kimi=True)
                kimi_probe = {
                    "qid": probe_row.get("qid") or probe_row.get("query"),
                    "reward": kimi_reward,
                    "reference_chars": len(extract_references(probe_row.get("user_prompt") or "")),
                }
            finally:
                if prev_k is None:
                    os.environ.pop("GRPO_V2_KIMI_K", None)
                else:
                    os.environ["GRPO_V2_KIMI_K"] = prev_k

    return {
        "ok": not missing and not failures and bool(checked),
        "rows_seen": len(rows),
        "rows_checked": len(checked),
        "missing": missing[:20],
        "failures": failures[:20],
        "kimi_probe": kimi_probe,
        "sample": checked[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--kimi", action="store_true", help="make one tiny online-Kimi reward probe")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"missing GRPO data: {data}")
    rows = read_jsonl(data)
    result = audit_rows(rows, n=args.n, use_kimi=args.kimi)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result["ok"]:
        preview = {
            "rows_seen": result.get("rows_seen"),
            "rows_checked": result.get("rows_checked"),
            "missing": result.get("missing", [])[:3],
            "failures": result.get("failures", [])[:3],
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(f"V2 GRPO reward audit failed -> {out}")
    print(f"V2 GRPO reward audit OK: rows_checked={result['rows_checked']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
