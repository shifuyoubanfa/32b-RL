"""swift GRPO 自定义 reward 插件 —— 把我们的 reward.py 接进 swift GRPO。

swift 用法（见 grpo.sh）：
    swift rlhf --rlhf_type grpo --external_plugins swift/grpo_reward_plugin.py --reward_funcs humanness ...

reward 函数对每个 completion 返回一个标量：复用 reward.score_rollout（两级门控 + humanness 表面项）。
- gold_answer / user_prompt 作为数据列随 batch 传入，从 kwargs 读（见 step13 构数据）。
- 在线 reward 用【表面项】（门控 + 照抄检测）；answer 段重 KL 在 swift KL 里兜底守准。
  PMI 已在 RFT/DPO 离线接入；在线 PMI 需独立尺子服务，作增量（见下方 GRPO_ONLINE_PMI 钩子）。

⚠️ swift 4.0.1 的 ORM 基类与 kwargs 列名以 `swift/plugin` 实际实现为准，上线前 smoke 确认。
"""

import contextlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ 根，便于 import config / pipeline
from config import REWARD_TAU_ACC, THINK_MIN_CHARS, THINK_MAX_CHARS
from pipeline import kimi_budget
from pipeline import reward as R
from pipeline import reward_v3 as R3
from pipeline.judgecal_common import judge_clean_score
from pipeline.rules_v6 import answer_in_v1_pool, detect_rag_style

# swift 的 ORM 注册接口（版本路径可能不同：swift.plugin / swift.rewards 都试）。
# 只 catch ModuleNotFoundError——绝不用裸 except：否则"装了 swift 但导入路径不对"会被静默吞成空注册，
# humanness 永远注册不进 swift 真正的表，swift 解析 --reward_funcs humanness 时才报错(难排查)。
_SWIFT_OK = True
try:
    from swift.plugin import ORM, orms            # ms-swift 常见路径
except ModuleNotFoundError:
    try:
        from swift.rewards import ORM, orms        # 部分版本重构到这
    except ModuleNotFoundError:
        _SWIFT_OK = False
        print("[grpo_reward_plugin] WARNING: swift 的 ORM/orms 未导入(swift.plugin 与 swift.rewards 都没)——"
              "humanness 未注册，仅本地单测可用；在 zhjg_rl 训练环境里不该走到这。", file=sys.stderr)
        class ORM:  # type: ignore
            pass
        orms = {}

GRPO_ONLINE_PMI = os.environ.get("GRPO_ONLINE_PMI", "0") == "1"  # 默认关（需独立尺子服务才开）
V2_KIMI_K = int(os.environ.get("GRPO_V2_KIMI_K", os.environ.get("GRPO_KIMI_K", "2")))
V2_USE_KIMI = os.environ.get("GRPO_V2_USE_KIMI", "1") == "1"
V2_KIMI_REQUIRED = os.environ.get("GRPO_V2_KIMI_REQUIRED", "1") == "1"
V2_THINK_MIN = int(os.environ.get("GRPO_V2_THINK_MIN", "40"))
V2_THINK_MAX = int(os.environ.get("GRPO_V2_THINK_MAX", "2200"))
V2_STRICT_TAGS = os.environ.get("GRPO_V2_STRICT_TAGS", "1") == "1"
V2_KIMI_LOCK = os.environ.get("GRPO_V2_KIMI_LOCK", "1") == "1"
V2_KIMI_MIN_INTERVAL = float(os.environ.get("GRPO_V2_KIMI_MIN_INTERVAL", "0.0"))
_LOCK_DEFAULT = Path(os.environ.get("ZHJG_LOG_DIR", "/home/nvme01/zhjg/logs")) / "v2_grpo_online" / "kimi_online.lock"
V2_KIMI_LOCK_PATH = Path(os.environ.get("GRPO_V2_KIMI_LOCK_PATH", str(_LOCK_DEFAULT)))
_V2_ONLINE_CALLS = 0


def _extract(completion):
    """swift 不同版本 completion 可能是 str 或 [{'role','content'}]。统一取文本。"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _expand_column(vals, n: int, name: str):
    """Align a swift dataset column with completions.

    ms-swift GRPO commonly gives completions length = prompts × K while extra
    columns keep prompt length.  The old plugin already relied on prompt-major
    ordering; V2 keeps that contract and fails loudly if it changes.
    """
    if vals is None:
        raise KeyError(f"GRPO 数据缺 {name} 列，拒绝继续。")
    if not isinstance(vals, list):
        vals = [vals]
    if len(vals) == n:
        return vals
    if len(vals) and n % len(vals) == 0:
        k = n // len(vals)
        return [v for v in vals for _ in range(k)]
    raise ValueError(f"completions({n}) 与 {name}({len(vals)}) 不匹配且不整除，"
                     "请 smoke 确认 swift 列传入方式后调整。")


def _parse_answers(value) -> list[str]:
    """Read V1 answer pool from a JSON string/list column."""
    if isinstance(value, list):
        return [str(x) for x in value if str(x or "").strip()]
    if isinstance(value, tuple):
        return [str(x) for x in value if str(x or "").strip()]
    s = str(value or "").strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x or "").strip()]
    except Exception:
        pass
    return [s]


def _norm_kimi(score: float | None) -> float:
    if score is None:
        return 0.0
    return max(0.0, min(1.0, float(score) / 10.0))


@contextlib.contextmanager
def _online_kimi_guard():
    """Serialize online Kimi calls across torch ranks.

    ms-swift reward plugins may be imported by several distributed workers.
    A tiny file lock keeps DashScope traffic predictable.  It is intentionally
    scoped only to GRPO online reward, so offline eval can keep using its own
    worker pool.
    """
    if not V2_KIMI_LOCK:
        yield
        return
    try:
        import fcntl  # Linux training host; unavailable on Windows unit tests.
    except Exception:
        yield
        return

    V2_KIMI_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = V2_KIMI_LOCK_PATH.with_suffix(".stamp")
    with V2_KIMI_LOCK_PATH.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            if V2_KIMI_MIN_INTERVAL > 0 and stamp.exists():
                try:
                    last = float(stamp.read_text(encoding="utf-8").strip() or "0")
                    wait = V2_KIMI_MIN_INTERVAL - (time.time() - last)
                    if wait > 0:
                        time.sleep(wait)
                except Exception:
                    pass
            yield
            try:
                stamp.write_text(str(time.time()), encoding="utf-8")
            except Exception:
                pass
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _kimi_clean_score(reference: str, think: str, k: int = None) -> dict:
    """Kimi clean score for online GRPO.

    Online GRPO uses k=2 by default.  This is intentionally not a 2σ/3σ
    selector; it is only a within-group ranking signal.
    """
    kk = max(1, int(k or V2_KIMI_K))
    cached = kimi_budget.cache_get(reference, think, kk)
    if cached is not None:
        return cached
    vals = []
    last_err = None
    for _ in range(kk):
        try:
            with _online_kimi_guard():
                vals.append(float(judge_clean_score(reference, think)["clean_score"]))
        except Exception as e:
            last_err = e
    if not vals:
        if V2_KIMI_REQUIRED:
            raise RuntimeError(f"V2 online GRPO Kimi 打分全失败（k={kk}）: {last_err!r}")
        return {"clean_score": None, "n": 0, "sd": 0.0}
    result = {
        "clean_score": sum(vals) / len(vals),
        "n": len(vals),
        "sd": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
    }
    kimi_budget.cache_put(reference, think, kk, result)
    return result


def _repeat_penalty(text: str) -> float:
    """Cheap guard against max-token loops / mantra reward hacking."""
    s = re.sub(r"\s+", "", text or "")
    if len(s) < 80:
        return 0.0
    penalty = 0.0
    for n, threshold, weight in ((8, 5, 0.35), (12, 4, 0.35), (20, 3, 0.25)):
        counts = {}
        for i in range(0, max(0, len(s) - n + 1)):
            g = s[i:i + n]
            counts[g] = counts.get(g, 0) + 1
        if counts and max(counts.values()) >= threshold:
            penalty += weight
    return min(0.8, penalty)


def _length_penalty(think: str) -> float:
    n = len((think or "").strip())
    if n <= V2_THINK_MAX:
        return 0.0
    return min(0.6, (n - V2_THINK_MAX) / 1000.0 * 0.3)


def _v2_reward_one(text: str, user_prompt: str, v1_answers: list[str], *, use_kimi: bool) -> float:
    """V2 GRPO reward: answer/format are hard gates; Kimi only steers think."""
    if V2_STRICT_TAGS:
        open_think = text.find("<think>")
        close_think = text.find("</think>")
        open_answer = text.find("<answer>", close_think + len("</think>") if close_think != -1 else 0)
        close_answer = text.rfind("</answer>")
        if not (0 <= open_think < close_think < open_answer < close_answer):
            return -4.0
    parsed = R.parse_think_answer_diagnostic(text)
    think, answer = parsed["think"], parsed["answer"]
    if (not parsed["format_ok"]) or len(think.strip()) < V2_THINK_MIN:
        return -4.0
    if not v1_answers:
        raise ValueError("V2 GRPO reward 缺 V1 answer pool；不能在线训练。")

    ad = answer_in_v1_pool(answer, v1_answers)
    if not ad["in_pool"]:
        return -3.0
    if not ad.get("comparable", True):
        # Historical eval keeps non-empty/no-slot answers as in_pool for metric
        # comparability.  Online RL cannot: otherwise the model can maximize
        # reward by saying less and avoiding concrete taxable/amount/date facts.
        return -2.0

    rule = detect_rag_style(think)
    kimi = None
    call_kimi_on_rule_fail = os.environ.get("GRPO_V2_KIMI_ON_RULE_FAIL", "0") == "1"
    if use_kimi and V2_USE_KIMI and ((not rule["has_rag_style"]) or call_kimi_on_rule_fail):
        refs = R.extract_references(user_prompt)
        kimi = _kimi_clean_score(refs, think, k=V2_KIMI_K)["clean_score"]
    kn = _norm_kimi(kimi)
    penalty = _repeat_penalty(think) + _length_penalty(think)

    if rule["has_rag_style"]:
        # 规则 think 未过时，不能让高 Kimi 分反超规则通过样本。
        return round(0.50 + 0.20 * kn - penalty, 4)
    return round(1.00 + 0.40 * kn - penalty, 4)


def _v2_call(completions, kwargs, *, use_kimi: bool):
    n = len(completions)
    ups = _expand_column(kwargs.get("user_prompt"), n, "user_prompt")
    pools_raw = kwargs.get("v1_answers_json")
    if pools_raw is None:
        pools_raw = kwargs.get("v1_answers")
    pools = _expand_column(pools_raw, n, "v1_answers_json")
    rewards = []
    for comp, up, pool in zip(completions, ups, pools):
        rewards.append(float(_v2_reward_one(_extract(comp), str(up or ""), _parse_answers(pool), use_kimi=use_kimi)))
    return rewards


class HumannessReward(ORM):
    """组内每个候选返回 reward（两级门控 + 表面 humanness）。"""

    def __call__(self, completions, **kwargs):
        n = len(completions)
        golds = kwargs.get("gold_answer")
        ups = kwargs.get("user_prompt")
        # 缺列即报错——绝不静默兜空串（空 gold 会让 answer_drift 返回 1.0，等于关掉守准门）。
        if golds is None or ups is None:
            raise KeyError(f"GRPO 数据缺 gold_answer/user_prompt 列，拒绝继续。现有列: {list(kwargs.keys())}")
        # ms-swift 每 prompt 采 num_generations=K 个：completions 长=prompt×K，而列长=prompt 数。
        # 不等且整除→按 K 展开对齐；否则报错（不静默截断）。上线 smoke 时打印两者长度确认。
        if len(golds) != n:
            if len(golds) and n % len(golds) == 0:
                k = n // len(golds)
                golds = [g for g in golds for _ in range(k)]
                ups = [u for u in ups for _ in range(k)]
            else:
                raise ValueError(f"completions({n}) 与 gold_answer({len(golds)}) 不匹配且不整除，"
                                 "请 smoke 确认 swift 列传入方式后调整对齐。")
        rewards = []
        for comp, gold, up in zip(completions, golds, ups):
            text = _extract(comp)
            sc = R.score_rollout(text, up, gold, tau_acc=REWARD_TAU_ACC,
                                 think_min=THINK_MIN_CHARS, think_max=THINK_MAX_CHARS, s_pmi=None)
            rewards.append(float(sc["reward"]))
        return rewards


class DeragV4Reward(ORM):
    """derag_v4 deterministic reward: trace removal with fact/answer guards."""

    def __call__(self, completions, **kwargs):
        n = len(completions)
        golds = kwargs.get("gold_answer")
        ups = kwargs.get("user_prompt")
        queries = kwargs.get("query") or [""] * (len(golds) if golds is not None else n)
        if golds is None or ups is None:
            raise KeyError(f"GRPO 数据缺 gold_answer/user_prompt 列，拒绝继续。现有列: {list(kwargs.keys())}")
        if len(golds) != n:
            if len(golds) and n % len(golds) == 0:
                k = n // len(golds)
                golds = [g for g in golds for _ in range(k)]
                ups = [u for u in ups for _ in range(k)]
                queries = [q for q in queries for _ in range(k)]
            else:
                raise ValueError(f"completions({n}) 与 gold_answer({len(golds)}) 不匹配且不整除")
        rewards = []
        for comp, gold, up, q in zip(completions, golds, ups, queries):
            val, _ = R3.derag_reward(_extract(comp), up, gold, q)
            rewards.append(float(val))
        return rewards


class V2RuleWarmupReward(ORM):
    """Short warmup reward: only format/answer/rule, no online Kimi calls.

    This is deliberately short-lived.  It stabilizes format and the answer pool
    before the expensive Kimi-online stage, but should not be run long enough to
    Goodhart the surface rules.
    """

    def __call__(self, completions, **kwargs):
        return _v2_call(completions, kwargs, use_kimi=False)


class V2OnlineReward(ORM):
    """V2 online GRPO reward.

    Lexicographic contract:
      format fail < answer-out-of-pool < rule-think fail < rule-think pass,
    and Kimi clean score (k=2 by default) only ranks candidates inside the safe
    region.  A pretty but answer-drifting think can never compensate the answer
    hard gate.
    """

    def __call__(self, completions, **kwargs):
        global _V2_ONLINE_CALLS
        _V2_ONLINE_CALLS += 1
        warmup_calls = int(os.environ.get("GRPO_RULE_WARMUP_CALLS", "0"))
        force_rule = os.environ.get("GRPO_RULE_ONLY", "0") == "1"
        use_kimi = (not force_rule) and (_V2_ONLINE_CALLS > warmup_calls)
        return _v2_call(completions, kwargs, use_kimi=use_kimi)


# 注册名 = grpo.sh 里 --reward_funcs 用的名字
orms["humanness"] = HumannessReward
orms["derag_v4"] = DeragV4Reward
orms["v2_rule_warmup"] = V2RuleWarmupReward
orms["v2_online"] = V2OnlineReward
orms["v2_online_kimi"] = V2OnlineReward
orms["v2_hard_rules"] = V2RuleWarmupReward
# 装了 swift 却没注册进去 = 致命（swift 会报 reward func 'humanness' not found）→ 启动即响、别静默。
assert (not _SWIFT_OK) or orms.get("humanness") is HumannessReward, \
    "humanness 未注册进 swift 的 orms——导入路径可能不对，核对该 swift 版本的 ORM 注册接口（smoke 时务必确认）"
