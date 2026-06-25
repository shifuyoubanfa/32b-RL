"""项目级 Kimi 围栏（解耦：计量 + 无损去重缓存 + 预算硬闸）。绝不降低实验设计性能。

三件事，全程旁路、可被任意 step 复用：
1. **计量**：每次 Kimi 调用把返回里的 usage(prompt/completion tokens) 累加到 OUTPUT_DIR/kimi_budget.json，
   按标价折算 ¥（监控页直接读这个文件）。kimi_client.chat 成功后调 record() 上报。
2. **无损去重缓存**：同一个 (reference, think, k>=K_MIN) 的【整段干净分聚合结果】跨阶段/续跑只算一次。
   **不碰 k 次独立打分本身**（降噪照旧）；只是同一估计量不重复估 —— V1 原 think 这个 σ 脏锚点在
   冷启动/RFT/DPO 里被当对照重复评 ~7 次，缓存后只评 1 次，统计上完全等价（σ 判据用均值+标定表 σ，不用经验 sd）。
   只缓存 k>=KIMI_CACHE_MIN_K(默认16) 的选样分（k=2 粗筛短命、候选各异、不缓存，免缓存文件膨胀）。
3. **预算硬闸（围栏）**：累计 ¥ 超 KIMI_BUDGET_YUAN 即抛 KimiBudgetExceeded → step 退出码非0 → 编排器干净停，
   由人决定是否加预算续跑（缓存在，续跑不重烧）。这是**安全闸、不是降质**：到顶是停、不是偷工减料。
   KIMI_BUDGET_YUAN=0（默认）= 只计量不设上限。
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR

# ---- 配置（全部环境变量可调；价格=DashScope kimi-k2.6 标价）----
PRICE_IN = float(os.environ.get("KIMI_PRICE_IN_PER_M", "6.5"))    # ¥/百万输入 token
PRICE_OUT = float(os.environ.get("KIMI_PRICE_OUT_PER_M", "27"))   # ¥/百万输出 token
BUDGET_YUAN = float(os.environ.get("KIMI_BUDGET_YUAN", "0"))      # 0=只计量不硬闸
WARN_FRAC = float(os.environ.get("KIMI_BUDGET_WARN", "0.8"))
CACHE_ENABLED = os.environ.get("KIMI_CACHE_ENABLED", "1") == "1"
CACHE_MIN_K = int(os.environ.get("KIMI_CACHE_MIN_K", "16"))
CACHE_VERSION = os.environ.get("KIMI_CACHE_VERSION", "jcal_v1")   # 打分提示词若变，bump 此值或删缓存文件

METER_FILE = Path(OUTPUT_DIR) / "kimi_budget.json"
CACHE_FILE = Path(OUTPUT_DIR) / "kimi_score_cache.jsonl"

_lock = threading.Lock()
_meter = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "yuan": 0.0, "cache_hits": 0, "by_tag": {}}
_cache: dict[str, dict] = {}
_warned = False
_loaded = False


class KimiBudgetExceeded(BaseException):
    """累计花费超过 KIMI_BUDGET_YUAN 围栏。

    刻意继承 BaseException 而非 Exception：打分/改写路径里到处是 `except Exception: pass`
    （v2_common.score_think_kimi、step_v2_coldstart 等容错重试），若继承 Exception 会被它们静默吞掉、
    围栏形同虚设。BaseException 不被 `except Exception` 捕获 → 经 map_concurrent 的 fut.result() 干净上抛
    → step 子进程非0退出 → 编排器干净停。全仓已核 pipeline 下无 `except BaseException`/裸 `except`。
    """


def _yuan(in_tok: int, out_tok: int) -> float:
    return in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT


def _load() -> None:
    """进程启动时把上一步留下的累计计量 + 缓存读进来（step 顺序跑 → 跨步累计正确）。"""
    global _loaded
    if _loaded:
        return
    try:
        if METER_FILE.exists():
            saved = json.loads(METER_FILE.read_text(encoding="utf-8"))
            for k in ("calls", "in_tokens", "out_tokens", "cache_hits"):
                _meter[k] = int(saved.get(k, 0))
            _meter["yuan"] = float(saved.get("yuan", 0.0))
            _meter["by_tag"] = dict(saved.get("by_tag", {}))
    except Exception:
        pass
    if CACHE_ENABLED and CACHE_FILE.exists():
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            for line in f:                              # 逐行容错：一个坏行(kill 中途写的半行)只跳它，不丢后面全部缓存
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    _cache[e["key"]] = e["val"]
                except Exception:
                    continue
    _loaded = True


def _flush_meter() -> None:
    tmp = METER_FILE.with_suffix(".json.tmp")
    try:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(_meter, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(METER_FILE)
    except Exception:
        pass   # 计量落盘失败绝不影响真实调用


def record(usage: dict | None, *, tag: str = "") -> None:
    """kimi_client.chat 成功后上报一次调用的 token 用量；超预算抛 KimiBudgetExceeded（围栏）。"""
    global _warned
    should_warn = over = False
    with _lock:
        _load()
        it = int((usage or {}).get("prompt_tokens", 0) or 0)
        ot = int((usage or {}).get("completion_tokens", 0) or 0)
        _meter["calls"] += 1
        _meter["in_tokens"] += it
        _meter["out_tokens"] += ot
        _meter["yuan"] = round(_meter["yuan"] + _yuan(it, ot), 4)
        if tag:
            t = _meter["by_tag"].setdefault(tag, {"calls": 0, "yuan": 0.0})
            t["calls"] += 1
            t["yuan"] = round(t["yuan"] + _yuan(it, ot), 4)
        if _meter["calls"] % 20 == 0 or _meter["calls"] < 5:
            _flush_meter()                       # 周期落盘，监控页有近实时数
        cur = _meter["yuan"]
        if BUDGET_YUAN > 0:                       # _warned/over 判定在锁内（避免锁外 check-then-set 重复告警）
            if not _warned and cur >= WARN_FRAC * BUDGET_YUAN:
                _warned = should_warn = True
            if cur >= BUDGET_YUAN:
                over = True
                _flush_meter()                   # 抛前确保最新累计已落盘（续跑基线不丢、不重烧）
    if should_warn:
        print(f"[kimi-budget] WARN 已花 ¥{cur:.1f} / 围栏 ¥{BUDGET_YUAN:.0f}（{cur/BUDGET_YUAN:.0%}）", flush=True)
    if over:
        raise KimiBudgetExceeded(
            f"Kimi 累计 ¥{cur:.1f} 已达围栏 ¥{BUDGET_YUAN:.0f}。加大 KIMI_BUDGET_YUAN 后续跑"
            f"（缓存在，不重烧）；或设 KIMI_BUDGET_YUAN=0 关闸。")


# ---- 无损去重缓存（只缓存 k>=CACHE_MIN_K 的选样聚合分）----

def _key(reference: str, think: str, k: int) -> str:
    h = hashlib.sha1()
    h.update(CACHE_VERSION.encode("utf-8")); h.update(b"\x00")
    h.update((reference or "").encode("utf-8")); h.update(b"\x00")
    h.update((think or "").encode("utf-8")); h.update(b"\x00")
    h.update(str(k).encode("utf-8"))
    return h.hexdigest()


def cache_get(reference: str, think: str, k: int) -> dict | None:
    if not (CACHE_ENABLED and k >= CACHE_MIN_K):
        return None
    with _lock:
        _load()
        hit = _cache.get(_key(reference, think, k))
        if hit is not None:
            _meter["cache_hits"] += 1
            if _meter["cache_hits"] % 50 == 0:
                _flush_meter()               # 纯命中续跑也让监控页刷新 cache_hits（命中不经 record）
            return dict(hit)
        return None


def cache_put(reference: str, think: str, k: int, result: dict) -> None:
    if not (CACHE_ENABLED and k >= CACHE_MIN_K):
        return
    if not result or result.get("n", 0) < k or result.get("clean_score") is None:
        return   # 只缓存【打满 k 遍】的聚合分；n<k（部分 429 失败的退化均值）当次用、不进缓存——
                 # 否则 √n 收噪不足却被当 √k 锚永久缓存，confident_cleaner 用 k=16 档 σ 判会偏松、毒化下游选样
    key = _key(reference, think, k)
    with _lock:
        if key in _cache:
            return
        _cache[key] = dict(result)
        try:
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            with CACHE_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "val": result}, ensure_ascii=False) + "\n")
        except Exception:
            pass


def snapshot() -> dict:
    """监控用：当前累计 {calls,in_tokens,out_tokens,yuan,cache_hits,budget,pct}。"""
    with _lock:
        _load()
        s = dict(_meter)
    s["budget_yuan"] = BUDGET_YUAN
    s["pct"] = (s["yuan"] / BUDGET_YUAN) if BUDGET_YUAN > 0 else 0.0
    return s


def flush() -> None:
    with _lock:
        _flush_meter()


# 进程退出兜底落盘：每个 step 是独立子进程，正常退出无人调 flush，否则丢尾账（≤19 次）。
# map_concurrent 用 with-ThreadPoolExecutor，返回前已 join 全线程 → atexit 时无线程持锁、不死锁。
atexit.register(flush)
