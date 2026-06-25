"""本地 vLLM(V1) 推理客户端 —— OpenAI 兼容，调 localhost。

用于：阶段0 数据构建（V1 重产 think/answer）、RFT/DPO 的 rollout 采样、各阶段评测推理。
vLLM 服务由 scripts/serve_v1_vllm.sh 在 vllm_env 环境起；本进程(zhjg_rl)只发 HTTP。
"""

import concurrent.futures as cf
import time

import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import VLLM_BASE_URL, VLLM_MODEL, VLLM_TIMEOUT, VLLM_CALL_WORKERS, SYSTEM_PROMPT
from pipeline.logger import get_logger

log = get_logger("vllm_client")

_session = requests.Session()


def health() -> bool:
    """vLLM 服务是否就绪（/models 返回 200）。"""
    try:
        r = _session.get(VLLM_BASE_URL.rstrip("/") + "/models", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def wait_ready(max_wait: int = 1800, interval: int = 10) -> None:
    """阻塞等待 vLLM 起好（加载 65G + TP 初始化要 1~3 分钟）。"""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if health():
            log.info("vLLM 服务就绪 (%s)", VLLM_BASE_URL)
            return
        log.info("等待 vLLM 起服务... (%ds)", int(time.time() - t0))
        time.sleep(interval)
    raise RuntimeError(f"vLLM 服务 {max_wait}s 内未就绪: {VLLM_BASE_URL}")


def chat(messages: list[dict], *, model: str = None, n: int = 1, temperature: float = 0.0,
         top_p: float = 1.0, max_tokens: int = 1536, timeout: int = None, retries: int = 4) -> list[str]:
    """OpenAI 兼容 chat completions；n>1 时一次返回 n 个候选（rollout 用）。返回 content 列表。

    model 默认 VLLM_MODEL('v1')；评测各阶段 adapter 时传入该 adapter 在 vLLM 里的名字。
    """
    url = VLLM_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": model or VLLM_MODEL, "messages": messages, "n": n,
        "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "stream": False,
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = _session.post(url, json=payload, timeout=timeout or VLLM_TIMEOUT)
            r.raise_for_status()
            return [c["message"]["content"] for c in r.json()["choices"]]
        except Exception as e:
            last_err = e
            log.warning("vLLM 第 %d 次失败: %r", attempt + 1, e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"vLLM 调用失败（{retries} 次）: {last_err}")


def gen_one(user_prompt: str, *, system: str = None, **kw) -> str:
    """单条：system + user_prompt → V1 一条输出（数据构建/评测用，默认贪心）。"""
    msgs = [{"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}]
    return chat(msgs, n=1, **kw)[0]


def gen_k(user_prompt: str, *, k: int, system: str = None, temperature: float = None,
          top_p: float = None, **kw) -> list[str]:
    """单条 query 采 K 个候选（rollout 用）。默认带 RL 采样温度，避免贪心下 K 个候选雷同。"""
    from config import RL_TEMPERATURE, RL_TOP_P
    msgs = [{"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}]
    return chat(msgs, n=k,
                temperature=RL_TEMPERATURE if temperature is None else temperature,
                top_p=RL_TOP_P if top_p is None else top_p, **kw)


def map_concurrent(items: list, fn, *, workers: int = None, desc: str = "") -> list:
    """并发跑 fn(item)，保持输入顺序返回。vLLM 连续批处理吃得下高并发。"""
    workers = workers or VLLM_CALL_WORKERS
    results = [None] * len(items)
    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, it): i for i, it in enumerate(items)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(items):
                rate = done / max(time.time() - t0, 1e-3)
                log.info("[%s] %d/%d  %.1f it/s", desc, done, len(items), rate)
    return results
