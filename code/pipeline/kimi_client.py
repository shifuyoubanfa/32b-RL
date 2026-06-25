"""Kimi 裁判/改写客户端 —— 公网 Aliyun DashScope OpenAI 兼容模式。

红线（见技术方案 §4.6）：
- key 不落明文，从环境变量 DASHSCOPE_API_KEY 读，缺失即报错；
- 判分确定性显式锁定（temperature/top_p），不依赖任何模型名分支；
- 同一套 judge prompt / 同模型族，保证迁移前后口径可比。
"""

import os
import random
import time

import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    KIMI_BASE_URL, KIMI_MODEL, KIMI_API_KEY_ENV, KIMI_ENABLE_THINKING,
    JUDGE_TEMPERATURE, JUDGE_TOP_P, JUDGE_TIMEOUT,
)
from pipeline import kimi_budget   # 项目围栏：计量 + 预算硬闸（解耦旁路）
from pipeline.logger import get_logger

log = get_logger("kimi_client")

_session = requests.Session()


def _reset_session() -> None:
    """换新连接池，丢掉上次失败可能残留的坏连接（keep-alive 死链）。

    针对实测症状：DashScope 抽风后，另起的新进程 2.3s 就能打分，训练进程却一直 ReadTimeout——
    旧 session 的 keep-alive socket 在抖动里坏了。重试前换新 session 让下一次拨号是干净连接，
    DashScope 一恢复就能立刻接上、不再卡死。判分是网络瓶颈，复用连接省不了多少，重建代价可忽略。
    """
    global _session
    try:
        _session.close()
    except Exception:
        pass
    _session = requests.Session()


def _api_key() -> str:
    key = os.environ.get(KIMI_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"缺少 {KIMI_API_KEY_ENV} 环境变量（DashScope key）。请先 `export {KIMI_API_KEY_ENV}=sk-...`"
        )
    return key


def chat(messages: list[dict], *, model: str = None, temperature: float = None,
         top_p: float = None, max_tokens: int = 4096, timeout: int = None,
         retries: int = None) -> str:
    """标准 OpenAI 兼容 chat completions，返回 content 字符串（失败重试后抛出）。

    DashScope Kimi 易 429「EngineOverloadedError」(引擎过载)：对 429/5xx/超时做【更耐心】的退避重试
    （封顶 30s + 抖动，最多 8 次），配合低并发(JUDGE_CALL_WORKERS=3)避免被限流；4xx(非429)立即抛。
    retries 默认从环境变量 KIMI_RETRIES 读(默认 8)：在线 GRPO 想抗 DashScope 长抽风可调小(如 3)，
    配合小 JUDGE_TIMEOUT + GRPO_V2_KIMI_REQUIRED=0，让单次失败快速回退而不是把整机冻住。
    """
    if retries is None:
        retries = int(os.environ.get("KIMI_RETRIES", "8"))
    url = KIMI_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": model or KIMI_MODEL,
        "messages": messages,
        "temperature": JUDGE_TEMPERATURE if temperature is None else temperature,
        "top_p": JUDGE_TOP_P if top_p is None else top_p,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": KIMI_ENABLE_THINKING,   # kimi-k2.6 关思考→content 直接出答案（见 config 说明）
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + _api_key()}
    last_err = None
    content = usage = None
    for attempt in range(retries):
        if attempt > 0:
            _reset_session()   # 上次失败可能留下坏连接，重试前换新 session，保证 DashScope 恢复后能立刻接上
        try:
            r = _session.post(url, json=payload, headers=headers,
                              timeout=timeout or JUDGE_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage")
            break                              # 成功 → 跳出，计量在循环外做
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", 0)
            body = getattr(e.response, "text", "")
            # 4xx(除 429) 是永久性错误(模型名错/鉴权/参数)，立即抛出不空等重试
            if 400 <= code < 500 and code != 429:
                raise RuntimeError(f"Kimi 永久性错误 {code}（模型名/鉴权/参数？）: {body[:300]}")
            last_err = e
            log.warning("Kimi 第 %d 次失败(%s): %s", attempt + 1, code, body[:200])
            time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 3))
        except Exception as e:
            last_err = e
            log.warning("Kimi 第 %d 次失败: %r", attempt + 1, e)
            time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 3))
    else:
        raise RuntimeError(f"Kimi 调用失败（{retries} 次）: {last_err}")
    # 计量 + 预算围栏放在 try 之外：KimiBudgetExceeded 须直接上抛，不能被上面的 except 当失败重试吞掉。
    kimi_budget.record(usage)
    return content


def smoke() -> str:
    """最小连通冒烟：返回模型一句话回复。"""
    return chat([{"role": "user", "content": "回复两个字：在么"}], max_tokens=16)


if __name__ == "__main__":
    print("DashScope Kimi 冒烟:", smoke())
