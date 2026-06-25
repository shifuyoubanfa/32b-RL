"""共享的 mudgate Kimi 调用客户端（种子改写 step11 / 种子打分 step12 复用）。

复刻 step06 已验证的逻辑：绕系统代理、kimi 强制约束(temp=1/top_p=0.95/关 think)、多路径解析响应。
不改动 step06（它已跑通），只把可复用部分抽出来。
"""

import json
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_TEMPERATURE, JUDGE_TOP_P, JUDGE_TIMEOUT,
    mudgate_headers,
)

# 公司内网，绕开本地系统代理
_session = requests.Session()
_session.trust_env = False
_session.proxies = {"http": "", "https": ""}


def _build_payload(messages: list[dict]) -> dict:
    payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "temperature": JUDGE_TEMPERATURE,
        "top_p": JUDGE_TOP_P,
        "stream": False,
    }
    if JUDGE_MODEL.lower().startswith("kimi"):
        # Kimi 在 mudgate 上的强制约束（违反即 400）
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["temperature"] = 1
        payload["top_p"] = 0.95
        payload["do_sample"] = False
    return payload


def _extract_response_text(data) -> str:
    if not isinstance(data, dict):
        raise ValueError(f"响应不是 dict: type={type(data).__name__}")
    if data.get("success") is False:
        ctx = data.get("errorContext") or data.get("error") or data
        raise ValueError(f"gateway success=False: {ctx}")
    err = data.get("error") or data.get("err")
    if isinstance(err, dict) or (isinstance(err, str) and ("error" in err.lower() or "fail" in err.lower())):
        raise ValueError(f"gateway returned error body: {err}")

    def _try(d, *keys):
        cur = d
        for k in keys:
            if isinstance(k, int):
                if not isinstance(cur, list) or len(cur) <= k:
                    return None
                cur = cur[k]
            else:
                if not isinstance(cur, dict) or k not in cur:
                    return None
                cur = cur[k]
        return cur if isinstance(cur, str) and cur else None

    candidates = [
        _try(data, "choices", 0, "message", "content"),
        _try(data, "choices", 0, "text"),
        _try(data, "data", "choices", 0, "message", "content"),
        _try(data, "data", "choices", 0, "text"),
        _try(data, "data", "content"),
        _try(data, "data", "text"),
        _try(data, "data", "answer"),
        _try(data, "content"),
        _try(data, "text"),
        _try(data, "answer"),
        _try(data, "result"),
    ]
    for c in candidates:
        if c:
            return c
    raise ValueError(f"无法定位响应文本，顶层键={list(data.keys())[:10]}")


def call_kimi(messages: list[dict], timeout: int = JUDGE_TIMEOUT, retries: int = 3) -> str:
    """调 Kimi，返回文本。失败抛 RuntimeError(带 body 片段便于排查)。"""
    url = JUDGE_BASE_URL.rstrip("/") + "/chat/completions"
    headers = mudgate_headers()
    payload = _build_payload(messages)
    last_err = None
    last_body = None
    for attempt in range(retries):
        r = None
        try:
            r = _session.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return _extract_response_text(r.json())
        except Exception as e:
            last_err = e
            if r is not None:
                try:
                    last_body = (r.text or "")[:400]
                except Exception:
                    last_body = None
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Kimi 调用失败: {last_err} | body[:400]={last_body!r}")


def extract_json(text: str) -> dict:
    """从模型输出里抽第一个 JSON 对象（容忍 ```json 包裹）。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    l = text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1:
        raise ValueError(f"无法定位 JSON: {text[:200]}")
    return json.loads(text[l:r + 1])
