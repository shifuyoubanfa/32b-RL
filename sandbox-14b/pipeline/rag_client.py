"""复用 rag_demo.py 中的检索 / prompt 拼装 / 流式对话逻辑，去掉彩色打印。"""

import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RETRIEVE_URL, COMPANY_CHAT_URL, SYSTEM_PROMPT
from pipeline.logger import get_logger

log = get_logger("rag_client")

# 公司接口走内网，绕开系统代理，避免本地 127.0.0.1 代理把请求劫走
_session = requests.Session()
_session.trust_env = False
_session.proxies = {"http": "", "https": ""}


def retrieve(
    query: str,
    brand: str = "119",
    location: str = "1300",
    model_id: str = "kgTtV4Ytrn",
    app_id: str = "sk-REDACTED-ROTATE-ME",
    request_id: str = "0000000000000001",
    timeout: int = 120,
    retries: int = 3,
) -> list[dict]:
    payload = {
        "query": query,
        "dimension": {"brand": brand, "location": location},
        "modelId": model_id,
        "requestId": request_id,
        "appId": app_id,
    }
    last_err = None
    for attempt in range(retries):
        try:
            response = _session.post(RETRIEVE_URL, data=json.dumps(payload), timeout=timeout)
            response.raise_for_status()
            res = response.json()
            if not res.get("success"):
                raise RuntimeError(f"检索接口返回失败: {res}")
            return res.get("value", {}).get("retrieveRes", []) or []
        except Exception as e:
            last_err = e
            log.warning("retrieve 第 %d 次失败 query=%s... err=%r",
                        attempt + 1, query[:30], e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"retrieve 失败 query={query!r}: {last_err}")


def remove_html_except_img(html: str) -> str:
    if not html:
        return html
    html = html.replace("<div>", "<p>").replace("</div>", "</p>")
    cleaned_content = re.sub(r"<br/>", "</p><p>", html)
    html_content = "<p>" + cleaned_content + "</p>"
    soup = BeautifulSoup(html_content, "lxml")
    for img in soup.find_all("img"):
        if img.parent.name != "p":
            new_p = soup.new_tag("p")
            img.wrap(new_p)
    paragraphs = []
    for p in soup.find_all("p"):
        for img in p.find_all("img"):
            src = img.get("src")
            if src:
                img.replace_with(f'<img src="{src}">')
        for br in p.find_all("br"):
            br.replace_with("\n")
        paragraphs.append(p.get_text(strip=True))
    return "\n".join(paragraphs).strip()


def build_user_prompt(query: str, retrieve_items: list[dict], top_k: int | None = None) -> str:
    if top_k is not None:
        retrieve_items = retrieve_items[:top_k]
    parts = []
    for item in retrieve_items:
        title = (item.get("title") or "").strip()
        content_multiline = remove_html_except_img(item.get("content") or "")
        if not content_multiline:
            continue
        content_inline = content_multiline.replace("\n", "")
        parts.append(f"{{'问题1: {title}, 回答: {content_inline}\n{content_multiline}'}}")
    references_block = "[" + ", ".join(parts) + "]"
    return f"【参考问答对】\n{references_block}\n【问题】\n{query}"


def stream_chat(messages: list[dict], timeout: int = 300) -> tuple[str, str]:
    """流式调用公司内部微调模型，返回 (reasoning_content, content)。"""
    response = _session.post(
        COMPANY_CHAT_URL,
        json={"messages": messages, "stream": True},
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()
    reasoning_buf: list[str] = []
    content_buf: list[str] = []
    for raw in response.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
            continue
        data = raw[6:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = chunk["choices"][0].get("delta") or {}
        if delta.get("reasoning_content"):
            reasoning_buf.append(delta["reasoning_content"])
        if delta.get("content"):
            content_buf.append(delta["content"])
    return "".join(reasoning_buf), "".join(content_buf)


def rag_answer(query: str, top_k: int = 5) -> dict:
    """端到端：检索 + 拼 prompt + 调公司模型。返回包含原始片段的完整记录。"""
    retrieve_items = retrieve(query)
    user_prompt = build_user_prompt(query, retrieve_items, top_k=top_k)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    reasoning, content = stream_chat(messages)
    return {
        "query": query,
        "user_prompt": user_prompt,
        "reasoning_content": reasoning,
        "content": content,
        "top_k": top_k,
    }
