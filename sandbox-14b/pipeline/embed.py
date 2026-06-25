"""嵌入客户端：对接公司内网 bge-m3 API（不本地加载模型，省显存、避版本冲突）。

用于 Phase 2 奖励的语义信号(ΔRAG / 语义照抄)。OpenAI 兼容 embeddings 端口。
带：绕系统代理、批量、按文本 hash 缓存、重试、L2 归一化。

自测：python pipeline/embed.py   → 打印两句话的向量维度和余弦相似度。
"""

import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import EMBED_URL, EMBED_MODEL, EMBED_TIMEOUT, EMBED_MAX_RETRIES
from pipeline.logger import get_logger

log = get_logger("embed")

# 公司内网，绕系统代理（配置里 use_env_proxy:false）
_session = requests.Session()
_session.trust_env = False
_session.proxies = {"http": "", "https": ""}

_cache: dict[str, np.ndarray] = {}   # text -> normalized vec


def _post_embed(texts: list[str]) -> list[list[float]]:
    """单批调用，返回 embedding 列表。多路径解析响应(兼容不同壳子)。"""
    payload = {"model": EMBED_MODEL, "input": texts}
    last_err = None
    for attempt in range(EMBED_MAX_RETRIES):
        r = None
        try:
            r = _session.post(EMBED_URL, json=payload, timeout=EMBED_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            # 标准 OpenAI: {"data":[{"embedding":[...],"index":0}, ...]}
            items = data.get("data") if isinstance(data, dict) else None
            if items is None and isinstance(data, dict):
                items = (data.get("result") or {}).get("data") if isinstance(data.get("result"), dict) else None
            if not items:
                raise ValueError(f"响应无 data 字段，顶层键={list(data.keys()) if isinstance(data,dict) else type(data)}")
            items = sorted(items, key=lambda x: x.get("index", 0))
            return [it["embedding"] for it in items]
        except Exception as e:
            last_err = e
            body = (r.text[:300] if r is not None else "")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"embed API 调用失败: {last_err} | body[:300]={body!r}")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-8, None)


def encode(texts: list[str], batch: int = 32) -> np.ndarray:
    """批量编码 + 缓存 + L2 归一化。返回 [N, dim] float32。"""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    need = [t for t in texts if t not in _cache]
    uniq = list(dict.fromkeys(need))   # 去重保序
    for i in range(0, len(uniq), batch):
        chunk = uniq[i:i + batch]
        embs = _post_embed(chunk)
        for t, e in zip(chunk, embs):
            _cache[t] = _normalize(np.asarray(e, dtype=np.float32))
    return np.stack([_cache[t] for t in texts])


def cos_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:[N,d] b:[M,d] 均已归一化 -> [N,M] 余弦。"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T


if __name__ == "__main__":
    log.info("自测 bge-m3 端口：%s 模型=%s", EMBED_URL, EMBED_MODEL)
    vs = encode(["小规模纳税人增值税征收率是3%。", "今天天气不错。"])
    log.info("向量形状=%s（应为 [2, 1024]）", vs.shape)
    sim = float(cos_matrix(vs[:1], vs[1:])[0, 0])
    log.info("两句余弦相似度=%.3f（税务句 vs 天气句，应较低）", sim)
    log.info("端口可用 ✓" if vs.shape[0] == 2 else "端口异常")
