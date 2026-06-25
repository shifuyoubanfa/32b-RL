"""统一日志：INFO 及以上同时写到 logs/pipeline.log 和 stdout。

- 每个进程调用 ``get_logger(name)`` 即可拿到同一份配置；
- 日志文件追加写，按 20MB 滚动，保留 5 份历史；
- 控制台与文件格式一致，便于 grep。
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from config import LOG_FILE


_LEVEL = logging.INFO
_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(_LEVEL)

    # 清掉外部库可能装上的默认 handler，避免重复
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    fh = RotatingFileHandler(
        LOG_FILE, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(_LEVEL)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(_LEVEL)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # 调低部分第三方库噪声
    for noisy in ("urllib3", "requests", "asyncio", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str = "rl") -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
