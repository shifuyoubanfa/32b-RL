"""统一日志。

默认行为保持旧流水线兼容：同时输出到控制台和 logs/pipeline.log，级别 INFO 起。
新实验可通过环境变量隔离日志：
- ZHJG_LOG_FILE: 覆盖文件日志路径
- ZHJG_FILE_LOG_LEVEL: 文件日志级别，默认 INFO
- ZHJG_CONSOLE_LOG_LEVEL: 控制台日志级别，默认 INFO
"""

import logging
import os
import sys
from pathlib import Path

_CONFIGURED: set[str] = set()


def _level(name: str, default: str) -> int:
    raw = os.environ.get(name, default).upper()
    return getattr(logging, raw, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger
    from config import LOG_FILE

    log_file = os.environ.get("ZHJG_LOG_FILE", LOG_FILE)
    file_level = _level("ZHJG_FILE_LOG_LEVEL", "INFO")
    console_level = _level("ZHJG_CONSOLE_LOG_LEVEL", "INFO")
    logger.setLevel(min(file_level, console_level))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(file_level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(console_level)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    _CONFIGURED.add(name)
    return logger
