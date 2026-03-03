from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

# Standard LogRecord attributes we don't want to re-emit as "extras"
_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Merge extras: anything on record.__dict__ that's not reserved
        for k, v in record.__dict__.items():
            if k in _RESERVED:
                continue
            if k.startswith("_"):
                continue
            # don't overwrite core keys
            if k in payload:
                continue
            payload[k] = v

        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid double handlers in tests

    logger.setLevel(level.upper())

    handler = logging.FileHandler("data/sim.log")
    handler.setFormatter(JsonFormatter())

    logger.addHandler(handler)
    logger.propagate = False
    return logger
