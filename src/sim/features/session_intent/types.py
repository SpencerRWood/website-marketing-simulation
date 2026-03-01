from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True, frozen=True)
class SessionIntent:
    """
    Canonical intent message passed on the hot intent bus.

    intent_id: deterministic ID from IdsService (intent scope).
    ts_utc: authoritative UTC wall-clock timestamp (from WebsiteGraph clock).
    sim_time_s: sim time in seconds since env start (env.now).
    """

    intent_id: str
    ts_utc: datetime
    sim_time_s: float

    intent_source: str
    channel: str | None = None
    audience_id: str | None = None
    payload: dict[str, Any] | None = None
