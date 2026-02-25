from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

ALLOWED_EVENT_TYPES: set[str] = {
    # Marketing touches (if you keep them)
    "exposure",
    "click",
    # Session lifecycle + navigation
    "session_start",
    "page_view",
    "drop_off",
    "conversion",
    "session_end",
}

# Events that are expected to be tied to a session process
SESSION_SCOPED_EVENT_TYPES: set[str] = {
    "session_start",
    "page_view",
    "drop_off",
    "conversion",
    "session_end",
}


@dataclass(frozen=True, slots=True)
class Event:
    run_id: str
    event_id: str
    ts_utc: datetime
    sim_time_s: float

    event_type: str

    user_id: str | None = None
    session_id: str | None = None

    # Single acquisition dimension:
    # - baseline traffic: "direct"
    # - channel traffic: channel name, e.g. "search"
    channel: str | None = None

    page: str | None = None
    payload_json: str | None = None

    def as_row(self) -> dict[str, Any]:
        """
        Canonical DuckDB row representation matching your events table columns.
        """
        return {
            "run_id": self.run_id,
            "event_id": self.event_id,
            "ts_utc": self.ts_utc,
            "sim_time_s": float(self.sim_time_s),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "channel": self.channel,
            "page": self.page,
            "payload_json": self.payload_json,
        }


def json_dumps(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    # Stable JSON for deterministic outputs/diffs
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
