from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Keep this list tight; expand deliberately as you add features.
ALLOWED_EVENT_TYPES: set[str] = {
    # upstream drivers
    "session_intent",
    "user_created",
    # marketing touches
    "exposure",
    "click",
    # session lifecycle + navigation
    "session_start",
    "page_view",
    "drop_off",
    "conversion",
    "session_end",
    # run lifecycle (optional but useful for smoke tests)
    "run_started",
    "run_finished",
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

    intent_source: str | None = None
    channel: str | None = None
    page: str | None = None

    value_num: float | None = None
    value_str: str | None = None

    payload_json: str | None = None

    def as_row(self) -> dict[str, Any]:
        """Canonical DuckDB row representation matching persistence schema."""
        return {
            "run_id": self.run_id,
            "event_id": self.event_id,
            "ts_utc": self.ts_utc,
            "sim_time_s": float(self.sim_time_s),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "intent_source": self.intent_source,
            "channel": self.channel,
            "page": self.page,
            "value_num": self.value_num,
            "value_str": self.value_str,
            "payload_json": self.payload_json,
        }


def json_dumps(payload: Mapping[str, Any] | None) -> str | None:
    if payload is None:
        return None
    # Stable JSON for deterministic outputs/diffs
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
