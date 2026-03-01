from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class EventsEmitter(Protocol):
    """Canonical, minimal event emission interface for all features.

    Features should depend on this Protocol, not on concrete EventService or persistence.
    EventService is responsible for stamping run_id, event_id, ts_utc, sim_time_s and
    routing to cold storage.
    """

    def emit(
        self,
        event_type: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        intent_source: str | None = None,
        channel: str | None = None,
        page: str | None = None,
        value_num: float | None = None,
        value_str: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None: ...
