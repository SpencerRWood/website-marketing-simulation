from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sim.core.logging import get_logger  # adjust if your logger lives elsewhere

from .duckdb_adapter import DuckDBAdapter


@dataclass(frozen=True)
class Event:
    """Legacy in-memory event type (kept for backwards compatibility)."""

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
    payload: dict[str, Any] | None = None


class PersistenceService:
    """Buffered event sink + flush policy.

    This service is the *cold* layer boundary. Features should not call it directly.
    Only the events feature should depend on `.append(row)`.

    - Hot: buffer in memory
    - Cold: DuckDB
    """

    def __init__(
        self,
        *,
        adapter: DuckDBAdapter,
        every_n_events: int,
        or_every_seconds: float,
    ) -> None:
        self.adapter = adapter
        self.every_n_events = int(every_n_events)
        self.or_every_seconds = float(or_every_seconds)

        # Internal buffer holds tuples already shaped for DuckDBAdapter.write_events(...)
        self._buf_rows: list[tuple] = []
        self._logger = get_logger(__name__)

        self._is_open = False
        self._periodic_proc_started = False

    def open(self) -> None:
        if self._is_open:
            return
        self.adapter.open()
        self._is_open = True

    # ------------------------------------------------------------------
    # Canonical API used by sim.features.events
    # ------------------------------------------------------------------
    def append(self, row: dict[str, Any]) -> None:
        """Append a single canonical event row (dict) to the buffer."""
        if not self._is_open:
            raise RuntimeError("PersistenceService not open. Call open() during bootstrap.")

        self._buf_rows.append(self._rowdict_to_tuple(row))

        if self.every_n_events > 0 and len(self._buf_rows) >= self.every_n_events:
            self.flush(reason="count")

    # ------------------------------------------------------------------
    # Legacy API (kept so older adapters/tests don't immediately break)
    # ------------------------------------------------------------------
    def emit(self, e: Event) -> None:
        self.append(self._event_to_rowdict(e))

    def flush(self, *, reason: str) -> None:
        if not self._buf_rows:
            return

        rows = list(self._buf_rows)
        self._buf_rows.clear()

        result = self.adapter.write_events(rows)

        self._logger.info(
            "flush",
            extra={
                "event": "flush",
                "reason": reason,
                "duckdb_path": self.adapter.path,
                "num_events": result.num_events,
                "duration_ms": result.duration_ms,
            },
        )

    def close(self) -> None:
        self.flush(reason="shutdown")
        self.adapter.close()
        self._is_open = False

    def start_periodic_flush(self, env) -> None:
        if self._periodic_proc_started:
            return
        self._periodic_proc_started = True
        env.process(self._periodic_flush_proc(env))

    def _periodic_flush_proc(self, env):
        while True:
            yield env.timeout(self.or_every_seconds)
            self.flush(reason="timer")

    # ------------------------------------------------------------------
    # Row shaping
    # ------------------------------------------------------------------
    @staticmethod
    def _event_to_rowdict(e: Event) -> dict[str, Any]:
        return {
            "run_id": e.run_id,
            "event_id": e.event_id,
            "ts_utc": e.ts_utc,
            "sim_time_s": float(e.sim_time_s),
            "user_id": e.user_id,
            "session_id": e.session_id,
            "event_type": e.event_type,
            "intent_source": e.intent_source,
            "channel": e.channel,
            "page": e.page,
            "value_num": e.value_num,
            "value_str": e.value_str,
            "payload_json": json.dumps(
                e.payload, sort_keys=True, separators=(",", ":"), default=str
            )
            if e.payload
            else None,
        }

    @staticmethod
    def _rowdict_to_tuple(row: dict[str, Any]) -> tuple:
        # DuckDBAdapter.write_events expects this exact order.
        return (
            row.get("run_id"),
            row.get("event_id"),
            row.get("ts_utc"),
            float(row.get("sim_time_s")) if row.get("sim_time_s") is not None else None,
            row.get("user_id"),
            row.get("session_id"),
            row.get("event_type"),
            row.get("intent_source"),
            row.get("channel"),
            row.get("page"),
            row.get("value_num"),
            row.get("value_str"),
            row.get("payload_json"),
        )
