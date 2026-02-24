from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sim.core.logging import get_logger  # adjust if your logger lives elsewhere

from .duckdb_adapter import DuckDBAdapter


@dataclass(frozen=True)
class Event:
    """
    Minimal event contract. If you already have an Event type, replace this import.
    """

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
    """
    Buffered event sink + flush policy.
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

        self._buf: list[Event] = []
        self._logger = get_logger(__name__)

        self._is_open = False
        self._periodic_proc_started = False

    def open(self) -> None:
        if self._is_open:
            return
        self.adapter.open()
        self._is_open = True

    def emit(self, e: Event) -> None:
        if not self._is_open:
            raise RuntimeError("PersistenceService not open. Call open() during bootstrap.")

        self._buf.append(e)

        if self.every_n_events > 0 and len(self._buf) >= self.every_n_events:
            self.flush(reason="count")

    def flush(self, *, reason: str) -> None:
        if not self._buf:
            return

        rows = [self._event_to_row(e) for e in self._buf]
        self._buf.clear()

        result = self.adapter.write_events(rows)

        # Structured log (no wall-clock timestamps for determinism; duration is ok)
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
        # final flush
        self.flush(reason="shutdown")
        self.adapter.close()
        self._is_open = False

    def start_periodic_flush(self, env) -> None:
        """
        Start a SimPy process that flushes every `or_every_seconds`.
        Call once during bootstrap after env is created.
        """
        if self._periodic_proc_started:
            return
        self._periodic_proc_started = True
        env.process(self._periodic_flush_proc(env))

    def _periodic_flush_proc(self, env):
        while True:
            yield env.timeout(self.or_every_seconds)
            self.flush(reason="timer")

    @staticmethod
    def _event_to_row(e: Event) -> tuple:
        payload_json = json.dumps(e.payload, separators=(",", ":")) if e.payload else None
        return (
            e.run_id,
            e.event_id,
            e.ts_utc,
            float(e.sim_time_s),
            e.user_id,
            e.session_id,
            e.event_type,
            e.intent_source,
            e.channel,
            e.page,
            e.value_num,
            e.value_str,
            payload_json,
        )
