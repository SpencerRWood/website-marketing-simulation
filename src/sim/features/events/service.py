from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import simpy

from sim.features.events.schema import (
    ALLOWED_EVENT_TYPES,
    SESSION_SCOPED_EVENT_TYPES,
    Event,
    json_dumps,
)
from sim.features.events.types import EventsEmitter
from sim.features.site_graph.types import WebsiteGraph


class IdGenerator(Protocol):
    def next_event_id(self) -> str: ...


class PersistenceSink(Protocol):
    """Minimal surface area the events feature needs."""

    def append(self, row: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class CounterEventIdGenerator:
    """Deterministic, monotonic event ids scoped to a run."""

    run_id: str
    counter: int = 0

    def next_event_id(self) -> str:
        self.counter += 1
        return f"{self.run_id}_{self.counter:08d}"


class EventService(EventsEmitter):
    """Canonical event emitter.

    - Stamps: run_id, event_id, ts_utc, sim_time_s
    - Enforces: basic event contracts
    - Routes: to PersistenceService via .append(row)
    """

    def __init__(
        self,
        *,
        env: simpy.Environment,
        graph: WebsiteGraph,
        persistence: PersistenceSink,
        ids: IdGenerator,
        run_id: str,
        logger: Any | None = None,
    ) -> None:
        self._env = env
        self._graph = graph
        self._persistence = persistence
        self._ids = ids
        self._run_id = run_id
        self._logger = logger

    @property
    def run_id(self) -> str:
        return self._run_id

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
        payload: dict[str, Any] | None = None,
    ) -> Event:
        if event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError(
                f"Unsupported event_type={event_type!r}. Allowed={sorted(ALLOWED_EVENT_TYPES)}"
            )

        # Session-scoped events must have session_id
        if event_type in SESSION_SCOPED_EVENT_TYPES and session_id is None:
            raise ValueError(f"{event_type} requires session_id")

        # If a session_id is present, channel must be present (persist origin channel across session)
        if session_id is not None and channel is None:
            raise ValueError(
                "channel must be provided when session_id is set "
                "(persist origin channel across session events). "
                "For baseline sessions, use channel='direct'."
            )

        event = Event(
            run_id=self._run_id,
            event_id=self._ids.next_event_id(),
            ts_utc=self._graph.get_current_time(),
            sim_time_s=float(self._env.now),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            intent_source=intent_source,
            channel=channel,
            page=page,
            value_num=value_num,
            value_str=value_str,
            payload_json=json_dumps(payload),
        )

        self._persistence.append(event.as_row())

        if self._logger is not None and self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(
                "event_emitted",
                extra={
                    "run_id": event.run_id,
                    "event_type": event.event_type,
                    "event_id": event.event_id,
                    "sim_time_s": event.sim_time_s,
                    "user_id": event.user_id,
                    "session_id": event.session_id,
                    "channel": event.channel,
                    "page": event.page,
                },
            )
