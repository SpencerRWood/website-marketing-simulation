from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import simpy

from sim.features.events.schema import (
    ALLOWED_EVENT_TYPES,
    SESSION_SCOPED_EVENT_TYPES,
    Event,
    json_dumps,
)
from sim.features.site_graph.service import WebsiteGraph


class IdGenerator(Protocol):
    def next_event_id(self) -> str: ...


class PersistenceSink(Protocol):
    """
    Minimal surface area the events feature needs.
    Adapt your existing PersistenceService to expose `.append(row: dict)`.
    """

    def append(self, row: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class CounterEventIdGenerator:
    """
    Deterministic, monotonic event ids scoped to a run.
    """

    run_id: str
    counter: int = 0

    def next_event_id(self) -> str:
        self.counter += 1
        return f"{self.run_id}_{self.counter:08d}"


class EventService:
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
        *,
        event_type: str,
        user_id: str | None = None,
        session_id: str | None = None,
        channel: str | None = None,
        page: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        """
        Emits a single behavioral event into cold storage via the persistence buffer.

        Contracts enforced:
        - event_type must be in ALLOWED_EVENT_TYPES
        - If session_id is present, channel must be present (persist origin channel across the session)
        - For baseline sessions, use channel="direct"
        """
        if event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError(
                f"Unsupported event_type={event_type!r}. " f"Allowed={sorted(ALLOWED_EVENT_TYPES)}"
            )

        if session_id is not None and channel is None:
            raise ValueError(
                "channel must be provided when session_id is set "
                "(persist origin channel across session events)."
            )

        # Optional: strengthen discipline further
        if event_type in SESSION_SCOPED_EVENT_TYPES and session_id is None:
            raise ValueError(f"{event_type} requires session_id")

        event = Event(
            run_id=self._run_id,
            event_id=self._ids.next_event_id(),
            ts_utc=self._graph.get_current_time(),
            sim_time_s=float(self._env.now),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            channel=channel,
            page=page,
            payload_json=json_dumps(payload),
        )

        self._persistence.append(event.as_row())

        if self._logger is not None:
            self._logger.info(
                "event_emitted",
                event_type=event.event_type,
                run_id=event.run_id,
                event_id=event.event_id,
                sim_time_s=event.sim_time_s,
                user_id=event.user_id,
                session_id=event.session_id,
                channel=event.channel,
                page=event.page,
            )

        return event
