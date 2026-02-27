from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import simpy

from sim.core.ids import IdsService

from .types import SessionIntent


@dataclass(slots=True)
class SessionIntentService:
    env: simpy.Environment
    ids: IdsService
    capacity: int | None = None

    _store: simpy.Store = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # SimPy does NOT accept None for capacity; omit to get infinite capacity.
        if self.capacity is None:
            self._store = simpy.Store(self.env)
        else:
            self._store = simpy.Store(self.env, capacity=self.capacity)

    @property
    def store(self) -> simpy.Store:
        return self._store

    def build_intent(
        self,
        *,
        ts_utc: datetime,
        intent_source: str,
        channel: str | None = None,
        audience_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> SessionIntent:
        intent_id = self.ids.next_id("intent")
        return SessionIntent(
            intent_id=intent_id,
            ts_utc=ts_utc,
            intent_source=intent_source,
            channel=channel,
            audience_id=audience_id,
            payload=payload,
        )

    def publish(self, intent: SessionIntent) -> simpy.events.Event:
        return self._store.put(intent)

    def publish_new(
        self,
        *,
        ts_utc: datetime,
        intent_source: str,
        channel: str | None = None,
        audience_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> simpy.events.Event:
        intent = self.build_intent(
            ts_utc=ts_utc,
            intent_source=intent_source,
            channel=channel,
            audience_id=audience_id,
            payload=payload,
        )
        return self.publish(intent)

    def get(self) -> simpy.events.Event:
        return self._store.get()

    def size(self) -> int:
        return len(self._store.items)
