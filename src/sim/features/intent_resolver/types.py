from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from sim.features.events.types import EventsEmitter


class HasUserId(Protocol):
    user_id: str


# Canonical event surface for the resolver
EventSink = EventsEmitter


class SessionRunner(Protocol):
    """Downstream dependency. In feat/sessions, this becomes the real session process."""

    def start_session(
        self,
        *,
        user_id: str,
        session_id: str,
        intent,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class IntentResolverConfig:
    enabled: bool = True
