from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


class RngLike(Protocol):
    def random(self) -> float: ...

    # UsersStateService.select_existing_user uses rng.choices(...)
    def choices(self, population, weights=None, k: int = 1): ...


class IdsLike(Protocol):
    def next_id(self, prefix: str) -> str: ...


class HasUserId(Protocol):
    user_id: str


class EventSink(Protocol):
    """
    Minimal interface so resolver is not coupled to your events/persistence implementation.
    """

    def emit(
        self,
        *,
        ts_utc: datetime,
        sim_time_s: float,
        event_type: str,
        user_id: str | None = None,
        session_id: str | None = None,
        intent_id: str | None = None,
        intent_source: str | None = None,
        channel: str | None = None,
        audience_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None: ...


class UsersStateLike(Protocol):
    """
    Matches sim.features.users_state.service.UsersStateService public API.
    """

    def get_or_create_user_for_intent(
        self,
        *,
        now_utc: datetime,
        rng: RngLike,
    ) -> tuple[HasUserId, bool]: ...


class SessionRunner(Protocol):
    """
    Downstream dependency. In feat/sessions, this becomes the real session process.
    """

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
