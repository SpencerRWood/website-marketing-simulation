from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


class RngLike(Protocol):
    """Minimal RNG surface required for arrivals."""

    def random(self) -> float: ...
    def expovariate(self, rate: float) -> float: ...


class IdsLike(Protocol):
    def new_id(self, prefix: str) -> str: ...


class IntentBusLike(Protocol):
    def publish(self, intent: SessionIntent) -> None: ...


class EventsLike(Protocol):
    """
    Optional event sink. Keep intentionally loose to avoid coupling to a specific schema.
    """

    def emit(
        self,
        *,
        run_id: str,
        event_type: str,
        ts_utc: datetime,
        sim_time_s: float,
        user_id: str | None = None,
        session_id: str | None = None,
        intent_source: str | None = None,
        channel: str | None = None,
        page: str | None = None,
        value_num: float | None = None,
        value_str: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None: ...


class WebsiteGraphLike(Protocol):
    """Matches your canonical WebsiteGraph time API."""

    def get_current_time(self) -> datetime: ...


@dataclass(frozen=True)
class SessionIntent:
    """
    Upstream session intent emitted by arrivals.
    Downstream resolver attaches/chooses a user and spawns a session.
    """

    intent_id: str
    ts_utc: datetime
    sim_time_s: float
    intent_source: str  # "baseline" or "channel:<name>"
    channel: str | None = None


class ArrivalModel(Protocol):
    def start(self, env) -> None: ...
