from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

# ----------------------------
# Config models
# ----------------------------


@dataclass(frozen=True)
class InterPageTimeConfig:
    """
    Minimal distribution support:
      - fixed: always fixed_seconds
      - exponential: mean_seconds
    """

    dist: str = "fixed"
    fixed_seconds: float = 10.0
    mean_seconds: float = 10.0


@dataclass(frozen=True)
class SessionsConfig:
    """
    max_steps:
      - int  -> hard cap on number of page_view steps
      - None -> no cap (session ends only via dropoff/timeout/no_next_page)
    """

    inactivity_timeout_minutes: float = 30.0
    max_steps: int | None = 12
    entry_page: str = "home"
    inter_page_time: InterPageTimeConfig = InterPageTimeConfig()


# ----------------------------
# Protocols (keeps integration flexible)
# ----------------------------


class WebsiteGraphLike(Protocol):
    def get_current_time(self): ...
    def get_page(self, name: str): ...
    def next_page(self, current_name: str, rng: Any) -> str | None: ...


class RNGLike(Protocol):
    def random(self) -> float: ...
    def expovariate(self, lambd: float) -> float: ...


class EventsLike(Protocol):
    # supported shapes:
    # - emit(**fields)
    # - publish(mapping_or_dict)
    def emit(self, **kwargs: Any) -> None: ...
    def publish(self, event: Mapping[str, Any]) -> None: ...


class IdsLike(Protocol):
    # supported shapes (we probe in order)
    def event_id(self) -> str: ...
    def new_event_id(self) -> str: ...
    def next_id(self) -> str: ...


# ----------------------------
# Internal helpers
# ----------------------------


def _events_write(events: Any, payload: dict[str, Any]) -> None:
    """
    Write an event using either:
      - events.emit(**payload)
      - events.publish(payload)
    """
    if hasattr(events, "emit"):
        events.emit(**payload)
        return
    if hasattr(events, "publish"):
        events.publish(payload)
        return
    raise TypeError("events must provide emit(**kwargs) or publish(mapping)")


def _new_event_id(ids: Any) -> str:
    """
    Deterministic ID generation should come from your ids service.
    We probe common method names.
    """
    for name in ("event_id", "new_event_id", "next_id"):
        if hasattr(ids, name):
            fn = getattr(ids, name)
            return str(fn())
    raise TypeError("ids must provide event_id() or new_event_id() or next_id()")


def _sample_inter_page_delay_s(cfg: InterPageTimeConfig, rng: Any) -> float:
    dist = (cfg.dist or "fixed").lower().strip()

    if dist == "fixed":
        return float(cfg.fixed_seconds)

    if dist == "exponential":
        mean = float(cfg.mean_seconds)
        if mean <= 0:
            return 0.0

        # expovariate expects lambda = 1/mean
        if not hasattr(rng, "expovariate"):
            # fallback using inverse-CDF with rng.random()
            u = float(rng.random())
            u = min(max(u, 1e-12), 1.0 - 1e-12)
            return -math.log(1.0 - u) * mean

        return float(rng.expovariate(1.0 / mean))

    raise ValueError(f"Unsupported sessions.inter_page_time.dist: {cfg.dist!r}")


# ----------------------------
# Service
# ----------------------------


class SessionsService:
    """
    Spawns per-session SimPy processes that traverse the website graph.

    Emits:
      - session_start
      - page_view (one per visited page)
      - drop_off (if page dropoff triggers)
      - session_end (always)

    Timeout rule:
      If inter-page delay > inactivity timeout, session ends with session_end(value_str="timeout").
    """

    def __init__(
        self,
        *,
        env: Any,
        graph: WebsiteGraphLike,
        rng: RNGLike,
        events: EventsLike,
        ids: IdsLike,
        run_id: str,
        cfg: SessionsConfig,
    ) -> None:
        self.env = env
        self.graph = graph
        self.rng = rng
        self.events = events
        self.ids = ids
        self.run_id = run_id
        self.cfg = cfg

    def spawn(
        self,
        *,
        user_id: str,
        session_id: str,
        intent_source: str | None = None,
        entry_page: str | None = None,
    ) -> Any:
        """
        Spawn the SimPy process for this session.
        """
        page0 = entry_page or self.cfg.entry_page
        return self.env.process(
            self._run_session(
                user_id=user_id,
                session_id=session_id,
                intent_source=intent_source,
                entry_page=page0,
            )
        )

    def _emit(
        self,
        *,
        event_type: str,
        user_id: str,
        session_id: str,
        intent_source: str | None = None,
        page: str | None = None,
        value_num: float | None = None,
        value_str: str | None = None,
        payload_json: str | None = None,
    ) -> None:
        ts_utc = self.graph.get_current_time()
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "event_id": _new_event_id(self.ids),
            "ts_utc": ts_utc,
            "sim_time_s": float(self.env.now),
            "user_id": user_id,
            "session_id": session_id,
            "event_type": event_type,
            "intent_source": intent_source,
            "page": page,
            "value_num": value_num,
            "value_str": value_str,
            "payload_json": payload_json,
        }
        _events_write(self.events, payload)

    def _run_session(
        self,
        *,
        user_id: str,
        session_id: str,
        intent_source: str | None,
        entry_page: str,
    ):
        max_steps = self.cfg.max_steps  # int | None
        timeout_s = float(self.cfg.inactivity_timeout_minutes) * 60.0

        self._emit(
            event_type="session_start",
            user_id=user_id,
            session_id=session_id,
            intent_source=intent_source,
            page=entry_page,
        )

        current: str | None = entry_page
        steps = 0

        def _under_step_cap() -> bool:
            if max_steps is None:
                return True
            return steps < int(max_steps)

        while current is not None and _under_step_cap():
            steps += 1

            # Page view
            self._emit(
                event_type="page_view",
                user_id=user_id,
                session_id=session_id,
                intent_source=intent_source,
                page=current,
                value_num=float(steps),
            )

            # Drop-off check uses page.dropoff_p if present
            page_obj = self.graph.get_page(current)
            dropoff_p = 0.0
            if page_obj is not None and hasattr(page_obj, "dropoff_p"):
                dropoff_p = float(page_obj.dropoff_p or 0.0)

            u = float(self.rng.random())
            if u < dropoff_p:
                self._emit(
                    event_type="drop_off",
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=intent_source,
                    page=current,
                    value_num=dropoff_p,
                )
                self._emit(
                    event_type="session_end",
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=intent_source,
                    page=current,
                    value_str="drop_off",
                    value_num=float(steps),
                )
                return

            # Inter-page time
            delay_s = _sample_inter_page_delay_s(self.cfg.inter_page_time, self.rng)

            # Timeout if the user "goes idle" too long
            if timeout_s > 0 and delay_s > timeout_s:
                # advance to timeout boundary for consistent sim clock
                yield self.env.timeout(timeout_s)
                self._emit(
                    event_type="session_end",
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=intent_source,
                    page=current,
                    value_str="timeout",
                    value_num=float(steps),
                )
                return

            if delay_s > 0:
                yield self.env.timeout(delay_s)

            # Transition
            nxt = self.graph.next_page(current, self.rng)
            if nxt is None:
                self._emit(
                    event_type="session_end",
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=intent_source,
                    page=current,
                    value_str="no_next_page",
                    value_num=float(steps),
                )
                return

            current = nxt

        # Only possible exits here:
        #   - max_steps cap hit (when max_steps is not None)
        #   - current becomes None (should not happen given guards, but handle anyway)
        if max_steps is not None and steps >= int(max_steps):
            self._emit(
                event_type="session_end",
                user_id=user_id,
                session_id=session_id,
                intent_source=intent_source,
                page=current,
                value_str="max_steps",
                value_num=float(steps),
            )
            return

        # Defensive fallback if current somehow becomes None without earlier termination
        self._emit(
            event_type="session_end",
            user_id=user_id,
            session_id=session_id,
            intent_source=intent_source,
            page=current,
            value_str="ended",
            value_num=float(steps),
        )
