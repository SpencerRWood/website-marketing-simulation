from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sim.core.rng import RNG
from sim.features.conversion.service import ConversionService
from sim.features.events.types import EventsEmitter
from sim.features.site_graph.types import WebsiteGraph

# ----------------------------
# Config models
# ----------------------------


@dataclass(frozen=True)
class InterPageTimeConfig:
    """Minimal distribution support:
    - fixed: always fixed_seconds
    - exponential: mean_seconds
    """

    dist: str = "fixed"
    fixed_seconds: float = 10.0
    mean_seconds: float = 10.0


@dataclass(frozen=True)
class SessionsConfig:
    """Session rules.

    max_steps:
      - int  -> hard cap on number of page_view steps
      - None -> no cap (session ends only via dropoff/timeout/no_next_page)
    """

    inactivity_timeout_minutes: float = 30.0
    max_steps: int | None = 12
    entry_page: str = "home"
    inter_page_time: InterPageTimeConfig = InterPageTimeConfig()


# ----------------------------
# Internal helpers
# ----------------------------


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
    """Spawns per-session SimPy processes that traverse the website graph.

    Emits via EventsEmitter:
      - session_start
      - page_view (one per visited page)
      - drop_off (if page dropoff triggers)
      - conversion (if applicable)
      - session_end (always)

    Timeout rule:
      If inter-page delay > inactivity timeout, session ends with session_end(value_str="timeout").
    """

    def __init__(
        self,
        *,
        env: Any,
        graph: WebsiteGraph,
        rng: RNG,
        events: EventsEmitter,
        cfg: SessionsConfig,
    ) -> None:
        self.env = env
        self.graph = graph
        self.rng = rng
        self.events = events
        self.cfg = cfg

    def spawn(
        self,
        *,
        user_id: str,
        session_id: str,
        intent_source: str | None = None,
        channel: str | None = None,
        entry_page: str | None = None,
        conversion: ConversionService | None = None,
        user_propensity: float | None = None,
        dropoff_multiplier: float = 1.0,
        conversion_logit_shift: float = 0.0,
    ) -> Any:
        page0 = entry_page or self.cfg.entry_page

        # Enforce channel for session-scoped events. Baseline traffic is "direct".
        channel_out = channel or "direct"

        return self.env.process(
            self._run_session(
                user_id=user_id,
                session_id=session_id,
                intent_source=intent_source,
                channel=channel_out,
                entry_page=page0,
                conversion=conversion,
                user_propensity=user_propensity,
                dropoff_multiplier=dropoff_multiplier,
                conversion_logit_shift=conversion_logit_shift,
            )
        )

    def _emit(
        self,
        *,
        event_type: str,
        user_id: str,
        session_id: str,
        intent_source: str | None,
        channel: str,
        page: str | None = None,
        value_num: float | None = None,
        value_str: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.events.emit(
            event_type,
            user_id=user_id,
            session_id=session_id,
            intent_source=intent_source,
            channel=channel,
            page=page,
            value_num=value_num,
            value_str=value_str,
            payload=payload,
        )

    def _run_session(
        self,
        *,
        user_id: str,
        session_id: str,
        intent_source: str | None,
        channel: str,
        entry_page: str,
        conversion: ConversionService | None,
        user_propensity: float | None,
        dropoff_multiplier: float,
        conversion_logit_shift: float,
    ):
        max_steps = self.cfg.max_steps
        timeout_s = float(self.cfg.inactivity_timeout_minutes) * 60.0

        self._emit(
            event_type="session_start",
            user_id=user_id,
            session_id=session_id,
            intent_source=intent_source,
            channel=channel,
            page=entry_page,
        )

        current: str | None = entry_page
        steps = 0

        while current is not None:
            # page view (entry page is counted as a view)
            self._emit(
                event_type="page_view",
                user_id=user_id,
                session_id=session_id,
                intent_source=intent_source,
                channel=channel,
                page=current,
            )
            steps += 1

            # max steps cap
            if max_steps is not None and steps >= int(max_steps):
                self._emit(
                    event_type="session_end",
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=intent_source,
                    channel=channel,
                    page=current,
                    value_str="max_steps",
                )
                return

            # page-level dropoff
            page_obj = self.graph.get_page(current)
            drop_p = float(getattr(page_obj, "dropoff_p", 0.0)) if page_obj is not None else 0.0
            drop_p = max(0.0, min(1.0, drop_p * float(dropoff_multiplier)))

            if self.rng.random() < drop_p:
                self._emit(
                    event_type="drop_off",
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=intent_source,
                    channel=channel,
                    page=current,
                    value_num=drop_p,
                )
                self._emit(
                    event_type="session_end",
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=intent_source,
                    channel=channel,
                    page=current,
                    value_str="drop_off",
                )
                return

            # conversion check (optional)
            if conversion is not None:
                did_convert, p_conv = conversion.should_convert(
                    rng=self.rng,
                    propensity=float(user_propensity) if user_propensity is not None else 0.0,
                    logit_shift=float(conversion_logit_shift),
                )

                if did_convert:
                    self._emit(
                        event_type="conversion",
                        user_id=user_id,
                        session_id=session_id,
                        intent_source=intent_source,
                        channel=channel,
                        page=current,
                        value_num=float(p_conv),
                    )

            # select next page
            try:
                nxt = self.graph.next_page(current, self.rng)
            except TypeError:
                nxt = self.graph.next_page(current)

            if nxt is None:
                self._emit(
                    event_type="session_end",
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=intent_source,
                    channel=channel,
                    page=current,
                    value_str="no_next_page",
                )
                return

            # delay to next interaction
            delay_s = float(_sample_inter_page_delay_s(self.cfg.inter_page_time, self.rng))
            if delay_s > timeout_s:
                yield self.env.timeout(timeout_s)
                self._emit(
                    event_type="session_end",
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=intent_source,
                    channel=channel,
                    page=current,
                    value_str="timeout",
                )
                return

            yield self.env.timeout(delay_s)
            current = nxt
