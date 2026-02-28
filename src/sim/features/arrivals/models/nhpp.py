from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from sim.features.arrivals.types import (
    ArrivalModel,
    EventsLike,
    IdsLike,
    IntentBusLike,
    RngLike,
    SessionIntent,
    WebsiteGraphLike,
)

SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class GaussianPeakCurveConfig:
    peak_hour: float
    spread_hours: float
    floor: float = 0.05


@dataclass(frozen=True)
class NHPPBaselineArrivalsConfig:
    """
    NHPP baseline arrivals config (distribution-specific).
    daily_expected_intents is the mean number of baseline intents per day.
    intraday_curve defines relative intensity over the day; it is normalized
    so that the daily integral equals daily_expected_intents.
    """

    daily_expected_intents: float
    intraday_curve: GaussianPeakCurveConfig


def _assert_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _gaussian_shape(hour: float, cfg: GaussianPeakCurveConfig) -> float:
    if cfg.spread_hours <= 0:
        raise ValueError("spread_hours must be > 0")
    z = (hour - cfg.peak_hour) / cfg.spread_hours
    return float(cfg.floor) + math.exp(-0.5 * z * z)


def _normalize_daily_shape(
    cfg: GaussianPeakCurveConfig, grid_minutes: int = 1
) -> tuple[list[float], float, float]:
    """
    Precompute shape on a 24h grid; return (shape_values, shape_max, integral_hours).
    integral_hours approximates ∫ shape(t) dt over 24 hours.
    """
    if grid_minutes <= 0:
        raise ValueError("grid_minutes must be > 0")

    n = int((24 * 60) / grid_minutes)
    if n <= 0:
        raise ValueError("invalid grid_minutes")

    shapes: list[float] = []
    for i in range(n):
        minute = (i + 0.5) * grid_minutes
        hour = minute / 60.0
        shapes.append(_gaussian_shape(hour, cfg))

    shape_max = max(shapes)
    dt_hours = grid_minutes / 60.0
    integral_hours = sum(shapes) * dt_hours
    return shapes, shape_max, integral_hours


def _shape_at_second(shapes: list[float], second_in_day: float, grid_minutes: int) -> float:
    second_in_day = max(0.0, min(float(SECONDS_PER_DAY) - 1e-9, float(second_in_day)))
    minute = second_in_day / 60.0
    idx = int(minute // grid_minutes)
    idx = max(0, min(idx, len(shapes) - 1))
    return shapes[idx]


class NHPPBaselineArrivalsModel(ArrivalModel):
    """
    NHPP baseline arrivals using thinning with a normalized intraday curve.
    Emits SessionIntent(intent_source="baseline").
    """

    def __init__(
        self,
        *,
        run_id: str,
        rng: RngLike,
        ids: IdsLike,
        graph: WebsiteGraphLike,
        intent_bus: IntentBusLike,
        cfg: NHPPBaselineArrivalsConfig,
        num_days: int,
        events: EventsLike | None = None,
        grid_minutes: int = 1,
    ) -> None:
        if num_days <= 0:
            raise ValueError("num_days must be > 0")
        if cfg.daily_expected_intents < 0:
            raise ValueError("daily_expected_intents must be >= 0")

        self.run_id = run_id
        self.rng = rng
        self.ids = ids
        self.graph = graph
        self.intent_bus = intent_bus
        self.cfg = cfg
        self.num_days = num_days
        self.events = events
        self.grid_minutes = grid_minutes

    def start(self, env) -> None:
        env.process(self._run(env))

    def _run(self, env):
        daily_expected = float(self.cfg.daily_expected_intents)
        horizon_s = float(self.num_days * SECONDS_PER_DAY)

        if daily_expected <= 0:
            # no events; just advance time
            yield env.timeout(horizon_s)
            return

        shapes, shape_max, integral_hours = _normalize_daily_shape(
            self.cfg.intraday_curve, grid_minutes=self.grid_minutes
        )
        if integral_hours <= 0:
            raise ValueError("intraday curve integral must be > 0")

        # intensity(hour) = daily_expected * shape(hour) / integral_hours  (intents per hour)
        # for thinning, compute maximum intensity (per second)
        intensity_max_sec = (daily_expected * shape_max / integral_hours) / 3600.0
        if intensity_max_sec <= 0:
            yield env.timeout(horizon_s)
            return

        for _day in range(self.num_days):
            day_start = float(env.now)
            day_end = day_start + float(SECONDS_PER_DAY)

            t = day_start
            while True:
                dt = self.rng.expovariate(intensity_max_sec)
                t_candidate = t + float(dt)
                if t_candidate >= day_end:
                    break

                sec_in_day = t_candidate - day_start
                shape_t = _shape_at_second(shapes, sec_in_day, grid_minutes=self.grid_minutes)
                intensity_t_sec = (daily_expected * shape_t / integral_hours) / 3600.0

                accept_p = intensity_t_sec / intensity_max_sec
                if self.rng.random() < accept_p:
                    yield env.timeout(t_candidate - float(env.now))

                    ts = _assert_utc(self.graph.get_current_time())
                    intent = SessionIntent(
                        intent_id=str(self.ids.new_id("intent")),
                        ts_utc=ts,
                        sim_time_s=float(env.now),
                        intent_source="baseline",
                        channel=None,
                    )
                    self.intent_bus.publish(intent)

                    if self.events is not None:
                        self.events.emit(
                            run_id=self.run_id,
                            event_type="session_intent",
                            ts_utc=intent.ts_utc,
                            sim_time_s=intent.sim_time_s,
                            intent_source=intent.intent_source,
                            channel=intent.channel,
                            payload={
                                "intent_id": intent.intent_id,
                                "intent_source": intent.intent_source,
                                "channel": intent.channel,
                            },
                        )

                t = t_candidate

            if float(env.now) < day_end:
                yield env.timeout(day_end - float(env.now))
