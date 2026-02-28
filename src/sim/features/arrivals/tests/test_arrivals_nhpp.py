from __future__ import annotations

from datetime import UTC, datetime, timedelta

import simpy

from sim.features.arrivals.models.nhpp import (
    SECONDS_PER_DAY,
    GaussianPeakCurveConfig,
    NHPPBaselineArrivalsConfig,
    NHPPBaselineArrivalsModel,
)
from sim.features.arrivals.types import SessionIntent


class DummyIds:
    def __init__(self) -> None:
        self.n = 0

    def new_id(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}_{self.n}"


class DummyGraph:
    def __init__(self, env: simpy.Environment, start_dt: datetime) -> None:
        self.env = env
        self.start_dt = start_dt.astimezone(UTC)

    def get_current_time(self) -> datetime:
        return self.start_dt + timedelta(seconds=float(self.env.now))


class ListBus:
    def __init__(self) -> None:
        self.items: list[SessionIntent] = []

    def publish(self, intent: SessionIntent) -> None:
        self.items.append(intent)


class DeterministicRng:
    def __init__(self, seed: int) -> None:
        import random

        self._r = random.Random(seed)

    def random(self) -> float:
        return self._r.random()

    def expovariate(self, rate: float) -> float:
        return self._r.expovariate(rate)


def test_nhpp_reproducible_same_seed() -> None:
    env1 = simpy.Environment()
    env2 = simpy.Environment()

    cfg = NHPPBaselineArrivalsConfig(
        daily_expected_intents=400.0,
        intraday_curve=GaussianPeakCurveConfig(peak_hour=12.0, spread_hours=3.0, floor=0.05),
    )

    bus1 = ListBus()
    bus2 = ListBus()

    g1 = DummyGraph(env1, datetime(2026, 1, 1, tzinfo=UTC))
    g2 = DummyGraph(env2, datetime(2026, 1, 1, tzinfo=UTC))

    m1 = NHPPBaselineArrivalsModel(
        run_id="run_1",
        rng=DeterministicRng(123),
        ids=DummyIds(),
        graph=g1,
        intent_bus=bus1,
        cfg=cfg,
        num_days=1,
        events=None,
    )
    m2 = NHPPBaselineArrivalsModel(
        run_id="run_1",
        rng=DeterministicRng(123),
        ids=DummyIds(),
        graph=g2,
        intent_bus=bus2,
        cfg=cfg,
        num_days=1,
        events=None,
    )

    m1.start(env1)
    m2.start(env2)

    env1.run(until=SECONDS_PER_DAY)
    env2.run(until=SECONDS_PER_DAY)

    t1 = [round(x.sim_time_s, 6) for x in bus1.items]
    t2 = [round(x.sim_time_s, 6) for x in bus2.items]
    assert t1 == t2


def test_nhpp_peak_has_more_than_night() -> None:
    env = simpy.Environment()

    cfg = NHPPBaselineArrivalsConfig(
        daily_expected_intents=800.0,
        intraday_curve=GaussianPeakCurveConfig(peak_hour=12.0, spread_hours=2.5, floor=0.05),
    )

    bus = ListBus()
    graph = DummyGraph(env, datetime(2026, 1, 1, tzinfo=UTC))

    model = NHPPBaselineArrivalsModel(
        run_id="run_peak",
        rng=DeterministicRng(7),
        ids=DummyIds(),
        graph=graph,
        intent_bus=bus,
        cfg=cfg,
        num_days=1,
        events=None,
    )
    model.start(env)
    env.run(until=SECONDS_PER_DAY)

    assert len(bus.items) > 0

    def in_window(intent: SessionIntent, h0: float, h1: float) -> bool:
        hour = (intent.sim_time_s / 3600.0) % 24.0
        return (hour >= h0) and (hour < h1)

    night = sum(1 for it in bus.items if in_window(it, 2.0, 4.0))
    peak = sum(1 for it in bus.items if in_window(it, 11.0, 13.0))

    assert peak > night


def test_nhpp_zero_expected_intents_produces_none() -> None:
    env = simpy.Environment()

    cfg = NHPPBaselineArrivalsConfig(
        daily_expected_intents=0.0,
        intraday_curve=GaussianPeakCurveConfig(peak_hour=12.0, spread_hours=3.0, floor=0.05),
    )

    bus = ListBus()
    graph = DummyGraph(env, datetime(2026, 1, 1, tzinfo=UTC))

    model = NHPPBaselineArrivalsModel(
        run_id="run_zero",
        rng=DeterministicRng(1),
        ids=DummyIds(),
        graph=graph,
        intent_bus=bus,
        cfg=cfg,
        num_days=1,
        events=None,
    )
    model.start(env)
    env.run(until=SECONDS_PER_DAY)

    assert bus.items == []
