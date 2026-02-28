from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import simpy

from sim.features.arrivals.models.nhpp import GaussianPeakCurveConfig
from sim.features.arrivals.service import ArrivalsService, BaselineArrivalsConfig
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


def test_dispatch_rejects_unknown_model() -> None:
    env = simpy.Environment()

    cfg = BaselineArrivalsConfig(
        model="not_a_model",
        daily_expected_intents=10.0,
        intraday_curve=GaussianPeakCurveConfig(peak_hour=12.0, spread_hours=3.0, floor=0.05),
    )

    with pytest.raises(ValueError, match="Unsupported arrivals\\.baseline_arrivals\\.model"):
        _ = ArrivalsService(
            run_id="run_x",
            rng=DeterministicRng(1),
            ids=DummyIds(),
            graph=DummyGraph(env, datetime(2026, 1, 1, tzinfo=UTC)),
            intent_bus=ListBus(),
            baseline_arrivals=cfg,
            num_days=1,
            events=None,
        )
