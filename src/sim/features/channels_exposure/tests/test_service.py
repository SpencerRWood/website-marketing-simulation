from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import simpy

from sim.features.channels_exposure.channels.paid_search import build as build_paid_search
from sim.features.channels_exposure.service import ChannelsExposureService
from sim.features.channels_exposure.types import ChannelConfig, ChannelsExposureConfig


@dataclass
class PredefinedRNG:
    values: list[float]
    idx: int = 0

    def random(self) -> float:
        if self.idx >= len(self.values):
            raise AssertionError("PredefinedRNG exhausted; add more values.")
        v = float(self.values[self.idx])
        self.idx += 1
        return v


@dataclass
class FakeLogger:
    def info(self, *_args, **_kwargs):  # pragma: no cover
        return None


@dataclass
class FakeCtx:
    run_id: str
    rng: Any
    logger: Any


@dataclass
class FakeWebsiteGraph:
    env: simpy.Environment
    start_dt: datetime

    def get_current_time(self) -> datetime:
        import datetime as _dt

        return self.start_dt + _dt.timedelta(seconds=float(self.env.now))


@dataclass
class FakePersistence:
    events: list[dict[str, Any]]

    def emit(self, **kwargs) -> None:
        self.events.append(kwargs)


@dataclass
class FakeUsersState:
    users: dict[str, Any]


def test_service_disabled_no_events():
    env = simpy.Environment()
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    website_graph = FakeWebsiteGraph(env=env, start_dt=start_dt)
    persistence = FakePersistence(events=[])
    intents = simpy.Store(env)
    users_state = FakeUsersState(users={"u1": {}})

    cfg = ChannelConfig(
        name="paid_search",
        exposure_rate_per_user_per_day=1.0,
        click_through_rate=1.0,
        incremental_intent=True,
        params={"in_market_share_per_day": 1.0},
    )
    channel = build_paid_search(cfg)

    svc = ChannelsExposureService(
        cfg=ChannelsExposureConfig(enabled=False),
        channels=[channel],
        persistence=persistence,
        website_graph=website_graph,
        intents_store=intents,
        ctx=FakeCtx(run_id="run_test", rng=PredefinedRNG([0.0]), logger=FakeLogger()),
    )

    svc.start(env, users_state=users_state, start_dt_utc=start_dt, num_days=1)
    env.run(until=24 * 60 * 60)

    assert persistence.events == []
    assert len(intents.items) == 0
