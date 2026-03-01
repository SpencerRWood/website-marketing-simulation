from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import simpy

from sim.features.channels_exposure.channels.paid_search import build as build_paid_search
from sim.features.channels_exposure.service import ChannelsExposureService
from sim.features.channels_exposure.types import ChannelConfig, ChannelsExposureConfig
from sim.features.session_intent.service import SessionIntentService


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
    rng: object
    logger: object


@dataclass
class FakeWebsiteGraph:
    env: simpy.Environment
    start_dt: datetime

    def get_current_time(self) -> datetime:
        import datetime as _dt

        return self.start_dt + _dt.timedelta(seconds=float(self.env.now))


@dataclass
class FakeEvents:
    events: list[dict]

    def emit(self, event_type: str, **kwargs) -> None:
        self.events.append({"event_type": event_type, **kwargs})


@dataclass
class FakeUsersState:
    users: dict


@dataclass
class DummyIds:
    n: int = 0

    def next_id(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}_{self.n:08d}"


def test_service_disabled_no_events():
    env = simpy.Environment()
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    website_graph = FakeWebsiteGraph(env=env, start_dt=start_dt)
    events = FakeEvents(events=[])
    intent_bus = SessionIntentService(env=env, ids=DummyIds(), capacity=None)
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
        events=events,
        website_graph=website_graph,
        intent_bus=intent_bus,
        ctx=FakeCtx(rng=PredefinedRNG([0.0]), logger=FakeLogger()),
    )

    svc.start(env, users_state=users_state, start_dt_utc=start_dt, num_days=1)
    env.run(until=24 * 60 * 60)

    assert events.events == []

    # Ensure no intent is available on the bus
    got: dict[str, object] = {"intent": None}

    def consumer():
        intent = yield intent_bus.get()
        got["intent"] = intent

    env.process(consumer())
    env.run(until=float(env.now) + 0.01)  # give the consumer a chance (should not receive)
    assert got["intent"] is None


def test_service_enabled_emits_exposure_click_and_intent():
    env = simpy.Environment()
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    website_graph = FakeWebsiteGraph(env=env, start_dt=start_dt)
    events = FakeEvents(events=[])
    intent_bus = SessionIntentService(env=env, ids=DummyIds(), capacity=None)
    users_state = FakeUsersState(users={"u1": {}})

    cfg = ChannelConfig(
        name="paid_search",
        exposure_rate_per_user_per_day=1.0,
        click_through_rate=1.0,
        incremental_intent=True,
        params={"in_market_share_per_day": 1.0},
    )
    channel = build_paid_search(cfg)

    # RNG stream:
    # - exposure scheduling + selection (channel internals)
    # - click draw
    rng = PredefinedRNG([0.9, 0.0, 0.0, 0.0, 0.0, 0.0])

    svc = ChannelsExposureService(
        cfg=ChannelsExposureConfig(enabled=True),
        channels=[channel],
        events=events,
        website_graph=website_graph,
        intent_bus=intent_bus,
        ctx=FakeCtx(rng=rng, logger=FakeLogger()),
    )

    got: dict[str, object] = {"intent": None}

    def consumer():
        intent = yield intent_bus.get()
        got["intent"] = intent

    env.process(consumer())
    svc.start(env, users_state=users_state, start_dt_utc=start_dt, num_days=1)
    env.run(until=24 * 60 * 60)

    types = [e["event_type"] for e in events.events]
    assert "exposure" in types
    assert "click" in types
    assert "session_intent" in types

    assert got["intent"] is not None
    intent = got["intent"]
    assert intent.intent_source == "channel:paid_search"
    assert intent.channel == "paid_search"
    assert hasattr(intent, "intent_id")
    assert hasattr(intent, "ts_utc")
    assert hasattr(intent, "sim_time_s")
