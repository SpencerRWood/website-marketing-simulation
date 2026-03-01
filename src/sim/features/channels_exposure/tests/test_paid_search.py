from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import simpy

from sim.features.channels_exposure.channels.paid_search import build as build_paid_search
from sim.features.channels_exposure.service import ChannelsExposureService
from sim.features.channels_exposure.types import (
    ChannelConfig,
    ChannelsExposureConfig,
    DeliveryPlan,
    DeliverySlice,
)


@dataclass
class PredefinedRNG:
    values: list[float]
    idx: int = 0

    def random(self) -> float:
        if self.idx >= len(self.values):
            raise AssertionError("PredefinedRNG exhausted; add more values to the test stream.")
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


def _count(events: list[dict[str, Any]], event_type: str) -> int:
    return sum(1 for e in events if e.get("event_type") == event_type)


def test_rate_mode_in_market_zero_emits_nothing():
    env = simpy.Environment()
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    website_graph = FakeWebsiteGraph(env=env, start_dt=start_dt)
    persistence = FakePersistence(events=[])
    intents = simpy.Store(env)

    users_state = FakeUsersState(users={"u1": {}, "u2": {}})

    cfg = ChannelConfig(
        name="paid_search",
        exposure_rate_per_user_per_day=5.0,
        click_through_rate=1.0,
        incremental_intent=True,
        params={
            "in_market_share_per_day": 0.0,  # no one eligible
            "freq_cap_per_user_per_day": 10,
            "incremental_click_share": 1.0,
        },
    )
    channel = build_paid_search(cfg)

    # in-market checks (u1, u2) — values don’t matter; all fail because share=0.0
    rng = PredefinedRNG(values=[0.1, 0.2])

    svc = ChannelsExposureService(
        cfg=ChannelsExposureConfig(enabled=True),
        channels=[channel],
        persistence=persistence,
        website_graph=website_graph,
        intents_store=intents,
        ctx=FakeCtx(run_id="run_test", rng=rng, logger=FakeLogger()),
    )

    svc.start(env, users_state=users_state, start_dt_utc=start_dt, num_days=1)
    env.run(until=24 * 60 * 60)

    assert persistence.events == []
    assert len(intents.items) == 0


def test_rate_mode_freq_cap_clamps_exposures():
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
        params={
            "in_market_share_per_day": 1.0,  # eligible
            "freq_cap_per_user_per_day": 2,  # clamp
            "brand_share": 0.0,
            "brand_ctr_multiplier": 1.0,
            "nonbrand_ctr_multiplier": 1.0,
            "incremental_click_share": 0.0,  # clicks but no intents
        },
    )
    channel = build_paid_search(cfg)

    # RNG usage (rate mode for 1 user):
    # 1) in-market check
    # 2-5) Poisson lam=1.0 -> force N=3 (0.9,0.9,0.9,0.4) => 3, then clamp to 2
    # For each of 2 exposures:
    # - offset
    # - brand draw
    # - click draw
    # - incremental draw
    rng = PredefinedRNG(
        values=[
            0.0,  # in-market
            0.9,
            0.9,
            0.9,
            0.4,  # poisson -> 3
            0.1,
            0.9,
            0.2,
            0.7,  # exp1
            0.2,
            0.9,
            0.3,
            0.8,  # exp2
        ]
    )

    svc = ChannelsExposureService(
        cfg=ChannelsExposureConfig(enabled=True),
        channels=[channel],
        persistence=persistence,
        website_graph=website_graph,
        intents_store=intents,
        ctx=FakeCtx(run_id="run_test", rng=rng, logger=FakeLogger()),
    )

    svc.start(env, users_state=users_state, start_dt_utc=start_dt, num_days=1)
    env.run(until=24 * 60 * 60)

    assert _count(persistence.events, "exposure") == 2
    assert _count(persistence.events, "click") == 2
    assert _count(persistence.events, "session_intent") == 0
    assert len(intents.items) == 0


def test_delivery_mode_campaign_id_and_counts():
    """
    Delivery-driven:
      - Provide one slice with impressions=3 at time 10s
      - Force in-market gating to pass for exactly 2 of them (third fails)
      - Force ctr=1 so clicks happen when exposed
      - Force incremental_click_share=1 so intents happen when clicked
      - Verify campaign_id propagates into payload_json for exposure/click/intent
    """
    env = simpy.Environment()
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    website_graph = FakeWebsiteGraph(env=env, start_dt=start_dt)
    persistence = FakePersistence(events=[])
    intents = simpy.Store(env)
    users_state = FakeUsersState(users={"u1": {}, "u2": {}})

    cfg = ChannelConfig(
        name="paid_search",
        exposure_rate_per_user_per_day=0.0,  # irrelevant in delivery mode
        click_through_rate=1.0,
        incremental_intent=True,
        params={
            # KEY FIX: make failure possible; 0.99 should fail when threshold is 0.5
            "in_market_share_per_day": 0.5,
            "freq_cap_per_user_per_day": 10,
            "brand_share": 0.0,
            "incremental_click_share": 1.0,
        },
    )
    channel = build_paid_search(cfg)

    def provider(day_idx: int, day_start_s: float, seconds_per_day: float):
        return [
            DeliveryPlan(
                channel="paid_search",
                slices=[
                    DeliverySlice(
                        at_s=10.0,
                        impressions=3,
                        campaign_id="c1",
                        ctr=1.0,
                        incremental_click_share=1.0,
                    )
                ],
            )
        ]

    # Delivery mode RNG consumption per impression (PaidSearchChannel):
    # - pick user_id (rng)
    # - in-market gate (rng)
    # - brand draw (rng) ONLY IF in-market passes
    # then at execution time (for each scheduled exposure):
    # - click draw (rng)
    # - incremental draw (rng)
    #
    # We want 2 exposures pass in-market, 1 fails:
    # - gate values: 0.0 pass, 0.0 pass, 0.99 fail (since threshold=0.5)
    rng = PredefinedRNG(
        values=[
            # impression 1: pick u1, in-market pass, brand draw
            0.1,
            0.0,
            0.9,
            # impression 2: pick u2, in-market pass, brand draw
            0.6,
            0.0,
            0.9,
            # impression 3: pick u1, in-market FAIL (>= 0.5)
            0.2,
            0.99,
            # execution for 2 exposures at t=10:
            0.0,
            0.0,  # exp1 click, incremental
            0.0,
            0.0,  # exp2 click, incremental
        ]
    )

    svc = ChannelsExposureService(
        cfg=ChannelsExposureConfig(enabled=True),
        channels=[channel],
        persistence=persistence,
        website_graph=website_graph,
        intents_store=intents,
        ctx=FakeCtx(run_id="run_test", rng=rng, logger=FakeLogger()),
        delivery_plan_provider=provider,
    )

    svc.start(env, users_state=users_state, start_dt_utc=start_dt, num_days=1)
    env.run(until=24 * 60 * 60)

    assert _count(persistence.events, "exposure") == 2
    assert _count(persistence.events, "click") == 2
    assert _count(persistence.events, "session_intent") == 2
    assert len(intents.items) == 2

    for e in persistence.events:
        if e["event_type"] in ("exposure", "click", "session_intent"):
            assert e["payload_json"] == {"campaign_id": "c1"}
