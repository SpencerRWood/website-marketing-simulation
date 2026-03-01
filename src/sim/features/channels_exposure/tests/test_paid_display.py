from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import simpy

from sim.features.channels_exposure.channels.paid_display import build as build_paid_display
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


def test_paid_display_builder():
    cfg = ChannelConfig(
        name="paid_display",
        exposure_rate_per_user_per_day=0.0,
        click_through_rate=0.001,
        incremental_intent=True,
        params={"viewable_share": 0.6, "freq_cap_per_user_per_day": 3},
    )
    ch = build_paid_display(cfg)
    assert ch.cfg.name == "paid_display"


def test_paid_display_delivery_mode_viewability_and_cap_and_campaign_id():
    """
    Delivery-driven:
      - 5 impressions at t=10s
      - viewable_share=0.6 -> make exactly 3 viewable, 2 not viewable
      - freq_cap_per_user_per_day=2 -> force one chosen user to exceed cap, reducing exposures to 2
      - ctr=1 and incremental=1 -> click + intent for every exposure
      - campaign_id propagated in payload_json
    """
    env = simpy.Environment()
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    website_graph = FakeWebsiteGraph(env=env, start_dt=start_dt)
    persistence = FakePersistence(events=[])
    intents = simpy.Store(env)
    users_state = FakeUsersState(users={"u1": {}, "u2": {}})

    cfg = ChannelConfig(
        name="paid_display",
        exposure_rate_per_user_per_day=0.0,
        click_through_rate=1.0,
        incremental_intent=True,
        params={
            "viewable_share": 0.6,
            "freq_cap_per_user_per_day": 2,
            "incremental_click_share": 1.0,
        },
    )
    channel = build_paid_display(cfg)

    def provider(day_idx: int, day_start_s: float, seconds_per_day: float):
        return [
            DeliveryPlan(
                channel="paid_display",
                slices=[
                    DeliverySlice(
                        at_s=10.0,
                        impressions=5,
                        campaign_id="disp1",
                        ctr=1.0,
                        incremental_click_share=1.0,
                    )
                ],
            )
        ]

    # For each impression (PaidDisplay delivery mode):
    # - pick user_id (rng)
    # - viewability draw (rng) -> pass if < 0.6
    #
    # For each scheduled exposure at execution:
    # - click draw (rng)
    # - incremental draw (rng)
    #
    # Construct:
    # 5 impressions:
    #  1) pick u1, viewable pass
    #  2) pick u1, viewable pass
    #  3) pick u1, viewable pass BUT cap=2 means this one gets dropped at scheduling time (after cap check)
    #  4) pick u2, viewable fail
    #  5) pick u2, viewable fail
    #
    # Then 2 exposures execute: each consumes click+incremental.
    rng = PredefinedRNG(
        values=[
            # imp1
            0.1,
            0.1,
            # imp2
            0.2,
            0.2,
            # imp3 (would be viewable but cap blocks; still consumes pick+viewable before cap check? In our code cap is checked before viewable.
            # IMPORTANT: our code checks cap before viewable, so we must make this pick u1 then see cap triggers and CONTINUE
            # without consuming viewability draw. Therefore provide only the pick value here:
            0.3,
            # imp4
            0.8,
            0.9,
            # imp5
            0.9,
            0.95,
            # execution for 2 exposures
            0.0,
            0.0,
            0.0,
            0.0,
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
            assert e["payload_json"] == {"campaign_id": "disp1"}
