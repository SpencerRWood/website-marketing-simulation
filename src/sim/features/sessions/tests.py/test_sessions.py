from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import simpy

from sim.features.conversion.service import ConversionConfig, ConversionService
from sim.features.sessions.service import (
    InterPageTimeConfig,
    SessionsConfig,
    SessionsService,
)

# ----------------------------
# Stubs
# ----------------------------


class DummyEvents:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def emit(self, event_type: str, **kwargs: Any) -> None:
        self.rows.append({"event_type": event_type, **kwargs})


class DummyRng:
    """Provide a deterministic stream of random() values."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.idx = 0

    def random(self) -> float:
        if self.idx >= len(self.values):
            return 0.999
        v = float(self.values[self.idx])
        self.idx += 1
        return v

    def expovariate(self, lambd: float) -> float:
        return 0.0


@dataclass
class Page:
    dropoff_p: float
    transitions: list[tuple[str, float]]


class DummyGraph:
    def __init__(self, env: simpy.Environment, pages: dict[str, Page], start_dt: datetime) -> None:
        self.env = env
        self.pages = pages
        self.start_dt = start_dt

    def get_current_time(self) -> datetime:
        return self.start_dt + timedelta(seconds=float(self.env.now))

    def get_page(self, name: str) -> Page | None:
        return self.pages.get(name)

    def next_page(self, current_name: str, rng: Any) -> str | None:
        page = self.get_page(current_name)
        if not page or not page.transitions:
            return None
        # deterministic: pick max weight
        return sorted(page.transitions, key=lambda t: t[1], reverse=True)[0][0]


# ----------------------------
# Tests
# ----------------------------


def test_session_emits_start_pageviews_end_no_next_page() -> None:
    env = simpy.Environment()

    pages = {
        "home": Page(dropoff_p=0.0, transitions=[("product", 1.0)]),
        "product": Page(dropoff_p=0.0, transitions=[]),
    }
    graph = DummyGraph(env=env, pages=pages, start_dt=datetime(2026, 1, 1, tzinfo=UTC))

    rng = DummyRng(values=[0.5, 0.5, 0.5])
    events = DummyEvents()

    cfg = SessionsConfig(
        inactivity_timeout_minutes=30,
        max_steps=12,
        entry_page="home",
        inter_page_time=InterPageTimeConfig(dist="fixed", fixed_seconds=0.0),
    )

    svc = SessionsService(env=env, graph=graph, rng=rng, events=events, cfg=cfg)

    svc.spawn(user_id="u1", session_id="s1", intent_source="baseline", channel="direct")
    env.run()

    types = [r["event_type"] for r in events.rows]
    assert types[0] == "session_start"
    assert types.count("page_view") == 2
    assert types[-1] == "session_end"

    end = events.rows[-1]
    assert end["value_str"] == "no_next_page"


def test_session_dropoff_emits_drop_off_and_end() -> None:
    env = simpy.Environment()

    pages = {"home": Page(dropoff_p=1.0, transitions=[("product", 1.0)])}
    graph = DummyGraph(env=env, pages=pages, start_dt=datetime(2026, 1, 1, tzinfo=UTC))

    rng = DummyRng(values=[0.0])  # triggers dropoff
    events = DummyEvents()

    cfg = SessionsConfig(
        inactivity_timeout_minutes=30,
        max_steps=12,
        entry_page="home",
        inter_page_time=InterPageTimeConfig(dist="fixed", fixed_seconds=0.0),
    )

    svc = SessionsService(env=env, graph=graph, rng=rng, events=events, cfg=cfg)

    svc.spawn(user_id="u1", session_id="s1", intent_source="baseline", channel="direct")
    env.run()

    types = [r["event_type"] for r in events.rows]
    assert "drop_off" in types
    assert types[-1] == "session_end"


def test_session_timeout_end_reason() -> None:
    env = simpy.Environment()

    pages = {
        "home": Page(dropoff_p=0.0, transitions=[("product", 1.0)]),
        "product": Page(dropoff_p=0.0, transitions=[]),
    }
    graph = DummyGraph(env=env, pages=pages, start_dt=datetime(2026, 1, 1, tzinfo=UTC))

    rng = DummyRng(values=[0.5, 0.5, 0.5])
    events = DummyEvents()

    cfg = SessionsConfig(
        inactivity_timeout_minutes=0.01,  # 0.6s
        max_steps=None,
        entry_page="home",
        inter_page_time=InterPageTimeConfig(dist="fixed", fixed_seconds=10.0),  # exceeds timeout
    )

    svc = SessionsService(env=env, graph=graph, rng=rng, events=events, cfg=cfg)

    svc.spawn(user_id="u1", session_id="s1", intent_source="baseline", channel="direct")
    env.run()

    assert events.rows[-1]["event_type"] == "session_end"
    assert events.rows[-1]["value_str"] == "timeout"


def test_session_conversion_emits_conversion() -> None:
    env = simpy.Environment()

    pages = {"home": Page(dropoff_p=0.0, transitions=[])}
    graph = DummyGraph(env=env, pages=pages, start_dt=datetime(2026, 1, 1, tzinfo=UTC))

    rng = DummyRng(values=[0.5, 0.0])  # no dropoff, conversion draw -> convert if p>0
    events = DummyEvents()

    conversion = ConversionService(
        ConversionConfig(model="logistic", cap=1.0, base_logit=10.0, propensity_coef=0.0)
    )

    cfg = SessionsConfig(
        inactivity_timeout_minutes=30,
        max_steps=12,
        entry_page="home",
        inter_page_time=InterPageTimeConfig(dist="fixed", fixed_seconds=0.0),
    )

    svc = SessionsService(env=env, graph=graph, rng=rng, events=events, cfg=cfg)

    svc.spawn(
        user_id="u1",
        session_id="s1",
        intent_source="baseline",
        channel="direct",
        conversion=conversion,
        user_propensity=0.5,
    )
    env.run()

    types = [r["event_type"] for r in events.rows]
    assert "conversion" in types
