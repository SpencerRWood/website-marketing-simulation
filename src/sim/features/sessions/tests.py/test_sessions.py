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


class DummyIds:
    def __init__(self) -> None:
        self._i = 0

    def new_event_id(self) -> str:
        self._i += 1
        return f"evt_{self._i:04d}"


class DummyEvents:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def emit(self, **kwargs: Any) -> None:
        self.rows.append(dict(kwargs))


class DummyRng:
    """
    Provide a deterministic stream of random() values.
    """

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
    ids = DummyIds()

    cfg = SessionsConfig(
        inactivity_timeout_minutes=30,
        max_steps=12,
        entry_page="home",
        inter_page_time=InterPageTimeConfig(dist="fixed", fixed_seconds=0.0),
    )

    svc = SessionsService(
        env=env,
        graph=graph,
        rng=rng,
        events=events,
        ids=ids,
        run_id="run_test",
        cfg=cfg,
    )

    svc.spawn(user_id="u1", session_id="s1", intent_source="baseline")
    env.run()

    types = [r["event_type"] for r in events.rows]
    assert types[0] == "session_start"
    assert "page_view" in types
    assert types[-1] == "session_end"
    assert events.rows[-1]["value_str"] == "no_next_page"

    pages_seen = [r["page"] for r in events.rows if r["event_type"] == "page_view"]
    assert pages_seen == ["home", "product"]


def test_session_dropoff_emits_drop_off_and_end() -> None:
    env = simpy.Environment()

    pages = {
        "home": Page(dropoff_p=1.0, transitions=[("product", 1.0)]),
        "product": Page(dropoff_p=0.0, transitions=[]),
    }
    graph = DummyGraph(env=env, pages=pages, start_dt=datetime(2026, 1, 1, tzinfo=UTC))

    rng = DummyRng(values=[0.0])
    events = DummyEvents()
    ids = DummyIds()

    cfg = SessionsConfig(
        inactivity_timeout_minutes=30,
        max_steps=12,
        entry_page="home",
        inter_page_time=InterPageTimeConfig(dist="fixed", fixed_seconds=0.0),
    )

    svc = SessionsService(
        env=env,
        graph=graph,
        rng=rng,
        events=events,
        ids=ids,
        run_id="run_test",
        cfg=cfg,
    )

    svc.spawn(user_id="u1", session_id="s1", intent_source="baseline")
    env.run()

    types = [r["event_type"] for r in events.rows]
    assert "drop_off" in types
    assert types[-1] == "session_end"
    assert events.rows[-1]["value_str"] == "drop_off"


def test_session_timeout_when_interpage_delay_exceeds_timeout() -> None:
    env = simpy.Environment()

    pages = {
        "home": Page(dropoff_p=0.0, transitions=[("product", 1.0)]),
        "product": Page(dropoff_p=0.0, transitions=[]),
    }
    graph = DummyGraph(env=env, pages=pages, start_dt=datetime(2026, 1, 1, tzinfo=UTC))

    rng = DummyRng(values=[0.999])
    events = DummyEvents()
    ids = DummyIds()

    cfg = SessionsConfig(
        inactivity_timeout_minutes=1.0,
        max_steps=12,
        entry_page="home",
        inter_page_time=InterPageTimeConfig(dist="fixed", fixed_seconds=999.0),
    )

    svc = SessionsService(
        env=env,
        graph=graph,
        rng=rng,
        events=events,
        ids=ids,
        run_id="run_test",
        cfg=cfg,
    )

    svc.spawn(user_id="u1", session_id="s1", intent_source="baseline")
    env.run()

    assert events.rows[-1]["event_type"] == "session_end"
    assert events.rows[-1]["value_str"] == "timeout"
    assert env.now == 60.0


def test_session_conversion_emits_conversion_and_ends() -> None:
    env = simpy.Environment()

    pages = {
        "home": Page(dropoff_p=0.0, transitions=[("product", 1.0)]),
        "product": Page(dropoff_p=0.0, transitions=[]),
    }
    graph = DummyGraph(env=env, pages=pages, start_dt=datetime(2026, 1, 1, tzinfo=UTC))

    # random() stream:
    #  - dropoff check (avoid): 0.999
    #  - conversion check (convert): 0.0
    rng = DummyRng(values=[0.999, 0.0])
    events = DummyEvents()
    ids = DummyIds()

    cfg = SessionsConfig(
        inactivity_timeout_minutes=30,
        max_steps=12,
        entry_page="home",
        inter_page_time=InterPageTimeConfig(dist="fixed", fixed_seconds=0.0),
    )

    conv = ConversionService(
        ConversionConfig(model="logistic", cap=1.0, base_logit=10.0, propensity_coef=0.0)
    )

    svc = SessionsService(
        env=env,
        graph=graph,
        rng=rng,
        events=events,
        ids=ids,
        run_id="run_test",
        cfg=cfg,
    )

    svc.spawn(
        user_id="u1",
        session_id="s1",
        intent_source="baseline",
        conversion=conv,
        user_propensity=0.5,
    )
    env.run()

    types = [r["event_type"] for r in events.rows]
    assert "conversion" in types
    assert types[-1] == "session_end"
    assert events.rows[-1]["value_str"] == "conversion"
