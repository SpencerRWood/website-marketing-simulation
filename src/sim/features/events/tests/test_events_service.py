from __future__ import annotations

from datetime import UTC, datetime

import pytest
import simpy

from sim.core.rng import RNG

UTC = UTC
rng = RNG(seed=123)


class DummySink:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def append(self, row: dict) -> None:
        self.rows.append(row)


def make_graph(env: simpy.Environment):
    from sim.features.site_graph.service import WebsiteGraph

    return WebsiteGraph(env=env, rng=rng, pages={}, start_dt=datetime(2026, 1, 1, tzinfo=UTC))


def test_emit_persists_row_and_ids():
    from sim.features.events.service import CounterEventIdGenerator, EventService

    env = simpy.Environment()
    graph = make_graph(env)
    sink = DummySink()
    ids = CounterEventIdGenerator(run_id="run_x")

    svc = EventService(env=env, graph=graph, persistence=sink, ids=ids, run_id="run_x")

    env.run(until=12.0)

    evt = svc.emit(
        event_type="session_start",
        user_id="u1",
        session_id="s1",
        channel="direct",
        payload={"foo": "bar"},
    )

    assert evt.event_id == "run_x_00000001"
    assert len(sink.rows) == 1

    row = sink.rows[0]
    assert row["run_id"] == "run_x"
    assert row["event_type"] == "session_start"
    assert row["user_id"] == "u1"
    assert row["session_id"] == "s1"
    assert row["channel"] == "direct"
    assert row["sim_time_s"] == 12.0
    assert row["payload_json"] == '{"foo":"bar"}'
    assert row["ts_utc"].tzinfo is not None


def test_invalid_event_type_raises():
    from sim.features.events.service import CounterEventIdGenerator, EventService

    env = simpy.Environment()
    graph = make_graph(env)
    sink = DummySink()
    ids = CounterEventIdGenerator(run_id="run_x")

    svc = EventService(env=env, graph=graph, persistence=sink, ids=ids, run_id="run_x")

    with pytest.raises(ValueError):
        svc.emit(event_type="user_created", user_id="u1")


def test_session_events_require_session_id():
    from sim.features.events.service import CounterEventIdGenerator, EventService

    env = simpy.Environment()
    graph = make_graph(env)
    sink = DummySink()
    ids = CounterEventIdGenerator(run_id="run_x")
    svc = EventService(env=env, graph=graph, persistence=sink, ids=ids, run_id="run_x")

    with pytest.raises(ValueError):
        svc.emit(event_type="page_view", user_id="u1", page="home", channel="direct")


def test_channel_required_when_session_id_set():
    from sim.features.events.service import CounterEventIdGenerator, EventService

    env = simpy.Environment()
    graph = make_graph(env)
    sink = DummySink()
    ids = CounterEventIdGenerator(run_id="run_x")
    svc = EventService(env=env, graph=graph, persistence=sink, ids=ids, run_id="run_x")

    with pytest.raises(ValueError):
        svc.emit(event_type="session_start", user_id="u1", session_id="s1", channel=None)


def test_payload_json_is_stable():
    from sim.features.events.service import CounterEventIdGenerator, EventService

    env = simpy.Environment()
    graph = make_graph(env)
    sink = DummySink()
    ids = CounterEventIdGenerator(run_id="run_x")
    svc = EventService(env=env, graph=graph, persistence=sink, ids=ids, run_id="run_x")

    svc.emit(
        event_type="session_end",
        user_id="u1",
        session_id="s1",
        channel="direct",
        payload={"end_reason": "timeout", "b": 2, "a": 1},
    )

    row = sink.rows[0]
    assert row["payload_json"] == '{"a":1,"b":2,"end_reason":"timeout"}'


def test_deterministic_ids_increment():
    from sim.features.events.service import CounterEventIdGenerator, EventService

    env = simpy.Environment()
    graph = make_graph(env)
    sink = DummySink()
    ids = CounterEventIdGenerator(run_id="run_det")
    svc = EventService(env=env, graph=graph, persistence=sink, ids=ids, run_id="run_det")

    svc.emit(event_type="session_start", user_id="u1", session_id="s1", channel="search")
    svc.emit(event_type="page_view", user_id="u1", session_id="s1", channel="search", page="home")
    svc.emit(
        event_type="session_end",
        user_id="u1",
        session_id="s1",
        channel="search",
        payload={"end_reason": "drop_off"},
    )

    assert [r["event_id"] for r in sink.rows] == [
        "run_det_00000001",
        "run_det_00000002",
        "run_det_00000003",
    ]
