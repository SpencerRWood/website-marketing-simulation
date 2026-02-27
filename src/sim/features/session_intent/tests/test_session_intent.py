from __future__ import annotations

from datetime import UTC, datetime

import simpy


class DummyIds:
    def __init__(self) -> None:
        self._n = 0

    def next_id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n:06d}"


def test_publish_and_get_roundtrip() -> None:
    from sim.features.session_intent.service import SessionIntentService

    env = simpy.Environment()
    ids = DummyIds()
    bus = SessionIntentService(env=env, ids=ids, capacity=None)

    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    got = {}

    def producer():
        yield bus.publish_new(
            ts_utc=t0,
            intent_source="baseline",
            payload={"k": "v"},
        )

    def consumer():
        intent = yield bus.get()
        got["intent"] = intent

    env.process(producer())
    env.process(consumer())
    env.run()

    intent = got["intent"]
    assert intent.intent_id == "intent_000001"
    assert intent.ts_utc == t0
    assert intent.intent_source == "baseline"
    assert intent.channel is None
    assert intent.audience_id is None
    assert intent.payload == {"k": "v"}


def test_capacity_blocks_if_full() -> None:
    from sim.features.session_intent.service import SessionIntentService

    env = simpy.Environment()
    ids = DummyIds()
    bus = SessionIntentService(env=env, ids=ids, capacity=1)

    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    done = {"second_put_done": False}

    def producer():
        # first fills capacity
        yield bus.publish_new(ts_utc=t0, intent_source="baseline")
        # second should block until a consumer drains
        yield bus.publish_new(ts_utc=t0, intent_source="baseline")
        done["second_put_done"] = True

    def consumer():
        # wait a bit then drain one item
        yield env.timeout(1)
        _ = yield bus.get()

    env.process(producer())
    env.process(consumer())

    env.run(until=0.5)
    assert done["second_put_done"] is False  # blocked

    env.run(until=2.0)
    assert done["second_put_done"] is True
