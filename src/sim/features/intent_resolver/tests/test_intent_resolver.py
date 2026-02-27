from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import simpy

from sim.features.intent_resolver.service import IntentResolverService
from sim.features.intent_resolver.types import IntentResolverConfig
from sim.features.session_intent.service import SessionIntentService

# -----------------------
# Test doubles
# -----------------------


@dataclass
class DummyIds:
    n: int = 0

    def next_id(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}_{self.n:08d}"


@dataclass
class _DummyUser:
    user_id: str


@dataclass
class DummyUsers:
    users: list[str] | None = None
    created: int = 0

    def __post_init__(self) -> None:
        if self.users is None:
            self.users = []

    def num_users(self) -> int:
        return len(self.users)

    def get_or_create_user_for_intent(self, *, now_utc: datetime, rng: Any):
        if not self.users:
            self.created += 1
            uid = f"user_{self.created}"
            self.users.append(uid)
            return _DummyUser(uid), True

        idx = int(rng.random() * len(self.users))
        if idx == len(self.users):
            idx -= 1
        return _DummyUser(self.users[idx]), False


@dataclass
class RecordingSink:
    events: list[dict[str, Any]]

    def emit(self, **kwargs) -> None:
        self.events.append(dict(kwargs))


@dataclass
class RecordingSessionRunner:
    started: list[tuple[str, str, Any]]

    def start_session(self, *, user_id: str, session_id: str, intent):
        self.started.append((user_id, session_id, intent))
        # No actual process required for these tests
        return None


# -----------------------
# Tests
# -----------------------


def test_resolver_creates_user_when_empty_and_spawns_session() -> None:
    env = simpy.Environment()
    ids = DummyIds()
    rng = random.Random(123)

    intents = SessionIntentService(env=env, ids=ids, capacity=None)
    users = DummyUsers()
    sink = RecordingSink(events=[])
    runner = RecordingSessionRunner(started=[])

    resolver = IntentResolverService(
        env=env,
        cfg=IntentResolverConfig(enabled=True),
        ids=ids,
        rng=rng,
        intents=intents,
        users=users,
        session_runner=runner,
        sink=sink,
    )
    resolver.start()

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    intents.publish_new(ts_utc=t0, intent_source="baseline")
    env.run(until=1)

    assert users.num_users() == 1
    assert len(runner.started) == 1
    assert len(sink.events) >= 2  # session_intent + session_start (and possibly user_created)


def test_resolver_selects_existing_when_new_user_share_zero() -> None:
    env = simpy.Environment()
    ids = DummyIds()
    rng = random.Random(42)

    intents = SessionIntentService(env=env, ids=ids, capacity=None)
    users = DummyUsers(users=["user_a", "user_b"])
    sink = RecordingSink(events=[])
    runner = RecordingSessionRunner(started=[])

    resolver = IntentResolverService(
        env=env,
        cfg=IntentResolverConfig(enabled=True),
        ids=ids,
        rng=rng,
        intents=intents,
        users=users,
        session_runner=runner,
        sink=sink,
    )
    resolver.start()

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    intents.publish_new(ts_utc=t0, intent_source="channel:search", channel="search")
    env.run(until=1)

    assert users.num_users() == 2
    assert len(runner.started) == 1
    # picked from existing pool
    assert runner.started[0][0] in {"user_a", "user_b"}


def test_resolver_disabled_emits_nothing_and_spawns_nothing() -> None:
    env = simpy.Environment()
    ids = DummyIds()
    rng = random.Random(123)

    intents = SessionIntentService(env=env, ids=ids, capacity=None)
    users = DummyUsers(users=["user_a"])
    sink = RecordingSink(events=[])
    runner = RecordingSessionRunner(started=[])

    resolver = IntentResolverService(
        env=env,
        cfg=IntentResolverConfig(enabled=False),
        ids=ids,
        rng=rng,
        intents=intents,
        users=users,
        session_runner=runner,
        sink=sink,
    )
    resolver.start()

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    intents.publish_new(ts_utc=t0, intent_source="baseline")
    env.run(until=1)

    assert sink.events == []
    assert runner.started == []
