from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sim.features.users_state.service import (
    DiscoveryModeConfig,
    PropensityInitConfig,
    UsersConfig,
    UsersSelectionConfig,
    UsersStateService,
)


class DummyRNG:
    """
    Deterministic minimal RNG for tests.
    - random() cycles through a fixed sequence
    - choices() uses cumulative weights
    """

    def __init__(self, seq):
        self.seq = list(seq)
        self.i = 0

    def random(self):
        v = self.seq[self.i % len(self.seq)]
        self.i += 1
        return float(v)

    def choices(self, population, weights, k=1):
        assert k == 1
        total = sum(weights)
        if total <= 0:
            # uniform fallback
            idx = int(self.random() * len(population))
            return [population[idx]]
        r = self.random() * total
        c = 0.0
        for item, w in zip(population, weights, strict=False):
            c += w
            if r <= c:
                return [item]
        return [population[-1]]


UTC = UTC


def test_creates_user_when_empty():
    cfg = UsersConfig(new_user_share=0.0)
    svc = UsersStateService(cfg)
    rng = DummyRNG([0.99])

    now = datetime(2026, 1, 1, tzinfo=UTC)
    u, is_new = svc.get_or_create_user_for_intent(now_utc=now, rng=rng)

    assert is_new is True
    assert u.user_id.startswith("u_")
    assert svc.get_user(u.user_id) is not None
    assert u.created_ts_utc == now
    assert u.last_seen_ts_utc == now


def test_new_user_share_one_always_creates():
    cfg = UsersConfig(new_user_share=1.0)
    svc = UsersStateService(cfg)
    rng = DummyRNG([0.2, 0.2, 0.2])

    now = datetime(2026, 1, 1, tzinfo=UTC)
    u1, n1 = svc.get_or_create_user_for_intent(now_utc=now, rng=rng)
    u2, n2 = svc.get_or_create_user_for_intent(now_utc=now, rng=rng)

    assert n1 is True and n2 is True
    assert u1.user_id != u2.user_id
    assert len(list(svc.all_users())) == 2


def test_new_user_share_zero_selects_existing_when_available():
    cfg = UsersConfig(new_user_share=0.0)
    svc = UsersStateService(cfg)

    # First call creates because empty; second should select existing (since p_new=0)
    rng = DummyRNG([0.99, 0.01])  # first random used for propensity init; second used for selection

    now = datetime(2026, 1, 1, tzinfo=UTC)
    u1, n1 = svc.get_or_create_user_for_intent(now_utc=now, rng=rng)
    u2, n2 = svc.get_or_create_user_for_intent(now_utc=now, rng=rng)

    assert n1 is True
    assert n2 is False
    assert u2.user_id == u1.user_id
    assert len(list(svc.all_users())) == 1


def test_recency_weighting_prefers_recent_user():
    cfg = UsersConfig(
        new_user_share=1.0,  # <-- change: ensure we can create 2 users
        selection=UsersSelectionConfig(
            mode="recency_propensity_weighted",
            recency_half_life_hours=1.0,
            recency_weight=1.0,
            propensity_weight=0.0,
        ),
        propensity_init=PropensityInitConfig(dist="uniform"),
        discovery_mode=DiscoveryModeConfig(enabled=False),
    )
    svc = UsersStateService(cfg)

    rng = DummyRNG([0.5] * 1000)

    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    u_old, _ = svc.get_or_create_user_for_intent(now_utc=t0, rng=rng)
    u_new, _ = svc.get_or_create_user_for_intent(now_utc=t0, rng=rng)

    assert u_old.user_id != u_new.user_id

    # Make u_old stale by 4 hours; keep u_new fresh
    svc.get_user(u_old.user_id).last_seen_ts_utc = t0 - timedelta(hours=4)
    svc.get_user(u_new.user_id).last_seen_ts_utc = t0

    picks = {u_old.user_id: 0, u_new.user_id: 0}
    for _ in range(200):
        u = svc.select_existing_user(now_utc=t0, rng=rng)
        picks[u.user_id] += 1

    assert picks[u_new.user_id] > picks[u_old.user_id]
    assert picks[u_new.user_id] >= 180


def test_discovery_graduation_after_n_sessions():
    cfg = UsersConfig(
        new_user_share=1.0,
        discovery_mode=DiscoveryModeConfig(enabled=True, graduation_sessions=2),
    )
    svc = UsersStateService(cfg)
    rng = DummyRNG([0.7, 0.7, 0.7])

    now = datetime(2026, 1, 1, tzinfo=UTC)
    u, is_new = svc.get_or_create_user_for_intent(now_utc=now, rng=rng)
    assert is_new is True
    assert u.discovery_mode is True

    svc.mark_session_end(user_id=u.user_id, now_utc=now + timedelta(minutes=10))
    assert svc.get_user(u.user_id).discovery_mode is True

    svc.mark_session_end(user_id=u.user_id, now_utc=now + timedelta(minutes=20))
    assert svc.get_user(u.user_id).discovery_mode is False
