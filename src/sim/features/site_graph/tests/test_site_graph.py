from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sim.core.rng import RNG
from sim.features.site_graph.service import SiteGraphFactory


class DummyEnv:
    def __init__(self, now: float = 0.0):
        self.now = now


def _cfg_basic() -> dict:
    return {
        "pages": {
            "home": {
                "dropoff_p": 0.25,
                "transitions": [["product", 0.55], ["pricing", 0.25]],
            },
            "product": {
                "dropoff_p": 0.15,
                "transitions": [["pricing", 0.35]],
            },
            "pricing": {
                "dropoff_p": 0.10,
                "transitions": [],
            },
        }
    }


def test_factory_builds_pages_and_fields() -> None:
    env = DummyEnv()
    rng = RNG(seed=123)
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    g = SiteGraphFactory().build(env=env, cfg_site_graph=_cfg_basic(), start_dt=start_dt, rng=rng)

    assert g.get_page("home") is not None
    assert g.get_page("home").dropoff_p == pytest.approx(0.25)  # type: ignore
    assert g.get_page("home").transitions == [("product", 0.55), ("pricing", 0.25)]  # type: ignore
    assert g.get_page("missing") is None


def test_strict_mode_unknown_transition_target_raises() -> None:
    env = DummyEnv()
    rng = RNG(seed=1)
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    cfg = {
        "pages": {
            "home": {"dropoff_p": 0.1, "transitions": [["does_not_exist", 1.0]]},
        }
    }

    with pytest.raises(ValueError, match="unknown page"):
        SiteGraphFactory(strict=True).build(env=env, cfg_site_graph=cfg, start_dt=start_dt, rng=rng)


def test_dropoff_bounds_validation() -> None:
    env = DummyEnv()
    rng = RNG(seed=1)
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    cfg = {"pages": {"home": {"dropoff_p": 1.5}}}
    with pytest.raises(ValueError, match="dropoff_p must be in \\[0, 1\\]"):
        SiteGraphFactory().build(env=env, cfg_site_graph=cfg, start_dt=start_dt, rng=rng)


def test_negative_transition_weight_rejected() -> None:
    env = DummyEnv()
    rng = RNG(seed=1)
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    cfg = {"pages": {"home": {"dropoff_p": 0.1, "transitions": [["product", -0.1]]}}}
    with pytest.raises(ValueError, match="weight must be >= 0"):
        SiteGraphFactory().build(env=env, cfg_site_graph=cfg, start_dt=start_dt, rng=rng)


def test_transitions_must_be_list_of_pairs() -> None:
    env = DummyEnv()
    rng = RNG(seed=1)
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    cfg = {"pages": {"home": {"dropoff_p": 0.1, "transitions": "nope"}}}
    with pytest.raises(TypeError, match="transitions must be a list"):
        SiteGraphFactory().build(env=env, cfg_site_graph=cfg, start_dt=start_dt, rng=rng)

    cfg2 = {"pages": {"home": {"dropoff_p": 0.1, "transitions": [["product"]]}}}
    with pytest.raises(TypeError, match="must be a 2-item list/tuple"):
        SiteGraphFactory().build(env=env, cfg_site_graph=cfg2, start_dt=start_dt, rng=rng)


def test_next_page_deterministic_with_seeded_rng() -> None:
    env = DummyEnv()
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    cfg = {
        "pages": {
            "home": {
                "dropoff_p": 0.0,
                "transitions": [["a", 1.0], ["b", 3.0], ["c", 6.0]],
            },
            "a": {"dropoff_p": 0.0, "transitions": []},
            "b": {"dropoff_p": 0.0, "transitions": []},
            "c": {"dropoff_p": 0.0, "transitions": []},
        }
    }

    g1 = SiteGraphFactory().build(env=env, cfg_site_graph=cfg, start_dt=start_dt, rng=RNG(seed=42))
    g2 = SiteGraphFactory().build(env=env, cfg_site_graph=cfg, start_dt=start_dt, rng=RNG(seed=42))

    seq1 = [g1.next_page("home") for _ in range(25)]
    seq2 = [g2.next_page("home") for _ in range(25)]

    assert seq1 == seq2
    assert all(x in {"a", "b", "c"} for x in seq1)


def test_next_page_zero_sum_weights_returns_none() -> None:
    env = DummyEnv()
    rng = RNG(seed=1)
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)

    cfg = {
        "pages": {
            "home": {"dropoff_p": 0.0, "transitions": [["a", 0.0], ["b", 0.0]]},
            "a": {"dropoff_p": 0.0, "transitions": []},
            "b": {"dropoff_p": 0.0, "transitions": []},
        }
    }

    g = SiteGraphFactory().build(env=env, cfg_site_graph=cfg, start_dt=start_dt, rng=rng)

    assert g.next_page("home") is None


def test_get_current_time_is_utc_and_matches_env_now_seconds() -> None:
    env = DummyEnv(now=0.0)
    rng = RNG(seed=1)

    # Naive start_dt should be coerced to UTC by WebsiteGraph
    start_dt_naive = datetime(2026, 1, 1, 12, 0, 0)  # naive

    g = SiteGraphFactory().build(
        env=env, cfg_site_graph=_cfg_basic(), start_dt=start_dt_naive, rng=rng
    )

    t0 = g.get_current_time()
    assert t0.tzinfo is not None
    assert t0.tzinfo == UTC
    # naive was interpreted as UTC in your spec
    assert t0 == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    env.now = 90.5
    t1 = g.get_current_time()
    assert t1 == t0 + timedelta(seconds=90.5)
