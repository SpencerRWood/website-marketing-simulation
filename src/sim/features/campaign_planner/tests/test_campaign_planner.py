from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sim.features.campaign_planner.service import CampaignPlannerService


def _mk_service(raw):
    # simulation start_dt doesn't have to equal campaign dates; it just anchors the sim clock elsewhere
    start_dt = datetime(2026, 1, 1, tzinfo=UTC)
    return CampaignPlannerService.from_raw_config(start_dt_utc=start_dt, raw=raw)


def test_disabled_returns_zero():
    svc = _mk_service({"campaigns": {"enabled": False, "campaigns": []}})
    t = datetime(2026, 1, 10, 12, tzinfo=UTC)
    assert svc.raw_spend_per_hour("search", t) == 0.0
    assert svc.raw_spend_between("search", t, t + timedelta(hours=3)) == 0.0


def test_total_budget_uniform_sums_to_total():
    raw = {
        "campaigns": {
            "enabled": True,
            "time_resolution": "hour",
            "campaigns": [
                {
                    "name": "c1",
                    "channel": "search",
                    "start_date": "2026-01-05",
                    "end_date": "2026-01-06",  # inclusive (2 days)
                    "total_budget": 480.0,
                    "pacing": "uniform",
                }
            ],
        }
    }
    svc = _mk_service(raw)

    # Total window: 2 days * 24 hours = 48 hours; uniform => $10/hour
    t0 = datetime(2026, 1, 5, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 7, 0, tzinfo=UTC)
    total = svc.raw_spend_between("search", t0, t1)
    assert total == pytest.approx(480.0, rel=1e-12)

    # Spot check: one hour bucket
    assert svc.raw_spend_per_hour(
        "search", datetime(2026, 1, 5, 13, 42, tzinfo=UTC)
    ) == pytest.approx(10.0)


def test_activation_hours_limits_spend():
    raw = {
        "campaigns": {
            "enabled": True,
            "time_resolution": "hour",
            "campaigns": [
                {
                    "name": "c_hours",
                    "channel": "social",
                    "start_date": "2026-01-05",
                    "end_date": "2026-01-05",
                    "total_budget": 120.0,
                    "pacing": "uniform",
                    "activation": {"hours": [8, 9, 10, 11]},  # 4 active hours
                }
            ],
        }
    }
    svc = _mk_service(raw)

    # Should spend only within those 4 hours, => $30/hour
    for hr in [8, 9, 10, 11]:
        t = datetime(2026, 1, 5, hr, 15, tzinfo=UTC)
        assert svc.raw_spend_per_hour("social", t) == pytest.approx(30.0)

    for hr in [0, 7, 12, 23]:
        t = datetime(2026, 1, 5, hr, 15, tzinfo=UTC)
        assert svc.raw_spend_per_hour("social", t) == 0.0

    t0 = datetime(2026, 1, 5, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 6, 0, tzinfo=UTC)
    assert svc.raw_spend_between("social", t0, t1) == pytest.approx(120.0)


def test_daily_budget_applies_each_day():
    raw = {
        "campaigns": {
            "enabled": True,
            "time_resolution": "hour",
            "campaigns": [
                {
                    "name": "c_daily",
                    "channel": "search",
                    "start_date": "2026-01-05",
                    "end_date": "2026-01-07",  # 3 days
                    "daily_budget": 100.0,
                    "pacing": "uniform",
                    "activation": {"hours": [12, 13, 14, 15, 16]},  # 5 hours/day
                }
            ],
        }
    }
    svc = _mk_service(raw)

    # 3 days * $100/day = $300 total
    t0 = datetime(2026, 1, 5, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 8, 0, tzinfo=UTC)
    assert svc.raw_spend_between("search", t0, t1) == pytest.approx(300.0)

    # per active hour: $100 / 5 = $20
    assert svc.raw_spend_per_hour(
        "search", datetime(2026, 1, 6, 12, 30, tzinfo=UTC)
    ) == pytest.approx(20.0)
    assert svc.raw_spend_per_hour("search", datetime(2026, 1, 6, 9, 30, tzinfo=UTC)) == 0.0


def test_front_loaded_biases_early_hours():
    raw = {
        "campaigns": {
            "enabled": True,
            "time_resolution": "hour",
            "campaigns": [
                {
                    "name": "c_front",
                    "channel": "display",
                    "start_date": "2026-01-05",
                    "end_date": "2026-01-05",
                    "total_budget": 240.0,
                    "pacing": "front_loaded",
                }
            ],
        }
    }
    svc = _mk_service(raw)

    # Compare first vs last hour of the day
    first = svc.raw_spend_per_hour("display", datetime(2026, 1, 5, 0, 10, tzinfo=UTC))
    last = svc.raw_spend_per_hour("display", datetime(2026, 1, 5, 23, 10, tzinfo=UTC))
    assert first > last

    # Total should still equal the budget
    t0 = datetime(2026, 1, 5, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 6, 0, tzinfo=UTC)
    assert svc.raw_spend_between("display", t0, t1) == pytest.approx(240.0)


def test_invalid_budget_spec_raises():
    raw = {
        "campaigns": {
            "enabled": True,
            "time_resolution": "hour",
            "campaigns": [
                {
                    "name": "bad",
                    "channel": "search",
                    "start_date": "2026-01-05",
                    "end_date": "2026-01-05",
                    # both set => invalid
                    "total_budget": 100.0,
                    "daily_budget": 10.0,
                }
            ],
        }
    }
    with pytest.raises(ValueError):
        _mk_service(raw)
