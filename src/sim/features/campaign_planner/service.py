from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

# ----------------------------
# Config dataclasses
# ----------------------------

_ALLOWED_PACING = {"uniform", "front_loaded", "back_loaded"}


def _parse_date(s: str) -> date:
    # Expect "YYYY-MM-DD"
    return date.fromisoformat(s)


def _dow_str_to_int(s: str) -> int:
    # Python weekday(): Monday=0 ... Sunday=6
    m = {
        "mon": 0,
        "tue": 1,
        "wed": 2,
        "thu": 3,
        "fri": 4,
        "sat": 5,
        "sun": 6,
    }
    key = s.strip().lower()
    if key not in m:
        raise ValueError(f"Invalid day_of_week '{s}'. Use: {sorted(m.keys())}")
    return m[key]


def _floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class CampaignActivation:
    days_of_week: tuple[int, ...] | None = None  # 0..6
    hours: tuple[int, ...] | None = None  # 0..23

    def is_active_hour(self, dt_hour_utc: datetime) -> bool:
        if dt_hour_utc.tzinfo is None:
            raise ValueError("dt_hour_utc must be timezone-aware (UTC).")

        if self.days_of_week is not None and dt_hour_utc.weekday() not in self.days_of_week:
            return False

        if self.hours is not None and dt_hour_utc.hour not in self.hours:
            return False

        return True


@dataclass(frozen=True)
class CampaignSpec:
    name: str
    channel: str
    start_date: date
    end_date: date  # inclusive date
    pacing: str = "uniform"  # uniform | front_loaded | back_loaded
    total_budget: float | None = None
    daily_budget: float | None = None
    activation: CampaignActivation = CampaignActivation()

    def validate(self) -> None:
        if self.pacing not in _ALLOWED_PACING:
            raise ValueError(
                f"campaign '{self.name}' pacing must be one of {_ALLOWED_PACING}; got '{self.pacing}'"
            )

        if self.end_date < self.start_date:
            raise ValueError(f"campaign '{self.name}' end_date < start_date")

        if (self.total_budget is None) == (self.daily_budget is None):
            raise ValueError(
                f"campaign '{self.name}' must set exactly one of total_budget or daily_budget"
            )

        if self.total_budget is not None and self.total_budget < 0:
            raise ValueError(f"campaign '{self.name}' total_budget must be >= 0")

        if self.daily_budget is not None and self.daily_budget < 0:
            raise ValueError(f"campaign '{self.name}' daily_budget must be >= 0")


@dataclass(frozen=True)
class CampaignPlannerConfig:
    enabled: bool = False
    time_resolution: str = "hour"  # currently hour-only (day not implemented)
    default_currency: str = "USD"
    campaigns: tuple[CampaignSpec, ...] = ()

    def validate(self) -> None:
        if self.time_resolution != "hour":
            raise ValueError("campaign_planner currently supports time_resolution: 'hour' only")
        for c in self.campaigns:
            c.validate()


# ----------------------------
# Service
# ----------------------------


class CampaignPlannerService:
    """
    Compiles a deterministic spend schedule into hourly UTC buckets.

    Output is raw spend ($) per channel per hour bucket.
    This is intended to be consumed by adstock/saturation (to produce multipliers),
    and then by channels-exposure (to scale exposure rates / CTR).
    """

    def __init__(self, start_dt_utc: datetime, cfg: CampaignPlannerConfig):
        self.start_dt_utc = _ensure_utc(start_dt_utc)
        self.cfg = cfg
        self.cfg.validate()

        # channel -> {hour_bucket_start_dt_utc: spend_amount_usd_for_that_hour}
        self._spend_buckets: dict[str, dict[datetime, float]] = {}
        if self.cfg.enabled:
            self._spend_buckets = self._compile_buckets(self.cfg.campaigns)

    @staticmethod
    def from_raw_config(*, start_dt_utc: datetime, raw: dict[str, Any]) -> CampaignPlannerService:
        sec = raw.get("campaigns", {}) or {}
        enabled = bool(sec.get("enabled", False))
        time_resolution = str(sec.get("time_resolution", "hour"))
        default_currency = str(sec.get("default_currency", "USD"))

        campaigns_raw = sec.get("campaigns", []) or []
        campaigns: list[CampaignSpec] = []
        for idx, item in enumerate(campaigns_raw):
            if not isinstance(item, dict):
                raise TypeError(f"campaigns.campaigns[{idx}] must be a mapping")

            name = str(item.get("name"))
            channel = str(item.get("channel"))
            start_date = _parse_date(str(item.get("start_date")))
            end_date = _parse_date(str(item.get("end_date")))
            pacing = str(item.get("pacing", "uniform"))

            total_budget = item.get("total_budget", None)
            daily_budget = item.get("daily_budget", None)
            total_budget_f = float(total_budget) if total_budget is not None else None
            daily_budget_f = float(daily_budget) if daily_budget is not None else None

            act_raw = item.get("activation", {}) or {}
            days_of_week_raw = act_raw.get("days_of_week", None)
            hours_raw = act_raw.get("hours", None)

            days_of_week: tuple[int, ...] | None = None
            if days_of_week_raw is not None:
                if not isinstance(days_of_week_raw, list):
                    raise TypeError(f"campaign '{name}' activation.days_of_week must be a list")
                days_of_week = tuple(_dow_str_to_int(x) for x in days_of_week_raw)

            hours: tuple[int, ...] | None = None
            if hours_raw is not None:
                if not isinstance(hours_raw, list):
                    raise TypeError(f"campaign '{name}' activation.hours must be a list")
                hrs: list[int] = []
                for h in hours_raw:
                    hh = int(h)
                    if hh < 0 or hh > 23:
                        raise ValueError(
                            f"campaign '{name}' activation.hours contains invalid hour {hh}"
                        )
                    hrs.append(hh)
                hours = tuple(hrs)

            activation = CampaignActivation(days_of_week=days_of_week, hours=hours)

            campaigns.append(
                CampaignSpec(
                    name=name,
                    channel=channel,
                    start_date=start_date,
                    end_date=end_date,
                    pacing=pacing,
                    total_budget=total_budget_f,
                    daily_budget=daily_budget_f,
                    activation=activation,
                )
            )

        cfg = CampaignPlannerConfig(
            enabled=enabled,
            time_resolution=time_resolution,
            default_currency=default_currency,
            campaigns=tuple(campaigns),
        )
        return CampaignPlannerService(start_dt_utc=_ensure_utc(start_dt_utc), cfg=cfg)

    def raw_spend_per_hour(self, channel: str, t_utc: datetime) -> float:
        """
        Returns USD/hour for the hour bucket containing t_utc.
        """
        if not self.cfg.enabled:
            return 0.0
        t_utc = _ensure_utc(t_utc)
        bucket = _floor_to_hour(t_utc)
        return float(self._spend_buckets.get(channel, {}).get(bucket, 0.0))

    def raw_spend_between(self, channel: str, t0_utc: datetime, t1_utc: datetime) -> float:
        """
        Integrate spend over [t0_utc, t1_utc) using hourly buckets.
        Assumes spend within a bucket is uniform across the hour.
        """
        if not self.cfg.enabled:
            return 0.0

        t0 = _ensure_utc(t0_utc)
        t1 = _ensure_utc(t1_utc)
        if t1 <= t0:
            return 0.0

        buckets = self._spend_buckets.get(channel, {})
        if not buckets:
            return 0.0

        cur = _floor_to_hour(t0)
        end = _floor_to_hour(t1)
        total = 0.0

        # handle first partial bucket
        while cur <= end:
            bucket_start = cur
            bucket_end = bucket_start + timedelta(hours=1)

            overlap_start = max(t0, bucket_start)
            overlap_end = min(t1, bucket_end)
            if overlap_end > overlap_start:
                frac = (overlap_end - overlap_start).total_seconds() / 3600.0
                total += buckets.get(bucket_start, 0.0) * frac

            cur += timedelta(hours=1)
        return float(total)

    # ----------------------------
    # Compilation
    # ----------------------------

    def _compile_buckets(
        self, campaigns: Iterable[CampaignSpec]
    ) -> dict[str, dict[datetime, float]]:
        out: dict[str, dict[datetime, float]] = {}
        for c in campaigns:
            c.validate()
            ch = c.channel
            out.setdefault(ch, {})

            if c.total_budget is not None:
                self._apply_total_budget_campaign(out[ch], c)
            else:
                self._apply_daily_budget_campaign(out[ch], c)

        return out

    def _campaign_window_hours(self, c: CampaignSpec) -> list[datetime]:
        """
        Enumerate all UTC hour bucket starts from start_date 00:00 through (end_date+1) 00:00.
        Filter by activation.
        """
        start_dt = datetime.combine(c.start_date, time(0, 0), tzinfo=UTC)
        end_excl = datetime.combine(c.end_date + timedelta(days=1), time(0, 0), tzinfo=UTC)

        hours: list[datetime] = []
        cur = start_dt
        while cur < end_excl:
            if c.activation.is_active_hour(cur):
                hours.append(cur)
            cur += timedelta(hours=1)
        return hours

    def _weights(self, n: int, pacing: str) -> list[float]:
        if n <= 0:
            return []
        if pacing == "uniform":
            return [1.0] * n

        # linear ramp (deterministic)
        # front_loaded: higher weights early; back_loaded: higher weights late
        if n == 1:
            return [1.0]

        if pacing == "front_loaded":
            # from 1.0 down to 0.2
            hi = 1.0
            lo = 0.2
            return [hi - (hi - lo) * (i / (n - 1)) for i in range(n)]

        if pacing == "back_loaded":
            hi = 1.0
            lo = 0.2
            return [lo + (hi - lo) * (i / (n - 1)) for i in range(n)]

        raise ValueError(f"Unsupported pacing '{pacing}'")

    def _apply_total_budget_campaign(
        self, channel_buckets: dict[datetime, float], c: CampaignSpec
    ) -> None:
        hours = self._campaign_window_hours(c)
        if not hours or c.total_budget is None:
            return

        w = self._weights(len(hours), c.pacing)
        s = sum(w)
        if s <= 0:
            return

        for dt_hr, wi in zip(hours, w, strict=True):
            amt = float(c.total_budget) * (wi / s)
            channel_buckets[dt_hr] = channel_buckets.get(dt_hr, 0.0) + amt

    def _apply_daily_budget_campaign(
        self, channel_buckets: dict[datetime, float], c: CampaignSpec
    ) -> None:
        if c.daily_budget is None:
            return

        # iterate each day, distribute within that day's active hours
        cur_day = c.start_date
        end_day = c.end_date
        while cur_day <= end_day:
            # collect active hours in this day
            day_start = datetime.combine(cur_day, time(0, 0), tzinfo=UTC)
            day_hours: list[datetime] = []
            for h in range(24):
                dt_hr = day_start + timedelta(hours=h)
                if c.activation.is_active_hour(dt_hr):
                    day_hours.append(dt_hr)

            if day_hours:
                w = self._weights(len(day_hours), c.pacing)
                s = sum(w)
                if s > 0:
                    for dt_hr, wi in zip(day_hours, w, strict=True):
                        amt = float(c.daily_budget) * (wi / s)
                        channel_buckets[dt_hr] = channel_buckets.get(dt_hr, 0.0) + amt

            cur_day += timedelta(days=1)
