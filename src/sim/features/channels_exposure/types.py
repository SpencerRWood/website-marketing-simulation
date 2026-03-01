from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChannelConfig:
    name: str
    exposure_rate_per_user_per_day: float
    click_through_rate: float
    incremental_intent: bool = True
    params: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChannelsExposureConfig:
    enabled: bool = True


# ----------------------------
# Delivery-driven (campaign-fed)
# ----------------------------
@dataclass(frozen=True)
class DeliverySlice:
    """
    A time-indexed delivery instruction for a channel.
    `impressions` here means "opportunities to expose" (channel-specific interpretation).
    """

    at_s: float  # absolute sim seconds since sim start
    impressions: int
    campaign_id: str | None = None

    # Optional overrides for this slice:
    ctr: float | None = None
    incremental_click_share: float | None = None
    params: dict[str, Any] | None = None


@dataclass(frozen=True)
class DeliveryPlan:
    channel: str
    slices: list[DeliverySlice]


@dataclass(frozen=True)
class SessionIntent:
    intent_source: str
    channel: str | None
    ts_utc: Any
    sim_time_s: float
