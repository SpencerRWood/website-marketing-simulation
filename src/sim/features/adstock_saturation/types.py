from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AdstockModel = Literal["geometric"]
SaturationModel = Literal["hill"]


@dataclass(frozen=True)
class AdstockConfig:
    model: AdstockModel = "geometric"
    # Half-life measured in DAYS for carryover decay.
    # If <= 0, adstock is treated as "no memory" (i.e., carryover=0 each step).
    half_life_days: float = 0.0


@dataclass(frozen=True)
class SaturationConfig:
    model: SaturationModel = "hill"
    # Hill parameters:
    #   response = max_effect * x^alpha / (x^alpha + gamma^alpha)
    # where x is the adstocked spend (or adstocked input)
    alpha: float = 1.0
    gamma: float = 1.0
    # Optional scale for response (defaults to 1.0)
    max_effect: float = 1.0


@dataclass(frozen=True)
class ChannelTransformConfig:
    channel: str
    adstock: AdstockConfig
    saturation: SaturationConfig


@dataclass(frozen=True)
class AdstockSaturationConfig:
    enabled: bool = True
    # If True, state key includes campaign_id; else state is per channel only.
    per_campaign_state: bool = False
    channels: dict[str, ChannelTransformConfig] | None = None


@dataclass(frozen=True)
class AdstockStepResult:
    channel: str
    campaign_id: str | None
    dt_days: float

    spend_in: float
    carryover_in: float
    decay: float
    carryover_out: float
    adstocked: float


@dataclass(frozen=True)
class SaturationStepResult:
    channel: str
    campaign_id: str | None

    x_in: float
    response: float


@dataclass(frozen=True)
class TransformStepResult:
    adstock: AdstockStepResult
    saturation: SaturationStepResult
