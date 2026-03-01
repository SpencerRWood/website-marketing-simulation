from __future__ import annotations

from sim.features.adstock_saturation.types import AdstockConfig, SaturationConfig


def geometric_decay_from_half_life(*, half_life_days: float, dt_days: float) -> float:
    """
    Convert half-life into a decay factor for a given time step.

    If half_life_days <= 0 => decay = 0 (no carryover).
    If dt_days <= 0 => decay = 1 (no time elapsed).
    """
    if dt_days <= 0:
        return 1.0
    if half_life_days <= 0:
        return 0.0
    return float(0.5 ** (float(dt_days) / float(half_life_days)))


def apply_geometric_adstock(
    *,
    spend: float,
    carryover_prev: float,
    cfg: AdstockConfig,
    dt_days: float,
) -> tuple[float, float, float]:
    """
    Geometric adstock:
      carryover_out = decay * carryover_prev + spend
      adstocked = carryover_out
    Returns (carryover_out, adstocked, decay)
    """
    s = float(spend)
    c_prev = float(carryover_prev)
    decay = geometric_decay_from_half_life(
        half_life_days=float(cfg.half_life_days), dt_days=float(dt_days)
    )
    carryover_out = decay * c_prev + s
    adstocked = carryover_out
    return carryover_out, adstocked, decay


def apply_hill_saturation(*, x: float, cfg: SaturationConfig) -> float:
    """
    Hill saturation:
      response = max_effect * x^alpha / (x^alpha + gamma^alpha)

    Notes:
      - If x <= 0 => response = 0
      - If gamma <= 0 => response = max_effect (immediate saturation)
      - alpha <= 0 is treated as alpha=1.0 for stability
    """
    xin = float(x)
    if xin <= 0.0:
        return 0.0

    alpha = float(cfg.alpha)
    if alpha <= 0.0:
        alpha = 1.0

    gamma = float(cfg.gamma)
    max_effect = float(cfg.max_effect)

    if gamma <= 0.0:
        return max_effect

    xa = xin**alpha
    ga = gamma**alpha
    return max_effect * (xa / (xa + ga))
