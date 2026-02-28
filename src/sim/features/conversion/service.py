from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class RNGLike(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True)
class ConversionConfig:
    """
    Logistic conversion model:

      p = sigmoid(base_logit + propensity_coef * propensity + logit_shift)
      p = min(p, cap)

    propensity is expected in [0, 1], but we clamp defensively.
    """

    model: str = "logistic"
    cap: float = 0.35
    base_logit: float = -3.0
    propensity_coef: float = 2.0


class ConversionService:
    def __init__(self, cfg: ConversionConfig) -> None:
        model = (cfg.model or "").strip().lower()
        if model != "logistic":
            raise ValueError(f"Unsupported conversion.model: {cfg.model!r}")
        if not (0.0 <= float(cfg.cap) <= 1.0):
            raise ValueError("conversion.cap must be in [0, 1]")
        self.cfg = cfg

    @staticmethod
    def _sigmoid(x: float) -> float:
        # Numerically stable sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def probability(self, *, propensity: float, logit_shift: float = 0.0) -> float:
        p = float(propensity)
        p = 0.0 if p < 0.0 else 1.0 if p > 1.0 else p

        logit = (
            float(self.cfg.base_logit) + float(self.cfg.propensity_coef) * p + float(logit_shift)
        )
        prob = self._sigmoid(logit)
        cap = float(self.cfg.cap)
        if prob > cap:
            prob = cap
        return 0.0 if prob < 0.0 else 1.0 if prob > 1.0 else prob

    def should_convert(
        self, *, propensity: float, logit_shift: float = 0.0, rng: RNGLike
    ) -> tuple[bool, float]:
        prob = self.probability(propensity=propensity, logit_shift=logit_shift)
        u = float(rng.random())
        return (u < prob), prob
