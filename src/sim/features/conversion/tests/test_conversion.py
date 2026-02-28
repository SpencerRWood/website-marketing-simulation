from __future__ import annotations

from sim.features.conversion.service import ConversionConfig, ConversionService


class DummyRng:
    def __init__(self, u: float) -> None:
        self._u = float(u)

    def random(self) -> float:
        return self._u


def test_probability_monotone_in_propensity() -> None:
    svc = ConversionService(
        ConversionConfig(model="logistic", cap=1.0, base_logit=-2.0, propensity_coef=4.0)
    )
    p0 = svc.probability(propensity=0.0)
    p1 = svc.probability(propensity=1.0)
    assert p1 > p0


def test_cap_applied() -> None:
    svc = ConversionService(
        ConversionConfig(model="logistic", cap=0.10, base_logit=10.0, propensity_coef=0.0)
    )
    assert abs(svc.probability(propensity=0.5) - 0.10) < 1e-12


def test_logit_shift_reduces_probability() -> None:
    svc = ConversionService(
        ConversionConfig(model="logistic", cap=1.0, base_logit=-2.0, propensity_coef=0.0)
    )
    base = svc.probability(propensity=0.5, logit_shift=0.0)
    shifted = svc.probability(propensity=0.5, logit_shift=-1.0)
    assert shifted < base


def test_should_convert_uses_rng_threshold() -> None:
    svc = ConversionService(
        ConversionConfig(model="logistic", cap=1.0, base_logit=0.0, propensity_coef=0.0)
    )
    # sigmoid(0)=0.5
    did, prob = svc.should_convert(propensity=0.5, rng=DummyRng(0.49))
    assert prob == 0.5
    assert did is True

    did, _ = svc.should_convert(propensity=0.5, rng=DummyRng(0.50))
    assert did is False
