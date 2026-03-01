from __future__ import annotations

from sim.features.adstock_saturation.transforms import apply_hill_saturation
from sim.features.adstock_saturation.types import SaturationConfig


def test_hill_saturation_basic_properties():
    cfg = SaturationConfig(model="hill", alpha=1.5, gamma=10.0, max_effect=1.0)

    assert apply_hill_saturation(x=0.0, cfg=cfg) == 0.0
    assert apply_hill_saturation(x=-1.0, cfg=cfg) == 0.0

    y1 = apply_hill_saturation(x=1.0, cfg=cfg)
    y2 = apply_hill_saturation(x=2.0, cfg=cfg)
    y3 = apply_hill_saturation(x=5.0, cfg=cfg)
    y4 = apply_hill_saturation(x=50.0, cfg=cfg)

    assert 0.0 < y1 < y2 < y3 < y4 < 1.0


def test_hill_saturation_max_effect_and_gamma_edge_cases():
    # gamma <= 0 => immediate saturation at max_effect for x>0
    cfg = SaturationConfig(model="hill", alpha=1.0, gamma=0.0, max_effect=3.0)
    assert apply_hill_saturation(x=1.0, cfg=cfg) == 3.0
    assert apply_hill_saturation(x=100.0, cfg=cfg) == 3.0

    # alpha<=0 gets stabilized to 1
    cfg2 = SaturationConfig(model="hill", alpha=0.0, gamma=10.0, max_effect=1.0)
    y = apply_hill_saturation(x=10.0, cfg=cfg2)
    assert 0.0 < y < 1.0
