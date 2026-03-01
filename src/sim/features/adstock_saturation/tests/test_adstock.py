from __future__ import annotations

from sim.features.adstock_saturation.transforms import (
    apply_geometric_adstock,
    geometric_decay_from_half_life,
)
from sim.features.adstock_saturation.types import AdstockConfig


def test_geometric_decay_half_life_identity_points():
    # dt=0 => decay=1
    assert geometric_decay_from_half_life(half_life_days=5.0, dt_days=0.0) == 1.0

    # half_life<=0 => no carryover
    assert geometric_decay_from_half_life(half_life_days=0.0, dt_days=1.0) == 0.0
    assert geometric_decay_from_half_life(half_life_days=-1.0, dt_days=1.0) == 0.0

    # dt=half_life => decay=0.5
    d = geometric_decay_from_half_life(half_life_days=4.0, dt_days=4.0)
    assert abs(d - 0.5) < 1e-12

    # dt=2*half_life => decay=0.25
    d2 = geometric_decay_from_half_life(half_life_days=4.0, dt_days=8.0)
    assert abs(d2 - 0.25) < 1e-12


def test_apply_geometric_adstock_recursion():
    cfg = AdstockConfig(model="geometric", half_life_days=2.0)
    dt = 1.0
    decay = geometric_decay_from_half_life(half_life_days=2.0, dt_days=1.0)

    # step 1: carry=0, spend=10 => out=10
    c1, a1, d1 = apply_geometric_adstock(spend=10.0, carryover_prev=0.0, cfg=cfg, dt_days=dt)
    assert abs(d1 - decay) < 1e-12
    assert abs(c1 - 10.0) < 1e-12
    assert abs(a1 - 10.0) < 1e-12

    # step 2: spend=0 => out=decay*10
    c2, a2, _ = apply_geometric_adstock(spend=0.0, carryover_prev=c1, cfg=cfg, dt_days=dt)
    assert abs(c2 - (decay * 10.0)) < 1e-12
    assert abs(a2 - c2) < 1e-12

    # step 3: spend=5 => out=decay*c2 + 5
    c3, a3, _ = apply_geometric_adstock(spend=5.0, carryover_prev=c2, cfg=cfg, dt_days=dt)
    assert abs(c3 - (decay * c2 + 5.0)) < 1e-12
    assert abs(a3 - c3) < 1e-12
