from __future__ import annotations

from sim.core.config import load_config
from sim.features.bootstrap.service import BootstrapResult, bootstrap_run


def run(config_path: str) -> BootstrapResult:
    cfg = load_config(config_path)
    return bootstrap_run(cfg, config_path=config_path)
