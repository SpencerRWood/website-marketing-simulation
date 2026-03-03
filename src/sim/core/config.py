from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    seed: int
    start_date: str
    num_days: int


@dataclass(frozen=True)
class FlushConfig:
    every_n_events: int = 5000
    or_every_seconds: float = 30.0


@dataclass(frozen=True)
class StorageConfig:
    duckdb_path: str
    clean_slate: bool = True
    flush: FlushConfig = field(default_factory=FlushConfig)


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class SimulationConfig:
    run: RunConfig
    storage: StorageConfig
    logging: LoggingConfig
    raw: dict[str, Any]


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config YAML must parse to a dict at the top level.")
    return data


def parse_config(data: dict[str, Any]) -> SimulationConfig:
    # --- required top-level sections ---
    for key in ["run", "storage", "logging"]:
        if key not in data:
            raise ValueError(f"Missing required top-level config section: '{key}'")

    run = data.get("run") or {}
    storage = data.get("storage") or {}
    logging_cfg = data.get("logging") or {}

    # --- run ---
    run_cfg = RunConfig(
        run_id=str(run.get("run_id", "auto")),
        seed=int(run["seed"]),
        start_date=str(run["start_date"]),
        num_days=int(run["num_days"]),
    )

    # --- storage.flush (THIS WAS MISSING) ---
    flush_raw = storage.get("flush") or {}
    flush_cfg = FlushConfig(
        every_n_events=int(flush_raw.get("every_n_events", FlushConfig.every_n_events)),
        or_every_seconds=float(flush_raw.get("or_every_seconds", FlushConfig.or_every_seconds)),
    )

    storage_cfg = StorageConfig(
        duckdb_path=str(storage["duckdb_path"]),
        clean_slate=bool(storage.get("clean_slate", True)),
        flush=flush_cfg,
    )

    # --- logging ---
    log_cfg = LoggingConfig(level=str(logging_cfg.get("level", "INFO")).upper())

    return SimulationConfig(run=run_cfg, storage=storage_cfg, logging=log_cfg, raw=data)


def load_config(path: str | Path) -> SimulationConfig:
    data = load_yaml(path)
    return parse_config(data)
