from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import simpy

from sim.core.config import SimulationConfig
from sim.core.ids import deterministic_run_id_from_config
from sim.core.logging import get_logger
from sim.core.rng import RNG
from sim.core.types import RunContext
from sim.features.persistence.duckdb_adapter import DuckDBAdapter
from sim.features.persistence.service import Event, PersistenceService


@dataclass(frozen=True)
class BootstrapResult:
    ctx: RunContext
    duckdb_path: str


def bootstrap_run(cfg: SimulationConfig, config_path: str | None = None) -> BootstrapResult:
    if cfg.run.run_id == "auto":
        run_id = deterministic_run_id_from_config(cfg.raw)
    else:
        run_id = cfg.run.run_id

    logger = get_logger("sim", cfg.logging.level)
    rng = RNG(cfg.run.seed)

    start_dt_utc = datetime.fromisoformat(cfg.run.start_date).replace(tzinfo=UTC)
    ctx = RunContext(run_id=run_id, seed=cfg.run.seed, start_dt_utc=start_dt_utc)

    def eid(n: int) -> str:
        return f"evt_{n:08d}"

    env = simpy.Environment()

    adapter = DuckDBAdapter(
        path=cfg.storage.duckdb_path,
        clean_slate=cfg.storage.clean_slate,
    )
    persistence = PersistenceService(
        adapter=adapter,
        every_n_events=cfg.storage.flush.every_n_events,
        or_every_seconds=cfg.storage.flush.or_every_seconds,
    )

    persistence.open()
    persistence.start_periodic_flush(env)

    try:
        logger.info(
            "bootstrap_start",
            extra={"run_id": ctx.run_id, "feature": "bootstrap", "event_type": "run_started"},
        )
        persistence.emit(
            Event(
                run_id=ctx.run_id,
                event_id=eid(1),
                ts_utc=start_dt_utc,
                sim_time_s=float(env.now),
                event_type="run_started",
            )
        )

        horizon_s = int(cfg.run.num_days) * 24 * 60 * 60
        env.run(until=horizon_s)

        end_ts = start_dt_utc + timedelta(seconds=horizon_s)

        persistence.emit(
            Event(
                run_id=ctx.run_id,
                event_id=eid(2),
                ts_utc=end_ts,
                sim_time_s=float(env.now),
                event_type="run_finished",
            )
        )
        logger.info(
            "bootstrap_finish",
            extra={"run_id": ctx.run_id, "feature": "bootstrap", "event_type": "run_finished"},
        )
    finally:
        persistence.close()

    _ = rng
    return BootstrapResult(ctx=ctx, duckdb_path=cfg.storage.duckdb_path)
