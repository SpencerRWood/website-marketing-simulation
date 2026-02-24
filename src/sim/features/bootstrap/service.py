from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import simpy

from sim.core.config import SimulationConfig
from sim.core.ids import deterministic_run_id_from_config
from sim.core.logging import get_logger
from sim.core.rng import RNG
from sim.core.types import RunContext

EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS events (
  run_id        VARCHAR,
  event_id      VARCHAR,
  ts_utc        TIMESTAMP,
  sim_time_s    DOUBLE,
  user_id       VARCHAR,
  session_id    VARCHAR,
  event_type    VARCHAR,
  intent_source VARCHAR,
  channel       VARCHAR,
  page          VARCHAR,
  value_num     DOUBLE,
  value_str     VARCHAR,
  payload_json  VARCHAR
);
"""


@dataclass(frozen=True)
class BootstrapResult:
    ctx: RunContext
    duckdb_path: str


def _clean_slate_if_needed(path: Path, clean_slate: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if clean_slate and path.exists():
        path.unlink()


def _emit_event(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    event_id: str,
    ts_utc: datetime,
    sim_time_s: float,
    event_type: str,
) -> None:
    con.execute(
        """
        INSERT INTO events (
          run_id, event_id, ts_utc, sim_time_s, user_id, session_id, event_type,
          intent_source, channel, page, value_num, value_str, payload_json
        )
        VALUES (?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, NULL, NULL, NULL, NULL)
        """,
        [run_id, event_id, ts_utc, sim_time_s, event_type],
    )


def bootstrap_run(cfg: SimulationConfig, config_path: str | None = None) -> BootstrapResult:
    # --- run_id policy ---
    if cfg.run.run_id == "auto":
        run_id = deterministic_run_id_from_config(cfg.raw)
    else:
        run_id = cfg.run.run_id

    logger = get_logger("sim", cfg.logging.level)
    rng = RNG(cfg.run.seed)

    # authoritative UTC start time derived from config start_date at midnight UTC
    start_dt_utc = datetime.fromisoformat(cfg.run.start_date).replace(tzinfo=UTC)
    ctx = RunContext(run_id=run_id, seed=cfg.run.seed, start_dt_utc=start_dt_utc)

    db_path = Path(cfg.storage.duckdb_path)
    _clean_slate_if_needed(db_path, cfg.storage.clean_slate)

    con = duckdb.connect(str(db_path))
    con.execute(EVENTS_DDL)

    # event ids: deterministic sequence for bootstrap
    # (later you can replace with a central id generator feature)
    def eid(n: int) -> str:
        return f"evt_{n:08d}"

    env = simpy.Environment()

    logger.info(
        "bootstrap_start",
        extra={"run_id": ctx.run_id, "feature": "bootstrap", "event_type": "run_started"},
    )
    _emit_event(
        con,
        run_id=ctx.run_id,
        event_id=eid(1),
        ts_utc=start_dt_utc,
        sim_time_s=float(env.now),
        event_type="run_started",
    )

    # run empty sim for configured horizon
    horizon_s = int(cfg.run.num_days) * 24 * 60 * 60
    env.run(until=horizon_s)

    end_ts = start_dt_utc.replace() + (
        datetime.fromtimestamp(horizon_s, tz=UTC) - datetime.fromtimestamp(0, tz=UTC)
    )

    _emit_event(
        con,
        run_id=ctx.run_id,
        event_id=eid(2),
        ts_utc=end_ts,
        sim_time_s=float(env.now),
        event_type="run_finished",
    )
    logger.info(
        "bootstrap_finish",
        extra={"run_id": ctx.run_id, "feature": "bootstrap", "event_type": "run_finished"},
    )

    con.close()
    _ = rng  # placeholder: ensures rng instantiated; used by later features

    return BootstrapResult(ctx=ctx, duckdb_path=str(db_path))
