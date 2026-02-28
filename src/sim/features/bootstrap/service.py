from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import simpy

from sim.core.config import SimulationConfig
from sim.core.ids import IdsService, deterministic_run_id_from_config
from sim.core.logging import get_logger
from sim.core.rng import RNG
from sim.core.types import RunContext
from sim.features.arrivals.models.nhpp import GaussianPeakCurveConfig
from sim.features.arrivals.service import ArrivalsService, BaselineArrivalsConfig
from sim.features.intent_resolver.service import IntentResolverService
from sim.features.intent_resolver.types import IntentResolverConfig
from sim.features.persistence.duckdb_adapter import DuckDBAdapter
from sim.features.persistence.service import Event, PersistenceService
from sim.features.session_intent.service import SessionIntentService
from sim.features.users_state.service import UsersConfig, UsersStateService


@dataclass(frozen=True)
class BootstrapResult:
    ctx: RunContext
    duckdb_path: str


def _path_exists(d: dict[str, Any], path: list[str]) -> bool:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def _get_path(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def bootstrap_run(cfg: SimulationConfig, config_path: str | None = None) -> BootstrapResult:
    # ----- run identity -----
    if cfg.run.run_id == "auto":
        run_id = deterministic_run_id_from_config(cfg.raw)
    else:
        run_id = cfg.run.run_id

    logger = get_logger("sim", cfg.logging.level)

    # Single seeded RNG (no global random)
    rng = RNG(cfg.run.seed)

    # Single IDs service (shared across features)
    ids = IdsService(run_id=run_id)

    start_dt_utc = datetime.fromisoformat(cfg.run.start_date).replace(tzinfo=UTC)
    ctx = RunContext(run_id=run_id, seed=cfg.run.seed, start_dt_utc=start_dt_utc)

    env = simpy.Environment()

    # ----- cold storage -----
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

    # ----- minimal sink adapter (resolver/session -> persistence) -----
    class PersistenceEventSink:
        def __init__(
            self, *, ctx: RunContext, ids: IdsService, persistence: PersistenceService
        ) -> None:
            self._ctx = ctx
            self._ids = ids
            self._p = persistence

        def emit(
            self,
            *,
            ts_utc: datetime,
            sim_time_s: float,
            event_type: str,
            user_id: str | None = None,
            session_id: str | None = None,
            intent_id: str | None = None,
            intent_source: str | None = None,
            channel: str | None = None,
            audience_id: str | None = None,
            payload: dict | None = None,
        ) -> None:
            self._p.emit(
                Event(
                    run_id=self._ctx.run_id,
                    event_id=self._ids.next_id("evt"),
                    ts_utc=ts_utc,
                    sim_time_s=sim_time_s,
                    event_type=event_type,
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=intent_source,
                    channel=channel,
                )
            )

    sink = PersistenceEventSink(ctx=ctx, ids=ids, persistence=persistence)

    # ----------------------------------------------------------------------------------
    # IMPORTANT: Keep bootstrap "minimal by default".
    # The bootstrap tests expect only run_started + run_finished in the events table
    # unless the config explicitly enables simulation activity.
    # ----------------------------------------------------------------------------------
    raw = cfg.raw if isinstance(cfg.raw, dict) else {}
    has_users = "users" in raw
    has_intent_resolver = "intent_resolver" in raw
    has_baseline_arrivals = _path_exists(raw, ["arrivals", "baseline_arrivals"])

    enable_activity = has_users or has_intent_resolver or has_baseline_arrivals

    if enable_activity:
        # ----- session-intent bus -----
        session_intents = SessionIntentService(env=env, ids=ids, capacity=None)

        # Adapter: arrivals publishes SessionIntent objects; SessionIntentService expects publish_new(...)
        class SessionIntentBusAdapter:
            def __init__(self, bus: SessionIntentService) -> None:
                self._bus = bus

            def publish(self, intent) -> None:
                """
                Be tolerant to evolving SessionIntentService signatures.
                """
                # Prefer passing intent_id/channel if supported; fall back gracefully.
                try:
                    self._bus.publish_new(
                        ts_utc=intent.ts_utc,
                        intent_source=intent.intent_source,
                        channel=intent.channel,
                        intent_id=intent.intent_id,
                    )
                    return
                except TypeError:
                    pass

                try:
                    self._bus.publish_new(
                        ts_utc=intent.ts_utc,
                        intent_source=intent.intent_source,
                        channel=intent.channel,
                    )
                    return
                except TypeError:
                    pass

                # Minimal required shape
                self._bus.publish_new(ts_utc=intent.ts_utc, intent_source=intent.intent_source)

        intent_bus = SessionIntentBusAdapter(session_intents)

        # ----- users hot state (REAL) -----
        users_cfg = getattr(cfg, "users", None)
        if users_cfg is None:
            users_cfg = UsersConfig()
        users = UsersStateService(cfg=users_cfg)

        # ----- downstream session runner (stub until feat/sessions) -----
        class NoopSessionRunner:
            def __init__(
                self, env: simpy.Environment, sink: PersistenceEventSink, users: UsersStateService
            ) -> None:
                self._env = env
                self._sink = sink
                self._users = users

            def start_session(
                self, *, user_id: str, session_id: str, intent
            ) -> simpy.events.Process:
                return self._env.process(
                    self._run(user_id=user_id, session_id=session_id, intent=intent)
                )

            def _run(self, *, user_id: str, session_id: str, intent):
                now_utc = intent.ts_utc
                self._sink.emit(
                    ts_utc=now_utc,
                    sim_time_s=float(self._env.now),
                    event_type="session_end",
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=getattr(intent, "intent_source", None),
                    channel=getattr(intent, "channel", None),
                )
                self._users.mark_session_end(user_id=user_id, now_utc=now_utc)
                yield self._env.timeout(0)

        session_runner = NoopSessionRunner(env=env, sink=sink, users=users)

        # ----- intent resolver -----
        resolver_cfg = IntentResolverConfig(
            enabled=getattr(getattr(cfg, "intent_resolver", None), "enabled", True)
        )
        resolver = IntentResolverService(
            env=env,
            cfg=resolver_cfg,
            ids=ids,
            rng=rng,
            intents=session_intents,
            users=users,
            session_runner=session_runner,
            sink=sink,
        )
        resolver.start()

        # ----- baseline arrivals (NHPP) -----
        # Minimal graph/clock adapter: arrivals only needs get_current_time()
        class ClockGraph:
            def __init__(self, env: simpy.Environment, start_dt_utc: datetime) -> None:
                self._env = env
                self._start = start_dt_utc

            def get_current_time(self) -> datetime:
                return self._start + timedelta(seconds=float(self._env.now))

        if has_baseline_arrivals:
            b = _get_path(raw, ["arrivals", "baseline_arrivals"], default={}) or {}

            # Defaults are intentionally conservative; if you omit baseline_arrivals entirely,
            # bootstrap stays minimal (enable_activity stays False).
            model = str(b.get("model", "nhpp"))
            daily_expected_intents = float(b.get("daily_expected_intents", 0.0))

            curve = b.get("intraday_curve", {}) or {}
            peak_hour = float(curve.get("peak_hour", 12.0))
            spread_hours = float(curve.get("spread_hours", 3.0))
            floor = float(curve.get("floor", 0.05))

            arrivals = ArrivalsService(
                run_id=ctx.run_id,
                rng=rng,
                ids=ids,
                graph=ClockGraph(env=env, start_dt_utc=start_dt_utc),
                intent_bus=intent_bus,
                baseline_arrivals=BaselineArrivalsConfig(
                    model=model,
                    daily_expected_intents=daily_expected_intents,
                    intraday_curve=GaussianPeakCurveConfig(
                        peak_hour=peak_hour,
                        spread_hours=spread_hours,
                        floor=floor,
                    ),
                ),
                num_days=int(cfg.run.num_days),
                events=sink,  # sink has .emit(...) compatible with arrivals EventsLike
            )
            arrivals.start(env)

    # ----- run lifecycle -----
    try:
        logger.info(
            "bootstrap_start",
            extra={"run_id": ctx.run_id, "feature": "bootstrap", "event_type": "run_started"},
        )
        persistence.emit(
            Event(
                run_id=ctx.run_id,
                event_id=ids.next_id("evt"),
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
                event_id=ids.next_id("evt"),
                ts_utc=end_ts,
                sim_time_s=float(env.now),
                event_type="run_finished",
            )
        )
        logger.info(
            "bootstrap_finish",
            extra={"run_id": ctx.run_id, "feature": "bootstrap", "event_type": "run_finished"},
        )

        # Ensure run_finished isn't stranded
        persistence.flush(reason="bootstrap_finish")

    finally:
        persistence.close()

    return BootstrapResult(ctx=ctx, duckdb_path=cfg.storage.duckdb_path)
