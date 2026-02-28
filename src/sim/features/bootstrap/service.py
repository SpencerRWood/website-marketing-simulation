from __future__ import annotations

from collections.abc import Mapping
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
from sim.features.conversion.service import ConversionConfig, ConversionService
from sim.features.intent_resolver.service import IntentResolverService
from sim.features.intent_resolver.types import IntentResolverConfig
from sim.features.persistence.duckdb_adapter import DuckDBAdapter
from sim.features.persistence.service import Event, PersistenceService
from sim.features.session_intent.service import SessionIntentService
from sim.features.sessions.service import InterPageTimeConfig, SessionsConfig, SessionsService
from sim.features.site_graph.service import SiteGraphFactory
from sim.features.users_state.service import UsersConfig, UsersStateService


@dataclass(frozen=True)
class BootstrapResult:
    ctx: RunContext
    duckdb_path: str


def bootstrap_run(cfg: SimulationConfig, config_path: str | None = None) -> BootstrapResult:
    raw: dict[str, Any] = cfg.raw if isinstance(cfg.raw, dict) else {}

    # ----- run identity -----
    run_id = deterministic_run_id_from_config(raw) if cfg.run.run_id == "auto" else cfg.run.run_id
    logger = get_logger("sim", cfg.logging.level)

    rng = RNG(cfg.run.seed)
    ids = IdsService(run_id=run_id)

    start_dt_utc = datetime.fromisoformat(cfg.run.start_date).replace(tzinfo=UTC)
    ctx = RunContext(run_id=run_id, seed=cfg.run.seed, start_dt_utc=start_dt_utc)

    env = simpy.Environment()

    # ----- cold storage -----
    adapter = DuckDBAdapter(path=cfg.storage.duckdb_path, clean_slate=cfg.storage.clean_slate)
    persistence = PersistenceService(
        adapter=adapter,
        every_n_events=cfg.storage.flush.every_n_events,
        or_every_seconds=cfg.storage.flush.or_every_seconds,
    )
    persistence.open()
    persistence.start_periodic_flush(env)

    # ------------------------------------------------------------------------------
    # Event sink used by IntentResolverService: MUST match its EventSink Protocol:
    #   emit(*, ts_utc, sim_time_s, event_type, user_id?, session_id?, intent_id?,
    #        intent_source?, channel?, audience_id?, payload?)
    # ------------------------------------------------------------------------------
    class ResolverEventSink:
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
            payload: dict[str, Any] | None = None,
        ) -> None:
            # We store intent_id/audience_id in payload so we don't need schema changes.
            payload_out = payload or {}
            if intent_id is not None:
                payload_out = dict(payload_out)
                payload_out["intent_id"] = intent_id
            if audience_id is not None:
                payload_out = dict(payload_out)
                payload_out["audience_id"] = audience_id

            persistence.emit(
                Event(
                    run_id=ctx.run_id,
                    event_id=ids.next_id("evt"),
                    ts_utc=ts_utc,
                    sim_time_s=sim_time_s,
                    event_type=event_type,
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=intent_source,
                    channel=channel,
                    page=None,
                    value_num=None,
                    value_str=None,
                    payload=payload_out if payload_out else None,
                )
            )

    sink = ResolverEventSink()

    # ----- users hot state -----
    users_cfg = getattr(cfg, "users", None)
    if users_cfg is None:
        users_cfg = UsersConfig()
    users = UsersStateService(cfg=users_cfg)

    # ----- session intents store -----
    session_intents = SessionIntentService(env=env, ids=ids, capacity=None)

    # Arrivals expects an intent bus with publish(intent)
    class IntentBusAdapter:
        def __init__(self, bus: SessionIntentService) -> None:
            self._bus = bus

        def publish(self, intent) -> None:
            # Your publish_new does NOT accept intent_id; only pass supported kwargs.
            try:
                self._bus.publish_new(
                    ts_utc=intent.ts_utc,
                    intent_source=intent.intent_source,
                    channel=intent.channel,
                )
            except TypeError:
                self._bus.publish_new(ts_utc=intent.ts_utc, intent_source=intent.intent_source)

    intent_bus = IntentBusAdapter(session_intents)

    # ----- graph + sessions (only if configured) -----
    sessions_svc: SessionsService | None = None
    if "site_graph" in raw and "sessions" in raw:
        sg_factory = SiteGraphFactory(strict=False)
        graph = sg_factory.build(
            env=env,
            cfg_site_graph=raw.get("site_graph"),
            start_dt=start_dt_utc,
            rng=rng,  # <-- SiteGraphFactory expects RNG, not an adapter
        )

        s_raw = raw.get("sessions") or {}
        ipt_raw = s_raw.get("inter_page_time") or {}

        sess_cfg = SessionsConfig(
            inactivity_timeout_minutes=float(s_raw.get("inactivity_timeout_minutes", 30.0)),
            max_steps=None
            if s_raw.get("max_steps", 12) is None
            else int(s_raw.get("max_steps", 12)),
            entry_page=str(s_raw.get("entry_page", "home")),
            inter_page_time=InterPageTimeConfig(
                dist=str(ipt_raw.get("dist", "fixed")),
                fixed_seconds=float(ipt_raw.get("fixed_seconds", 10.0)),
                mean_seconds=float(ipt_raw.get("mean_seconds", 10.0)),
            ),
        )

        # SessionsService expects an EventsLike with emit(**kwargs) and publish(Mapping)
        class SessionsEventsAdapter:
            def emit(self, **kwargs: Any) -> None:
                # This adapter writes directly to persistence using your events schema.
                persistence.emit(
                    Event(
                        run_id=ctx.run_id,
                        event_id=ids.next_id("evt"),
                        ts_utc=kwargs["ts_utc"],
                        sim_time_s=float(kwargs["sim_time_s"]),
                        event_type=str(kwargs["event_type"]),
                        user_id=kwargs.get("user_id"),
                        session_id=kwargs.get("session_id"),
                        intent_source=kwargs.get("intent_source"),
                        channel=kwargs.get("channel"),
                        page=kwargs.get("page"),
                        value_num=kwargs.get("value_num"),
                        value_str=kwargs.get("value_str"),
                        payload=kwargs.get("payload"),
                    )
                )

            def publish(self, event: Mapping[str, Any]) -> None:
                # Protocol wants Mapping, not dict
                self.emit(**dict(event))

        sessions_events = SessionsEventsAdapter()

        sessions_svc = SessionsService(
            env=env,
            graph=graph,
            rng=rng,  # <-- Sessions RNGLike expects expovariate(lambd), RNG matches
            events=sessions_events,
            ids=ids,  # SessionsService handles next_id(prefix) internally
            run_id=ctx.run_id,
            cfg=sess_cfg,
        )

    # ----- conversion -----
    conversion: ConversionService | None = None
    if "conversion" in raw:
        c_raw = raw.get("conversion") or {}
        conversion = ConversionService(
            ConversionConfig(
                model=str(c_raw.get("model", "logistic")),
                cap=float(c_raw.get("cap", 0.35)),
                base_logit=float(c_raw.get("base_logit", -3.0)),
                propensity_coef=float(c_raw.get("propensity_coef", 2.0)),
            )
        )

    # ----- session runner used by intent resolver -----
    class NoopSessionRunner:
        def __init__(self, env_: simpy.Environment) -> None:
            self._env = env_

        def start_session(self, *, user_id: str, session_id: str, intent) -> Any:
            return self._env.process(
                self._run(user_id=user_id, session_id=session_id, intent=intent)
            )

        def _run(self, *, user_id: str, session_id: str, intent):
            # Minimal end + user update
            sink.emit(
                ts_utc=intent.ts_utc,
                sim_time_s=float(self._env.now),
                event_type="session_end",
                user_id=user_id,
                session_id=session_id,
                intent_id=getattr(intent, "intent_id", None),
                intent_source=getattr(intent, "intent_source", None),
                channel=getattr(intent, "channel", None),
                audience_id=None,
                payload=None,
            )
            users.mark_session_end(user_id=user_id, now_utc=intent.ts_utc)
            yield self._env.timeout(0)

    class RealSessionRunner:
        def __init__(self, env_: simpy.Environment, sessions_: SessionsService) -> None:
            self._env = env_
            self._sessions = sessions_

        def start_session(self, *, user_id: str, session_id: str, intent) -> Any:
            return self._env.process(
                self._run(user_id=user_id, session_id=session_id, intent=intent)
            )

        def _run(self, *, user_id: str, session_id: str, intent):
            u = users.get_user(user_id)
            propensity = float(getattr(u, "propensity", 0.5)) if u is not None else 0.5
            is_disc = bool(getattr(u, "discovery_mode", False)) if u is not None else False

            drop_mult = 1.0
            logit_shift = 0.0
            if is_disc and bool(getattr(users.cfg.discovery_mode, "enabled", False)):
                drop_mult = float(users.cfg.discovery_mode.dropoff_multiplier)
                logit_shift = float(users.cfg.discovery_mode.conversion_logit_shift)

            proc = self._sessions.spawn(
                user_id=user_id,
                session_id=session_id,
                intent_source=getattr(intent, "intent_source", None),
                channel=getattr(intent, "channel", None),
                conversion=conversion,
                user_propensity=propensity,
                dropoff_multiplier=drop_mult,
                conversion_logit_shift=logit_shift,
            )
            yield proc

            end_ts = self._sessions.graph.get_current_time()
            # Ensure Pylance sees datetime, not Unknown
            if not isinstance(end_ts, datetime):
                end_ts = intent.ts_utc
            users.mark_session_end(user_id=user_id, now_utc=end_ts)

    session_runner: Any = (
        RealSessionRunner(env, sessions_svc) if sessions_svc is not None else NoopSessionRunner(env)
    )

    # ----- intent resolver -----
    resolver_cfg = IntentResolverConfig(
        enabled=getattr(getattr(cfg, "intent_resolver", None), "enabled", True)
    )
    resolver = IntentResolverService(
        env=env,
        cfg=resolver_cfg,
        ids=ids,  # Intent resolver expects ids.new_id(prefix) -> your IdsService supports next_id(prefix)
        rng=rng,  # pass real RNG to satisfy its RngLike
        intents=session_intents,
        users=users,
        session_runner=session_runner,
        sink=sink,  # sink signature matches EventSink
    )
    resolver.start()

    # ----- baseline arrivals -----
    class ClockGraph:
        def __init__(self, env_: simpy.Environment, start_: datetime) -> None:
            self.env = env_
            self.start_dt = start_

        def get_current_time(self) -> datetime:
            return self.start_dt + timedelta(seconds=float(self.env.now))

    a_raw = raw.get("arrivals") if isinstance(raw.get("arrivals"), dict) else None
    if isinstance(a_raw, dict):
        b_raw = a_raw.get("baseline_arrivals") or a_raw.get("baseline_intents")
        if isinstance(b_raw, dict):
            curve_raw = b_raw.get("intraday_curve", {}) or {}
            arrivals = ArrivalsService(
                run_id=ctx.run_id,
                rng=rng,
                ids=ids,
                graph=ClockGraph(env, start_dt_utc),
                intent_bus=intent_bus,
                baseline_arrivals=BaselineArrivalsConfig(
                    model=str(b_raw.get("model", "nhpp")),
                    daily_expected_intents=float(b_raw.get("daily_expected_intents", 0.0)),
                    intraday_curve=GaussianPeakCurveConfig(
                        peak_hour=float(curve_raw.get("peak_hour", 12.0)),
                        spread_hours=float(curve_raw.get("spread_hours", 3.0)),
                        floor=float(curve_raw.get("floor", 0.05)),
                    ),
                ),
                num_days=int(cfg.run.num_days),
                events=None,  # optional; avoids EventSink protocol mismatch
            )
            arrivals.start(env)

    # ----- run lifecycle -----
    try:
        persistence.emit(
            Event(
                run_id=ctx.run_id,
                event_id=ids.next_id("evt"),
                ts_utc=start_dt_utc,
                sim_time_s=float(env.now),
                event_type="run_started",
            )
        )

        horizon_s = int(cfg.run.num_days) * 86400
        logger.info("starting sim", extra={"run_id": ctx.run_id, "until_s": horizon_s})
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
        persistence.flush(reason="bootstrap_finish")
    finally:
        persistence.close()

    return BootstrapResult(ctx=ctx, duckdb_path=cfg.storage.duckdb_path)
