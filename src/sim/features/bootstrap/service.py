from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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
from sim.features.events.service import EventService
from sim.features.intent_resolver.service import IntentResolverService
from sim.features.intent_resolver.types import IntentResolverConfig
from sim.features.persistence.duckdb_adapter import DuckDBAdapter
from sim.features.persistence.service import PersistenceService
from sim.features.session_intent.service import SessionIntentService
from sim.features.sessions.service import InterPageTimeConfig, SessionsConfig, SessionsService
from sim.features.site_graph.service import SiteGraphFactory
from sim.features.site_graph.types import WebsiteGraph
from sim.features.users_state.service import (
    DiscoveryModeConfig,
    PropensityInitConfig,
    UsersConfig,
    UsersSelectionConfig,
    UsersStateService,
)


@dataclass(frozen=True)
class BootstrapResult:
    ctx: RunContext
    duckdb_path: str


class _IdsEventIdAdapter:
    """Adapts IdsService to the events feature IdGenerator protocol."""

    def __init__(self, ids: IdsService) -> None:
        self._ids = ids

    def next_event_id(self) -> str:
        return str(self._ids.next_id("evt"))


def _build_graph(
    *, env: simpy.Environment, raw: dict[str, Any], start_dt_utc: datetime, rng: RNG
) -> WebsiteGraph:
    sg_factory = SiteGraphFactory(strict=False)
    cfg_site = raw.get("site_graph") if isinstance(raw, dict) else None
    return sg_factory.build(env=env, cfg_site_graph=cfg_site, start_dt=start_dt_utc, rng=rng)


def _parse_users_config(raw: dict[str, Any]) -> UsersConfig:
    u_raw = raw.get("users") or {}

    sel_raw = u_raw.get("selection") or {}
    disc_raw = u_raw.get("discovery_mode") or {}

    # Back-compat: some older configs used users.propensity.init.*
    prop_init_raw = (
        u_raw.get("propensity_init") or (u_raw.get("propensity") or {}).get("init") or {}
    )

    return UsersConfig(
        new_user_share=float(u_raw.get("new_user_share", 0.6)),
        selection=UsersSelectionConfig(
            mode=str(sel_raw.get("mode", "recency_propensity_weighted")),
            recency_half_life_hours=float(sel_raw.get("recency_half_life_hours", 18.0)),
            propensity_weight=float(sel_raw.get("propensity_weight", 0.5)),
            recency_weight=float(sel_raw.get("recency_weight", 0.5)),
        ),
        propensity_init=PropensityInitConfig(
            dist=str(prop_init_raw.get("dist", "uniform")),
            alpha=float(prop_init_raw.get("alpha", 2.0)),
            beta=float(prop_init_raw.get("beta", 6.0)),
        ),
        discovery_mode=DiscoveryModeConfig(
            enabled=bool(disc_raw.get("enabled", True)),
            graduation_sessions=int(disc_raw.get("graduation_sessions", 2)),
            dropoff_multiplier=float(disc_raw.get("dropoff_multiplier", 1.25)),
            conversion_logit_shift=float(disc_raw.get("conversion_logit_shift", -0.8)),
        ),
    )


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

    # ----- website graph (authoritative clock) -----
    graph = _build_graph(env=env, raw=raw, start_dt_utc=start_dt_utc, rng=rng)

    # ----- canonical event service -----
    events = EventService(
        env=env,
        graph=graph,
        persistence=persistence,
        ids=_IdsEventIdAdapter(ids),
        run_id=ctx.run_id,
        logger=logger,
    )

    try:
        # ============================================================
        # Marketing pressure: Campaign Planner (pure, deterministic)
        # ============================================================
        # _campaign_planner = CampaignPlannerService.from_raw_config(
        #     start_dt_utc=start_dt_utc,
        #     raw=raw,
        # )
        # NOTE: campaign_planner is wired for downstream consumption (adstock/saturation → delivery plans).
        # It is intentionally not used here yet.

        # ----- users hot state -----
        users_cfg = _parse_users_config(raw)
        users = UsersStateService(cfg=users_cfg)

        # ----- session intents store -----
        session_intents = SessionIntentService(env=env, ids=ids, capacity=None)

        # Arrivals expects an intent bus with publish(intent)
        class IntentBusAdapter:
            def __init__(self, bus: SessionIntentService) -> None:
                self._bus = bus

            def publish(self, intent) -> None:
                self._bus.publish_new(
                    ts_utc=intent.ts_utc,
                    intent_source=intent.intent_source,
                    channel=intent.channel,
                    audience_id=getattr(intent, "audience_id", None),
                    payload=getattr(intent, "payload", None),
                )

        intent_bus = IntentBusAdapter(session_intents)

        # ----- arrivals (baseline intents) -----
        if "arrivals" in raw:
            a_raw = raw.get("arrivals") or {}
            # Back-compat: support either "baseline_intents" or "baseline_arrivals"
            b_raw = a_raw.get("baseline_intents") or a_raw.get("baseline_arrivals") or {}
            curve_raw = b_raw.get("intraday_curve") or {}
            baseline_cfg = BaselineArrivalsConfig(
                model=str(b_raw.get("model", "nhpp")),
                daily_expected_intents=float(b_raw.get("daily_expected_intents", 0.0)),
                intraday_curve=GaussianPeakCurveConfig(
                    peak_hour=float(curve_raw.get("peak_hour", 12.0)),
                    spread_hours=float(curve_raw.get("spread_hours", 3.0)),
                    floor=float(curve_raw.get("floor", 0.05)),
                ),
            )
            arrivals = ArrivalsService(
                run_id=ctx.run_id,
                rng=rng,
                ids=ids,
                graph=graph,
                intent_bus=intent_bus,
                baseline_arrivals=baseline_cfg,
                num_days=int(cfg.run.num_days),
                events=events,
            )
            arrivals.start(env)

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

        # ----- sessions -----
        s_raw = raw.get("sessions") or {}
        ipt_raw = s_raw.get("inter_page_time") or {}
        sessions_cfg = SessionsConfig(
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
        sessions_svc = SessionsService(
            env=env, graph=graph, rng=rng, events=events, cfg=sessions_cfg
        )

        # ----- session runner used by intent resolver -----
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
                if is_disc and bool(users.cfg.discovery_mode.enabled):
                    drop_mult = float(users.cfg.discovery_mode.dropoff_multiplier)
                    logit_shift = float(users.cfg.discovery_mode.conversion_logit_shift)

                channel_out = getattr(intent, "channel", None) or "direct"

                proc = self._sessions.spawn(
                    user_id=user_id,
                    session_id=session_id,
                    intent_source=getattr(intent, "intent_source", None),
                    channel=channel_out,
                    conversion=conversion,
                    user_propensity=propensity,
                    dropoff_multiplier=drop_mult,
                    conversion_logit_shift=logit_shift,
                )
                yield proc
                users.mark_session_end(user_id=user_id, now_utc=intent.ts_utc)

        session_runner = RealSessionRunner(env, sessions_svc)

        # ----- intent resolver -----
        resolver = IntentResolverService(
            env=env,
            cfg=IntentResolverConfig(enabled=True),
            ids=ids,
            rng=rng,
            intents=session_intents,
            users=users,
            session_runner=session_runner,
            sink=events,
        )
        resolver.start()

        # ============================================================
        # Channels Exposure (rate-driven mode)
        # ============================================================
        # Only start if config actually defines channels; this keeps minimal bootstrap tests passing.
        ch_list = raw.get("channels")
        if isinstance(ch_list, list) and len(ch_list) > 0:
            from sim.features.channels_exposure.service import (
                ChannelsExposureService,
                build_channels,
            )
            from sim.features.channels_exposure.types import ChannelConfig, ChannelsExposureConfig

            # ChannelsExposureConfig: default enabled=True unless explicitly disabled via channels_exposure.enabled
            ce_raw = raw.get("channels_exposure") or {}
            ce_enabled = bool(ce_raw.get("enabled", True))

            # Build ChannelConfig list from raw channel dicts (matching your YAML)
            cfgs: list[ChannelConfig] = []
            for idx, c in enumerate(ch_list):
                if not isinstance(c, dict):
                    raise TypeError(f"channels[{idx}] must be a mapping")

                cfgs.append(
                    ChannelConfig(
                        name=str(c.get("name")),
                        exposure_rate_per_user_per_day=float(
                            c.get("exposure_rate_per_user_per_day", 0.0)
                        ),
                        click_through_rate=float(c.get("click_through_rate", 0.0)),
                        incremental_intent=bool(c.get("incremental_intent", True)),
                        params=dict(c.get("params") or {}),
                    )
                )

            # Minimal ctx adapter expected by ChannelsExposureService (ctx.rng)
            class _ChannelsCtx:
                def __init__(self, rng_: RNG) -> None:
                    self.rng = rng_

            channels_svc = ChannelsExposureService(
                cfg=ChannelsExposureConfig(enabled=ce_enabled),
                channels=build_channels(cfgs),
                events=events,
                website_graph=graph,
                intent_bus=session_intents,
                ctx=_ChannelsCtx(rng),
                delivery_plan_provider=None,  # will be wired later (campaigns → adstock/sat → delivery plans)
            )
            channels_svc.start(
                env,
                users_state=users,
                start_dt_utc=start_dt_utc,
                num_days=int(cfg.run.num_days),
            )

        # ----- run lifecycle -----
        events.emit("run_started")

        horizon_s = int(cfg.run.num_days) * 86400
        logger.info(
            "starting sim",
            extra={
                "run_id": ctx.run_id,
                "until_s": horizon_s,
                "campaigns_enabled": bool(raw.get("campaigns", {}).get("enabled", False)),
            },
        )
        env.run(until=horizon_s)

        events.emit("run_finished")
        persistence.flush(reason="bootstrap_finish")
    finally:
        persistence.close()

    return BootstrapResult(ctx=ctx, duckdb_path=cfg.storage.duckdb_path)
