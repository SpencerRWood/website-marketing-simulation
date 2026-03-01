from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import simpy

from sim.features.channels_exposure.channels.base import Channel
from sim.features.channels_exposure.channels.paid_display import build as build_paid_display
from sim.features.channels_exposure.channels.paid_search import build as build_paid_search
from sim.features.channels_exposure.types import ChannelConfig, ChannelsExposureConfig, DeliveryPlan
from sim.features.events.types import EventsEmitter
from sim.features.session_intent.service import SessionIntentService

_SECONDS_PER_DAY = 24 * 60 * 60


DeliveryPlanProvider = Callable[[int, float, float], list[DeliveryPlan]]


@dataclass(frozen=True)
class ChannelsExposureService:
    cfg: ChannelsExposureConfig
    channels: list[Channel]
    events: EventsEmitter
    website_graph: any
    intent_bus: SessionIntentService
    ctx: any
    delivery_plan_provider: DeliveryPlanProvider | None = None

    def start(
        self, env: simpy.Environment, *, users_state: any, start_dt_utc: datetime, num_days: int
    ) -> None:
        if not self.cfg.enabled:
            return

        if start_dt_utc.tzinfo is None:
            start_dt_utc = start_dt_utc.replace(tzinfo=UTC)
        else:
            start_dt_utc = start_dt_utc.astimezone(UTC)

        env.process(self._run(env=env, users_state=users_state, num_days=int(num_days)))

    def _run(self, *, env: simpy.Environment, users_state: any, num_days: int):
        for day_idx in range(num_days):
            day_start_s = float(day_idx) * _SECONDS_PER_DAY

            if float(env.now) < day_start_s:
                yield env.timeout(day_start_s - float(env.now))

            user_ids = _snapshot_user_ids(users_state)

            # Delivery-driven mode if provider exists, else rate-driven
            if self.delivery_plan_provider is not None:
                plans = self.delivery_plan_provider(day_idx, day_start_s, float(_SECONDS_PER_DAY))
                plans_by_channel: dict[str, list[DeliveryPlan]] = {}
                for p in plans:
                    plans_by_channel.setdefault(p.channel, []).append(p)

                for ch in self.channels:
                    ch_plans = plans_by_channel.get(ch.cfg.name, [])
                    for plan in ch_plans:
                        ch.schedule_from_delivery_plan(
                            env=env,
                            plan=plan,
                            user_ids=user_ids,
                            emit_exposure=self._emit_exposure,
                            emit_click=self._emit_click,
                            emit_intent=self._emit_intent,
                            rng=self.ctx.rng,
                        )
            else:
                for ch in self.channels:
                    ch.schedule_for_day(
                        env=env,
                        day_start_s=day_start_s,
                        seconds_per_day=float(_SECONDS_PER_DAY),
                        user_ids=user_ids,
                        emit_exposure=self._emit_exposure,
                        emit_click=self._emit_click,
                        emit_intent=self._emit_intent,
                        rng=self.ctx.rng,
                    )

            next_day_s = float(day_idx + 1) * _SECONDS_PER_DAY
            if float(env.now) < next_day_s:
                yield env.timeout(next_day_s - float(env.now))

    # -----------------------
    # emission hooks
    # -----------------------
    def _emit_exposure(self, *, user_id: str, channel: str, campaign_id: str | None = None) -> None:
        self.events.emit(
            "exposure",
            user_id=user_id,
            channel=channel,
            payload={"campaign_id": campaign_id} if campaign_id else None,
        )

    def _emit_click(self, *, user_id: str, channel: str, campaign_id: str | None = None) -> None:
        self.events.emit(
            "click",
            user_id=user_id,
            channel=channel,
            payload={"campaign_id": campaign_id} if campaign_id else None,
        )

    def _emit_intent(self, *, user_id: str, channel: str, campaign_id: str | None = None) -> None:
        now_ts = self.website_graph.get_current_time()

        # publish onto canonical intent bus (assigns intent_id + enforces capacity)
        self.intent_bus.publish_new(
            ts_utc=now_ts,
            intent_source=f"channel:{channel}",
            channel=channel,
            payload={"campaign_id": campaign_id} if campaign_id else None,
        )

        # also emit cold event
        self.events.emit(
            event_type="session_intent",
            user_id=user_id,
            session_id=None,
            intent_source=f"channel:{channel}",
            channel=channel,
            page=None,
            payload={"campaign_id": campaign_id} if campaign_id else None,
        )


def build_channels(cfgs: list[ChannelConfig]) -> list[Channel]:
    built: list[Channel] = []
    for c in cfgs:
        if c.name == "paid_search":
            built.append(build_paid_search(c))
        elif c.name == "paid_display":
            built.append(build_paid_display(c))
        else:
            raise ValueError(f"Unknown channel: {c.name}")
    return built


def _snapshot_user_ids(users_state: any) -> list[str]:
    if hasattr(users_state, "iter_user_ids"):
        return list(users_state.iter_user_ids())
    if hasattr(users_state, "user_ids"):
        return list(users_state.user_ids())
    users = getattr(users_state, "users", None)
    if isinstance(users, dict):
        return list(users.keys())
    raise TypeError("UsersStateService must expose iter_user_ids(), user_ids(), or .users dict")
