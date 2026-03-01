from __future__ import annotations

from dataclasses import dataclass

import simpy

from sim.features.channels_exposure.channels.base import (
    Channel,
    _pick_user_id,
    _schedule_at,
)
from sim.features.channels_exposure.types import ChannelConfig, DeliveryPlan


@dataclass(frozen=True)
class PaidSearchChannel(Channel):
    """
    Paid search differences:
      - Only a fraction of users are "in-market" per day (or per impression in delivery-driven mode)
      - Frequency cap per user per day (applies in both modes)
      - Brand vs nonbrand mix modifies CTR
      - incremental_click_share models demand capture vs incrementality
    """

    cfg: ChannelConfig

    def schedule_for_day(
        self,
        *,
        env: simpy.Environment,
        day_start_s: float,
        seconds_per_day: float,
        user_ids: list[str],
        emit_exposure,
        emit_click,
        emit_intent,
        rng,
    ) -> None:
        rate = float(self.cfg.exposure_rate_per_user_per_day)
        rate = max(0.0, rate)

        # interpret as "at most one exposure per user per day"
        p_expose = 1.0 if rate >= 1.0 else rate

        for user_id in user_ids:
            # gating: does this user get an exposure today?
            if rng.random() >= p_expose:
                continue

            # schedule exposure time uniformly within the day
            t = day_start_s + rng.random() * float(seconds_per_day)

            def _proc(u: str, *, t_s: float = t) -> simpy.events.Event:
                # wait until scheduled time
                yield env.timeout(max(0.0, t_s - float(env.now)))

                emit_exposure(user_id=u, channel=self.cfg.name, campaign_id=None)

                # click
                if rng.random() < float(self.cfg.click_through_rate):
                    emit_click(user_id=u, channel=self.cfg.name, campaign_id=None)

                    # incremental intent
                    if bool(self.cfg.incremental_intent):
                        emit_intent(user_id=u, channel=self.cfg.name, campaign_id=None)

            # IMPORTANT: actually schedule the process
            env.process(_proc(user_id))

    def schedule_from_delivery_plan(
        self,
        *,
        env: simpy.Environment,
        plan: DeliveryPlan,
        user_ids: list[str],
        emit_exposure: callable,
        emit_click: callable,
        emit_intent: callable,
        rng,
    ) -> None:
        p0 = self.cfg.params or {}

        # Interpret each slice as "search opportunities" (impressions).
        # We pick users; only some are in-market; cap exposures per user per day.
        in_market_share = float(p0.get("in_market_share_per_day", 1.0))
        freq_cap = int(p0.get("freq_cap_per_user_per_day", 10_000))

        brand_share = float(p0.get("brand_share", 0.0))
        brand_mult = float(p0.get("brand_ctr_multiplier", 1.0))
        nonbrand_mult = float(p0.get("nonbrand_ctr_multiplier", 1.0))

        base_ctr = float(self.cfg.click_through_rate)
        base_incremental = float(p0.get("incremental_click_share", 1.0))
        if not bool(self.cfg.incremental_intent):
            base_incremental = 0.0

        channel_name = self.cfg.name

        per_user_exposures: dict[str, int] = {u: 0 for u in user_ids}

        for sl in plan.slices:
            sp = sl.params or {}
            # slice can override base params if desired
            in_market = float(sp.get("in_market_share_per_day", in_market_share))
            inc = (
                float(sl.incremental_click_share)
                if sl.incremental_click_share is not None
                else base_incremental
            )
            ctr_override = float(sl.ctr) if sl.ctr is not None else base_ctr

            for _ in range(int(sl.impressions)):
                user_id = _pick_user_id(rng, user_ids)

                if per_user_exposures[user_id] >= freq_cap:
                    continue

                # in-market gating can be per impression in delivery-driven mode
                if float(rng.random()) >= in_market:
                    continue

                per_user_exposures[user_id] += 1

                is_brand = float(rng.random()) < float(sp.get("brand_share", brand_share))
                bmult = float(sp.get("brand_ctr_multiplier", brand_mult))
                nmult = float(sp.get("nonbrand_ctr_multiplier", nonbrand_mult))
                ctr = ctr_override * (bmult if is_brand else nmult)

                env.process(
                    _paid_search_exposure(
                        env=env,
                        at_s=float(sl.at_s),
                        user_id=user_id,
                        channel_name=channel_name,
                        ctr=ctr,
                        incremental_click_share=inc,
                        campaign_id=sl.campaign_id,
                        emit_exposure=emit_exposure,
                        emit_click=emit_click,
                        emit_intent=emit_intent,
                        rng=rng,
                    )
                )


def _paid_search_exposure(
    *,
    env: simpy.Environment,
    at_s: float,
    user_id: str,
    channel_name: str,
    ctr: float,
    incremental_click_share: float,
    campaign_id: str | None,
    emit_exposure: callable,
    emit_click: callable,
    emit_intent: callable,
    rng,
):
    yield _schedule_at(env, at_s)

    emit_exposure(user_id=user_id, channel=channel_name, campaign_id=campaign_id)

    if float(rng.random()) < float(ctr):
        emit_click(user_id=user_id, channel=channel_name, campaign_id=campaign_id)

        if float(rng.random()) < float(incremental_click_share):
            emit_intent(user_id=user_id, channel=channel_name, campaign_id=campaign_id)


def build(cfg: ChannelConfig) -> PaidSearchChannel:
    if cfg.name != "paid_search":
        raise ValueError("paid_search channel must have name='paid_search'")
    return PaidSearchChannel(cfg=cfg)
