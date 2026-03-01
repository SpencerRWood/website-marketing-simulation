from __future__ import annotations

from dataclasses import dataclass

import simpy

from sim.features.channels_exposure.channels.base import (
    Channel,
    _pick_user_id,
    _poisson_knuth,
    _schedule_at,
)
from sim.features.channels_exposure.types import ChannelConfig, DeliveryPlan


@dataclass(frozen=True)
class PaidDisplayChannel(Channel):
    """
    Paid display / programmatic differences:
      - No in-market gating (broad reach)
      - Viewability filter: only a share of raw impressions become viewable exposures
      - Strong frequency caps
      - Typically low CTR
      - Incremental_click_share can be higher than search (config)
    """

    cfg: ChannelConfig

    def schedule_for_day(
        self,
        *,
        env: simpy.Environment,
        day_start_s: float,
        seconds_per_day: float,
        user_ids: list[str],
        emit_exposure: callable,
        emit_click: callable,
        emit_intent: callable,
        rng,
    ) -> None:
        p = self.cfg.params or {}

        freq_cap = int(p.get("freq_cap_per_user_per_day", 10))
        viewable_share = float(p.get("viewable_share", 1.0))
        ctr_mult = float(p.get("ctr_multiplier", 1.0))

        incremental_click_share = float(p.get("incremental_click_share", 1.0))
        if not bool(self.cfg.incremental_intent):
            incremental_click_share = 0.0

        lam = float(self.cfg.exposure_rate_per_user_per_day)
        base_ctr = float(self.cfg.click_through_rate) * ctr_mult
        channel_name = self.cfg.name

        # per-user cap tracking for the day
        per_user_exposures: dict[str, int] = {u: 0 for u in user_ids}

        # Rate-driven: per user Poisson impressions, viewable filter -> exposure
        for user_id in user_ids:
            if per_user_exposures[user_id] >= freq_cap:
                continue

            n = _poisson_knuth(rng, lam)
            for _ in range(n):
                if per_user_exposures[user_id] >= freq_cap:
                    break

                # viewability filter
                if float(rng.random()) >= viewable_share:
                    continue

                per_user_exposures[user_id] += 1
                at_s = float(day_start_s) + float(rng.random()) * float(seconds_per_day)

                env.process(
                    _paid_display_exposure(
                        env=env,
                        at_s=at_s,
                        user_id=user_id,
                        channel_name=channel_name,
                        ctr=base_ctr,
                        incremental_click_share=incremental_click_share,
                        campaign_id=None,
                        emit_exposure=emit_exposure,
                        emit_click=emit_click,
                        emit_intent=emit_intent,
                        rng=rng,
                    )
                )

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

        freq_cap = int(p0.get("freq_cap_per_user_per_day", 10))
        viewable_share = float(p0.get("viewable_share", 1.0))
        ctr_mult = float(p0.get("ctr_multiplier", 1.0))

        base_ctr = float(self.cfg.click_through_rate) * ctr_mult
        incremental_click_share = float(p0.get("incremental_click_share", 1.0))
        if not bool(self.cfg.incremental_intent):
            incremental_click_share = 0.0

        channel_name = self.cfg.name
        per_user_exposures: dict[str, int] = {u: 0 for u in user_ids}

        for sl in plan.slices:
            sp = sl.params or {}

            # allow slice overrides
            vshare = float(sp.get("viewable_share", viewable_share))
            cap = int(sp.get("freq_cap_per_user_per_day", freq_cap))
            ctr = (float(sl.ctr) if sl.ctr is not None else base_ctr) * float(
                sp.get("ctr_multiplier", 1.0)
            )
            inc = (
                float(sl.incremental_click_share)
                if sl.incremental_click_share is not None
                else incremental_click_share
            )

            for _ in range(int(sl.impressions)):
                user_id = _pick_user_id(rng, user_ids)

                if per_user_exposures[user_id] >= cap:
                    continue

                # viewability filter: impression becomes viewable exposure
                if float(rng.random()) >= vshare:
                    continue

                per_user_exposures[user_id] += 1

                env.process(
                    _paid_display_exposure(
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


def _paid_display_exposure(
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


def build(cfg: ChannelConfig) -> PaidDisplayChannel:
    if cfg.name != "paid_display":
        raise ValueError("paid_display channel must have name='paid_display'")
    return PaidDisplayChannel(cfg=cfg)
