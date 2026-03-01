from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import simpy

from sim.features.channels_exposure.types import ChannelConfig, DeliveryPlan


class Channel(Protocol):
    cfg: ChannelConfig

    # Rate-driven mode (legacy / simple)
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
    ) -> None: ...

    # Delivery-driven mode (campaign-fed)
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
    ) -> None: ...


def _poisson_knuth(rng, lam: float) -> int:
    if lam <= 0.0:
        return 0
    import math

    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= float(rng.random())
    return k - 1


def _pick_user_id(rng, user_ids: list[str]) -> str:
    # deterministic pick using rng.random()
    if not user_ids:
        raise ValueError("user_ids is empty")
    idx = int(float(rng.random()) * len(user_ids))
    if idx == len(user_ids):
        idx = len(user_ids) - 1
    return user_ids[idx]


def _schedule_at(env: simpy.Environment, at_s: float):
    delay = max(0.0, float(at_s) - float(env.now))
    if delay > 0.0:
        return env.timeout(delay)
    return env.timeout(0.0)


@dataclass(frozen=True)
class RateDrivenPoissonChannel(Channel):
    """
    Generic rate-driven exposure process:
      For each user:
        N ~ Poisson(exposure_rate_per_user_per_day)
        each exposure uniformly within day
        click with CTR
        if clicked and incremental_intent -> emit intent
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
        lam = float(self.cfg.exposure_rate_per_user_per_day)
        ctr = float(self.cfg.click_through_rate)
        incremental = bool(self.cfg.incremental_intent)
        channel_name = self.cfg.name

        for user_id in user_ids:
            n = _poisson_knuth(rng, lam)
            for _ in range(n):
                at_s = float(day_start_s) + float(rng.random()) * float(seconds_per_day)
                env.process(
                    _exposure_process(
                        env=env,
                        at_s=at_s,
                        user_id=user_id,
                        channel_name=channel_name,
                        ctr=ctr,
                        incremental_click_share=1.0 if incremental else 0.0,
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
        # Default delivery-driven behavior for generic channels:
        # Each impression -> pick a user uniformly -> expose -> click -> intent (per slice overrides).
        base_ctr = float(self.cfg.click_through_rate)
        base_incremental = 1.0 if bool(self.cfg.incremental_intent) else 0.0
        channel_name = self.cfg.name

        for sl in plan.slices:
            ctr = float(sl.ctr) if sl.ctr is not None else base_ctr
            inc = (
                float(sl.incremental_click_share)
                if sl.incremental_click_share is not None
                else base_incremental
            )

            for _ in range(int(sl.impressions)):
                user_id = _pick_user_id(rng, user_ids)
                env.process(
                    _exposure_process(
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


def _exposure_process(
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
