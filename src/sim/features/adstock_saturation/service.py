from __future__ import annotations

from dataclasses import dataclass

from sim.features.adstock_saturation.transforms import (
    apply_geometric_adstock,
    apply_hill_saturation,
)
from sim.features.adstock_saturation.types import (
    AdstockSaturationConfig,
    AdstockStepResult,
    SaturationStepResult,
    TransformStepResult,
)


@dataclass
class AdstockSaturationService:
    """
    Stateful transform engine:
      spend -> adstock (carryover) -> saturation (diminishing returns)

    State is maintained across calls. Use reset() between runs.
    """

    cfg: AdstockSaturationConfig
    _carryover_state: dict[tuple[str, str | None], float]

    @classmethod
    def from_config(cls, cfg: AdstockSaturationConfig) -> AdstockSaturationService:
        return cls(cfg=cfg, _carryover_state={})

    def reset(self) -> None:
        self._carryover_state.clear()

    def _key(self, channel: str, campaign_id: str | None) -> tuple[str, str | None]:
        if self.cfg.per_campaign_state:
            return (channel, campaign_id)
        return (channel, None)

    def apply(
        self,
        *,
        channel: str,
        spend: float,
        dt_days: float,
        campaign_id: str | None = None,
    ) -> TransformStepResult:
        """
        Apply transforms for a single channel (and optional campaign).
        """
        if not self.cfg.enabled:
            # Passthrough mode: no carryover, no saturation.
            ad = AdstockStepResult(
                channel=channel,
                campaign_id=campaign_id if self.cfg.per_campaign_state else None,
                dt_days=float(dt_days),
                spend_in=float(spend),
                carryover_in=0.0,
                decay=0.0,
                carryover_out=float(spend),
                adstocked=float(spend),
            )
            sat = SaturationStepResult(
                channel=channel,
                campaign_id=campaign_id if self.cfg.per_campaign_state else None,
                x_in=float(spend),
                response=float(spend),
            )
            return TransformStepResult(adstock=ad, saturation=sat)

        ch_cfg = (self.cfg.channels or {}).get(channel)
        if ch_cfg is None:
            raise KeyError(f"adstock_saturation missing channel config for '{channel}'")

        key = self._key(channel, campaign_id)
        carry_prev = float(self._carryover_state.get(key, 0.0))

        # --- adstock ---
        carry_out, adstocked, decay = apply_geometric_adstock(
            spend=float(spend),
            carryover_prev=carry_prev,
            cfg=ch_cfg.adstock,
            dt_days=float(dt_days),
        )
        self._carryover_state[key] = float(carry_out)

        ad = AdstockStepResult(
            channel=channel,
            campaign_id=key[1],
            dt_days=float(dt_days),
            spend_in=float(spend),
            carryover_in=float(carry_prev),
            decay=float(decay),
            carryover_out=float(carry_out),
            adstocked=float(adstocked),
        )

        # --- saturation ---
        response = apply_hill_saturation(x=float(adstocked), cfg=ch_cfg.saturation)
        sat = SaturationStepResult(
            channel=channel,
            campaign_id=key[1],
            x_in=float(adstocked),
            response=float(response),
        )

        return TransformStepResult(adstock=ad, saturation=sat)

    def apply_many(
        self,
        *,
        spends_by_channel: dict[str, float],
        dt_days: float,
        campaign_id: str | None = None,
    ) -> dict[str, TransformStepResult]:
        """
        Convenience: apply transforms across multiple channels for a timestep.
        """
        out: dict[str, TransformStepResult] = {}
        for ch, s in spends_by_channel.items():
            out[ch] = self.apply(
                channel=ch, spend=float(s), dt_days=float(dt_days), campaign_id=campaign_id
            )
        return out
