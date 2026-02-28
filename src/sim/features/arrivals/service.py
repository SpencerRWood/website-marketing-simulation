from __future__ import annotations

from dataclasses import dataclass

from sim.features.arrivals.models.nhpp import (
    GaussianPeakCurveConfig,
    NHPPBaselineArrivalsConfig,
    NHPPBaselineArrivalsModel,
)
from sim.features.arrivals.types import (
    ArrivalModel,
    EventsLike,
    IdsLike,
    IntentBusLike,
    RngLike,
    WebsiteGraphLike,
)


@dataclass(frozen=True)
class BaselineArrivalsConfig:
    """
    Feature-level config for baseline arrivals.

    model:
      - "nhpp" (implemented)
      - future: "piecewise_constant", "empirical_replay", ...
    """

    model: str
    daily_expected_intents: float
    intraday_curve: GaussianPeakCurveConfig


class ArrivalsService:
    """
    Feature orchestrator: selects an arrival model from config and starts it.
    """

    def __init__(
        self,
        *,
        run_id: str,
        rng: RngLike,
        ids: IdsLike,
        graph: WebsiteGraphLike,
        intent_bus: IntentBusLike,
        baseline_arrivals: BaselineArrivalsConfig,
        num_days: int,
        events: EventsLike | None = None,
        grid_minutes: int = 1,
    ) -> None:
        self.run_id = run_id
        self.rng = rng
        self.ids = ids
        self.graph = graph
        self.intent_bus = intent_bus
        self.baseline_arrivals = baseline_arrivals
        self.num_days = num_days
        self.events = events
        self.grid_minutes = grid_minutes

        self._model: ArrivalModel = self._build_model()

    def _build_model(self) -> ArrivalModel:
        model = (self.baseline_arrivals.model or "").strip().lower()
        if model == "nhpp":
            cfg = NHPPBaselineArrivalsConfig(
                daily_expected_intents=float(self.baseline_arrivals.daily_expected_intents),
                intraday_curve=self.baseline_arrivals.intraday_curve,
            )
            return NHPPBaselineArrivalsModel(
                run_id=self.run_id,
                rng=self.rng,
                ids=self.ids,
                graph=self.graph,
                intent_bus=self.intent_bus,
                cfg=cfg,
                num_days=self.num_days,
                events=self.events,
                grid_minutes=self.grid_minutes,
            )

        raise ValueError(
            f"Unsupported arrivals.baseline_arrivals.model={self.baseline_arrivals.model!r}"
        )

    def start(self, env) -> None:
        self._model.start(env)
