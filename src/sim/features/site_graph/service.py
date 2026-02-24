# src/sim/features/site_graph/service.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sim.core.rng import RNG
from sim.features.site_graph.types import Page, WebsiteGraph


@dataclass(frozen=True)
class SiteGraphFactory:
    """
    Builds WebsiteGraph from the YAML config structure:

    site_graph:
      pages:
        home:
          dropoff_p: 0.25
          transitions:
            - ["product", 0.55]
            - ["pricing", 0.25]
    """

    strict: bool = False  # if True, fail on unknown pages referenced in transitions

    def build(
        self, env: Any, cfg_site_graph: dict[str, Any] | None, start_dt: datetime, rng: RNG
    ) -> WebsiteGraph:
        pages = self._parse_pages(cfg_site_graph or {}, strict=self.strict)
        return WebsiteGraph(env=env, pages=pages, start_dt=start_dt, rng=rng)

    @staticmethod
    def _parse_pages(cfg_site_graph: dict[str, Any], strict: bool) -> dict[str, Page]:
        pages_cfg = cfg_site_graph.get("pages") or {}
        if not isinstance(pages_cfg, dict):
            raise TypeError("site_graph.pages must be a mapping/dict")

        pages: dict[str, Page] = {}

        # First pass: build page objects
        for page_name, page_cfg_any in pages_cfg.items():
            if not isinstance(page_name, str) or not page_name:
                raise ValueError("site_graph.pages keys must be non-empty strings")

            page_cfg = page_cfg_any or {}
            if not isinstance(page_cfg, dict):
                raise TypeError(f"site_graph.pages.{page_name} must be a mapping/dict")

            dropoff_p = page_cfg.get("dropoff_p", 0.0)
            try:
                dropoff_p_f = float(dropoff_p)
            except Exception as e:  # noqa: BLE001
                raise TypeError(f"site_graph.pages.{page_name}.dropoff_p must be numeric") from e
            if not (0.0 <= dropoff_p_f <= 1.0):
                raise ValueError(f"site_graph.pages.{page_name}.dropoff_p must be in [0, 1]")

            transitions_raw = page_cfg.get("transitions") or []
            transitions = SiteGraphFactory._parse_transitions(page_name, transitions_raw)

            pages[page_name] = Page(name=page_name, dropoff_p=dropoff_p_f, transitions=transitions)

        # Second pass: optionally validate transition targets exist
        if strict:
            for page in pages.values():
                for target, _w in page.transitions:
                    if target not in pages:
                        raise ValueError(
                            f"site_graph.pages.{page.name}.transitions references unknown page '{target}'"
                        )

        return pages

    @staticmethod
    def _parse_transitions(page_name: str, transitions_raw: Any) -> list[tuple[str, float]]:
        if transitions_raw is None:
            return []

        if not isinstance(transitions_raw, list):
            raise TypeError(f"site_graph.pages.{page_name}.transitions must be a list")

        out: list[tuple[str, float]] = []
        for idx, item in enumerate(transitions_raw):
            # Expect ["target", weight] (list/tuple length 2)
            if not isinstance(item, list | tuple) or len(item) != 2:
                raise TypeError(
                    f"site_graph.pages.{page_name}.transitions[{idx}] must be a 2-item list/tuple: [target, weight]"
                )
            target, weight = item[0], item[1]
            if not isinstance(target, str) or not target:
                raise ValueError(
                    f"site_graph.pages.{page_name}.transitions[{idx}][0] target must be a non-empty string"
                )
            try:
                w = float(weight)
            except Exception as e:  # noqa: BLE001
                raise TypeError(
                    f"site_graph.pages.{page_name}.transitions[{idx}][1] weight must be numeric"
                ) from e

            # Allow zero weights; WebsiteGraph.next_page will handle zero-sum
            # Disallow negative weights (usually a config mistake)
            if w < 0.0:
                raise ValueError(
                    f"site_graph.pages.{page_name}.transitions[{idx}] weight must be >= 0"
                )

            out.append((target, w))

        return out
