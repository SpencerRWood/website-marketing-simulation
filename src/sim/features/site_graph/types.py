from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sim.core.rng import RNG


@dataclass(frozen=True)
class Page:
    name: str
    dropoff_p: float
    transitions: list[tuple[str, float]]


class WebsiteGraph:
    """
    Holds the site structure and provides a simulation clock.
    env.now is seconds since sim start.
    """

    def __init__(
        self,
        env,
        pages: dict[str, Page],
        start_dt: datetime,
        rng: RNG,
    ):
        self.env = env
        self.pages = pages or {}
        self.rng = rng

        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)
        else:
            start_dt = start_dt.astimezone(UTC)

        self.start_dt = start_dt

    # ----- Authoritative timestamps (UTC) -----
    def get_current_time(self) -> datetime:
        return self.start_dt + timedelta(seconds=float(self.env.now))

    # ----- Page helpers -----
    def get_page(self, name: str) -> Page | None:
        return self.pages.get(name)

    def next_page(self, current_name: str) -> str | None:
        page = self.get_page(current_name)
        if not page or not page.transitions:
            return None

        names = [t[0] for t in page.transitions]
        probs = [t[1] for t in page.transitions]

        s = sum(probs)
        if s <= 0:
            return None

        probs = [p / s for p in probs]

        # critical: use seeded wrapper only
        return self.rng.choices(names, weights=probs, k=1)[0]
