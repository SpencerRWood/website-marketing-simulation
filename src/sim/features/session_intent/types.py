from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SessionIntent:
    """
    A single website-arrival signal that should be resolved into a concrete session.

    Option B-ready:
      - audience_id identifies the person in the marketing audience universe
      - channel is the channel name when intent_source is channel:<name>
    """

    intent_id: str
    ts_utc: datetime

    # "baseline" OR "channel:<name>"
    intent_source: str

    # Convenience for downstream filtering; can be redundant with intent_source.
    channel: str | None = None

    # Identity in the addressable audience universe (not yet a site user)
    audience_id: str | None = None

    # Freeform metadata (e.g., campaign_id, creative_id, placement, etc.)
    payload: dict[str, Any] | None = None
