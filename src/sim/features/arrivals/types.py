from __future__ import annotations

from typing import Protocol


class ArrivalModel(Protocol):
    def start(self, env) -> None: ...
