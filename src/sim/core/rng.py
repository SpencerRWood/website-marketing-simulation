from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass
class RNG:
    seed: int

    def __post_init__(self) -> None:
        self._r = random.Random(self.seed)

    def random(self) -> float:
        return self._r.random()

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def choice(self, seq: Sequence[T]) -> T:
        return self._r.choice(seq)

    def choices(
        self,
        population: Sequence[T],
        weights: Sequence[float] | None = None,
        k: int = 1,
    ) -> list[T]:
        return self._r.choices(population, weights=weights, k=k)
