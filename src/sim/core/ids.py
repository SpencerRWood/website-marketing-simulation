from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def canonical_json(obj: dict[str, Any]) -> str:
    # stable serialization for hashing
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def deterministic_run_id_from_config(cfg_raw: dict[str, Any], length: int = 12) -> str:
    """
    Deterministic run_id derived from the full config content.
    - If you run twice with the same YAML content, you get the same run_id.
    - If config changes, run_id changes.
    """
    s = canonical_json(cfg_raw).encode("utf-8")
    h = hashlib.sha1(s).hexdigest()
    return h[:length]


@dataclass(slots=True)
class IdsService:
    run_id: str
    _counters: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def next_id(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return f"{prefix}_{self.run_id}_{n:08d}"
