from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

import duckdb

from .schema import EVENTS_TABLE_NAME, create_schema


@dataclass(frozen=True)
class DuckDBWriteResult:
    num_events: int
    duration_ms: float


class DuckDBAdapter:
    """
    DuckDB persistence adapter. Owns the connection and schema.
    """

    def __init__(self, path: str, *, clean_slate: bool) -> None:
        self.path = path
        self.clean_slate = clean_slate
        self._conn: duckdb.DuckDBPyConnection | None = None

    def open(self) -> None:
        if self.clean_slate and os.path.exists(self.path):
            os.remove(self.path)

        # Ensure parent dir exists
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._conn = duckdb.connect(self.path)
        create_schema(self._conn)

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("DuckDBAdapter not opened. Call open() first.")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def write_events(self, rows: Sequence[tuple]) -> DuckDBWriteResult:
        """
        Writes a batch of rows matching the events schema.
        Returns count and duration.
        """
        if not rows:
            return DuckDBWriteResult(num_events=0, duration_ms=0.0)

        t0 = time.perf_counter()

        # Parameterized insert
        self.conn.executemany(
            f"""
            INSERT INTO {EVENTS_TABLE_NAME} (
                run_id, event_id,
                ts_utc, sim_time_s,
                user_id, session_id,
                event_type,
                intent_source, channel, page,
                value_num, value_str,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return DuckDBWriteResult(num_events=len(rows), duration_ms=dt_ms)

    def count_events(self, run_id: str) -> int:
        """
        Convenience method for sanity checks/tests.
        """
        res = self.conn.execute(
            f"SELECT COUNT(*) FROM {EVENTS_TABLE_NAME} WHERE run_id = ?",
            [run_id],
        ).fetchone()
        return int(res[0]) if res else 0
