from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    events: int
    users: int
    sessions: int
    conversions: int


class RunExplorerService:
    """
    Read-only query helper for the run-scoped DuckDB events table.
    Assumes table name: events
    """

    def __init__(self, *, duckdb_path: str) -> None:
        self._path = duckdb_path

    def _connect(self) -> duckdb.DuckDBPyConnection:
        # read_only helps guard against accidental writes
        return duckdb.connect(self._path, read_only=True)

    def list_run_ids(self, *, limit: int = 20) -> list[str]:
        q = """
        SELECT DISTINCT run_id
        FROM events
        ORDER BY run_id DESC
        LIMIT ?
        """
        with self._connect() as con:
            rows = con.execute(q, [int(limit)]).fetchall()
        return [r[0] for r in rows]

    def summary(self, *, run_id: str) -> RunSummary:
        q = """
        SELECT
          run_id,
          COUNT(*) AS events,
          COUNT(DISTINCT user_id) AS users,
          COUNT(DISTINCT session_id) AS sessions,
          SUM(CASE WHEN event_type='conversion' THEN 1 ELSE 0 END) AS conversions
        FROM events
        WHERE run_id = ?
        GROUP BY run_id
        """
        with self._connect() as con:
            row = con.execute(q, [run_id]).fetchone()

        if row is None:
            raise ValueError(f"run_id not found: {run_id}")

        return RunSummary(
            run_id=str(row[0]),
            events=int(row[1]),
            users=int(row[2]),
            sessions=int(row[3]),
            conversions=int(row[4]),
        )

    def event_counts(self, *, run_id: str) -> list[dict[str, Any]]:
        q = """
        SELECT
          event_type,
          COALESCE(channel, '') AS channel,
          COUNT(*) AS n
        FROM events
        WHERE run_id = ?
        GROUP BY 1,2
        ORDER BY n DESC, event_type, channel
        """
        with self._connect() as con:
            rows = con.execute(q, [run_id]).fetchall()

        out: list[dict[str, Any]] = []
        for event_type, channel, n in rows:
            out.append({"event_type": event_type, "channel": channel, "n": int(n)})
        return out

    def head_events(self, *, run_id: str, limit: int = 50) -> list[dict[str, Any]]:
        q = """
        SELECT *
        FROM events
        WHERE run_id = ?
        ORDER BY ts_utc, event_id
        LIMIT ?
        """
        with self._connect() as con:
            cur = con.execute(q, [run_id, int(limit)])
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({cols[i]: r[i] for i in range(len(cols))})
        return out
