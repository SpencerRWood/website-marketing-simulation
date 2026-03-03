from __future__ import annotations

from datetime import UTC, datetime

import simpy

from sim.features.persistence.duckdb_adapter import DuckDBAdapter
from sim.features.persistence.service import Event, PersistenceService
from sim.features.run_explorer.service import RunExplorerService


def test_run_explorer_summary_counts(tmp_path):
    db_path = tmp_path / "sim.duckdb"

    adapter = DuckDBAdapter(path=str(db_path), clean_slate=True)
    persistence = PersistenceService(
        adapter=adapter,
        every_n_events=1_000_000,
        or_every_seconds=1_000_000,
    )
    persistence.open()

    env = simpy.Environment()
    run_id = "run_test"
    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    # Emit a couple of events using the canonical Event model
    persistence.emit(
        Event(
            run_id=run_id,
            event_id="evt_1",
            ts_utc=t0,
            sim_time_s=float(env.now),
            user_id="u1",
            session_id="s1",
            event_type="session_start",
            intent_source=None,
            channel="paid_search",
            page="home",
            value_num=None,
            value_str=None,
            payload=None,
        )
    )
    persistence.emit(
        Event(
            run_id=run_id,
            event_id="evt_2",
            ts_utc=t0,
            sim_time_s=float(env.now) + 1.0,
            user_id="u1",
            session_id="s1",
            event_type="conversion",
            intent_source=None,
            channel="paid_search",
            page=None,
            value_num=1.0,
            value_str=None,
            payload=None,
        )
    )

    persistence.flush(reason="test")
    persistence.close()

    ex = RunExplorerService(duckdb_path=str(db_path))
    s = ex.summary(run_id=run_id)

    assert s.run_id == run_id
    assert s.events == 2
    assert s.users == 1
    assert s.sessions == 1
    assert s.conversions == 1

    counts = ex.event_counts(run_id=run_id)
    assert any(r["event_type"] == "conversion" and r["n"] == 1 for r in counts)
