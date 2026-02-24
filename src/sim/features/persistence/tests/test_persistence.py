from __future__ import annotations

from datetime import UTC, datetime

import simpy


def _mk_event(*, run_id: str, event_id: str, ts: datetime, sim_time_s: float, event_type: str):
    from sim.features.persistence.service import Event

    return Event(
        run_id=run_id,
        event_id=event_id,
        ts_utc=ts,
        sim_time_s=sim_time_s,
        event_type=event_type,
    )


def test_duckdb_adapter_clean_slate(tmp_path):
    from sim.features.persistence.duckdb_adapter import DuckDBAdapter

    db_path = tmp_path / "sim.duckdb"

    # First run: write something
    a1 = DuckDBAdapter(path=str(db_path), clean_slate=True)
    a1.open()
    a1.write_events(
        [
            (
                "run_a",
                "evt_00000001",
                datetime(2026, 1, 1, tzinfo=UTC),
                0.0,
                None,
                None,
                "run_started",
                None,
                None,
                None,
                None,
                None,
                None,
            )
        ]
    )
    a1.close()

    # Second run with clean_slate should remove old file and start empty
    a2 = DuckDBAdapter(path=str(db_path), clean_slate=True)
    a2.open()
    assert a2.count_events("run_a") == 0
    a2.close()


def test_persistence_flush_by_count(tmp_path):
    from sim.features.persistence.duckdb_adapter import DuckDBAdapter
    from sim.features.persistence.service import PersistenceService

    db_path = tmp_path / "sim.duckdb"
    adapter = DuckDBAdapter(path=str(db_path), clean_slate=True)

    svc = PersistenceService(adapter=adapter, every_n_events=3, or_every_seconds=10_000.0)
    svc.open()

    run_id = "run_count"
    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    # Emit 2 events -> should not flush yet
    svc.emit(
        _mk_event(run_id=run_id, event_id="evt_00000001", ts=t0, sim_time_s=0.0, event_type="a")
    )
    svc.emit(
        _mk_event(run_id=run_id, event_id="evt_00000002", ts=t0, sim_time_s=1.0, event_type="b")
    )
    assert adapter.count_events(run_id) == 0

    # Emit 3rd -> triggers flush
    svc.emit(
        _mk_event(run_id=run_id, event_id="evt_00000003", ts=t0, sim_time_s=2.0, event_type="c")
    )
    assert adapter.count_events(run_id) == 3

    svc.close()


def test_persistence_periodic_flush(tmp_path):
    from datetime import UTC, datetime

    from sim.features.persistence.duckdb_adapter import DuckDBAdapter
    from sim.features.persistence.service import PersistenceService

    db_path = tmp_path / "sim.duckdb"
    adapter = DuckDBAdapter(path=str(db_path), clean_slate=True)

    env = simpy.Environment()
    svc = PersistenceService(adapter=adapter, every_n_events=1_000_000, or_every_seconds=5.0)
    svc.open()
    svc.start_periodic_flush(env)

    run_id = "run_timer"
    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    svc.emit(
        _mk_event(run_id=run_id, event_id="evt_00000001", ts=t0, sim_time_s=0.0, event_type="a")
    )
    svc.emit(
        _mk_event(run_id=run_id, event_id="evt_00000002", ts=t0, sim_time_s=1.0, event_type="b")
    )
    assert adapter.count_events(run_id) == 0

    # run slightly past the boundary to ensure the timer event is processed
    env.run(until=5.000001)
    assert adapter.count_events(run_id) == 2

    svc.close()


def _count_events_from_disk(db_path, run_id: str) -> int:
    import duckdb

    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = $1",
            [run_id],
        ).fetchone()
        if row is None:
            return 0
        return int(row[0])
    finally:
        con.close()


def test_persistence_close_flushes_remaining(tmp_path):
    from datetime import UTC, datetime

    from sim.features.persistence.duckdb_adapter import DuckDBAdapter
    from sim.features.persistence.service import PersistenceService

    db_path = tmp_path / "sim.duckdb"
    adapter = DuckDBAdapter(path=str(db_path), clean_slate=True)

    svc = PersistenceService(adapter=adapter, every_n_events=1_000_000, or_every_seconds=10_000.0)
    svc.open()

    run_id = "run_close"
    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    svc.emit(
        _mk_event(run_id=run_id, event_id="evt_00000001", ts=t0, sim_time_s=0.0, event_type="a")
    )
    assert adapter.count_events(run_id) == 0

    svc.close()

    # adapter is closed; verify persisted rows by opening a new connection
    assert _count_events_from_disk(db_path, run_id) == 1
