import duckdb

from sim.core.config import parse_config
from sim.features.bootstrap.service import bootstrap_run


def test_bootstrap_creates_db_and_events(tmp_path):
    db_path = tmp_path / "sim.duckdb"
    cfg_dict = {
        "run": {"run_id": "auto", "seed": 123, "start_date": "2026-01-01", "num_days": 2},
        "storage": {"duckdb_path": str(db_path), "clean_slate": True},
        "logging": {"level": "INFO"},
    }
    cfg = parse_config(cfg_dict)

    res = bootstrap_run(cfg)

    assert db_path.exists()
    assert res.duckdb_path == str(db_path)

    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute("SELECT event_type, sim_time_s FROM events ORDER BY event_id").fetchall()
    con.close()

    assert rows[0][0] == "run_started"
    assert rows[0][1] == 0.0

    assert rows[1][0] == "run_finished"
    # 2 days in seconds
    assert rows[1][1] == 2 * 24 * 60 * 60
