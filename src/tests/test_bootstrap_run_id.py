from sim.core.config import parse_config
from sim.features.bootstrap.service import bootstrap_run


def test_run_id_auto_is_deterministic(tmp_path):
    cfg_dict = {
        "run": {"run_id": "auto", "seed": 123, "start_date": "2026-01-01", "num_days": 1},
        "storage": {"duckdb_path": str(tmp_path / "sim.duckdb"), "clean_slate": True},
        "logging": {"level": "INFO"},
    }
    cfg = parse_config(cfg_dict)

    r1 = bootstrap_run(cfg)
    r2 = bootstrap_run(cfg)

    assert r1.ctx.run_id == r2.ctx.run_id


def test_run_id_respects_explicit_value(tmp_path):
    cfg_dict = {
        "run": {"run_id": "my_run", "seed": 123, "start_date": "2026-01-01", "num_days": 1},
        "storage": {"duckdb_path": str(tmp_path / "sim.duckdb"), "clean_slate": True},
        "logging": {"level": "INFO"},
    }
    cfg = parse_config(cfg_dict)

    r = bootstrap_run(cfg)
    assert r.ctx.run_id == "my_run"
