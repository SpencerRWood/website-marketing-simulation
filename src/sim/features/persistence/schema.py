from __future__ import annotations

EVENTS_TABLE_NAME = "events"

EVENTS_DDL = f"""
CREATE TABLE IF NOT EXISTS {EVENTS_TABLE_NAME} (
    run_id TEXT NOT NULL,
    event_id TEXT NOT NULL,

    ts_utc TIMESTAMP NOT NULL,
    sim_time_s DOUBLE NOT NULL,

    user_id TEXT,
    session_id TEXT,

    event_type TEXT NOT NULL,

    intent_source TEXT,
    channel TEXT,
    page TEXT,

    value_num DOUBLE,
    value_str TEXT,

    payload_json TEXT
);
"""

# Optional but helpful for query speed
EVENTS_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS idx_events_run_id ON {EVENTS_TABLE_NAME}(run_id);",
    f"CREATE INDEX IF NOT EXISTS idx_events_event_type ON {EVENTS_TABLE_NAME}(event_type);",
    f"CREATE INDEX IF NOT EXISTS idx_events_ts_utc ON {EVENTS_TABLE_NAME}(ts_utc);",
]


def create_schema(conn) -> None:
    """
    Create tables/indexes. No migrations. Safe to call per run.
    """
    conn.execute(EVENTS_DDL)
    for ddl in EVENTS_INDEXES:
        conn.execute(ddl)
