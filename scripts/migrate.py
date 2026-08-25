#!/usr/bin/env python3
"""Additive schema migration.

Not Alembic yet.  Every change here is an additive ADD COLUMN, which SQLite
applies without rewriting the table and which leaves existing rows intact.
Anything destructive must wait for a real migration tool.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import inspect, text
from app.db.session import engine, init_db

ADDITIONS = {
    "camera": [
        ("line_x1", "FLOAT"), ("line_y1", "FLOAT"),
        ("line_x2", "FLOAT"), ("line_y2", "FLOAT"),
        ("inside_side", "INTEGER DEFAULT 1"),
        ("depth_grows_inward", "BOOLEAN DEFAULT 1"),
        ("min_travel", "FLOAT DEFAULT 0.06"),
    ],
    "recognition_event": [
        ("direction", "VARCHAR(16) DEFAULT 'UNKNOWN'"),
        ("direction_reason", "VARCHAR(96) DEFAULT ''"),
    ],
}

def main():
    init_db()
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in ADDITIONS.items():
            if table not in insp.get_table_names():
                print(f"  {table}: not present, created by init_db"); continue
            have = {c["name"] for c in insp.get_columns(table)}
            for name, decl in cols:
                if name in have:
                    continue
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {decl}"))
                print(f"  {table}.{name} added")
    print("migration complete")

if __name__ == "__main__":
    main()
