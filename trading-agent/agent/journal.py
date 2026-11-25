"""SQLite log of every decision and fill, so you can audit what the agent did."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class Journal:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                ts TEXT, mode TEXT, symbol TEXT, kind TEXT, action TEXT,
                detail TEXT, data TEXT
            );
            CREATE TABLE IF NOT EXISTS fills (
                ts TEXT, mode TEXT, symbol TEXT, side TEXT, qty REAL,
                price REAL, fee REAL, order_id TEXT, reason TEXT
            );
            CREATE TABLE IF NOT EXISTS equity (ts TEXT, mode TEXT, equity REAL, cash REAL);
            """
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def event(self, mode, symbol, kind, action, detail, data=None):
        self.db.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
            (self._now(), mode, symbol, kind, action, detail, json.dumps(data or {})),
        )
        self.db.commit()

    def fill(self, mode, fill, reason):
        self.db.execute(
            "INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?)",
            (self._now(), mode, fill.symbol, fill.side, fill.qty, fill.price, fill.fee, fill.order_id, reason),
        )
        self.db.commit()

    def equity(self, mode, equity, cash):
        self.db.execute("INSERT INTO equity VALUES (?,?,?,?)", (self._now(), mode, equity, cash))
        self.db.commit()
