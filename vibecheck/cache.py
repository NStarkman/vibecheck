"""Tiny SQLite key/value cache so repeat runs don't re-hit the APIs.

Audio features, lyrics and artist genres basically never change, so they are cached
for a long time. Misses (e.g. "no lyrics found") are cached too, for a shorter time,
so we don't hammer a service for tracks it doesn't have.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_MISSING = object()
DAY = 86_400


class Cache:
    def __init__(self, path: str | Path | None):
        """`path=None` gives an in-memory cache (handy for tests and the demo)."""
        self._lock = threading.Lock()
        if path is None:
            target = ":memory:"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            target = str(path)
        self._db = sqlite3.connect(target, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, expires REAL)"
        )
        self._db.commit()

    def get(self, key: str, default: Any = _MISSING) -> Any:
        with self._lock:
            row = self._db.execute("SELECT v, expires FROM kv WHERE k = ?", (key,)).fetchone()
        if row is None or (row[1] is not None and row[1] < time.time()):
            return default
        return json.loads(row[0])

    def has(self, key: str) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    def set(self, key: str, value: Any, ttl: float | None = 180 * DAY) -> None:
        expires = None if ttl is None else time.time() + ttl
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO kv (k, v, expires) VALUES (?, ?, ?)",
                (key, json.dumps(value), expires),
            )
            self._db.commit()

    def clear(self, prefix: str = "") -> int:
        with self._lock:
            cur = self._db.execute("DELETE FROM kv WHERE k LIKE ?", (prefix + "%",))
            self._db.commit()
            return cur.rowcount
