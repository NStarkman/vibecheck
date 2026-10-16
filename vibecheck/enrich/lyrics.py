"""Lyrics from LRCLIB (https://lrclib.net), a free, open lyrics database with no API key.

GET /api/get?track_name&artist_name&album_name&duration   (exact match, duration in seconds)
GET /api/search?track_name&artist_name                     (fuzzy fallback)
Responses carry plainLyrics, syncedLyrics ([mm:ss.xx] lines) and an `instrumental` flag.
Lyrics are only used locally for NLP features; vibecheck never displays them.
"""
from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import pandas as pd
import requests

from ..cache import DAY, Cache
from ..http import request_json

BASE = "https://lrclib.net/api"
_SUFFIX = re.compile(
    r"\s*(-\s*(\d{4}\s*)?(remaster(ed)?|live|mono|stereo|radio edit|single version|edit|version|mix|demo|acoustic)\b.*$"
    r"|\((feat\.?|ft\.?|with)\s[^)]*\)|\[(feat\.?|ft\.?)\s[^\]]*\])",
    re.I,
)
_TIMESTAMP = re.compile(r"^\[\d+:\d+(\.\d+)?\]\s?", re.M)


def clean_title(title: str) -> str:
    return _SUFFIX.sub("", title or "").strip()


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


class LRCLib:
    def __init__(self, session: requests.Session, cache: Cache, max_workers: int = 4):
        self.session = session
        self.cache = cache
        self.max_workers = max_workers

    def lyrics(self, title: str, artist: str, album: str | None = None, duration_s: float | None = None) -> dict | None:
        key = f"lrclib:{_norm(artist)}|{_norm(clean_title(title))}"
        hit = self.cache.get(key, "__miss__")
        if hit != "__miss__":
            return hit
        result = self._lookup(title, artist, album, duration_s)
        self.cache.set(key, result, ttl=365 * DAY if result else 30 * DAY)
        return result

    def _lookup(self, title, artist, album, duration_s) -> dict | None:
        if duration_s:
            params = {"track_name": title, "artist_name": artist, "duration": int(round(duration_s))}
            if album:
                params["album_name"] = album
            rec = request_json(self.session, "GET", f"{BASE}/get", params=params, not_found_ok=True)
            if rec:
                return _pack(rec)
        candidates = request_json(
            self.session, "GET", f"{BASE}/search",
            params={"track_name": clean_title(title), "artist_name": artist}, not_found_ok=True,
        ) or []
        best, best_score = None, 0.0
        for c in candidates:
            score = 0.6 * _similar(clean_title(title), clean_title(c.get("trackName", ""))) + 0.4 * _similar(
                artist, c.get("artistName", "")
            )
            if duration_s and c.get("duration"):
                score -= min(abs(c["duration"] - duration_s) / 60.0, 0.3)
            if score > best_score:
                best, best_score = c, score
        return _pack(best) if best and best_score >= 0.7 else None

    def add_lyrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def one(row):
            artist = (row["artists"] or [""])[0] if isinstance(row["artists"], list) else row["artists"]
            dur = row["duration_ms"] / 1000 if pd.notna(row.get("duration_ms")) else None
            try:
                return self.lyrics(row["name"], artist, row.get("album"), dur)
            except Exception:
                return None

        with ThreadPoolExecutor(self.max_workers) as pool:
            results = list(pool.map(one, [r for _, r in df.iterrows()]))
        df["lyrics"] = [r["text"] if r else None for r in results]
        df["instrumental"] = [bool(r and r["instrumental"]) for r in results]
        df["has_lyrics"] = df["lyrics"].notna() & (df["lyrics"].str.len() > 40)
        return df


def _pack(rec: dict) -> dict:
    text = rec.get("plainLyrics") or _TIMESTAMP.sub("", rec.get("syncedLyrics") or "")
    return {"text": text or None, "instrumental": bool(rec.get("instrumental"))}
