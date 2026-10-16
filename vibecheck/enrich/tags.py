"""Optional Last.fm tags (needs a free API key: https://www.last.fm/api/account/create).

Two uses: a genre fallback for artists Spotify gives no genres for, and crowd mood tags
("sad", "chill", "happy"...) as one more mood signal.
"""
from __future__ import annotations

import pandas as pd
import requests

from ..cache import DAY, Cache
from ..http import request_json

BASE = "https://ws.audioscrobbler.com/2.0/"

MOOD_TAGS = {
    # tag -> (valence, arousal) on 0-1 scales
    "sad": (0.15, 0.3), "melancholy": (0.2, 0.3), "melancholic": (0.2, 0.3), "depressing": (0.1, 0.25),
    "heartbreak": (0.15, 0.35), "breakup": (0.2, 0.4), "lonely": (0.15, 0.25), "dark": (0.25, 0.5),
    "happy": (0.85, 0.65), "feel good": (0.85, 0.6), "upbeat": (0.8, 0.75), "fun": (0.8, 0.7),
    "chill": (0.6, 0.25), "chillout": (0.6, 0.25), "mellow": (0.55, 0.25), "relaxing": (0.6, 0.2),
    "calm": (0.6, 0.15), "energetic": (0.65, 0.9), "party": (0.8, 0.85), "angry": (0.2, 0.9),
    "aggressive": (0.25, 0.95), "romantic": (0.7, 0.35), "love": (0.7, 0.4), "nostalgic": (0.5, 0.35),
    "dreamy": (0.55, 0.3), "epic": (0.6, 0.85), "uplifting": (0.85, 0.65), "emotional": (0.35, 0.45),
}


class LastFM:
    def __init__(self, api_key: str, session: requests.Session, cache: Cache):
        self.api_key = api_key
        self.session = session
        self.cache = cache

    def _tags(self, method: str, **params) -> list[tuple[str, int]]:
        key = f"lastfm:{method}:" + "|".join(f"{k}={v}".lower() for k, v in sorted(params.items()))
        hit = self.cache.get(key, None)
        if hit is not None:
            return [tuple(t) for t in hit]
        data = request_json(
            self.session, "GET", BASE,
            params={"method": method, "api_key": self.api_key, "format": "json", "autocorrect": 1, **params},
        ) or {}
        tags = [(t["name"].lower(), int(t.get("count", 0))) for t in (data.get("toptags") or {}).get("tag", [])[:15]]
        self.cache.set(key, tags, ttl=90 * DAY)
        return tags

    def artist_tags(self, artist: str) -> list[tuple[str, int]]:
        return self._tags("artist.gettoptags", artist=artist)

    def track_tags(self, artist: str, track: str) -> list[tuple[str, int]]:
        return self._tags("track.gettoptags", artist=artist, track=track)


def tag_mood(tags: list[tuple[str, int]]) -> tuple[float, float] | None:
    """Count-weighted (valence, arousal) from mood tags, or None if no mood tags."""
    num_v = num_a = den = 0.0
    for name, count in tags:
        if name in MOOD_TAGS:
            w = max(count, 1)
            v, a = MOOD_TAGS[name]
            num_v += w * v
            num_a += w * a
            den += w
    return (num_v / den, num_a / den) if den else None


def add_lastfm(df: pd.DataFrame, lastfm: LastFM, genre_fallback: bool = True) -> pd.DataFrame:
    df = df.copy()
    tag_v, tag_a = [], []
    for _, row in df.iterrows():
        artist = (row["artists"] or [""])[0]
        try:
            mood = tag_mood(lastfm.track_tags(artist, row["name"]))
        except Exception:
            mood = None
        tag_v.append(mood[0] if mood else float("nan"))
        tag_a.append(mood[1] if mood else float("nan"))
    df["tag_valence"], df["tag_arousal"] = tag_v, tag_a
    if genre_fallback and "genres" in df:
        def fill(row):
            if row["genres"]:
                return row["genres"]
            try:
                return [t for t, c in lastfm.artist_tags((row["artists"] or [""])[0])[:5] if t not in MOOD_TAGS]
            except Exception:
                return []
        df["genres"] = df.apply(fill, axis=1)
    return df
