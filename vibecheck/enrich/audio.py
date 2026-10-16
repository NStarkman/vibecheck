"""Audio features via ReccoBeats (free, no key), a stand-in for Spotify's removed /audio-features.

GET https://api.reccobeats.com/v1/audio-features?ids=<spotify ids, comma-separated, max 40>
returns {"content": [{"id": <reccobeats uuid>, "href": "https://open.spotify.com/track/<id>",
"danceability": ..., "valence": ..., ...}]}. Values are ReccoBeats' own estimates, not
Spotify's original numbers, and not every track is in their catalog.
"""
from __future__ import annotations

import re

import pandas as pd
import requests

from ..cache import DAY, Cache
from ..http import request_json

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "acousticness", "instrumentalness",
    "liveness", "speechiness", "loudness", "tempo", "key", "mode",
]
BASE = "https://api.reccobeats.com/v1"
_SPOTIFY_ID = re.compile(r"track/([A-Za-z0-9]{22})")


class ReccoBeats:
    def __init__(self, session: requests.Session, cache: Cache, batch_size: int = 40):
        self.session = session
        self.cache = cache
        self.batch_size = batch_size

    def _fetch(self, ids: list[str]) -> dict[str, dict]:
        data = request_json(self.session, "GET", f"{BASE}/audio-features", params={"ids": ",".join(ids)}) or {}
        found: dict[str, dict] = {}
        unmatched = []
        for item in data.get("content") or []:
            m = _SPOTIFY_ID.search(item.get("href") or "")
            feats = {k: item.get(k) for k in AUDIO_FEATURES}
            if m and m.group(1) in ids:
                found[m.group(1)] = feats
            else:
                unmatched.append(feats)
        if unmatched and len(ids) == 1:
            found[ids[0]] = unmatched[0]
        elif unmatched:
            # Couldn't map results back to IDs; ask again one at a time for the leftovers.
            for sid in ids:
                if sid not in found:
                    found.update(self._fetch([sid]))
        return found

    def audio_features(self, spotify_ids) -> pd.DataFrame:
        ids = [i for i in dict.fromkeys(spotify_ids) if isinstance(i, str) and i]
        result: dict[str, dict | None] = {}
        todo = []
        for sid in ids:
            hit = self.cache.get(f"recco:{sid}", "__miss__")
            if hit == "__miss__":
                todo.append(sid)
            else:
                result[sid] = hit
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i : i + self.batch_size]
            got = self._fetch(batch)
            for sid in batch:
                feats = got.get(sid)
                self.cache.set(f"recco:{sid}", feats, ttl=365 * DAY if feats else 14 * DAY)
                result[sid] = feats
        rows = [{"track_id": sid, **feats} for sid, feats in result.items() if feats]
        return pd.DataFrame(rows, columns=["track_id", *AUDIO_FEATURES]).set_index("track_id")


def add_audio_features(df: pd.DataFrame, provider: ReccoBeats) -> pd.DataFrame:
    """Fill audio-feature columns, keeping any values already present (e.g. from a CSV)."""
    df = df.copy()
    for col in AUDIO_FEATURES:
        if col not in df:
            df[col] = float("nan")
    need = df["valence"].isna() & df["track_id"].notna()
    if need.any():
        feats = provider.audio_features(df.loc[need, "track_id"].tolist())
        for col in AUDIO_FEATURES:
            df.loc[need, col] = df.loc[need, "track_id"].map(feats[col]) if len(feats) else float("nan")
    df["has_audio"] = df["valence"].notna()
    return df
