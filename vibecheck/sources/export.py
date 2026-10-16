"""Load data that doesn't need the Spotify API.

* Spotify privacy export (https://www.spotify.com/account/privacy/):
    - "Extended streaming history"  -> Streaming_History_Audio_*.json (or older endsong_*.json)
    - "Account data"                -> StreamingHistory_music_*.json (last year only, no IDs)
  Pass the zip, the folder, or a single JSON file.
* A playlist CSV from another tool (Exportify, Chosic's analyzer export, a spreadsheet...).
  Column names are matched loosely, and any audio-feature columns are kept.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_COLUMNS = [
    "ts", "track_id", "name", "artist", "album", "ms_played",
    "skipped", "reason_start", "reason_end", "shuffle", "platform", "source",
]


def _iter_json_blobs(path: Path):
    """Yield (filename, parsed JSON) for every history file in a zip / folder / file."""
    pattern = re.compile(r"(Streaming_History_Audio|endsong|StreamingHistory)[^/\\]*\.json$", re.I)
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if pattern.search(name):
                    with zf.open(name) as fh:
                        yield name, json.load(io.TextIOWrapper(fh, encoding="utf-8"))
    elif path.is_dir():
        for f in sorted(path.rglob("*.json")):
            if pattern.search(f.name):
                yield f.name, json.loads(f.read_text(encoding="utf-8"))
    elif path.is_file():
        yield path.name, json.loads(path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(path)


def _uri_to_id(uri) -> str | None:
    if not isinstance(uri, str):
        return None
    m = re.search(r"track[:/]([A-Za-z0-9]{22})", uri)
    if m:
        return m.group(1)
    return uri if re.fullmatch(r"[A-Za-z0-9]{22}", uri) else None


def load_streaming_history(path: str | Path, include_podcasts: bool = False) -> pd.DataFrame:
    """Return one row per play with UTC timestamps (convert with `.dt.tz_convert(...)`)."""
    rows = []
    for name, blob in _iter_json_blobs(Path(path)):
        if not isinstance(blob, list):
            continue
        for r in blob:
            if "ts" in r:  # extended history
                track = r.get("master_metadata_track_name")
                if track is None and not include_podcasts:
                    continue
                rows.append(
                    {
                        "ts": r.get("ts"),
                        "track_id": _uri_to_id(r.get("spotify_track_uri")),
                        "name": track or r.get("episode_name"),
                        "artist": r.get("master_metadata_album_artist_name") or r.get("episode_show_name"),
                        "album": r.get("master_metadata_album_album_name"),
                        "ms_played": r.get("ms_played"),
                        "skipped": r.get("skipped"),
                        "reason_start": r.get("reason_start"),
                        "reason_end": r.get("reason_end"),
                        "shuffle": r.get("shuffle"),
                        "platform": r.get("platform"),
                        "source": "extended",
                    }
                )
            elif "endTime" in r:  # account-data history (no IDs, no skip flags)
                rows.append(
                    {
                        "ts": r.get("endTime"),
                        "track_id": None,
                        "name": r.get("trackName"),
                        "artist": r.get("artistName"),
                        "album": None,
                        "ms_played": r.get("msPlayed"),
                        "skipped": None,
                        "reason_start": None,
                        "reason_end": None,
                        "shuffle": None,
                        "platform": None,
                        "source": "account",
                    }
                )
    df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    if df.empty:
        raise ValueError(f"No streaming-history records found in {path}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="mixed")
    df["ms_played"] = pd.to_numeric(df["ms_played"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates(subset=["ts", "name", "artist", "ms_played"]).sort_values("ts")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- CSV
_CSV_ALIASES = {
    "track_id": ["track uri", "spotify uri", "uri", "track id", "spotify id", "spotify track id", "id", "spotify_track_uri", "spotify link", "link"],
    "name": ["track name", "song", "song name", "title", "name", "track"],
    "artists": ["artist name(s)", "artist names", "artists", "artist", "artist name"],
    "album": ["album name", "album"],
    "release_date": ["release date", "album release date", "release_date", "release year", "year"],
    "genres": ["genres", "artist genres", "genre"],
    "duration_ms": ["duration (ms)", "duration_ms", "track duration (ms)", "duration"],
    "explicit": ["explicit"],
    "added_at": ["added at", "date added", "added_at"],
    "isrc": ["isrc"],
    "danceability": ["danceability"],
    "energy": ["energy"],
    "valence": ["valence", "happiness", "happy"],
    "acousticness": ["acousticness", "acoustic"],
    "instrumentalness": ["instrumentalness", "instrumental"],
    "liveness": ["liveness"],
    "speechiness": ["speechiness"],
    "loudness": ["loudness"],
    "tempo": ["tempo", "bpm"],
    "key": ["key"],
    "mode": ["mode"],
}
UNIT_FEATURES = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "liveness", "speechiness"]


def _split_list(value) -> list[str]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    sep = ";" if ";" in value else ","
    return [v.strip() for v in value.split(sep) if v.strip()]


def _duration_to_ms(v):
    if isinstance(v, str) and ":" in v:
        parts = [int(p) for p in v.split(":")]
        secs = parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) > 2 else 0)
        return secs * 1000
    v = pd.to_numeric(v, errors="coerce")
    if pd.isna(v):
        return None
    return int(v * 1000) if v < 3600 else int(v)  # seconds vs ms


def load_tracks_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in raw.columns}
    out = pd.DataFrame(index=raw.index)
    for target, aliases in _CSV_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                out[target] = raw[lower[alias]]
                break
    if "name" not in out:
        raise ValueError(f"Couldn't find a track-name column in {path}; columns are {list(raw.columns)}")
    out["track_id"] = out["track_id"].map(_uri_to_id) if "track_id" in out else None
    out["artists"] = out["artists"].map(_split_list) if "artists" in out else [[] for _ in range(len(out))]
    out["artist_ids"] = [[] for _ in range(len(out))]
    if "genres" in out:
        out["genres"] = out["genres"].map(_split_list)
    if "duration_ms" in out:
        out["duration_ms"] = out["duration_ms"].map(_duration_to_ms)
    if "release_date" in out:
        out["release_date"] = out["release_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    for col in UNIT_FEATURES:
        if col in out:
            vals = pd.to_numeric(out[col], errors="coerce")
            out[col] = vals / 100.0 if vals.max(skipna=True) > 1.5 else vals  # 0-100 scales
    for col in ("loudness", "tempo", "key", "mode"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "explicit" in out:
        out["explicit"] = out["explicit"].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes", "explicit") if pd.notna(v) else np.nan
        )
    out["position"] = np.arange(len(out))
    return out.reset_index(drop=True)
