"""Wrapped-style stats and listening-behaviour signals from the streaming-history export."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STREAM_MS = 30_000  # Spotify counts a stream after 30 seconds


def resolve_tz(tz):
    """'local' / None -> this computer's current UTC offset (DST-naive); otherwise pass through."""
    if tz in (None, "local"):
        import datetime as _dt

        return _dt.datetime.now().astimezone().tzinfo
    return tz


@dataclass
class Behavior:
    total_hours: float
    n_plays: int
    first: pd.Timestamp
    last: pd.Timestamp
    hour_share: pd.Series  # local hour -> share of listening time
    night_share: float  # share of listening between midnight and 5am
    skip_rate: float
    repeat_share: float  # share of plays that replay a track already played that day
    discovery_share: float  # share of plays that are a first-ever listen
    top_artists: pd.DataFrame
    top_tracks: pd.DataFrame
    weekly_hours: pd.Series
    yearly: pd.DataFrame = field(default_factory=pd.DataFrame)
    peak_binge: dict = field(default_factory=dict)


def listening_behavior(hist: pd.DataFrame, tz: str = "local", top_n: int = 10) -> Behavior:
    """`tz` is an IANA name like "America/New_York" (best), or "local" for this computer's offset."""
    h = hist.copy()
    h["local"] = h["ts"].dt.tz_convert(resolve_tz(tz))
    h["hours"] = h["ms_played"] / 3.6e6
    plays = h[h["ms_played"] >= STREAM_MS].copy()
    key = plays["track_id"].fillna(plays["artist"].fillna("") + " - " + plays["name"].fillna(""))
    plays["key"] = key

    hour_share = h.groupby(h["local"].dt.hour)["hours"].sum().reindex(range(24), fill_value=0)
    hour_share = hour_share / hour_share.sum()

    skip_info = h["skipped"].notna() | h["reason_end"].notna()
    skipped = (h["skipped"] == True) | ((h["reason_end"] == "fwdbtn") & (h["ms_played"] < STREAM_MS))  # noqa: E712
    skip_rate = float(skipped[skip_info].mean()) if skip_info.any() else np.nan

    plays["day"] = plays["local"].dt.date
    first_in_day = ~plays.duplicated(subset=["day", "key"])
    repeat_share = float(1 - first_in_day.mean()) if len(plays) else np.nan
    discovery_share = float((~plays.duplicated(subset=["key"])).mean()) if len(plays) else np.nan
    binge = plays.groupby(["day", "key"]).size()
    peak = {}
    if len(binge):
        (day, k), count = binge.idxmax(), int(binge.max())
        name = plays.loc[plays["key"] == k, ["name", "artist"]].iloc[0]
        peak = {"date": str(day), "track": name["name"], "artist": name["artist"], "plays": count}

    top_artists = (
        h.groupby("artist")["hours"].sum().sort_values(ascending=False).head(top_n).rename("hours").reset_index()
    )
    top_tracks = (
        plays.groupby(["name", "artist"]).agg(plays=("key", "size"), hours=("hours", "sum"))
        .sort_values("plays", ascending=False).head(top_n).reset_index()
    )
    weekly = h.set_index("local")["hours"].resample("W").sum()

    yearly_rows = []
    for year, g in plays.groupby(plays["local"].dt.year):
        yearly_rows.append(
            {
                "year": int(year),
                "hours": round(float(g["hours"].sum()), 1),
                "top_artist": g.groupby("artist")["hours"].sum().idxmax(),
                "top_track": g.groupby(["name", "artist"]).size().idxmax()[0],
                "unique_tracks": int(g["key"].nunique()),
            }
        )

    return Behavior(
        total_hours=float(h["hours"].sum()),
        n_plays=len(plays),
        first=h["local"].min(),
        last=h["local"].max(),
        hour_share=hour_share,
        night_share=float(hour_share.loc[0:4].sum()),
        skip_rate=skip_rate,
        repeat_share=repeat_share,
        discovery_share=discovery_share,
        top_artists=top_artists,
        top_tracks=top_tracks,
        weekly_hours=weekly,
        yearly=pd.DataFrame(yearly_rows),
        peak_binge=peak,
    )


def mood_timeline(hist: pd.DataFrame, track_moods: pd.DataFrame, tz: str = "local", freq: str = "W") -> pd.DataFrame:
    """Play-time-weighted valence/arousal per period, using whichever tracks have mood scores.

    `track_moods` needs track_id, mood_valence, mood_arousal (e.g. an enriched DataFrame of your
    most-played tracks). `coverage` says what share of listening time the estimate is based on.
    """
    h = hist[hist["ms_played"] >= STREAM_MS].copy()
    h["local"] = h["ts"].dt.tz_convert(resolve_tz(tz))
    moods = track_moods.dropna(subset=["mood_valence"]).drop_duplicates("track_id").set_index("track_id")
    h = h.join(moods[["mood_valence", "mood_arousal"]], on="track_id")
    h["w"] = h["ms_played"]
    h["has"] = h["mood_valence"].notna()
    g = h.set_index("local").groupby(pd.Grouper(freq=freq))

    def agg(x):
        m = x[x["has"]]
        tot = x["w"].sum()
        if m.empty or tot == 0:
            return pd.Series({"valence": np.nan, "arousal": np.nan, "coverage": 0.0, "hours": tot / 3.6e6})
        return pd.Series(
            {
                "valence": np.average(m["mood_valence"], weights=m["w"]),
                "arousal": np.average(m["mood_arousal"], weights=m["w"]),
                "coverage": m["w"].sum() / tot,
                "hours": tot / 3.6e6,
            }
        )

    return g.apply(agg)
