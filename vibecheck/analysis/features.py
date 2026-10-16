"""Per-track mood coordinates and the playlist-level feature vector everything else builds on."""
from __future__ import annotations

import datetime as _dt
import math
from collections import Counter

import numpy as np
import pandas as pd

from ..enrich.genres import genre_distribution, music_scores
from ..nlp.emotion import EMO_COLS
from ..nlp.lexicon import THEMES

# How much each signal counts toward a track's mood (renormalised over what's available).
VALENCE_WEIGHTS = {"audio": 0.6, "lyrics": 0.3, "tags": 0.1}
AROUSAL_WEIGHTS = {"audio": 0.8, "lyrics": 0.15, "tags": 0.05}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce") if name in df else pd.Series(np.nan, index=df.index)


def add_track_mood(df: pd.DataFrame) -> pd.DataFrame:
    """mood_valence / mood_arousal per track by blending audio, lyric and tag signals."""
    df = df.copy()
    tempo_norm = ((_col(df, "tempo") - 60) / 120).clip(0, 1)
    energy = _col(df, "energy")
    audio_a = 0.7 * energy + 0.3 * tempo_norm.fillna(energy)
    sources_v = {"audio": _col(df, "valence"), "lyrics": _col(df, "lyric_valence"), "tags": _col(df, "tag_valence")}
    sources_a = {"audio": audio_a, "lyrics": _col(df, "lyric_arousal"), "tags": _col(df, "tag_arousal")}

    def blend(sources, weights):
        num = pd.Series(0.0, index=df.index)
        den = pd.Series(0.0, index=df.index)
        for k, s in sources.items():
            ok = s.notna()
            num[ok] += weights[k] * s[ok]
            den[ok] += weights[k]
        return (num / den).where(den > 0)

    df["mood_valence"] = blend(sources_v, VALENCE_WEIGHTS)
    df["mood_arousal"] = blend(sources_a, AROUSAL_WEIGHTS)
    lyric_v = _col(df, "lyric_valence")
    df["sad_banger"] = (lyric_v < 0.4) & ((_col(df, "danceability") > 0.6) | (energy > 0.65))
    return df


def _entropy(shares) -> float:
    p = np.array([s for s in shares if s > 0], dtype=float)
    if len(p) <= 1:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(p)))


def playlist_features(df: pd.DataFrame, now_year: int | None = None) -> dict:
    """Flat dict of playlist statistics (used by the report, the roasts and the ML models)."""
    now_year = now_year or _dt.date.today().year
    n = len(df)
    f: dict = {"n_tracks": n}
    artists = Counter(a for lst in df.get("artists", []) for a in (lst[:1] if isinstance(lst, list) else []))
    f["n_artists"] = len(artists)
    if artists:
        top_artist, top_count = artists.most_common(1)[0]
        f["top_artist"], f["top_artist_share"] = top_artist, top_count / n
        shares = np.array(list(artists.values())) / n
        f["artist_hhi"] = float((shares ** 2).sum())
        f["artist_repeat_rate"] = 1 - len(artists) / n
    dist = genre_distribution(df) if "macro_genres" in df else pd.Series(dtype=float)
    f["genre_dist"] = dist
    f["genre_coverage"] = float(df["macro_genres"].map(bool).mean()) if "macro_genres" in df and n else 0.0
    if len(dist):
        f["top_genre"], f["top_genre_share"] = dist.index[0], float(dist.iloc[0])
        f["genre_entropy"] = _entropy(dist.values)
    f["music"] = music_scores(dist) if len(dist) else pd.Series(dtype=float)

    for col in ["valence", "energy", "danceability", "acousticness", "instrumentalness", "speechiness", "liveness", "tempo", "loudness"]:
        vals = _col(df, col)
        f[f"mean_{col}"] = float(vals.mean()) if vals.notna().any() else np.nan
    f["audio_coverage"] = float(_col(df, "valence").notna().mean()) if n else 0.0
    f["minor_share"] = float((_col(df, "mode") == 0).mean()) if _col(df, "mode").notna().any() else np.nan
    f["instrumental_share"] = float((_col(df, "instrumentalness") > 0.5).mean()) if f["audio_coverage"] else np.nan

    years = _col(df, "release_year").dropna()
    if len(years):
        f["median_year"] = int(years.median())
        f["year_iqr"] = (int(years.quantile(0.25)), int(years.quantile(0.75)))
        f["share_pre2000"] = float((years < 2000).mean())
        f["share_recent"] = float((years >= now_year - 2).mean())
        decades = (years // 10 * 10).astype(int).value_counts(normalize=True).sort_index()
        f["decade_shares"] = decades
        f["top_decade"] = int(decades.idxmax())
        mode_year = years.astype(int).value_counts().idxmax()
        f["mode_year"] = int(mode_year)
        f["share_mode_window"] = float(((years >= mode_year - 2) & (years <= mode_year + 2)).mean())

    if "explicit" in df and df["explicit"].notna().any():
        f["explicit_share"] = float(df["explicit"].astype(float).mean())
    dur = _col(df, "duration_ms")
    if dur.notna().any():
        f["avg_duration_min"] = float(dur.mean() / 60000)

    lyr = df[df["lyrics_english"]] if "lyrics_english" in df else df.iloc[0:0]
    f["lyrics_coverage"] = float(df.get("has_lyrics", pd.Series(False, index=df.index)).mean()) if n else 0.0
    if len(lyr):
        for c in EMO_COLS:
            f[f"mean_{c}"] = float(_col(lyr, c).mean())
        for c in ["repetition", "first_person", "profanity", "type_token_ratio"]:
            f[f"mean_{c}"] = float(_col(lyr, c).mean())
        themes = {t: float(_col(lyr, f"theme_{t}").mean()) for t in THEMES}
        f["themes"] = pd.Series(themes).sort_values(ascending=False)
        f["sad_lyrics_share"] = float((_col(lyr, "emo_sadness") > 0.35).mean())

    if "mood_valence" in df:
        mv = df["mood_valence"].dropna()
        f["low_valence_share"] = float((mv < 0.35).mean()) if len(mv) else np.nan
        both = df["lyric_valence"].notna() & df["valence"].notna() if "lyric_valence" in df and "valence" in df else None
        if both is not None and both.any():
            f["sad_banger_share"] = float(df.loc[both, "sad_banger"].mean())
            f["n_sad_bangers"] = int(df.loc[both, "sad_banger"].sum())
    return f


def model_vector(f: dict) -> dict[str, float]:
    """Numeric-only view of playlist_features(), for scikit-learn."""
    out = {}
    for k, v in f.items():
        if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            out[k] = float(v)
    for dim, val in (f.get("music") if isinstance(f.get("music"), pd.Series) else pd.Series(dtype=float)).items():
        out[f"music_{dim}"] = float(val)
    for g, val in (f.get("genre_dist") if isinstance(f.get("genre_dist"), pd.Series) else pd.Series(dtype=float)).items():
        out[f"genre_{g}"] = float(val)
    return out
