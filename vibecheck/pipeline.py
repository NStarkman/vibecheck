"""High-level API: paste a link (or hand over a DataFrame / history export), get a Profile back.

    from vibecheck import Vibecheck
    vc = Vibecheck()                       # reads SPOTIFY_CLIENT_ID etc. from .env
    profile = vc.analyze("https://open.spotify.com/playlist/...")
    print(profile.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .analysis import (
    AgeEstimate, AgeModel, Behavior, MoodEstimate, Personality, Wellbeing, add_track_mood, album_level_years,
    assess, estimate_mood, listening_behavior, mood_timeline, personality, playlist_features, roast, verdict,
)
from .cache import Cache
from .config import Settings
from .enrich import LastFM, LRCLib, ReccoBeats, add_audio_features, add_genres, add_lastfm, add_release_years
from .http import make_session
from .nlp.emotion import LexiconEmotion, add_lyric_features, get_emotion_model
from .sources import SpotifyAuth, SpotifyClient, load_streaming_history


@dataclass
class Profile:
    meta: dict
    tracks: pd.DataFrame
    features: dict
    mood: MoodEstimate
    age: AgeEstimate | None
    personality: Personality
    roast: list[str]
    verdict: str
    wellbeing: Wellbeing
    behavior: Behavior | None = None
    timeline: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        from .report import summary_markdown

        return summary_markdown(self)

    def to_markdown(self) -> str:
        from .report import full_markdown

        return full_markdown(self)


class Vibecheck:
    def __init__(self, settings: Settings | None = None, offline: bool = False, cache: Cache | None = None, verbose: bool = True):
        """`offline=True` never touches the network (used by the demo and tests)."""
        self.settings = settings or Settings.from_env()
        self.offline = offline
        self.verbose = verbose
        self.cache = cache or (Cache(None) if offline else Cache(self.settings.cache_dir / "cache.sqlite"))
        self.session = make_session(self.settings.user_agent)
        self.recco = ReccoBeats(self.session, self.cache)
        self.lrclib = LRCLib(self.session, self.cache)
        self.lastfm = LastFM(self.settings.lastfm_api_key, self.session, self.cache) if self.settings.lastfm_api_key else None
        self._spotify: SpotifyClient | None = None
        self._emotion = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[vibecheck] {msg}")

    # ---- lazily-built clients -----------------------------------------------------
    @property
    def spotify(self) -> SpotifyClient:
        if self.offline:
            raise RuntimeError("Spotify is unavailable in offline mode")
        if self._spotify is None:
            s = self.settings
            auth = SpotifyAuth(s.spotify_client_id, s.spotify_redirect_uri, s.spotify_scopes,
                               token_path=s.cache_dir / "spotify_token.json")
            self._spotify = SpotifyClient(auth=auth, session=self.session, cache=self.cache)
        return self._spotify

    @property
    def emotion_model(self):
        if self._emotion is None:
            self._emotion = LexiconEmotion() if self.offline else get_emotion_model(self.settings.emotion_backend)
        return self._emotion

    # ---- steps --------------------------------------------------------------------
    def load(self, link: str) -> tuple[dict, pd.DataFrame]:
        meta, df = self.spotify.load(link)
        self._log(f"loaded {len(df)} tracks from {meta.get('kind')} '{meta.get('name')}'")
        return meta, df

    def enrich(self, df: pd.DataFrame, audio: bool = True, lyrics: bool = True, genres: bool = True, tags: bool = True) -> pd.DataFrame:
        """Add release years, genres, audio features, Last.fm tags, lyrics + NLP, and per-track mood."""
        df = add_release_years(df)
        online = not self.offline
        if genres:
            needs_artists = online and "artist_ids" in df and df["artist_ids"].map(bool).any() and (
                "genres" not in df or not df["genres"].map(bool).any())
            info = None
            if needs_artists and self.settings.spotify_client_id:
                ids = [a for lst in df["artist_ids"] for a in (lst or [])]
                self._log(f"fetching genres for {len(set(ids))} artists")
                info = self.spotify.artists(ids)
            df = add_genres(df, info)
        if audio and online:
            self._log("fetching audio features (ReccoBeats)")
            df = add_audio_features(df, self.recco)
        elif "valence" in df:
            df["has_audio"] = df["valence"].notna()
        if tags and online and self.lastfm is not None:
            self._log("fetching Last.fm tags")
            df = add_lastfm(df, self.lastfm)
        if lyrics and online and "lyrics" not in df:
            self._log("fetching lyrics (LRCLIB)")
            df = self.lrclib.add_lyrics(df)
        if "lyrics" in df:
            if "has_lyrics" not in df:
                df["has_lyrics"] = df["lyrics"].notna() & (df["lyrics"].fillna("").str.len() > 40)
            self._log(f"scoring lyrics ({getattr(self.emotion_model, 'name', 'model')} emotion model)")
            df = add_lyric_features(df, self.emotion_model)
        return add_track_mood(df)

    def analyze(
        self,
        source: str | pd.DataFrame,
        meta: dict | None = None,
        history: str | Path | pd.DataFrame | None = None,
        tz: str = "local",
        age_model: AgeModel | None = None,
        roast_lines: int = 5,
        weights: pd.Series | None = None,
    ) -> Profile:
        """Full analysis. `weights` (per track, e.g. play time) weights the mood estimate."""
        if isinstance(source, str):
            meta, df = self.load(source)
        else:
            df, meta = source, meta or {"kind": "custom", "name": "Your tracks"}
        if df.empty:
            raise ValueError("No tracks to analyze.")
        if "mood_valence" not in df:
            df = self.enrich(df)

        f = playlist_features(df)
        mood = estimate_mood(df, weights=weights)
        f["mood_volatility"] = mood.volatility
        try:
            age = (age_model or AgeModel()).estimate(album_level_years(df))
        except ValueError:
            age = None
        pers = personality(f, mood.label)

        behavior = timeline = None
        if history is not None:
            hist = history if isinstance(history, pd.DataFrame) else load_streaming_history(history)
            behavior = listening_behavior(hist, tz=tz)
            timeline = mood_timeline(hist, df, tz=tz)
            f.update(behavior_features(behavior))

        return Profile(
            meta=meta,
            tracks=df,
            features=f,
            mood=mood,
            age=age,
            personality=pers,
            roast=roast(f, k=roast_lines),
            verdict=verdict(f, mood.label, pers.persona),
            wellbeing=assess(f, mood, behavior, timeline),
            behavior=behavior,
            timeline=timeline,
            notes=coverage_notes(df, f, self.emotion_model),
        )

    def analyze_history(self, history: str | Path | pd.DataFrame, tz: str = "local", top_n: int = 300, tracks: pd.DataFrame | None = None) -> Profile:
        """Analyze a streaming-history export: your most-played tracks + how you listen.

        Your top `top_n` tracks by listening time are enriched (audio features + lyrics; no
        Spotify login needed) and weighted by play time. Pass `tracks` to reuse an enriched set.
        """
        hist = history if isinstance(history, pd.DataFrame) else load_streaming_history(history)
        if tracks is None:
            tracks = top_tracks_frame(hist, top_n)
            if not self.offline and self.settings.spotify_client_id:
                # Fill release dates, ISRCs and artist IDs (needed for age and genres) from Spotify.
                self._log(f"fetching metadata for {len(tracks)} top tracks from Spotify")
                meta = self.spotify.track_rows(tracks["track_id"]).set_index("track_id")
                for col in ("artist_ids", "release_date", "isrc", "duration_ms", "explicit", "album_id"):
                    if col in meta:
                        tracks[col] = tracks["track_id"].map(meta[col])
                tracks["artist_ids"] = tracks["artist_ids"].map(lambda v: v if isinstance(v, list) else [])
            tracks = self.enrich(tracks)
        weights = hist.groupby("track_id")["ms_played"].sum()
        tracks = tracks.assign(play_ms=tracks["track_id"].map(weights).fillna(0))
        return self.analyze(
            tracks, meta={"kind": "history", "name": "Your listening history"}, history=hist, tz=tz,
            weights=tracks["play_ms"] + 1,
        )


def top_tracks_frame(hist: pd.DataFrame, top_n: int) -> pd.DataFrame:
    h = hist.dropna(subset=["track_id"])
    agg = (
        h.groupby("track_id")
        .agg(name=("name", "first"), artist=("artist", "first"), album=("album", "first"), ms=("ms_played", "sum"))
        .sort_values("ms", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return pd.DataFrame(
        {
            "track_id": agg["track_id"],
            "name": agg["name"],
            "artists": agg["artist"].map(lambda a: [a] if isinstance(a, str) else []),
            "artist_ids": [[] for _ in range(len(agg))],
            "album": agg["album"],
            "position": range(len(agg)),
        }
    )


def behavior_features(b: Behavior) -> dict:
    out = {
        "night_share": b.night_share,
        "skip_rate": b.skip_rate,
        "repeat_share": b.repeat_share,
        "discovery_share": b.discovery_share,
        "total_hours": b.total_hours,
    }
    if b.peak_binge:
        out.update({"peak_track": b.peak_binge["track"], "peak_plays": b.peak_binge["plays"], "peak_date": b.peak_binge["date"]})
    return out


def coverage_notes(df: pd.DataFrame, f: dict, emotion_model) -> list[str]:
    notes = []
    if f.get("audio_coverage", 0) < 0.6:
        notes.append(f"Audio features were found for only {f.get('audio_coverage', 0):.0%} of tracks; mood leans more on lyrics.")
    if 0 < f.get("lyrics_coverage", 0) < 0.4:
        notes.append(f"Lyrics were found for only {f['lyrics_coverage']:.0%} of tracks.")
    if f.get("genre_coverage", 1) < 0.5:
        notes.append(f"Only {f.get('genre_coverage', 0):.0%} of tracks have genre info (add a Last.fm key to fill gaps).")
    if getattr(emotion_model, "name", "") == "lexicon" and f.get("lyrics_coverage", 0) > 0:
        notes.append("Lyric emotions use the offline word-list model; install vibecheck[nlp] for the transformer model.")
    if len(df) < 15:
        notes.append(f"Only {len(df)} tracks, so treat every estimate here as a rough guess.")
    return notes
