"""Train a mood classifier without hand-labelling anything: playlist titles as weak labels.

People name playlists "sad girl autumn", "GYM 🔥", "lofi study". Each title gives a noisy label to
every track in it. We train a track-level classifier on audio + lyric + genre features and
evaluate with GroupKFold so tracks from the same playlist never sit in both train and test (that
leak would inflate scores massively). Then a playlist's mood = average of its tracks' predictions.

Data options: your own playlists (SpotifyClient.my_playlists(), only the ones you own are
readable), CSV exports, or slices of the Spotify Million Playlist Dataset (`load_mpd_slice`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..enrich.audio import AUDIO_FEATURES
from ..nlp.emotion import EMO_COLS
from ..nlp.lexicon import THEMES

TITLE_PATTERNS: dict[str, list[str]] = {
    "sad": [r"\bsad", r"\bcry(ing)?\b", r"\btears?\b", r"heart ?break", r"\bdepress", r"\bmelanchol", r"in my feelings",
            r"\bbreak ?up\b", r"\bdown bad\b", r"\bgloom", r"\brainy\b", r"😭", r"💔", r"😢", r"🥀"],
    "happy": [r"\bhappy\b", r"good vibes?", r"feel ?good", r"\bsunshine\b", r"\bjoy", r"\bupbeat\b", r"\bsmil", r"😊", r"☀", r"😁"],
    "chill": [r"\bchill", r"\blo-?fi\b", r"\brelax", r"\bcalm\b", r"\bsleep", r"\bmellow\b", r"\bcozy\b", r"\bslow\b", r"😌", r"🌙"],
    "hype": [r"\bhype\b", r"\bgym\b", r"\bworkout\b", r"\bpump", r"\brun(ning)?\b", r"\bparty\b", r"turn ?up", r"\bpregame\b",
             r"\bbangers?\b", r"🔥", r"💪"],
    "angry": [r"\bangry\b", r"\brage\b", r"\bmad\b", r"\bscream", r"😡", r"🤬"],
    "romantic": [r"\blove songs?\b", r"\bromantic\b", r"\bromance\b", r"\bcrush\b", r"date night", r"💕", r"😍"],
    "focus": [r"\bfocus", r"\bstudy", r"\bconcentrat", r"\bcoding\b", r"\bdeep work\b", r"📚"],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in TITLE_PATTERNS.items()}


def title_label(title: str) -> str | None:
    """Weak label from a playlist title; None if no match or ambiguous (matches 2+ moods)."""
    hits = [mood for mood, pats in _COMPILED.items() if any(p.search(title or "") for p in pats)]
    return hits[0] if len(hits) == 1 else None


TRACK_FEATURES = [
    *[c for c in AUDIO_FEATURES if c not in ("key",)],
    *EMO_COLS, "lyric_valence", "lyric_arousal", "first_person", "repetition", "profanity",
    *[f"theme_{t}" for t in THEMES],
]
GENRE_FEATURES = [
    "K/J-Pop", "Latin", "Gospel/Christian", "Hip-Hop", "R&B/Soul", "Metal", "Punk/Emo", "Electronic", "Indie/Alt",
    "Rock", "Afro/Caribbean", "Folk/Acoustic", "Country", "Jazz/Blues", "Classical", "Soundtrack", "Pop",
]


def track_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in TRACK_FEATURES:
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    macros = df["macro_genres"] if "macro_genres" in df else pd.Series([[]] * len(df), index=df.index)
    for g in GENRE_FEATURES:
        X[f"genre_{g}"] = macros.map(lambda ms, g=g: float(g in (ms or [])))
    X["has_lyrics"] = df["lyric_valence"].notna().astype(float) if "lyric_valence" in df else 0.0
    return X


def build_dataset(playlists: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """From (title, enriched tracks) pairs -> X, y, groups, and a summary of which titles were used."""
    Xs, ys, gs, used = [], [], [], []
    for gid, (title, df) in enumerate(playlists):
        label = title_label(title)
        used.append({"playlist": title, "label": label, "n_tracks": len(df)})
        if label is None or df.empty:
            continue
        X = track_matrix(df)
        Xs.append(X)
        ys.append(pd.Series(label, index=X.index))
        gs.append(pd.Series(gid, index=X.index))
    if not Xs:
        raise ValueError("No playlist titles matched a mood keyword; see TITLE_PATTERNS.")
    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    g = pd.concat(gs, ignore_index=True)
    X = X.loc[:, X.notna().any()]  # drop features nobody has
    return X, y, g, pd.DataFrame(used)


@dataclass
class MoodClassifier:
    model: object
    features: list[str]
    classes: list[str]
    cv: dict

    def predict_tracks(self, df: pd.DataFrame) -> pd.DataFrame:
        X = track_matrix(df).reindex(columns=self.features)
        return pd.DataFrame(self.model.predict_proba(X), columns=self.classes, index=df.index)

    def predict_playlist(self, df: pd.DataFrame) -> pd.Series:
        return self.predict_tracks(df).mean().sort_values(ascending=False)

    def top_features(self, n: int = 6) -> pd.DataFrame:
        """Largest positive coefficients per class (standardised features, so comparable)."""
        coefs = self.model[-1].coef_
        rows = []
        for cls, row in zip(self.classes, coefs):
            order = np.argsort(row)[::-1][:n]
            rows.append({"mood": cls, "top features": ", ".join(f"{self.features[i]} ({row[i]:+.2f})" for i in order)})
        return pd.DataFrame(rows)


def train_mood_classifier(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5, C: float = 0.5) -> MoodClassifier:
    def make():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=3000),
        )

    n_splits = max(2, min(n_splits, groups.nunique()))
    preds = pd.Series(index=y.index, dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if y.iloc[tr].nunique() < 2:
            continue
        m = make().fit(X.iloc[tr], y.iloc[tr])
        preds.iloc[te] = m.predict(X.iloc[te])
    ok = preds.notna()
    classes = sorted(y.unique())
    majority = y.value_counts().idxmax()
    cv = {
        "n_tracks": int(len(y)),
        "n_playlists": int(groups.nunique()),
        "folds": n_splits,
        "accuracy": float(accuracy_score(y[ok], preds[ok])),
        "macro_f1": float(f1_score(y[ok], preds[ok], average="macro")),
        "baseline_accuracy": float((y[ok] == majority).mean()),
        "baseline_macro_f1": float(f1_score(y[ok], [majority] * int(ok.sum()), average="macro")),
        "confusion": pd.DataFrame(confusion_matrix(y[ok], preds[ok], labels=classes), index=classes, columns=classes),
        "report": classification_report(y[ok], preds[ok], zero_division=0),
    }
    final = make().fit(X, y)
    return MoodClassifier(model=final, features=list(X.columns), classes=list(final.classes_), cv=cv)


def load_mpd_slice(path: str | Path, max_playlists: int | None = None) -> list[tuple[str, pd.DataFrame]]:
    """Read one mpd.slice.*.json file from the Spotify Million Playlist Dataset (AIcrowd).

    Returns (title, tracks) pairs with track IDs, ready for ReccoBeats/LRCLIB enrichment. Only
    playlists whose titles match a mood keyword are kept, since the rest can't be labelled.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for p in data.get("playlists", []):
        if title_label(p.get("name", "")) is None:
            continue
        rows = [
            {
                "track_id": t["track_uri"].split(":")[-1],
                "name": t.get("track_name"),
                "artists": [t.get("artist_name")],
                "artist_ids": [t.get("artist_uri", "").split(":")[-1]],
                "album": t.get("album_name"),
                "album_id": t.get("album_uri", "").split(":")[-1],
                "duration_ms": t.get("duration_ms"),
                "position": t.get("pos"),
            }
            for t in p.get("tracks", [])
        ]
        out.append((p["name"], pd.DataFrame(rows)))
        if max_playlists and len(out) >= max_playlists:
            break
    return out
