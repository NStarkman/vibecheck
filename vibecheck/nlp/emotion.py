"""Lyric NLP: emotion probabilities, valence/arousal, themes and simple text statistics.

Two interchangeable emotion backends, both returning probabilities over the same 7 labels
(Ekman's six basic emotions + neutral):

* TransformerEmotion: j-hartmann/emotion-english-distilroberta-base via Hugging Face
  `transformers` (pip install "vibecheck[nlp]"). Lyrics are split into stanza chunks, scored,
  and averaged by chunk length.
* LexiconEmotion: offline fallback - hand-written emotion word lists plus VADER sentiment.

Only English lyrics are scored; others get NaN emotions but still get text stats.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from .lexicon import EMOTION_WORDS, ENGLISH_STOPWORDS, FIRST_PERSON, PROFANITY, THEMES

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMO_COLS = [f"emo_{e}" for e in EMOTIONS]
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def looks_english(text: str, threshold: float = 0.12) -> bool:
    toks = tokenize(text)
    if len(toks) < 20:
        return False
    return sum(t in ENGLISH_STOPWORDS for t in toks) / len(toks) >= threshold


def emotions_to_va(p: dict[str, float]) -> tuple[float, float]:
    """Map emotion probabilities onto Russell's valence/arousal circumplex (0-1 scales)."""
    v = 0.5 + 0.5 * (p["joy"] + 0.3 * p["surprise"] - p["sadness"] - 0.7 * p["anger"] - 0.7 * p["fear"] - 0.6 * p["disgust"])
    a = 0.35 + 0.65 * (p["anger"] + p["fear"] + p["surprise"] + 0.5 * p["joy"] + 0.3 * p["disgust"]) - 0.25 * p["sadness"]
    return float(np.clip(v, 0, 1)), float(np.clip(a, 0, 1))


def text_stats(text: str) -> dict:
    toks = tokenize(text)
    lines = [ln.strip().lower() for ln in (text or "").splitlines() if ln.strip()]
    n = max(len(toks), 1)
    line_counts = Counter(lines)
    return {
        "n_words": len(toks),
        "type_token_ratio": len(set(toks)) / n if toks else np.nan,
        "repetition": (sum(c for c in line_counts.values() if c > 1) / len(lines)) if lines else np.nan,
        "first_person": sum(t in FIRST_PERSON for t in toks) / n if toks else np.nan,
        "profanity": sum(t in PROFANITY for t in toks) / n if toks else np.nan,
    }


def theme_scores(text: str) -> dict[str, float]:
    """Theme hits per 100 words (phrases matched on word boundaries)."""
    low = " " + re.sub(r"[^a-z' ]+", " ", (text or "").lower()) + " "
    n = max(len(tokenize(text)), 1)
    out = {}
    for theme, phrases in THEMES.items():
        hits = sum(len(re.findall(rf"(?<![a-z']){re.escape(p)}(?![a-z'])", low)) for p in phrases)
        out[theme] = 100.0 * hits / n
    return out


class LexiconEmotion:
    name = "lexicon"

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()
        except ImportError:  # pragma: no cover
            self._vader = None

    def score(self, texts: list[str]) -> list[dict[str, float]]:
        out = []
        for text in texts:
            toks = tokenize(text)
            counts = {e: sum(t in words for t in toks) for e, words in EMOTION_WORDS.items()}
            neutral_mass = max(1.0, 0.03 * len(toks))
            total = sum(counts.values()) + neutral_mass
            probs = {e: counts.get(e, 0) / total for e in EMOTIONS if e != "neutral"}
            probs["neutral"] = neutral_mass / total
            if self._vader is not None:
                lines = [ln for ln in text.splitlines() if ln.strip()]
                comp = np.mean([self._vader.polarity_scores(ln)["compound"] for ln in lines]) if lines else 0.0
                probs["_vader"] = float(comp)
            out.append(probs)
        return out


class TransformerEmotion:
    name = "transformer"
    MODEL = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self, model: str = MODEL, device: int | str | None = None, max_chunks: int = 12):
        from transformers import pipeline  # optional dependency

        self._clf = pipeline("text-classification", model=model, top_k=None, truncation=True, device=device)
        self.max_chunks = max_chunks

    @staticmethod
    def chunks(text: str, target: int = 500) -> list[str]:
        stanzas = [s.strip() for s in re.split(r"\n\s*\n", text or "") if s.strip()]
        if len(stanzas) <= 1:
            lines = [ln for ln in (text or "").splitlines() if ln.strip()]
            stanzas = ["\n".join(lines[i : i + 4]) for i in range(0, len(lines), 4)]
        out, cur = [], ""
        for s in stanzas:
            if cur and len(cur) + len(s) > target:
                out.append(cur)
                cur = s
            else:
                cur = f"{cur}\n{s}" if cur else s
        if cur:
            out.append(cur)
        return out

    def score(self, texts: list[str]) -> list[dict[str, float]]:
        flat, owner = [], []
        for i, t in enumerate(texts):
            for c in self.chunks(t)[: self.max_chunks]:
                flat.append(c)
                owner.append(i)
        preds = self._clf(flat, batch_size=16) if flat else []
        acc = [np.zeros(len(EMOTIONS)) for _ in texts]
        wts = [0.0 for _ in texts]
        for i, chunk, pred in zip(owner, flat, preds):
            vec = np.array([{d["label"]: d["score"] for d in pred}.get(e, 0.0) for e in EMOTIONS])
            acc[i] += len(chunk) * vec
            wts[i] += len(chunk)
        out = []
        for a, w in zip(acc, wts):
            vec = a / w if w else np.full(len(EMOTIONS), np.nan)
            out.append(dict(zip(EMOTIONS, vec.tolist())))
        return out


def get_emotion_model(backend: str = "auto"):
    if backend in ("auto", "transformer"):
        try:
            return TransformerEmotion()
        except Exception as e:  # ImportError, or no network to download weights
            if backend == "transformer":
                raise
            print(f"[vibecheck] transformer emotion model unavailable ({type(e).__name__}); using offline lexicon model")
    return LexiconEmotion()


def add_lyric_features(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Add emotion probabilities, lyric valence/arousal, themes and text stats for tracks with lyrics."""
    df = df.copy()
    model = model or LexiconEmotion()
    has = df.get("has_lyrics", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    english = pd.Series(False, index=df.index)
    english[has] = df.loc[has, "lyrics"].map(looks_english)
    df["lyrics_english"] = english

    for col in [*EMO_COLS, "lyric_valence", "lyric_arousal"]:
        df[col] = np.nan
    idx = df.index[english]
    if len(idx):
        scored = model.score(df.loc[idx, "lyrics"].tolist())
        for i, probs in zip(idx, scored):
            for e in EMOTIONS:
                df.at[i, f"emo_{e}"] = probs.get(e, np.nan)
            if not np.isnan(probs.get("joy", np.nan)):
                v, a = emotions_to_va(probs)
                if "_vader" in probs:  # lexicon backend: blend in line-level VADER sentiment
                    v = 0.5 * v + 0.5 * (probs["_vader"] + 1) / 2
                df.at[i, "lyric_valence"], df.at[i, "lyric_arousal"] = v, a

    stats = [text_stats(t) if h else {} for t, h in zip(df.get("lyrics", [None] * len(df)), has)]
    for col in ["n_words", "type_token_ratio", "repetition", "first_person", "profanity"]:
        df[col] = [s.get(col, np.nan) for s in stats]
    themes = [theme_scores(t) if e else {} for t, e in zip(df.get("lyrics", [None] * len(df)), english)]
    for theme in THEMES:
        df[f"theme_{theme}"] = [t.get(theme, np.nan) for t in themes]
    df.attrs["emotion_backend"] = getattr(model, "name", type(model).__name__)
    return df
