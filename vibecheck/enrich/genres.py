"""Collapse Spotify's micro-genres ("midwest emo", "dark trap") into macro genres, and score the
MUSIC five-factor model of music preference (Rentfrow, Goldberg & Levitin, 2011):
Mellow, Unpretentious, Sophisticated, Intense, Contemporary.

The macro->MUSIC weights are a hand-built heuristic, not values from the paper.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

# Order matters: earlier rules "consume" the words they match, so "k-pop" doesn't also count as Pop.
MACRO_RULES: list[tuple[str, list[str]]] = [
    ("K/J-Pop", ["k-pop", "kpop", "j-pop", "j-rock", "korean", "japanese", "anime", "c-pop", "mandopop", "cantopop", "city pop", "vocaloid", "k-rap", "k-indie"]),
    ("Latin", ["latin", "latino", "reggaeton", "urbano", "bachata", "salsa", "cumbia", "corrido", "corridos tumbados", "banda", "mexican", "sertanejo", "funk carioca", "brazilian", "mpb", "tango", "dembow", "regional mexican"]),
    ("Gospel/Christian", ["worship", "christian", "gospel", "ccm"]),
    ("Hip-Hop", ["hip hop", "hip-hop", "rap", "trap", "drill", "grime", "boom bap", "phonk", "plugg", "rage"]),
    ("R&B/Soul", ["r&b", "rnb", "neo soul", "soul", "funk", "motown", "quiet storm", "new jack swing"]),
    ("Metal", ["metalcore", "deathcore", "metal", "djent", "grindcore", "doom", "sludge", "nu metal"]),
    ("Punk/Emo", ["pop punk", "post-hardcore", "punk", "midwest emo", "emo", "hardcore", "screamo", "riot grrrl"]),
    ("Electronic", ["edm", "house", "techno", "trance", "dubstep", "drum and bass", "dnb", "electronica", "electronic", "electro", "synthwave", "future bass", "hyperpop", "uk garage", "idm", "ambient", "downtempo", "lo-fi", "lofi", "chillwave", "breakbeat", "jungle", "eurodance", "hardstyle"]),
    ("Indie/Alt", ["indie", "alternative", "alt", "shoegaze", "dream pop", "bedroom pop", "post-punk", "new wave", "art pop", "slowcore", "noise pop"]),
    ("Rock", ["classic rock", "hard rock", "rock", "grunge", "britpop", "psychedelic", "prog", "rockabilly"]),
    ("Afro/Caribbean", ["afrobeats", "afrobeat", "afropop", "amapiano", "afro", "reggae", "dancehall", "dub", "ska", "soca"]),
    ("Folk/Acoustic", ["singer-songwriter", "folk", "acoustic", "americana", "bluegrass", "celtic"]),
    ("Country", ["country", "honky tonk", "outlaw", "red dirt", "nashville"]),
    ("Jazz/Blues", ["vocal jazz", "jazz", "blues", "bebop", "swing", "big band", "bossa nova"]),
    ("Classical", ["classical", "orchestra", "orchestral", "baroque", "opera", "choral", "string quartet", "early music", "minimalism", "neoclassical", "compositional", "piano"]),
    ("Soundtrack", ["soundtrack", "score", "video game", "musicals", "broadway", "show tunes", "disney"]),
    ("Pop", ["dance pop", "electropop", "synthpop", "pop", "boy band", "girl group", "adult contemporary", "teen"]),
]
_COMPILED = [
    (macro, [re.compile(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])") for kw in kws]) for macro, kws in MACRO_RULES
]

MUSIC_DIMS = ["Mellow", "Unpretentious", "Sophisticated", "Intense", "Contemporary"]
MACRO_TO_MUSIC: dict[str, dict[str, float]] = {
    "Pop": {"Contemporary": 0.6, "Mellow": 0.3, "Unpretentious": 0.1},
    "Hip-Hop": {"Contemporary": 1.0},
    "R&B/Soul": {"Mellow": 0.6, "Contemporary": 0.4},
    "Electronic": {"Contemporary": 0.7, "Mellow": 0.2, "Sophisticated": 0.1},
    "Latin": {"Contemporary": 0.8, "Unpretentious": 0.2},
    "K/J-Pop": {"Contemporary": 0.8, "Mellow": 0.2},
    "Afro/Caribbean": {"Contemporary": 0.7, "Mellow": 0.3},
    "Rock": {"Intense": 0.7, "Unpretentious": 0.3},
    "Metal": {"Intense": 1.0},
    "Punk/Emo": {"Intense": 0.9, "Contemporary": 0.1},
    "Indie/Alt": {"Intense": 0.4, "Mellow": 0.3, "Sophisticated": 0.3},
    "Folk/Acoustic": {"Unpretentious": 0.6, "Mellow": 0.3, "Sophisticated": 0.1},
    "Country": {"Unpretentious": 1.0},
    "Gospel/Christian": {"Unpretentious": 0.7, "Mellow": 0.3},
    "Jazz/Blues": {"Sophisticated": 0.7, "Mellow": 0.3},
    "Classical": {"Sophisticated": 1.0},
    "Soundtrack": {"Sophisticated": 0.6, "Mellow": 0.4},
}


def macro_genres(micro: str) -> list[str]:
    text = (micro or "").lower()
    found = []
    for macro, patterns in _COMPILED:
        hit = False
        for p in patterns:
            if p.search(text):
                text = p.sub(" ", text)
                hit = True
        if hit:
            found.append(macro)
    return found


def track_macros(genres: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for g in genres or []:
        for m in macro_genres(g):
            seen[m] = None
    return list(seen)


def add_genres(df: pd.DataFrame, artist_info: dict[str, dict] | None = None) -> pd.DataFrame:
    """Attach artist genres (if not already present) and derived macro genres."""
    df = df.copy()
    if artist_info is not None:
        def genres_for(ids):
            out: dict[str, None] = {}
            for aid in ids or []:
                for g in (artist_info.get(aid) or {}).get("genres", []):
                    out[g] = None
            return list(out)
        existing = df["genres"] if "genres" in df else pd.Series([[]] * len(df), index=df.index)
        df["genres"] = [
            g if isinstance(g, list) and g else genres_for(ids) for g, ids in zip(existing, df["artist_ids"])
        ]
    if "genres" not in df:
        df["genres"] = [[] for _ in range(len(df))]
    df["genres"] = df["genres"].map(lambda g: g if isinstance(g, list) else [])
    df["macro_genres"] = df["genres"].map(track_macros)
    return df


def genre_distribution(df: pd.DataFrame, weights: pd.Series | None = None) -> pd.Series:
    """Share of the playlist in each macro genre (each track splits one unit across its genres)."""
    c: Counter = Counter()
    w = weights if weights is not None else pd.Series(1.0, index=df.index)
    for macros, wt in zip(df["macro_genres"], w):
        if not macros:
            continue
        for m in macros:
            c[m] += wt / len(macros)
    s = pd.Series(c, dtype=float)
    return (s / s.sum()).sort_values(ascending=False) if len(s) else s


def micro_genre_counts(df: pd.DataFrame, top: int = 15) -> pd.Series:
    c = Counter(g for gs in df["genres"] for g in gs)
    return pd.Series(dict(c.most_common(top)), dtype=int)


def music_scores(dist: pd.Series) -> pd.Series:
    """MUSIC dimension scores (sum to 1) from a macro-genre distribution."""
    out = pd.Series(0.0, index=MUSIC_DIMS)
    for macro, share in dist.items():
        for dim, w in MACRO_TO_MUSIC.get(macro, {}).items():
            out[dim] += share * w
    total = out.sum()
    return out / total if total else out
