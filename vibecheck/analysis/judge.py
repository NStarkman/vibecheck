"""The roast. Rule-based, Pudding-style: each rule looks at one stat, and the most extreme ones win.

Rules only ever poke at taste. Nothing here jokes about the listener's mental health.
"""
from __future__ import annotations

import datetime as _dt
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Rule:
    name: str
    applies: Callable[[dict], bool]
    salience: Callable[[dict], float]  # roughly 0-1: how extreme the stat is
    lines: list[str]  # str.format()-ed with the feature dict


def _g(f, k, default=np.nan):
    v = f.get(k, default)
    return default if v is None else v


def _theme_top(f, name):
    t = f.get("themes")
    return t is not None and len(t) and t.index[0] == name and t.iloc[0] > 0.6


def _music_top(f, name):
    m = f.get("music")
    return m is not None and len(m) and m.idxmax() == name and m.max() > 0.4


RULES: list[Rule] = [
    Rule("fan account", lambda f: _g(f, "top_artist_share", 0) > 0.25, lambda f: f["top_artist_share"],
         ["{top_artist_share:.0%} of this is {top_artist}. This isn't a playlist, it's a fan account.",
          "{top_artist} appears so often they should be getting royalties from you."]),
    Rule("loyal", lambda f: 0.12 < _g(f, "top_artist_share", 0) <= 0.25, lambda f: f["top_artist_share"] * 2,
         ["You and {top_artist} should probably open a joint bank account."]),
    Rule("monoculture", lambda f: _g(f, "genre_entropy", 1) < 0.3 and f.get("n_tracks", 0) >= 15, lambda f: 1 - f["genre_entropy"],
         ["Genre diversity of a single-crop farm. It's {top_genre} all the way down."]),
    Rule("genre chaos", lambda f: _g(f, "genre_entropy", 0) > 0.88, lambda f: f["genre_entropy"] - 0.3,
         ["This playlist has the attention span of a shuffle button on a sugar high."]),
    Rule("old soul", lambda f: _g(f, "share_pre2000", 0) > 0.5, lambda f: f["share_pre2000"],
         ["{share_pre2000:.0%} of this predates Wi-Fi. You'd be insufferable in a record store and you know it."]),
    Rule("new only", lambda f: _g(f, "share_recent", 0) > 0.6, lambda f: f["share_recent"],
         ["{share_recent:.0%} of this came out in the last two years. Did music start in {two_years_ago}?"]),
    Rule("time capsule", lambda f: _g(f, "share_mode_window", 0) > 0.45 and _g(f, "mode_year", 0) < _dt.date.today().year - 5,
         lambda f: f["share_mode_window"],
         ["Your musical clock stopped around {mode_year}. The rest of us moved on; this playlist did not."]),
    Rule("sad bangers", lambda f: _g(f, "sad_banger_share", 0) > 0.2, lambda f: f["sad_banger_share"] * 1.5,
         ["{n_sad_bangers} sad bangers detected. Crying, but make it danceable."]),
    Rule("gloomy", lambda f: _g(f, "mean_valence", 1) < 0.3, lambda f: 0.6 - f["mean_valence"],
         ["Average happiness score: {mean_valence:.2f}. Even the upbeat ones sound like they're staring out a rainy window."]),
    Rule("relentless", lambda f: _g(f, "mean_valence", 0) > 0.7, lambda f: f["mean_valence"] - 0.2,
         ["Relentlessly cheerful. This playlist has never had a bad day, and honestly it's a little suspicious."]),
    Rule("loud", lambda f: _g(f, "mean_energy", 0) > 0.8, lambda f: f["mean_energy"] - 0.2,
         ["Average energy {mean_energy:.2f}. Your neighbours know every song on here, and not by choice."]),
    Rule("nap", lambda f: _g(f, "mean_energy", 1) < 0.35, lambda f: 0.8 - f["mean_energy"],
         ["This playlist is basically a nap with a bassline."]),
    Rule("coffee shop", lambda f: _g(f, "mean_acousticness", 0) > 0.6, lambda f: f["mean_acousticness"] - 0.1,
         ["Enough acoustic guitar to open a coffee shop with mismatched mugs."]),
    Rule("instrumental", lambda f: _g(f, "instrumental_share", 0) > 0.4, lambda f: f["instrumental_share"],
         ["Mostly instrumentals. Either you're studying, or the lyrics were getting a little too relatable."]),
    Rule("explicit", lambda f: _g(f, "explicit_share", 0) > 0.6, lambda f: f["explicit_share"] - 0.1,
         ["{explicit_share:.0%} explicit. Your parental advisory sticker has a parental advisory sticker."]),
    Rule("clean", lambda f: _g(f, "explicit_share", 1) < 0.03 and f.get("n_tracks", 0) >= 20, lambda f: 0.5,
         ["Zero explicit tracks. Squeaky clean. Grandma-approved."]),
    Rule("long songs", lambda f: _g(f, "avg_duration_min", 0) > 6, lambda f: (f["avg_duration_min"] - 4) / 4,
         ["Average song length {avg_duration_min:.1f} minutes. Some of these have intermissions."]),
    Rule("short songs", lambda f: _g(f, "avg_duration_min", 9) < 2.6, lambda f: (3.5 - f["avg_duration_min"]) / 2,
         ["The average track is {avg_duration_min:.1f} minutes. The short-form video era has fully rewired you."]),
    Rule("cardio", lambda f: _g(f, "mean_tempo", 0) > 138, lambda f: (f["mean_tempo"] - 110) / 60,
         ["Average {mean_tempo:.0f} BPM. This isn't a playlist, it's a cardio class."]),
    Rule("main character", lambda f: _g(f, "mean_first_person", 0) > 0.09, lambda f: f["mean_first_person"] * 5,
         ["The lyrics say 'I' and 'me' a lot. Main-character energy: confirmed."]),
    Rule("hooks", lambda f: _g(f, "mean_repetition", 0) > 0.6, lambda f: f["mean_repetition"] - 0.2,
         ["The average song here repeats its hook so often it could double as a ringtone."]),
    Rule("heartbreak", lambda f: _theme_top(f, "heartbreak"), lambda f: 0.7,
         ["Top lyrical theme: heartbreak. Whoever they were, they're living here rent-free."]),
    Rule("flex", lambda f: _theme_top(f, "party & flex"), lambda f: 0.7,
         ["Lyrically it's mostly money, bottles and VIP sections. Respectfully: whose money?"]),
    Rule("nostalgia", lambda f: _theme_top(f, "nostalgia"), lambda f: 0.6,
         ["Top lyrical theme: nostalgia. Homesick for a decade you may not have even lived through."]),
    Rule("goth", lambda f: _theme_top(f, "darkness"), lambda f: 0.6,
         ["Lots of ghosts and graves in these lyrics. Very goth of you."]),
    Rule("tiny", lambda f: f.get("n_tracks", 99) < 10, lambda f: 0.8,
         ["{n_tracks} tracks isn't a playlist, it's a mood board."]),
    Rule("huge", lambda f: f.get("n_tracks", 0) > 500, lambda f: 0.8,
         ["{n_tracks} tracks. This isn't a playlist, it's an archive."]),
    Rule("sophisticated", lambda f: _music_top(f, "Sophisticated"), lambda f: 0.5,
         ["Heavy on jazz and classical. You have definitely said 'it's about the space between the notes.'"]),
    Rule("intense", lambda f: _music_top(f, "Intense"), lambda f: 0.5,
         ["Loud, distorted and proud. Your ears ring and you call it peace."]),
    Rule("unpretentious", lambda f: _music_top(f, "Unpretentious"), lambda f: 0.5,
         ["Wholesome and earnest. This playlist would help you move apartments."]),
    Rule("contemporary", lambda f: _music_top(f, "Contemporary"), lambda f: 0.45,
         ["Chart-adjacent and algorithm-approved. The recommendation engine thanks you for your service."]),
    Rule("mellow", lambda f: _music_top(f, "Mellow"), lambda f: 0.45,
         ["Smooth and agreeable. This playlist has never started an argument."]),
    # --- listening-history rules (only fire when streaming history was loaded) ---
    Rule("night owl", lambda f: _g(f, "night_share", 0) > 0.18, lambda f: f["night_share"] * 2.5,
         ["{night_share:.0%} of your listening happens between midnight and 5am. Sleep is also a genre; look into it."]),
    Rule("skipper", lambda f: _g(f, "skip_rate", 0) > 0.4, lambda f: f["skip_rate"],
         ["You skip {skip_rate:.0%} of songs. Commitment issues, but make it audio."]),
    Rule("binge", lambda f: _g(f, "peak_plays", 0) >= 15, lambda f: min(f["peak_plays"] / 30, 1),
         ["You played '{peak_track}' {peak_plays} times on {peak_date}. That's not a song any more, it's a personality."]),
    Rule("comfort food", lambda f: _g(f, "discovery_share", 1) < 0.1, lambda f: 0.6,
         ["Only {discovery_share:.0%} of your plays were songs you'd never heard before. Comfort food, every meal."]),
]


def roast(f: dict, k: int = 5, seed: int | None = 0) -> list[str]:
    """The k most salient roast lines for a playlist."""
    ctx = dict(f)
    ctx["two_years_ago"] = _dt.date.today().year - 2
    rng = random.Random(seed)
    scored = []
    for rule in RULES:
        try:
            if rule.applies(ctx):
                scored.append((float(rule.salience(ctx)), rule))
        except (KeyError, TypeError, ValueError):
            continue
    scored.sort(key=lambda x: -x[0])
    out = []
    for _, rule in scored[:k]:
        try:
            out.append(rng.choice(rule.lines).format(**ctx))
        except (KeyError, ValueError):
            continue
    return out or ["Honestly? Perfectly fine. Annoyingly fine. Nothing to roast here."]


def verdict(f: dict, mood_label: str, persona: str) -> str:
    """One-line headline."""
    parts = []
    if f.get("top_genre"):
        parts.append(f"{'Mostly' if f.get('top_genre_share', 0) >= 0.4 else 'Leans'} {f['top_genre']}")
    if f.get("median_year"):
        parts.append(f"centred on {f['median_year']}")
    mood = mood_label.lower()
    article = "an" if mood[0] in "aeiou" else "a"
    parts.append(f"with {article} {mood} streak")
    return f"{persona}. " + ", ".join(parts) + "."
