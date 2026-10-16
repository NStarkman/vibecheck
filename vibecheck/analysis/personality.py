"""Listening personality: MUSIC-model profile, a persona name, loose Big Five hints, playlist craft score.

The Big Five links come from music-preference research (Rentfrow & Gosling, 2003; Rentfrow et al.,
2011): e.g. people who like "sophisticated" music tend to score higher on Openness. Those
correlations are real but small (|r| around .1-.3), so the hints are phrased as tendencies, and
should be read as entertainment rather than a personality test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERSONAS = {
    ("Mellow", "Euphoric"): "The Golden-Hour Optimist",
    ("Mellow", "Angsty"): "The Slow-Burn Romantic",
    ("Mellow", "Melancholy"): "The Rainy-Window Philosopher",
    ("Mellow", "Mellow"): "The Sunday-Morning Soul",
    ("Mellow", "Balanced"): "The Smooth Operator",
    ("Unpretentious", "Euphoric"): "The Road-Trip Captain",
    ("Unpretentious", "Angsty"): "The Heartland Heartbreaker",
    ("Unpretentious", "Melancholy"): "The Porch-Light Storyteller",
    ("Unpretentious", "Mellow"): "The Campfire Regular",
    ("Unpretentious", "Balanced"): "The Earnest Everyperson",
    ("Sophisticated", "Euphoric"): "The Concert-Hall Enthusiast",
    ("Sophisticated", "Angsty"): "The Tortured Aesthete",
    ("Sophisticated", "Melancholy"): "The Late-Night Jazz Ghost",
    ("Sophisticated", "Mellow"): "The Vinyl Sommelier",
    ("Sophisticated", "Balanced"): "The Liner-Notes Reader",
    ("Intense", "Euphoric"): "The Mosh-Pit Optimist",
    ("Intense", "Angsty"): "The Feedback-Loop Rebel",
    ("Intense", "Melancholy"): "The Basement-Show Poet",
    ("Intense", "Mellow"): "The Quiet-Loud Enigma",
    ("Intense", "Balanced"): "The Distortion Diplomat",
    ("Contemporary", "Euphoric"): "The Main-Stage Hype Machine",
    ("Contemporary", "Angsty"): "The 3 A.M. Drive Specialist",
    ("Contemporary", "Melancholy"): "The Sad-Banger Connoisseur",
    ("Contemporary", "Mellow"): "The Lo-Fi Lounge Lizard",
    ("Contemporary", "Balanced"): "The Algorithm's Favourite Child",
}

TRAIT_LINKS = {
    # MUSIC dimension -> [(trait, direction)]
    "Sophisticated": [("Openness", +1)],
    "Intense": [("Openness", +1)],
    "Contemporary": [("Extraversion", +1)],
    "Unpretentious": [("Agreeableness", +1), ("Conscientiousness", +1), ("Openness", -1)],
    "Mellow": [("Agreeableness", +1)],
}


@dataclass
class Personality:
    persona: str
    music: pd.Series
    trait_hints: list[str]
    craft_score: float  # 0-5, Chosic-style "how well made is this playlist"
    craft_breakdown: dict


def craft_score(f: dict) -> tuple[float, dict]:
    """0-5 stars: artist variety, genre focus, mood coherence, length."""
    variety = 1 - min(f.get("artist_repeat_rate", 0) / 0.6, 1)
    ent = f.get("genre_entropy", 0.5)
    focus = 1 - abs(ent - 0.45) / 0.55  # some focus is good; a single genre or total chaos less so
    length = min(f.get("n_tracks", 0) / 40, 1)
    breakdown = {"artist variety": variety, "genre focus": float(np.clip(focus, 0, 1)), "length": length}
    if "mood_volatility" in f:
        breakdown["mood coherence"] = float(np.clip(1 - (f["mood_volatility"] - 0.1) / 0.2, 0, 1))
    score = 5 * float(np.mean(list(breakdown.values())))
    return round(score, 1), breakdown


def personality(f: dict, mood_label: str) -> Personality:
    music: pd.Series = f.get("music", pd.Series(dtype=float))
    if music is None or music.empty or music.sum() == 0:
        dominant = "Contemporary"
        music = pd.Series(dtype=float)
    else:
        dominant = music.idxmax()
    hints: dict[str, float] = {}
    for dim, share in music.items():
        if share <= 0.25:  # only lean on styles you clearly favour (0.2 = even split across 5)
            continue
        for trait, sign in TRAIT_LINKS.get(dim, []):
            hints[trait] = hints.get(trait, 0) + sign * (share - 0.2)
    lines = []
    for trait, val in sorted(hints.items(), key=lambda kv: -abs(kv[1])):
        if abs(val) < 0.08:
            continue
        direction = "higher" if val > 0 else "lower"
        lines.append(f"People with this mix tend to score a bit {direction} on {trait}.")
    score, breakdown = craft_score(f)
    return Personality(
        persona=PERSONAS.get((dominant, mood_label), "The Unclassifiable"),
        music=music,
        trait_hints=lines[:3],
        craft_score=score,
        craft_breakdown=breakdown,
    )
