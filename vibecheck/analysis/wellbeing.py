"""Emotional-tone signals. Descriptive, NOT diagnostic.

What the research supports: people often choose music that matches or regulates their mood, and
some studies find group-level links between depressive symptoms and preferences for sadder music or
more repetitive / late-night listening. Those effects are small and noisy at the individual level,
and listening to sad music is common and often helps. So this module only describes the music and
compares listening against the person's *own* baseline; it never labels the listener.

All thresholds are heuristics chosen for readability, not validated cut-offs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DISCLAIMER = (
    "These are patterns in the music, not a read on your mental health. Sad songs are often how "
    "people feel better, and a playlist can't tell the difference."
)
CHECK_IN = (
    "If things have actually felt heavy lately, talking to someone you trust (or a professional) "
    "will help more than any playlist."
)


@dataclass
class Signal:
    name: str
    value: float
    level: str  # "typical" | "elevated"
    note: str


@dataclass
class Wellbeing:
    tone: str
    signals: list[Signal]
    trend: dict | None
    disclaimer: str = DISCLAIMER
    check_in: str | None = None

    @property
    def elevated(self) -> list[Signal]:
        return [s for s in self.signals if s.level == "elevated"]


def _sig(name, value, threshold, note) -> Signal | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return Signal(name, float(value), "elevated" if value >= threshold else "typical", note)


def valence_trend(timeline: pd.DataFrame, recent_periods: int = 4, baseline_periods: int = 26, min_coverage: float = 0.3) -> dict | None:
    """Compare the last few periods' valence against the listener's own trailing baseline."""
    t = timeline[(timeline["coverage"] >= min_coverage) & timeline["valence"].notna()]
    if len(t) < recent_periods + 6:
        return None
    recent = t["valence"].iloc[-recent_periods:]
    base = t["valence"].iloc[-(recent_periods + baseline_periods) : -recent_periods]
    sd = base.std(ddof=1)
    z = (recent.mean() - base.mean()) / sd if sd and sd > 0 else 0.0
    return {"recent": float(recent.mean()), "baseline": float(base.mean()), "z": float(z), "periods": len(base)}


def assess(f: dict, mood, behavior=None, timeline: pd.DataFrame | None = None) -> Wellbeing:
    signals = [
        _sig("low-valence share", f.get("low_valence_share"), 0.5, "share of tracks that sound and read as sad or dark"),
        _sig("sad lyrics share", f.get("sad_lyrics_share"), 0.4, "share of lyric-bearing tracks where sadness is the leading emotion signal"),
    ]
    themes = f.get("themes")
    if isinstance(themes, pd.Series) and len(themes):
        heavy = sum(themes.get(t, 0) for t in ("loneliness", "anxiety & self-doubt", "darkness"))
        signals.append(_sig("heavy lyrical themes", heavy, 1.5, "loneliness / anxiety / darkness words per 100 lyric words"))
    if behavior is not None:
        signals += [
            _sig("late-night listening", behavior.night_share, 0.15, "share of listening time between midnight and 5am"),
            _sig("same-day repeats", behavior.repeat_share, 0.35, "share of plays that replay a song already played that day"),
        ]
    trend = valence_trend(timeline) if timeline is not None else None
    if trend:
        signals.append(
            Signal("recent mood shift", trend["z"], "elevated" if trend["z"] <= -1.5 else "typical",
                   "last 4 weeks vs your own previous ~6 months (z-score; negative = sadder than usual)")
        )
    signals = [s for s in signals if s is not None]

    tone = f"Overall the music reads as {mood.label.lower()}: {mood.description}."
    if mood.volatility > 0.2:
        tone += " It swings a lot between highs and lows."
    n_elev = sum(s.level == "elevated" for s in signals)
    return Wellbeing(tone=tone, signals=signals, trend=trend, check_in=CHECK_IN if n_elev >= 2 else None)
