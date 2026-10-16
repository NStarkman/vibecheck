"""Playlist mood on Russell's circumplex (valence x arousal), with bootstrap uncertainty."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

QUADRANTS = {
    "Euphoric": ("bright, high-energy, feel-good", "(>_<)/"),
    "Angsty": ("intense, restless, a little stormy", "(>_<)"),
    "Melancholy": ("low-key, wistful, in your feelings", "(._.)"),
    "Mellow": ("calm, warm, content", "(-_-)"),
    "Balanced": ("all over the emotional map, evenly", "(o_o)"),
}


def quadrant(v: float, a: float, center_radius: float = 0.06) -> str:
    if np.hypot(v - 0.5, a - 0.5) < center_radius:
        return "Balanced"
    if v >= 0.5:
        return "Euphoric" if a >= 0.5 else "Mellow"
    return "Angsty" if a >= 0.5 else "Melancholy"


@dataclass
class MoodEstimate:
    valence: float
    arousal: float
    valence_ci: tuple[float, float]
    arousal_ci: tuple[float, float]
    label: str
    description: str
    volatility: float  # SD of track valence: how much the playlist swings
    n_tracks: int
    quadrant_shares: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def confident(self) -> bool:
        """True if the 90% interval doesn't straddle the quadrant boundary."""
        return not (self.valence_ci[0] < 0.5 < self.valence_ci[1] or self.arousal_ci[0] < 0.5 < self.arousal_ci[1])


def estimate_mood(df: pd.DataFrame, weights: pd.Series | None = None, n_boot: int = 1000, seed: int = 0) -> MoodEstimate:
    ok = df["mood_valence"].notna() & df["mood_arousal"].notna()
    v = df.loc[ok, "mood_valence"].to_numpy(float)
    a = df.loc[ok, "mood_arousal"].to_numpy(float)
    if len(v) == 0:
        raise ValueError("No tracks have mood signals (audio features or lyrics).")
    w = (weights[ok].to_numpy(float) if weights is not None else np.ones(len(v)))
    w = w / w.sum()
    mv, ma = float(w @ v), float(w @ a)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(v), size=(n_boot, len(v)), p=w)
    bv, ba = v[idx].mean(axis=1), a[idx].mean(axis=1)
    label = quadrant(mv, ma)
    shares = pd.Series([quadrant(x, y) for x, y in zip(v, a)]).value_counts(normalize=True)
    return MoodEstimate(
        valence=mv,
        arousal=ma,
        valence_ci=(float(np.quantile(bv, 0.05)), float(np.quantile(bv, 0.95))),
        arousal_ci=(float(np.quantile(ba, 0.05)), float(np.quantile(ba, 0.95))),
        label=label,
        description=QUADRANTS[label][0],
        volatility=float(np.sqrt(w @ (v - mv) ** 2)),
        n_tracks=len(v),
        quadrant_shares=shares,
    )
