"""Guess a listener's age from the release years of what they listen to.

Idea: the "reminiscence bump". People keep a lifelong preference for music from their teens and
early twenties (e.g. Krumhansl & Zupnick, 2013; Stephens-Davidowitz's 2018 analysis of Spotify
data found favourite songs cluster around the early-to-mid teens). So we model each release year y as drawn from
a three-part mixture, given birth year b:

    p(y | b) = pi_bump * Normal(y; b + bump_age, bump_sd)    # music from your formative years
             + pi_now  * Recent(y)                           # whatever is current (any age)
             + pi_bg   * Uniform(1950..now)                  # classics, parents' records, noise

The mixture weights are fit per candidate b by a few EM steps, and the posterior over b combines
that likelihood with a broad prior over listener age. Because tracks from one album/artist are not
independent, the log-likelihood is tempered so at most ~`tempering_n` observations count. All the
constants are tunable, and `fit()` calibrates them on labelled (years, true age) examples.
"""
from __future__ import annotations

import datetime as _dt
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AgeEstimate:
    age: float  # posterior median
    interval80: tuple[int, int]
    posterior: pd.Series  # age -> probability
    bump_year: int  # center of the estimated "formative years" window
    mixture: dict  # MAP mixture weights: how much is nostalgia vs current vs background
    n_obs: int

    def __str__(self) -> str:
        lo, hi = self.interval80
        return f"~{self.age:.0f} years old (80% interval {lo}-{hi}, from {self.n_obs} release years)"


class AgeModel:
    def __init__(
        self,
        bump_age: float = 17.0,
        bump_sd: float = 5.0,
        recent_scale: float = 2.5,
        tempering_n: float = 12.0,
        prior_mean: float = 28.0,
        prior_sd: float = 13.0,
        min_age: int = 13,
        max_age: int = 75,
        now_year: int | None = None,
        em_iters: int = 25,
    ):
        self.bump_age = bump_age
        self.bump_sd = bump_sd
        self.recent_scale = recent_scale
        self.tempering_n = tempering_n
        self.prior_mean = prior_mean
        self.prior_sd = prior_sd
        self.min_age = min_age
        self.max_age = max_age
        self.now_year = now_year or _dt.date.today().year
        self.em_iters = em_iters

    # ---- core --------------------------------------------------------------------
    @property
    def ages(self) -> np.ndarray:
        return np.arange(self.min_age, self.max_age + 1)

    def _log_prior(self) -> np.ndarray:
        a = self.ages
        return -0.5 * ((a - self.prior_mean) / self.prior_sd) ** 2

    def _component_densities(self, years: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        births = self.now_year - self.ages  # (A,)
        mu = births[:, None] + self.bump_age  # (A,1)
        bump = np.exp(-0.5 * ((years[None, :] - mu) / self.bump_sd) ** 2) / (self.bump_sd * np.sqrt(2 * np.pi))
        lag = np.clip(self.now_year - years, 0, None)
        recent = np.exp(-lag / self.recent_scale) / self.recent_scale
        span = self.now_year - 1950 + 1
        bg = np.where((years >= 1950) & (years <= self.now_year), 1.0 / span, 1e-4 / span)
        return bump, recent, bg

    def _log_lik(self, years: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-age log-likelihood (A,) and fitted mixture weights (A, 3)."""
        bump, recent, bg = self._component_densities(years)
        A = len(self.ages)
        comps = np.stack([bump, np.broadcast_to(recent, bump.shape), np.broadcast_to(bg, bump.shape)], axis=1)  # (A,3,N)
        comps = comps + 1e-12
        pi = np.full((A, 3), 1 / 3)
        alpha = np.array([2.0, 2.0, 2.0])  # Dirichlet prior keeps weights away from 0/1
        for _ in range(self.em_iters):
            num = pi[:, :, None] * comps
            resp = num / num.sum(axis=1, keepdims=True)
            pi = resp.sum(axis=2) + (alpha - 1)
            pi = pi / pi.sum(axis=1, keepdims=True)
        ll = np.log((pi[:, :, None] * comps).sum(axis=1)).sum(axis=1)
        return ll, pi

    def estimate(self, years) -> AgeEstimate:
        years = np.asarray(pd.Series(years).dropna().astype(float))
        years = years[(years >= 1900) & (years <= self.now_year + 1)]
        if len(years) < 3:
            raise ValueError("Need at least 3 release years to estimate age.")
        ll, pi = self._log_lik(years)
        temper = min(1.0, self.tempering_n / len(years))
        logp = self._log_prior() + temper * ll
        p = np.exp(logp - logp.max())
        p /= p.sum()
        post = pd.Series(p, index=self.ages)
        cdf = post.cumsum()
        median = float(self.ages[np.searchsorted(cdf.values, 0.5)])
        lo = int(self.ages[np.searchsorted(cdf.values, 0.10)])
        hi = int(self.ages[min(np.searchsorted(cdf.values, 0.90), len(self.ages) - 1)])
        best = int(np.argmax(p))
        return AgeEstimate(
            age=median,
            interval80=(lo, hi),
            posterior=post,
            bump_year=int(self.now_year - self.ages[best] + self.bump_age),
            mixture=dict(zip(["formative years", "current releases", "everything else"], pi[best].round(3).tolist())),
            n_obs=len(years),
        )

    # ---- calibration -----------------------------------------------------------
    def score(self, samples: list[tuple[np.ndarray, float]]) -> dict:
        """Evaluate on (years, true_age) pairs: MAE, 80%-interval coverage, mean log loss."""
        errs, covered, nll = [], [], []
        for years, true_age in samples:
            est = self.estimate(years)
            errs.append(abs(est.age - true_age))
            covered.append(est.interval80[0] <= true_age <= est.interval80[1])
            p = est.posterior.get(int(round(true_age)), 0.0)
            nll.append(-np.log(max(p, 1e-6)))
        return {"mae": float(np.mean(errs)), "coverage80": float(np.mean(covered)), "log_loss": float(np.mean(nll)), "n": len(samples)}

    def fit(
        self,
        samples: list[tuple[np.ndarray, float]],
        bump_ages=(13, 15, 17, 19, 21),
        bump_sds=(3, 4, 5, 7),
        tempering_ns=(8, 12, 25),
    ) -> pd.DataFrame:
        """Grid-search the bump location/width and tempering by log loss. Returns the search table."""
        rows = []
        for ba, bs, tn in itertools.product(bump_ages, bump_sds, tempering_ns):
            self.bump_age, self.bump_sd, self.tempering_n = ba, bs, tn
            rows.append({"bump_age": ba, "bump_sd": bs, "tempering_n": tn, **self.score(samples)})
        table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
        best = table.iloc[0]
        self.bump_age, self.bump_sd, self.tempering_n = float(best.bump_age), float(best.bump_sd), float(best.tempering_n)
        return table


def album_level_years(df: pd.DataFrame) -> pd.Series:
    """One year per album (median), so a 20-track album doesn't count 20 times."""
    if "release_year" not in df:
        return pd.Series(dtype=float)
    years = pd.to_numeric(df["release_year"], errors="coerce")
    key = pd.Series([f"_row{i}" for i in range(len(df))], index=df.index, dtype=object)
    for col in ("album", "album_id"):  # album_id wins, then album name, then the row itself
        if col in df:
            key = df[col].astype(object).where(df[col].notna(), key)
    return years.groupby(key).median().dropna()
