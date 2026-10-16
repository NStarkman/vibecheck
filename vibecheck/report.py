"""Text summaries and matplotlib charts for a Profile.

Charts follow one small design system: a single accent hue for single-series charts, a second hue
only when two series need telling apart (validated for colour-blind separation), hairline grids,
and light/dark themes via `set_theme()`. Every chart's numbers are also in profile.tracks /
profile.features, so nothing is readable only from a picture.
"""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", band="#f0efec"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", s1="#3987e5", s2="#d95926", band="#383835"),
}
T = dict(THEMES["light"])
FONT = ["system-ui", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def set_theme(name: str = "light") -> None:
    T.clear()
    T.update(THEMES[name])


def _fig(w=7.0, h=4.0, ncols=1):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=T["surface"])
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(T["surface"])
    return fig, axes


def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=T["muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(T["ink2"])
    ax.yaxis.label.set_color(T["ink2"])
    if grid_axis:
        ax.grid(axis=grid_axis, color=T["grid"], linewidth=0.6)
        ax.set_axisbelow(True)


def _title(ax, title, subtitle=None):
    ax.set_title(title, loc="left", color=T["ink"], fontsize=12, fontweight="bold", pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=T["ink2"], fontsize=9, va="bottom")


def _hbars(ax, series: pd.Series, fmt="{:.0%}", label_top: int = 1, color=None):
    s = series.iloc[::-1]
    ax.barh(s.index, s.values, height=0.62, color=color or T["s1"], edgecolor=T["surface"], linewidth=2)
    _style(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", colors=T["ink2"], labelsize=9)
    top = series.head(label_top)
    for name, val in top.items():
        ax.text(val, name, "  " + fmt.format(val), va="center", color=T["ink"], fontsize=9)
    ax.set_xlim(0, max(series.max() * 1.18, 1e-6))


# --------------------------------------------------------------------------- text
def _pct_list(s: pd.Series, n=3) -> str:
    return ", ".join(f"{k} {v:.0%}" for k, v in s.head(n).items()) if s is not None and len(s) else "n/a"


def summary_markdown(p) -> str:
    f, m = p.features, p.mood
    lines = [f"## {p.meta.get('name', 'Your music')}: {p.personality.persona}", "", f"**Verdict:** {p.verdict}", ""]
    rows = [
        ("Mood", f"{m.label} ({m.description}); valence {m.valence:.2f}, arousal {m.arousal:.2f}"
                 + ("" if m.confident else " (borderline)")),
        ("Estimated age", str(p.age) if p.age else "not enough release-year data"),
        ("Era", f"median {f.get('median_year', 'n/a')}, mostly the {f.get('top_decade', '?')}s" if f.get("median_year") else "n/a"),
        ("Top genres", _pct_list(f.get("genre_dist"))),
        ("Listening style (MUSIC)", _pct_list(p.personality.music.sort_values(ascending=False), 2)),
        ("Playlist craft", f"{p.personality.craft_score}/5"),
    ]
    if p.behavior is not None:
        b = p.behavior
        rows.append(("Listening", f"{b.total_hours:,.0f} h; {b.night_share:.0%} after midnight; skips {b.skip_rate:.0%}"))
    lines += ["| | |", "|---|---|", *[f"| {k} | {v} |" for k, v in rows], "", "### The roast"]
    lines += [f"- {r}" for r in p.roast]
    w = p.wellbeing
    lines += ["", "### Emotional tone (not a diagnosis)", w.tone]
    if w.elevated:
        lines.append("Stands out: " + ", ".join(s.name for s in w.elevated) + " (see the signals table).")
    lines += ["", f"_{w.disclaimer}_"]
    if w.check_in:
        lines += ["", w.check_in]
    if p.notes:
        lines += ["", "**Data notes:** " + " ".join(p.notes)]
    return "\n".join(lines)


def full_markdown(p) -> str:
    f = p.features
    out = [summary_markdown(p), "", "### Personality hints", *(f"- {h}" for h in p.personality.trait_hints or ["No strong leanings."])]
    out += ["_Small, population-level correlations from music-preference research; for fun, not a test._", ""]
    if "mean_emo_sadness" in f:
        emos = {k.replace("mean_emo_", ""): v for k, v in f.items() if k.startswith("mean_emo_")}
        out += ["### Lyrics", "Emotions: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(emos.items(), key=lambda kv: -kv[1])[:4])]
        if isinstance(f.get("themes"), pd.Series):
            out.append("Top themes: " + ", ".join(f"{k}" for k in f["themes"].head(3).index))
        out.append("")
    out += ["### Signals", "| signal | value | level |", "|---|---|---|"]
    out += [f"| {s.name} | {s.value:.2f} | {s.level} |" for s in p.wellbeing.signals]
    if p.behavior is not None and len(p.behavior.yearly):
        out += ["", "### Year by year", p.behavior.yearly.to_markdown(index=False) if _has_tabulate() else p.behavior.yearly.to_string(index=False)]
    return "\n".join(out)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- charts
def plot_card(p):
    """A Wrapped-style summary card."""
    fig, ax = _fig(6.2, 6.6)
    ax.axis("off")
    f = p.features
    y = 0.97
    ax.text(0.02, y, p.meta.get("name", "Your music").upper(), color=T["muted"], fontsize=9, fontweight="bold", transform=ax.transAxes)
    y -= 0.07
    ax.text(0.02, y, textwrap.fill(p.personality.persona, 26), color=T["ink"], fontsize=22, fontweight="bold", va="top", transform=ax.transAxes)
    y -= 0.07 * (1 + p.personality.persona.count(" ") // 3) + 0.03
    tiles = [
        ("MOOD", p.mood.label),
        ("EST. AGE", f"~{p.age.age:.0f}" if p.age else "?"),
        ("TOP GENRE", f.get("top_genre", "?")),
        ("CRAFT", f"{p.personality.craft_score}/5"),
    ]
    for i, (k, v) in enumerate(tiles):
        x = 0.02 + i * 0.245
        ax.add_patch(Rectangle((x, y - 0.115), 0.225, 0.115, transform=ax.transAxes, facecolor=T["band"], edgecolor="none"))
        ax.text(x + 0.015, y - 0.025, k, color=T["muted"], fontsize=7.5, fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(x + 0.015, y - 0.065, textwrap.shorten(str(v), 13, placeholder="…"), color=T["ink"], fontsize=12.5,
                fontweight="bold", transform=ax.transAxes, va="top")
    y -= 0.17
    ax.text(0.02, y, "THE VERDICT", color=T["s1"], fontsize=9, fontweight="bold", transform=ax.transAxes)
    y -= 0.035
    for r in p.roast[:5]:
        wrapped = textwrap.fill(r, 64)
        ax.text(0.02, y, wrapped, color=T["ink"], fontsize=10, va="top", transform=ax.transAxes, linespacing=1.3)
        y -= 0.042 * (wrapped.count("\n") + 1) + 0.022
    ax.text(0.02, 0.01, textwrap.fill(p.wellbeing.tone, 70), color=T["ink2"], fontsize=8.5, transform=ax.transAxes)
    return fig


def plot_mood_map(p):
    df = p.tracks.dropna(subset=["mood_valence", "mood_arousal"])
    fig, ax = _fig(6.2, 5.4)
    _style(ax, grid_axis=None)
    ax.axhline(0.5, color=T["axis"], lw=0.8)
    ax.axvline(0.5, color=T["axis"], lw=0.8)
    for (x, y, name, ha, va) in [(0.98, 0.98, "Euphoric", "right", "top"), (0.02, 0.98, "Angsty", "left", "top"),
                                 (0.02, 0.02, "Melancholy", "left", "bottom"), (0.98, 0.02, "Mellow", "right", "bottom")]:
        ax.text(x, y, name, color=T["muted"], fontsize=10, ha=ha, va=va, transform=ax.transAxes, fontweight="bold")
    banger = df.get("sad_banger", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    ax.scatter(df.loc[~banger, "mood_valence"], df.loc[~banger, "mood_arousal"], s=46, color=T["s1"], alpha=0.85,
               edgecolor=T["surface"], linewidth=2, label="tracks")
    if banger.any():
        ax.scatter(df.loc[banger, "mood_valence"], df.loc[banger, "mood_arousal"], s=46, color=T["s2"], alpha=0.9,
                   edgecolor=T["surface"], linewidth=2, label="sad bangers (sad lyrics, upbeat sound)")
    m = p.mood
    (v0, v1), (a0, a1) = m.valence_ci, m.arousal_ci
    ax.add_patch(Rectangle((v0, a0), v1 - v0, a1 - a0, facecolor=T["ink"], alpha=0.12, edgecolor="none", zorder=4))
    ax.plot(m.valence, m.arousal, "D", ms=11, color=T["ink"], mec=T["surface"], mew=2, zorder=5)
    ax.annotate("playlist average\n(box = 90% interval)", (m.valence, m.arousal), xytext=(12, 10),
                textcoords="offset points", color=T["ink"], fontsize=9, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc=T["surface"], ec="none", alpha=0.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("valence (sad → happy)")
    ax.set_ylabel("arousal (calm → energetic)")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(T["ink2"])
    _title(ax, "Mood map", f"{m.label}: {m.description}")
    fig.tight_layout()
    return fig


def plot_era(p):
    years = pd.to_numeric(p.tracks.get("release_year"), errors="coerce").dropna().astype(int)
    fig, (ax1, ax2) = _fig(10, 3.8, ncols=2)
    _style(ax1)
    if len(years):
        lo, hi = years.min(), years.max()
        counts = years.value_counts().reindex(range(lo, hi + 1), fill_value=0)
        if p.age:
            ax1.axvspan(p.age.bump_year - 4.5, p.age.bump_year + 4.5, color=T["band"], zorder=0)
            ax1.text(p.age.bump_year, counts.max() * 1.05, "your formative years?", ha="center", color=T["ink2"], fontsize=9)
        ax1.bar(counts.index, counts.values, width=0.85, color=T["s1"], edgecolor=T["surface"], linewidth=1)
        ax1.set_ylim(0, counts.max() * 1.18)
        ax1.set_ylabel("tracks")
    _title(ax1, "Release years", "earliest of album date and ISRC year")
    _style(ax2)
    if p.age:
        post = p.age.posterior
        lo, hi = p.age.interval80
        ax2.axvspan(lo - 0.5, hi + 0.5, color=T["band"], zorder=0)
        ax2.bar(post.index, post.values, width=0.85, color=T["s1"], edgecolor=T["surface"], linewidth=1)
        ax2.annotate(f"~{p.age.age:.0f}", (p.age.age, post.max()), xytext=(0, 6), textcoords="offset points",
                     ha="center", color=T["ink"], fontweight="bold")
        ax2.set_xlim(max(post.index.min(), lo - 15), min(post.index.max(), hi + 15))
        ax2.set_xlabel("age")
        ax2.set_ylabel("probability")
        _title(ax2, "Estimated age", f"shaded: 80% interval {lo}-{hi}")
    else:
        ax2.axis("off")
    fig.tight_layout()
    return fig


def plot_genres(p, top: int = 10):
    dist = p.features.get("genre_dist")
    fig, ax = _fig(6.2, 0.35 * min(top, len(dist) if dist is not None else 1) + 1.4)
    if dist is None or not len(dist):
        ax.axis("off")
        ax.text(0.5, 0.5, "no genre data", ha="center", color=T["muted"])
        return fig
    _hbars(ax, dist.head(top))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _title(ax, "Genres", "Spotify micro-genres grouped into families")
    fig.tight_layout()
    return fig


def plot_music(p):
    music = p.personality.music
    fig, ax = _fig(6.2, 2.9)
    if music is None or music.empty:
        ax.axis("off")
        return fig
    _hbars(ax, music.sort_values(ascending=False))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _title(ax, "MUSIC preference profile", "Rentfrow et al.'s five dimensions of musical taste")
    fig.tight_layout()
    return fig


def plot_emotions(p):
    f = p.features
    emos = pd.Series({k.replace("mean_emo_", ""): v for k, v in f.items() if k.startswith("mean_emo_")}).dropna()
    fig, ax = _fig(6.2, 3.2)
    if emos.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "no lyrics scored", ha="center", color=T["muted"])
        return fig
    _hbars(ax, emos.sort_values(ascending=False))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _title(ax, "Lyric emotions", f"average over {int(p.tracks.get('lyrics_english', pd.Series()).sum())} tracks with English lyrics")
    fig.tight_layout()
    return fig


def plot_listening_clock(p):
    if p.behavior is None:
        raise ValueError("Load a streaming history to see the listening clock.")
    hs = p.behavior.hour_share
    fig, ax = _fig(8, 3.2)
    _style(ax)
    colors = [T["s1"] if h < 5 else T["axis"] for h in hs.index]
    ax.bar(hs.index, hs.values, width=0.8, color=colors, edgecolor=T["surface"], linewidth=1)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _title(ax, "When you listen", f"midnight-5am highlighted: {p.behavior.night_share:.0%} of listening time")
    fig.tight_layout()
    return fig


def plot_mood_timeline(p, min_coverage: float = 0.3):
    if p.timeline is None:
        raise ValueError("Load a streaming history to see the mood timeline.")
    t = p.timeline[(p.timeline["coverage"] >= min_coverage)].dropna(subset=["valence"])
    fig, ax = _fig(9, 3.4)
    _style(ax)
    tr = p.wellbeing.trend
    if tr:
        base = t["valence"].iloc[-(4 + tr["periods"]):-4]
        ax.axhspan(base.mean() - base.std(), base.mean() + base.std(), color=T["band"], zorder=0)
        ax.text(t.index[0], base.mean() + base.std(), " your usual range", va="bottom", color=T["ink2"], fontsize=9)
    ax.plot(t.index, t["valence"], color=T["s1"], lw=2)
    ax.plot(t.index[-1], t["valence"].iloc[-1], "o", ms=8, color=T["s1"], mec=T["surface"], mew=2)
    ax.annotate(f"{t['valence'].iloc[-1]:.2f}", (t.index[-1], t["valence"].iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", color=T["ink"], fontsize=9)
    ax.set_ylabel("valence (play-time weighted)")
    _title(ax, "Mood over time", "weekly; compared only with your own past listening")
    fig.tight_layout()
    return fig


def save_all(p, folder="vibecheck_report", dpi=150) -> list[str]:
    import os

    os.makedirs(folder, exist_ok=True)
    paths = []
    plots = {"card": plot_card, "mood_map": plot_mood_map, "era": plot_era, "genres": plot_genres,
             "music": plot_music, "emotions": plot_emotions}
    if p.behavior is not None:
        plots.update({"listening_clock": plot_listening_clock, "mood_timeline": plot_mood_timeline})
    for name, fn in plots.items():
        fig = fn(p)
        path = os.path.join(folder, f"{name}.png")
        fig.savefig(path, dpi=dpi, facecolor=T["surface"], bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(full_markdown(p))
    return paths
