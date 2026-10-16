"""Synthetic data so everything runs offline: fake artists, invented lyrics, a fake streaming
history, labelled playlists for the weak-supervision demo and labelled ages for calibration.

All artists, titles and lyric lines here are made up for this project.
"""
from __future__ import annotations

import datetime as _dt
import string

import numpy as np
import pandas as pd

LINES = {
    "sad": [
        "I left the porch light on but you never came home",
        "Every song on the radio sounds like the last time",
        "I keep your sweater in the back of the closet",
        "The kitchen is so quiet I can hear the clock",
        "I'm sorry for the things I never said out loud",
        "Cold coffee and a phone that doesn't ring",
        "I miss the version of me that you knew",
        "Tears on the window of the late night train",
        "Alone again, counting cracks in the ceiling",
        "It hurts to say goodbye to someone who's already gone",
    ],
    "happy": [
        "Windows down and the summer in our hair",
        "We were laughing till the sun came up again",
        "Dance with me in the middle of the street",
        "Everything is golden when you smile like that",
        "Turn it up, tonight we're feeling alive",
        "Lucky me, lucky you, lucky everything",
        "Sunshine spilling over every word we say",
        "Celebrate the little things, the sweet things",
    ],
    "angry": [
        "I hate the way you twist the truth",
        "Burn the bridge and watch it fall",
        "Scream it loud until the walls give in",
        "You betrayed me and I won't forget",
        "This is war and I'm not backing down",
        "Fight me in the parking lot of my mind",
    ],
    "party": [
        "Bottles up, we own the club tonight",
        "Money in the air like confetti",
        "VIP, we never wait in line",
        "Shots on the table, lights in my eyes",
        "Turn up, the party never stops",
        "Cash in my pocket and my friends by my side",
    ],
    "love": [
        "Hold you close on a slow Sunday morning",
        "Your hand in mine, the city fading out",
        "Baby, stay a little longer, the rain can wait",
        "Kiss me where the streetlights hum",
        "Forever feels like this, soft and easy",
        "Love is a quiet room with the window open",
    ],
    "nostalgia": [
        "Remember when we used to own the summer",
        "Back then the whole world fit in my hometown",
        "Memories of bikes and broken fences",
        "Those days are gone but I still drive by",
        "We were younger than the songs we sang",
    ],
}

ARTISTS = [
    dict(name="The Paper Moths", genres=["indie folk", "chamber pop"], era=2014, v=0.30, e=0.35, ac=0.75, sp=0.04, inst=0.05, tempo=100, explicit=0.0, lyrics={"sad": 0.6, "nostalgia": 0.4}),
    dict(name="Mara Lune", genres=["bedroom pop", "indie pop"], era=2021, v=0.36, e=0.45, ac=0.45, sp=0.05, inst=0.05, tempo=110, explicit=0.1, lyrics={"sad": 0.6, "love": 0.4}),
    dict(name="Kilo Sunday", genres=["melodic rap", "emo rap"], era=2019, v=0.30, e=0.72, ac=0.15, sp=0.22, inst=0.0, tempo=140, explicit=0.9, lyrics={"sad": 0.55, "party": 0.45}, dance=0.78),
    dict(name="Neon Hymnal", genres=["synthpop", "dance pop"], era=2023, v=0.76, e=0.82, ac=0.08, sp=0.05, inst=0.02, tempo=124, explicit=0.1, lyrics={"happy": 0.7, "party": 0.3}, dance=0.8),
    dict(name="Ash & Ivory", genres=["neo soul", "alternative r&b"], era=2017, v=0.52, e=0.45, ac=0.35, sp=0.07, inst=0.02, tempo=92, explicit=0.3, lyrics={"love": 0.8, "sad": 0.2}),
    dict(name="Coastal Ghosts", genres=["midwest emo", "pop punk"], era=2008, v=0.35, e=0.82, ac=0.05, sp=0.06, inst=0.03, tempo=150, explicit=0.4, lyrics={"angry": 0.4, "sad": 0.3, "nostalgia": 0.3}),
    dict(name="Sable Nine", genres=["uk garage", "house"], era=2022, v=0.62, e=0.86, ac=0.03, sp=0.06, inst=0.55, tempo=132, explicit=0.0, lyrics={"party": 1.0}, dance=0.85),
    dict(name="Low Tide Choir", genres=["ambient", "neoclassical"], era=2016, v=0.22, e=0.15, ac=0.9, sp=0.03, inst=0.92, tempo=70, explicit=0.0, lyrics={}),
    dict(name="Bricklane Saints", genres=["britpop", "classic rock"], era=1996, v=0.62, e=0.72, ac=0.2, sp=0.04, inst=0.02, tempo=118, explicit=0.0, lyrics={"nostalgia": 0.5, "happy": 0.5}),
    dict(name="Juno Vale", genres=["contemporary country", "country pop"], era=2015, v=0.62, e=0.6, ac=0.4, sp=0.04, inst=0.0, tempo=108, explicit=0.0, lyrics={"love": 0.5, "nostalgia": 0.5}),
]
TITLE_WORDS = ["Static", "Bloom", "Kitchen", "Light", "Velvet", "Highway", "Paper", "Echo", "Midnight", "Orchard",
               "Signal", "Glass", "River", "Satellite", "Ember", "Harbor", "Neon", "Ghost", "Honey", "Weather"]
_B62 = string.ascii_letters + string.digits


def _fake_id(rng) -> str:
    return "".join(rng.choice(list(_B62), 22))


def _lyrics(rng, style: dict) -> str | None:
    if not style:
        return None
    moods, probs = list(style), np.array(list(style.values()), dtype=float)
    probs /= probs.sum()

    def line():
        return rng.choice(LINES[rng.choice(moods, p=probs)])

    chorus = "\n".join([line(), line()] * 2)
    verse = lambda: "\n".join(line() for _ in range(4))  # noqa: E731
    return "\n\n".join([verse(), chorus, verse(), chorus, chorus])


def _track(rng, a: dict, year: int, idx: int) -> dict:
    clip = lambda x: float(np.clip(x, 0, 1))  # noqa: E731
    v = clip(rng.normal(a["v"], 0.1))
    e = clip(rng.normal(a["e"], 0.1))
    lyr = _lyrics(rng, a["lyrics"]) if rng.random() > a["inst"] else None
    return {
        "track_id": _fake_id(rng),
        "name": f"{rng.choice(TITLE_WORDS)} {rng.choice(TITLE_WORDS)}",
        "artists": [a["name"]],
        "artist_ids": [a["name"].lower().replace(" ", "-")],
        "album": f"{rng.choice(TITLE_WORDS)} {rng.choice(['Sessions', 'Hours', 'Tapes', 'Season', 'Room'])}",
        "album_id": f"{a['name'][:4]}-{year}",
        "release_date": f"{year}-{rng.integers(1, 13):02d}-{rng.integers(1, 28):02d}",
        "isrc": f"QZ{rng.choice(list('ABCDEFG'))}{rng.integers(10, 99)}{year % 100:02d}{rng.integers(0, 99999):05d}",
        "duration_ms": int(np.clip(rng.normal(205, 40), 95, 420) * 1000),
        "explicit": bool(rng.random() < a["explicit"]),
        "genres": list(a["genres"]),
        "danceability": clip(rng.normal(a.get("dance", 0.55), 0.1)),
        "energy": e,
        "valence": v,
        "acousticness": clip(rng.normal(a["ac"], 0.1)),
        "instrumentalness": clip(rng.normal(a["inst"], 0.05)),
        "liveness": clip(rng.normal(0.12, 0.05)),
        "speechiness": clip(rng.normal(a["sp"], 0.03)),
        "loudness": float(-30 + 25 * e + rng.normal(0, 1.5)),
        "tempo": float(rng.normal(a["tempo"], 8)),
        "key": int(rng.integers(0, 12)),
        "mode": int(rng.random() < (0.35 + 0.5 * v)),
        "lyrics": lyr,
        "instrumental": lyr is None,
        "position": idx,
    }


def demo_playlist(n: int = 60, seed: int = 7, artist_weights: dict | None = None) -> pd.DataFrame:
    """A playlist for a fictional ~24-year-old with a sad-indie / sad-rap streak."""
    rng = np.random.default_rng(seed)
    weights = artist_weights or {
        "The Paper Moths": 3, "Mara Lune": 4, "Kilo Sunday": 5, "Neon Hymnal": 2, "Ash & Ivory": 2,
        "Coastal Ghosts": 1.5, "Sable Nine": 1, "Low Tide Choir": 1, "Bricklane Saints": 0.5, "Juno Vale": 1,
    }
    names = list(weights)
    p = np.array([weights[k] for k in names], dtype=float)
    p /= p.sum()
    by_name = {a["name"]: a for a in ARTISTS}
    rows = []
    start = pd.Timestamp("2023-01-01", tz="UTC")
    for i in range(n):
        a = by_name[rng.choice(names, p=p)]
        year = int(np.clip(round(rng.normal(a["era"], 2)), 1960, _dt.date.today().year))
        row = _track(rng, a, year, i)
        row["added_at"] = (start + pd.Timedelta(days=int(rng.integers(0, 900)))).isoformat()
        rows.append(row)
    return pd.DataFrame(rows)


def demo_history(tracks: pd.DataFrame, days: int = 420, seed: int = 3, dip_weeks: int = 5) -> pd.DataFrame:
    """Fake extended streaming history over `tracks`. The last `dip_weeks` lean sadder and later at night,
    so the mood-timeline and own-baseline comparison have something to find."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(_dt.date.today(), tz="America/New_York")
    start = end - pd.Timedelta(days=days)
    pop = rng.zipf(1.6, len(tracks)).astype(float)
    pop = np.minimum(pop, 50)
    sad = (tracks["valence"].fillna(0.5) < 0.4).to_numpy()
    rows = []
    for d in range(days):
        day = start + pd.Timedelta(days=d)
        in_dip = d >= days - 7 * dip_weeks
        w = pop * np.where(sad, 2.2 if in_dip else 1.0, 0.6 if in_dip else 1.0)
        w /= w.sum()
        for _ in range(int(rng.poisson(18))):
            hour = int(rng.choice([1, 2, 23]) if (in_dip and rng.random() < 0.3) else rng.choice(
                [7, 8, 9, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1], p=[.06, .08, .06, .07, .06, .07, .1, .1, .1, .09, .07, .06, .04, .02, .02]))
            ts = day + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
            t = tracks.iloc[rng.choice(len(tracks), p=w)]
            skipped = rng.random() < 0.22
            ms = int(rng.integers(3000, 25000)) if skipped else int(t["duration_ms"])
            rows.append({
                "ts": ts.tz_convert("UTC"), "track_id": t["track_id"], "name": t["name"], "artist": t["artists"][0],
                "album": t["album"], "ms_played": ms, "skipped": skipped,
                "reason_start": "trackdone", "reason_end": "fwdbtn" if skipped else "trackdone",
                "shuffle": bool(rng.random() < 0.5), "platform": "android", "source": "extended",
            })
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


MOOD_SPECS = {
    "sad": dict(v=0.25, e=0.35, dance=0.45, ac=0.55, inst=0.05, tempo=95, lyrics={"sad": 0.8, "nostalgia": 0.2}, genres=["indie folk", "bedroom pop"]),
    "happy": dict(v=0.78, e=0.72, dance=0.72, ac=0.2, inst=0.03, tempo=120, lyrics={"happy": 0.8, "love": 0.2}, genres=["dance pop", "pop"]),
    "chill": dict(v=0.55, e=0.3, dance=0.55, ac=0.6, inst=0.4, tempo=85, lyrics={"love": 0.6, "nostalgia": 0.4}, genres=["lo-fi beats", "chillwave"]),
    "hype": dict(v=0.6, e=0.88, dance=0.75, ac=0.05, inst=0.1, tempo=138, lyrics={"party": 0.8, "angry": 0.2}, genres=["trap", "edm"]),
    "angry": dict(v=0.25, e=0.9, dance=0.45, ac=0.03, inst=0.05, tempo=150, lyrics={"angry": 0.9, "sad": 0.1}, genres=["metalcore", "punk"]),
    "romantic": dict(v=0.6, e=0.42, dance=0.6, ac=0.4, inst=0.03, tempo=95, lyrics={"love": 0.9, "happy": 0.1}, genres=["neo soul", "r&b"]),
    "focus": dict(v=0.45, e=0.3, dance=0.4, ac=0.7, inst=0.85, tempo=80, lyrics={"love": 0.5, "nostalgia": 0.5}, genres=["neoclassical", "ambient"]),
}
MOOD_TITLES = {
    "sad": ["sad hours 😭", "in my feelings", "crying in the car", "heartbreak szn", "rainy day"],
    "happy": ["happy happy", "good vibes only", "feel good friday", "sunshine ☀️"],
    "chill": ["chill sunday", "lofi nights", "cozy", "relax"],
    "hype": ["GYM 🔥", "pregame", "workout", "party mix"],
    "angry": ["rage mode", "angry", "scream into the void 🤬"],
    "romantic": ["date night", "love songs", "crush 💕"],
    "focus": ["study", "deep focus", "coding 📚"],
}


def demo_labelled_playlists(per_mood: int = 6, tracks_per: int = 25, noise: float = 0.2, seed: int = 11) -> list[tuple[str, pd.DataFrame]]:
    """Playlists whose titles carry a mood, with `noise` share of tracks drawn from other moods."""
    rng = np.random.default_rng(seed)
    moods = list(MOOD_SPECS)
    out = []
    for mood in moods:
        for k in range(per_mood):
            rows = []
            for i in range(tracks_per):
                m = mood if rng.random() > noise else rng.choice(moods)
                spec = MOOD_SPECS[m]
                a = dict(name=f"{m}-artist-{rng.integers(0, 8)}", genres=spec["genres"], era=int(rng.integers(1995, 2025)),
                         v=spec["v"], e=spec["e"], ac=spec["ac"], sp=0.05, inst=spec["inst"], tempo=spec["tempo"],
                         explicit=0.2, lyrics=spec["lyrics"], dance=spec["dance"])
                rows.append(_track(rng, a, a["era"], i))
            title = MOOD_TITLES[mood][k % len(MOOD_TITLES[mood])]
            out.append((f"{title}" + (f" {k}" if k >= len(MOOD_TITLES[mood]) else ""), pd.DataFrame(rows)))
    out += [("random stuff", demo_playlist(20, seed=99)), ("2019", demo_playlist(20, seed=98))]  # unlabelled
    return out


def demo_age_samples(n: int = 150, seed: int = 5, bump_age: float = 15.0, bump_sd: float = 4.0) -> list[tuple[np.ndarray, float]]:
    """(album release years, true age) pairs from a known generative process, for testing calibration."""
    rng = np.random.default_rng(seed)
    now = _dt.date.today().year
    out = []
    for _ in range(n):
        age = float(rng.integers(16, 61))
        n_albums = int(rng.integers(15, 80))
        pi = rng.dirichlet([4, 2, 1.5])
        comp = rng.choice(3, size=n_albums, p=pi)
        bump = rng.normal(now - age + bump_age, bump_sd, n_albums)
        recent = now - rng.exponential(2.5, n_albums)
        bg = rng.uniform(1950, now, n_albums)
        years = np.round(np.where(comp == 0, bump, np.where(comp == 1, recent, bg)))
        out.append((np.clip(years, 1950, now), age))
    return out
