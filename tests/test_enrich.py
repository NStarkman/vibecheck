import numpy as np
import pandas as pd
import pytest

from vibecheck.cache import Cache
from vibecheck.enrich import LRCLib, ReccoBeats, add_audio_features, best_year, isrc_year, macro_genres, music_scores
from vibecheck.enrich.lyrics import clean_title
from vibecheck.enrich.tags import tag_mood

from .fakes import FakeResponse, FakeSession

A, B = "a" * 22, "b" * 22


def _features(sid, v):
    return {"id": "uuid-" + sid[:3], "href": f"https://open.spotify.com/track/{sid}", "valence": v, "energy": 0.5,
            "danceability": 0.6, "acousticness": 0.1, "instrumentalness": 0.0, "liveness": 0.1, "speechiness": 0.05,
            "loudness": -6.0, "tempo": 120.0, "key": 5, "mode": 1}


def test_reccobeats_maps_by_href_and_caches_misses():
    session = FakeSession([(lambda m, u, p: u.endswith("/audio-features"), FakeResponse(200, {"content": [_features(A, 0.9)]}))])
    rb = ReccoBeats(session, Cache(None))
    df = add_audio_features(pd.DataFrame({"track_id": [A, B], "valence": [np.nan, np.nan]}), rb)
    assert df.loc[0, "valence"] == 0.9 and np.isnan(df.loc[1, "valence"])
    assert list(df["has_audio"]) == [True, False]
    rb.audio_features([A, B])
    assert len(session.calls) == 1  # both the hit and the miss came from cache


def test_reccobeats_keeps_existing_values():
    session = FakeSession([])
    df = add_audio_features(pd.DataFrame({"track_id": [A], "valence": [0.3]}), ReccoBeats(session, Cache(None)))
    assert df.loc[0, "valence"] == 0.3 and session.calls == []


def test_reccobeats_batches_of_40():
    ids = [f"{i:022d}" for i in range(85)]
    session = FakeSession([(lambda m, u, p: True, lambda m, u, p: FakeResponse(200, {"content": [_features(s, 0.5) for s in p["ids"].split(",")]}))])
    out = ReccoBeats(session, Cache(None)).audio_features(ids)
    assert len(out) == 85 and [len(c[2]["ids"].split(",")) for c in session.calls] == [40, 40, 5]


def test_lrclib_exact_then_search_fallback():
    exact = FakeResponse(404, {"code": 404})
    search = FakeResponse(200, [
        {"trackName": "Different Song", "artistName": "Nobody", "duration": 200, "plainLyrics": "no"},
        {"trackName": "Kitchen Light", "artistName": "The Paper Moths", "duration": 201, "plainLyrics": "line one\nline two", "instrumental": False},
    ])
    session = FakeSession([
        (lambda m, u, p: u.endswith("/api/get"), exact),
        (lambda m, u, p: u.endswith("/api/search"), search),
    ])
    lib = LRCLib(session, Cache(None))
    res = lib.lyrics("Kitchen Light - 2019 Remaster", "The Paper Moths", duration_s=200)
    assert res["text"].startswith("line one")
    assert session.calls[1][2]["track_name"] == "Kitchen Light"  # suffix stripped for search


def test_lrclib_strips_synced_timestamps():
    session = FakeSession([(lambda m, u, p: u.endswith("/api/get"), FakeResponse(200, {"plainLyrics": None, "syncedLyrics": "[00:01.20] hello\n[00:03.00] world"}))])
    res = LRCLib(session, Cache(None)).lyrics("x", "y", duration_s=100)
    assert res["text"] == "hello\nworld"


@pytest.mark.parametrize("title,clean", [
    ("Song (feat. Someone)", "Song"), ("Song - Remastered 2011", "Song"), ("Song - 2009 Remaster", "Song"),
    ("Song - Live", "Song"), ("Plain Song", "Plain Song"),
])
def test_clean_title(title, clean):
    assert clean_title(title) == clean


def test_isrc_year_and_best_year():
    assert isrc_year("USRC17607839", now_year=2026) == 1976
    assert isrc_year("GBAYE0601498", now_year=2026) == 2006
    assert isrc_year("bad") is None
    assert best_year("2015-05-01", "USRC17607839") == 1976  # compilation re-release vs original recording
    assert best_year("1999", None) == 1999


@pytest.mark.parametrize("micro,expected", [
    ("k-pop", ["K/J-Pop"]), ("emo rap", ["Hip-Hop", "Punk/Emo"]), ("indie rock", ["Indie/Alt", "Rock"]),
    ("dance pop", ["Pop"]), ("trap latino", ["Latin", "Hip-Hop"]), ("skate punk", ["Punk/Emo"]),
    ("dubstep", ["Electronic"]), ("alternative r&b", ["R&B/Soul", "Indie/Alt"]), ("polka", []),
])
def test_macro_genres(micro, expected):
    assert macro_genres(micro) == expected


def test_music_scores_sum_to_one():
    s = music_scores(pd.Series({"Hip-Hop": 0.5, "Classical": 0.3, "Metal": 0.2}))
    assert s.sum() == pytest.approx(1.0) and s.idxmax() == "Contemporary"


def test_tag_mood():
    assert tag_mood([("rock", 100)]) is None
    v, a = tag_mood([("sad", 100), ("happy", 100)])
    assert 0.4 < v < 0.6
