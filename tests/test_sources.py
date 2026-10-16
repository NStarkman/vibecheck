import json
import zipfile

import pandas as pd
import pytest

from vibecheck.cache import Cache
from vibecheck.http import request_json
from vibecheck.sources import LinkError, PlaylistNotAccessible, SpotifyClient, load_streaming_history, load_tracks_csv, parse_link
from vibecheck.sources.spotify_auth import SpotifyAuth, make_pkce_pair

from .fakes import FakeResponse, FakeSession

PID = "37i9dQZF1DXcBWIGoYBM5M"


@pytest.mark.parametrize(
    "text,kind,id_",
    [
        (f"https://open.spotify.com/playlist/{PID}?si=abc123", "playlist", PID),
        (f"https://open.spotify.com/intl-de/playlist/{PID}", "playlist", PID),
        (f"open.spotify.com/album/{PID}", "album", PID),
        (f"spotify:track:{PID}", "track", PID),
        (f"spotify:user:someone:playlist:{PID}", "playlist", PID),
        (PID, "playlist", PID),
        ("liked", "liked", None),
        ("top:short_term", "top", "short_term"),
        ("https://spotify.link/AbCdEf", "short", "https://spotify.link/AbCdEf"),
    ],
)
def test_parse_link(text, kind, id_):
    ref = parse_link(text)
    assert (ref.kind, ref.id) == (kind, id_)


def test_parse_link_rejects_junk():
    with pytest.raises(LinkError):
        parse_link("https://example.com/not-spotify")


def test_request_json_retries_429_with_retry_after():
    responses = iter([FakeResponse(429, None, {"Retry-After": "3"}), FakeResponse(200, {"ok": True})])
    session = FakeSession([(lambda *a: True, lambda *a: next(responses))])
    slept = []
    assert request_json(session, "GET", "https://x", sleep=slept.append) == {"ok": True}
    assert slept == [3.0]


def test_pkce_pair_shape():
    verifier, challenge = make_pkce_pair()
    assert 43 <= len(verifier) <= 128 and "=" not in challenge


def test_authorize_url_has_pkce_params(tmp_path):
    auth = SpotifyAuth("cid", scopes=["user-top-read"], token_path=tmp_path / "t.json")
    url = auth.authorize_url("CHALLENGE", "STATE")
    assert "code_challenge=CHALLENGE" in url and "code_challenge_method=S256" in url
    assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A8888%2Fcallback" in url


def test_token_refresh_keeps_refresh_token(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"access_token": "old", "refresh_token": "R", "expires_at": 0, "scope": "user-top-read"}))
    session = FakeSession([(lambda m, u, p: "api/token" in u, FakeResponse(200, {"access_token": "new", "expires_in": 3600}))])
    auth = SpotifyAuth("cid", scopes=["user-top-read"], token_path=path, session=session)
    assert auth.access_token() == "new"
    saved = json.loads(path.read_text())
    assert saved["refresh_token"] == "R" and saved["scope"] == "user-top-read"
    assert session.calls[0][3]["grant_type"] == "refresh_token"


def _track(i, isrc="USRC17607839"):
    return {
        "type": "track", "id": f"track{i:017d}", "name": f"Song {i}", "duration_ms": 200000, "explicit": False,
        "artists": [{"id": "artistA", "name": "Artist A"}], "external_ids": {"isrc": isrc},
        "album": {"id": "alb1", "name": "Album", "release_date": "2015-05-01", "album_type": "compilation"},
    }


def test_playlist_tracks_new_items_shape_and_pagination():
    page1 = {"items": [{"added_at": "2024-01-01T00:00:00Z", "item": _track(0)}],
             "next": f"https://api.spotify.com/v1/playlists/{PID}/items?offset=1"}
    page2 = {"items": [{"added_at": "2024-01-02T00:00:00Z", "item": _track(1)}, {"item": {"type": "episode"}}], "next": None}
    meta = {"name": "Mine", "owner": {"display_name": "me"}, "items": page1}
    session = FakeSession([
        (lambda m, u, p: u.endswith(f"/playlists/{PID}"), FakeResponse(200, meta)),
        (lambda m, u, p: "offset=1" in u, FakeResponse(200, page2)),
    ])
    client = SpotifyClient(token="T", session=session, cache=Cache(None))
    info, df = client.load(f"https://open.spotify.com/playlist/{PID}")
    assert info["name"] == "Mine" and len(df) == 2  # the podcast episode is dropped
    assert df.loc[0, "isrc"] == "USRC17607839" and df.loc[1, "added_at"] == "2024-01-02T00:00:00Z"


def test_playlist_not_owned_raises_helpful_error():
    meta = {"name": "Someone else's", "owner": {"display_name": "them"}}  # no `items` for non-owned playlists
    session = FakeSession([(lambda m, u, p: u.endswith(f"/playlists/{PID}"), FakeResponse(200, meta))])
    client = SpotifyClient(token="T", session=session, cache=Cache(None))
    with pytest.raises(PlaylistNotAccessible, match="own or collaborate"):
        client.load(PID)


def test_legacy_tracks_shape_still_works():
    meta = {"name": "Legacy", "tracks": {"items": [{"track": _track(0)}], "next": None}}
    session = FakeSession([(lambda m, u, p: u.endswith(f"/playlists/{PID}"), FakeResponse(200, meta))])
    _, df = SpotifyClient(token="T", session=session, cache=Cache(None)).load(PID)
    assert df.loc[0, "name"] == "Song 0"


def test_artists_fetched_individually_and_cached():
    session = FakeSession([(lambda m, u, p: "/artists/" in u, lambda m, u, p: FakeResponse(200, {"id": u.rsplit("/", 1)[1], "name": "x", "genres": ["indie folk"]}))])
    client = SpotifyClient(token="T", session=session, cache=Cache(None))
    info = client.artists(["a1", "a2", "a1"])
    assert set(info) == {"a1", "a2"} and info["a1"]["genres"] == ["indie folk"]
    client.artists(["a1"])
    assert len(session.calls) == 2  # third lookup served from cache


def test_load_extended_history_from_zip(tmp_path):
    recs = [
        {"ts": "2024-03-01T23:30:00Z", "ms_played": 200000, "master_metadata_track_name": "A", "master_metadata_album_artist_name": "X",
         "master_metadata_album_album_name": "Al", "spotify_track_uri": "spotify:track:" + "a" * 22, "reason_start": "clickrow",
         "reason_end": "trackdone", "shuffle": False, "skipped": None, "platform": "ios"},
        {"ts": "2024-03-01T23:35:00Z", "ms_played": 4000, "master_metadata_track_name": "B", "master_metadata_album_artist_name": "Y",
         "spotify_track_uri": "spotify:track:" + "b" * 22, "reason_end": "fwdbtn", "skipped": True},
        {"ts": "2024-03-02T10:00:00Z", "ms_played": 900000, "master_metadata_track_name": None, "episode_name": "Pod"},
    ]
    z = tmp_path / "my_spotify_data.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("Spotify Extended Streaming History/Streaming_History_Audio_2024.json", json.dumps(recs))
    h = load_streaming_history(z)
    assert len(h) == 2 and h["track_id"].iloc[0] == "a" * 22
    assert str(h["ts"].dt.tz) == "UTC"


def test_load_account_history(tmp_path):
    f = tmp_path / "StreamingHistory_music_0.json"
    f.write_text(json.dumps([{"endTime": "2024-01-31 23:59", "artistName": "X", "trackName": "A", "msPlayed": 1234}]))
    h = load_streaming_history(f)
    assert h.loc[0, "artist"] == "X" and h.loc[0, "source"] == "account"


def test_csv_loader_maps_loose_columns(tmp_path):
    p = tmp_path / "export.csv"
    pd.DataFrame({
        "Track URI": ["spotify:track:" + "c" * 22], "Track Name": ["Song"], "Artist Name(s)": ["A, B"],
        "Release Date": ["2001-02-03"], "Happiness": [80], "Energy": [55], "BPM": [120], "Duration": ["3:30"],
        "Genres": ["indie rock;shoegaze"],
    }).to_csv(p, index=False)
    df = load_tracks_csv(p)
    row = df.iloc[0]
    assert row["track_id"] == "c" * 22 and row["artists"] == ["A", "B"]
    assert row["valence"] == pytest.approx(0.8) and row["energy"] == pytest.approx(0.55) and row["tempo"] == 120
    assert row["duration_ms"] == 210000 and row["genres"] == ["indie rock", "shoegaze"]
