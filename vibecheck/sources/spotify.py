"""Spotify Web API client written for the post-February-2026 Development Mode rules.

What that means in practice (see README for links):
  * playlist contents only come back for playlists the logged-in user owns or collaborates on;
  * batch endpoints (GET /tracks, /artists, /albums) are gone, so artists are fetched one by one
    (and cached);
  * playlists use `items`/`item` instead of `tracks`/`track` (we accept both, so the client also
    works for Extended Quota apps);
  * popularity, followers and audio features are not available.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator

import pandas as pd
import requests

from ..cache import DAY, Cache
from ..http import HTTPError, request_json
from .links import SpotifyRef, parse_link, resolve_short_link

API = "https://api.spotify.com/v1"


class PlaylistNotAccessible(RuntimeError):
    """Raised when Spotify withholds a playlist's tracks (not owned / not collaborative)."""


NOT_ACCESSIBLE_HELP = (
    "Spotify only lets Development Mode apps read the tracks of playlists you own or collaborate on.\n"
    "Workarounds:\n"
    "  1. In Spotify, open the playlist, select all tracks (Ctrl+A on desktop), right-click ->\n"
    "     'Add to playlist' -> 'New playlist', then paste the link to your copy.\n"
    "  2. Export the playlist to CSV with a tool that can read it (e.g. Chosic's playlist analyzer\n"
    "     has a CSV export) and load it with vibecheck.load_tracks_csv()."
)


def track_to_row(track: dict, added_at: str | None = None, position: int | None = None, album: dict | None = None) -> dict | None:
    if not track or track.get("type", "track") != "track":
        return None  # podcast episodes, removed tracks
    alb = track.get("album") or album or {}
    artists = track.get("artists") or []
    return {
        "track_id": track.get("id"),
        "name": track.get("name"),
        "artists": [a.get("name") for a in artists],
        "artist_ids": [a.get("id") for a in artists if a.get("id")],
        "album": alb.get("name"),
        "album_id": alb.get("id"),
        "album_type": alb.get("album_type"),
        "release_date": alb.get("release_date"),
        "isrc": (track.get("external_ids") or {}).get("isrc"),
        "duration_ms": track.get("duration_ms"),
        "explicit": track.get("explicit"),
        "is_local": track.get("is_local", False),
        "added_at": added_at,
        "position": position,
    }


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([r for r in rows if r])
    if df.empty:
        return df
    return df.reset_index(drop=True)


class SpotifyClient:
    def __init__(
        self,
        auth=None,
        token: str | None = None,
        session: requests.Session | None = None,
        cache: Cache | None = None,
        max_workers: int = 4,
    ):
        """Pass a SpotifyAuth (normal use) or a raw access `token` (quick experiments)."""
        if auth is None and token is None:
            raise ValueError("SpotifyClient needs an auth object or an access token")
        self.auth = auth
        self._static_token = token
        self.session = session or requests.Session()
        self.cache = cache or Cache(None)
        self.max_workers = max_workers

    # ---- low level ------------------------------------------------------------
    def _headers(self, force_refresh: bool = False) -> dict:
        tok = self._static_token or self.auth.access_token(force_refresh=force_refresh)
        return {"Authorization": f"Bearer {tok}"}

    def get(self, path_or_url: str, params: dict | None = None, not_found_ok: bool = False) -> Any:
        url = path_or_url if path_or_url.startswith("http") else f"{API}{path_or_url}"
        try:
            return request_json(self.session, "GET", url, params=params, headers=self._headers(), not_found_ok=not_found_ok)
        except HTTPError as e:
            if e.status == 401 and self.auth is not None:
                return request_json(
                    self.session, "GET", url, params=params, headers=self._headers(force_refresh=True), not_found_ok=not_found_ok
                )
            raise

    def paginate(self, path: str, params: dict | None = None, max_items: int | None = None, first_page: dict | None = None) -> Iterator[dict]:
        page = first_page if first_page is not None else self.get(path, params)
        count = 0
        while page:
            for item in page.get("items") or []:
                yield item
                count += 1
                if max_items and count >= max_items:
                    return
            nxt = page.get("next")
            page = self.get(nxt) if nxt else None

    # ---- profile / library ---------------------------------------------------------
    def me(self) -> dict:
        return self.get("/me")

    def my_playlists(self) -> pd.DataFrame:
        me_id = self.me().get("id")
        rows = []
        for p in self.paginate("/me/playlists", {"limit": 50}):
            if not p:
                continue
            owner = (p.get("owner") or {}).get("id")
            paging = p.get("items") or p.get("tracks") or {}
            rows.append(
                {
                    "playlist_id": p.get("id"),
                    "name": p.get("name"),
                    "description": p.get("description"),
                    "owner_id": owner,
                    "readable": owner == me_id or bool(p.get("collaborative")),
                    "collaborative": p.get("collaborative"),
                    "n_tracks": paging.get("total"),
                }
            )
        return pd.DataFrame(rows)

    def saved_tracks(self, max_items: int | None = 500) -> pd.DataFrame:
        rows = [
            track_to_row(e.get("item") or e.get("track"), added_at=e.get("added_at"), position=i)
            for i, e in enumerate(self.paginate("/me/tracks", {"limit": 50}, max_items=max_items))
        ]
        return rows_to_frame(rows)

    def top_tracks(self, time_range: str = "medium_term", limit: int = 50) -> pd.DataFrame:
        page = self.get("/me/top/tracks", {"time_range": time_range, "limit": min(limit, 50)})
        rows = [track_to_row(t, position=i) for i, t in enumerate(page.get("items") or [])]
        return rows_to_frame(rows)

    def top_artists(self, time_range: str = "medium_term", limit: int = 50) -> pd.DataFrame:
        page = self.get("/me/top/artists", {"time_range": time_range, "limit": min(limit, 50)})
        rows = []
        for i, a in enumerate(page.get("items") or []):
            self.cache.set(f"spotify:artist:{a['id']}", _slim_artist(a), ttl=30 * DAY)
            rows.append({"rank": i + 1, "artist_id": a["id"], "name": a.get("name"), "genres": a.get("genres") or []})
        return pd.DataFrame(rows)

    def recently_played(self) -> pd.DataFrame:
        page = self.get("/me/player/recently-played", {"limit": 50})
        rows = []
        for i, e in enumerate(page.get("items") or []):
            row = track_to_row(e.get("item") or e.get("track"), position=i)
            if row:
                row["played_at"] = e.get("played_at")
                rows.append(row)
        return rows_to_frame(rows)

    # ---- catalog ----------------------------------------------------------------
    def playlist(self, playlist_id: str) -> dict:
        return self.get(f"/playlists/{playlist_id}")

    def playlist_tracks(self, playlist_id: str, meta: dict | None = None) -> pd.DataFrame:
        meta = meta or self.playlist(playlist_id)
        embedded = meta.get("items") if isinstance(meta.get("items"), dict) else meta.get("tracks")
        if not isinstance(embedded, dict):
            # Per the Feb 2026 rules, `items` is simply absent for playlists you don't own.
            raise PlaylistNotAccessible(
                f"Spotify returned no tracks for '{meta.get('name', playlist_id)}'.\n" + NOT_ACCESSIBLE_HELP
            )
        first_page = embedded if "items" in embedded else None
        rows = []
        try:
            entries = self.paginate(f"/playlists/{playlist_id}/items", {"limit": 50}, first_page=first_page)
            for i, e in enumerate(entries):
                row = track_to_row(e.get("item") or e.get("track"), added_at=e.get("added_at"), position=i)
                if row and not row["is_local"]:
                    rows.append(row)
        except HTTPError as e:
            if e.status in (403, 404):
                raise PlaylistNotAccessible(NOT_ACCESSIBLE_HELP) from e
            raise
        return rows_to_frame(rows)

    def album_tracks(self, album_id: str) -> tuple[dict, pd.DataFrame]:
        album = self.get(f"/albums/{album_id}")
        slim_album = {k: album.get(k) for k in ("id", "name", "release_date", "album_type")}
        rows = [
            track_to_row(t, position=i, album=slim_album)
            for i, t in enumerate(self.paginate(f"/albums/{album_id}/tracks", first_page=album.get("tracks")))
        ]
        return album, rows_to_frame(rows)

    def track(self, track_id: str) -> dict:
        return self.get(f"/tracks/{track_id}")

    def track_rows(self, track_ids) -> pd.DataFrame:
        """Full metadata rows for many track IDs (one cached request each; batch GET /tracks is gone)."""
        def one(tid):
            key = f"spotify:track:{tid}"
            hit = self.cache.get(key, None)
            if hit is None:
                hit = track_to_row(self.get(f"/tracks/{tid}", not_found_ok=True) or {}) or {}
                self.cache.set(key, hit, ttl=90 * DAY)
            return hit

        ids = [t for t in dict.fromkeys(track_ids) if t]
        with ThreadPoolExecutor(self.max_workers) as pool:
            return rows_to_frame([r for r in pool.map(one, ids) if r])

    def artist(self, artist_id: str) -> dict:
        key = f"spotify:artist:{artist_id}"
        hit = self.cache.get(key, None)
        if hit is not None:
            return hit
        data = self.get(f"/artists/{artist_id}", not_found_ok=True) or {"id": artist_id, "genres": []}
        slim = _slim_artist(data)
        self.cache.set(key, slim, ttl=30 * DAY)
        return slim

    def artists(self, artist_ids) -> dict[str, dict]:
        """Fetch many artists (one request each since batch GET /artists is gone; cached)."""
        ids = list(dict.fromkeys(i for i in artist_ids if i))
        with ThreadPoolExecutor(self.max_workers) as pool:
            return dict(zip(ids, pool.map(self.artist, ids)))

    # ---- convenience --------------------------------------------------------------
    def load(self, link: str | SpotifyRef, max_items: int | None = 1000) -> tuple[dict, pd.DataFrame]:
        """Load anything pasteable. Returns (metadata, tracks DataFrame)."""
        ref = link if isinstance(link, SpotifyRef) else parse_link(link)
        if ref.kind == "short":
            ref = resolve_short_link(self.session, ref.id)
        if ref.kind == "playlist":
            meta = self.playlist(ref.id)
            df = self.playlist_tracks(ref.id, meta=meta)
            info = {
                "kind": "playlist",
                "id": ref.id,
                "name": meta.get("name"),
                "description": meta.get("description"),
                "owner": (meta.get("owner") or {}).get("display_name"),
            }
            return info, df
        if ref.kind == "album":
            album, df = self.album_tracks(ref.id)
            return {"kind": "album", "id": ref.id, "name": album.get("name")}, df
        if ref.kind == "track":
            t = self.track(ref.id)
            return {"kind": "track", "id": ref.id, "name": t.get("name")}, rows_to_frame([track_to_row(t)])
        if ref.kind == "liked":
            return {"kind": "liked", "name": "Liked Songs"}, self.saved_tracks(max_items=max_items)
        if ref.kind == "recent":
            return {"kind": "recent", "name": "Recently played"}, self.recently_played()
        if ref.kind == "top":
            return {"kind": "top", "name": f"Top tracks ({ref.id})"}, self.top_tracks(ref.id)
        raise ValueError(f"Don't know how to analyze a {ref.kind} link yet")


def _slim_artist(a: dict) -> dict:
    return {"id": a.get("id"), "name": a.get("name"), "genres": a.get("genres") or []}
