"""Parse whatever the user pastes: open.spotify.com URLs, spotify: URIs, bare IDs, or keywords."""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

KINDS = ("playlist", "album", "track", "artist")
_ID = r"[A-Za-z0-9]{22}"
KEYWORDS = {
    "liked": ("liked", None),
    "liked songs": ("liked", None),
    "saved": ("liked", None),
    "recent": ("recent", None),
    "recently played": ("recent", None),
    "top": ("top", "medium_term"),
    "top:short_term": ("top", "short_term"),
    "top:medium_term": ("top", "medium_term"),
    "top:long_term": ("top", "long_term"),
}


@dataclass(frozen=True)
class SpotifyRef:
    kind: str  # playlist | album | track | artist | liked | recent | top | short
    id: str | None = None

    def __str__(self) -> str:
        return f"{self.kind}:{self.id}" if self.id else self.kind


class LinkError(ValueError):
    pass


def parse_link(text: str) -> SpotifyRef:
    """Turn a pasted link/URI/ID into a SpotifyRef.

    >>> parse_link("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc")
    SpotifyRef(kind='playlist', id='37i9dQZF1DXcBWIGoYBM5M')
    >>> parse_link("spotify:album:4LH4d3cOWNNsVw41Gqt2kv").kind
    'album'
    """
    s = text.strip()
    low = s.lower()
    if low in KEYWORDS:
        kind, arg = KEYWORDS[low]
        return SpotifyRef(kind, arg)

    m = re.fullmatch(rf"spotify:({'|'.join(KINDS)}):({_ID})", s)
    if m:
        return SpotifyRef(m.group(1), m.group(2))
    # Legacy user-scoped playlist URIs: spotify:user:<name>:playlist:<id>
    m = re.fullmatch(rf"spotify:user:[^:]+:playlist:({_ID})", s)
    if m:
        return SpotifyRef("playlist", m.group(1))

    if re.fullmatch(_ID, s):
        return SpotifyRef("playlist", s)

    if "://" not in s and ("spotify.com" in low or "spotify.link" in low):
        s = "https://" + s
    url = urlparse(s)
    host = (url.hostname or "").lower()
    if host in ("spotify.link", "spotify.app.link"):
        return SpotifyRef("short", s)
    if host.endswith("spotify.com"):
        parts = [p for p in url.path.split("/") if p]
        # Drop locale segments like "intl-de" and embed prefixes.
        parts = [p for p in parts if not p.startswith("intl-") and p != "embed"]
        for i, part in enumerate(parts[:-1]):
            if part in KINDS and re.fullmatch(_ID, parts[i + 1]):
                return SpotifyRef(part, parts[i + 1])
        if parts[:1] == ["collection"] and parts[1:2] == ["tracks"]:
            return SpotifyRef("liked", None)

    raise LinkError(
        f"Couldn't read a Spotify playlist/album/track from {text!r}. "
        "Paste a link like https://open.spotify.com/playlist/<id>."
    )


def resolve_short_link(session, url: str) -> SpotifyRef:
    """spotify.link URLs redirect to open.spotify.com; follow them to find the real target."""
    resp = session.get(url, allow_redirects=True, timeout=15)
    for hop in [*resp.history, resp]:
        try:
            ref = parse_link(hop.url)
        except LinkError:
            continue
        if ref.kind != "short":
            return ref
    # Some short links land on an HTML page that embeds the canonical URL.
    m = re.search(r"https://open\.spotify\.com/(playlist|album|track|artist)/([A-Za-z0-9]{22})", resp.text)
    if m:
        return SpotifyRef(m.group(1), m.group(2))
    raise LinkError(f"Couldn't resolve short link {url}")
