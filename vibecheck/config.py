"""Settings, read from environment variables (or a .env file in the working directory)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SCOPES = (
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-top-read",
    "user-read-recently-played",
    "user-library-read",
)


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines) so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    spotify_client_id: str | None = None
    # Spotify requires a loopback IP literal (not "localhost") for http redirect URIs.
    spotify_redirect_uri: str = "http://127.0.0.1:8888/callback"
    spotify_scopes: tuple[str, ...] = DEFAULT_SCOPES
    lastfm_api_key: str | None = None
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".vibecheck")
    user_agent: str = "vibecheck/0.1 (personal music-analysis project)"
    # "auto" uses the transformer model if `transformers` is installed, else the lexicon model.
    emotion_backend: str = "auto"

    @classmethod
    def from_env(cls, dotenv: bool = True) -> "Settings":
        if dotenv:
            _load_dotenv()
        s = cls()
        s.spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID") or None
        s.spotify_redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", s.spotify_redirect_uri)
        s.lastfm_api_key = os.getenv("LASTFM_API_KEY") or None
        if os.getenv("VIBECHECK_CACHE_DIR"):
            s.cache_dir = Path(os.environ["VIBECHECK_CACHE_DIR"])
        s.emotion_backend = os.getenv("VIBECHECK_EMOTION_BACKEND", s.emotion_backend)
        return s
