"""Spotify Authorization Code + PKCE flow (no client secret needed), with token caching.

The first call opens a browser for you to log in; the token (and refresh token) are then
cached under ~/.vibecheck/spotify_token.json and refreshed automatically.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from ..http import request_json

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"


def make_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


class SpotifyAuth:
    def __init__(
        self,
        client_id: str,
        redirect_uri: str = "http://127.0.0.1:8888/callback",
        scopes: tuple[str, ...] | list[str] = (),
        token_path: str | Path | None = None,
        session: requests.Session | None = None,
        open_browser: bool = True,
        login_timeout: float = 300,
    ):
        if not client_id:
            raise ValueError(
                "SPOTIFY_CLIENT_ID is not set. Create an app at "
                "https://developer.spotify.com/dashboard and put its Client ID in .env"
            )
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.scopes = list(scopes)
        self.token_path = Path(token_path) if token_path else Path.home() / ".vibecheck" / "spotify_token.json"
        self.session = session or requests.Session()
        self.open_browser = open_browser
        self.login_timeout = login_timeout
        self._token: dict | None = None
        self._lock = threading.Lock()

    # ---- public -------------------------------------------------------------
    def access_token(self, force_refresh: bool = False) -> str:
        with self._lock:
            tok = self._token or self._load()
            if tok and set(self.scopes) - set(tok.get("scope", "").split()):
                tok = None  # cached token lacks scopes we now need -> log in again
            if tok and (force_refresh or tok["expires_at"] - 60 < time.time()):
                tok = self._refresh(tok) if tok.get("refresh_token") else None
            if not tok:
                tok = self.login()
            self._token = tok
            return tok["access_token"]

    def authorize_url(self, challenge: str, state: str) -> str:
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
            "state": state,
            "scope": " ".join(self.scopes),
        }
        return f"{AUTH_URL}?{urlencode(params)}"

    def login(self) -> dict:
        verifier, challenge = make_pkce_pair()
        state = secrets.token_urlsafe(16)
        url = self.authorize_url(challenge, state)
        code = self._wait_for_code(url, state)
        tok = self._token_request(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "client_id": self.client_id,
                "code_verifier": verifier,
            }
        )
        self._save(tok)
        return tok

    def logout(self) -> None:
        self._token = None
        if self.token_path.exists():
            self.token_path.unlink()

    # ---- internals ----------------------------------------------------------
    def _wait_for_code(self, url: str, state: str) -> str:
        print("Log in to Spotify to continue:\n  " + url)
        if self.open_browser:
            webbrowser.open(url)
        parsed = urlparse(self.redirect_uri)
        try:
            params = _catch_redirect(parsed.hostname, parsed.port or 80, parsed.path, self.login_timeout)
        except OSError:
            # Port busy / remote kernel: fall back to pasting the redirected URL.
            pasted = input("After approving, paste the full URL your browser was sent to: ")
            params = {k: v[0] for k, v in parse_qs(urlparse(pasted).query).items()}
        if params.get("error"):
            raise RuntimeError(f"Spotify login failed: {params['error']}")
        if params.get("state") != state:
            raise RuntimeError("Spotify login failed: state mismatch (possible CSRF); try again.")
        return params["code"]

    def _refresh(self, tok: dict) -> dict | None:
        try:
            new = self._token_request(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": tok["refresh_token"],
                    "client_id": self.client_id,
                }
            )
        except Exception:
            return None
        new.setdefault("refresh_token", tok["refresh_token"])  # Spotify may not rotate it
        new.setdefault("scope", tok.get("scope", ""))
        self._save(new)
        return new

    def _token_request(self, data: dict) -> dict:
        tok = request_json(
            self.session,
            "POST",
            TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        tok["expires_at"] = time.time() + int(tok.get("expires_in", 3600))
        return tok

    def _load(self) -> dict | None:
        try:
            return json.loads(self.token_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _save(self, tok: dict) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(tok), encoding="utf-8")


def _catch_redirect(host: str, port: int, path: str, timeout: float) -> dict:
    """Serve one request on the redirect URI and return its query parameters."""
    result: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            if u.path != path:
                self.send_response(404)
                self.end_headers()
                return
            result.update({k: v[0] for k, v in parse_qs(u.query).items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<h2>vibecheck: you're logged in. You can close this tab.</h2>")

        def log_message(self, *args):  # silence default logging
            pass

    server = HTTPServer((host, port), Handler)
    server.timeout = 1
    deadline = time.time() + timeout
    try:
        while not result and time.time() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if not result:
        raise TimeoutError("Timed out waiting for the Spotify login redirect.")
    return result
