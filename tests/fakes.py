"""A fake requests.Session that serves canned JSON by URL, so network code is testable offline."""
from __future__ import annotations

import json
from urllib.parse import urlencode


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, url=""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.url = url
        self.content = b"" if payload is None else json.dumps(payload).encode()
        self.text = self.content.decode()
        self.history = []

    def json(self):
        return self._payload


class FakeSession:
    """routes: list of (predicate(method, url, params) -> bool, response or callable)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        self.headers = {}

    def request(self, method, url, params=None, headers=None, data=None, timeout=None):
        self.calls.append((method, url, dict(params or {}), dict(data or {})))
        full = url + ("?" + urlencode(params) if params else "")
        for pred, resp in self.routes:
            if pred(method, url, params or {}):
                r = resp(method, url, params or {}) if callable(resp) else resp
                r.url = full
                return r
        return FakeResponse(404, {"error": "no route"}, url=full)

    def get(self, url, **kw):
        return self.request("GET", url, **kw)
