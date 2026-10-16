"""Shared HTTP helper: retries on 429 (honouring Retry-After) and transient 5xx errors."""
from __future__ import annotations

import time
from typing import Any

import requests


class HTTPError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status
        self.url = url
        self.body = body


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    data: dict | None = None,
    max_retries: int = 5,
    timeout: float = 20,
    not_found_ok: bool = False,
    sleep=time.sleep,
) -> Any:
    """Return parsed JSON, or None for a 404 when `not_found_ok`.

    Raises HTTPError for other non-2xx responses once retries are exhausted.
    """
    attempt = 0
    while True:
        resp = session.request(
            method, url, params=params, headers=headers, data=data, timeout=timeout
        )
        status = resp.status_code
        if status == 404 and not_found_ok:
            return None
        if status == 429 or 500 <= status < 600:
            if attempt >= max_retries:
                raise HTTPError(status, url, resp.text)
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else 2 ** attempt
            except ValueError:
                wait = 2 ** attempt
            sleep(min(wait, 60))
            attempt += 1
            continue
        if status >= 400:
            raise HTTPError(status, url, resp.text)
        if status == 204 or not resp.content:
            return None
        return resp.json()


def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent
    return s
