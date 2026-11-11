"""Bounded, paced HTTP reads for public metadata surfaces."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

_USER_AGENT = "watch-authenticity/0.1 (research collector)"


class PacedBoundedHTTP:
    def __init__(
        self,
        http_client: Any,
        requests_per_minute: int,
        error_type: type[RuntimeError],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            isinstance(requests_per_minute, bool)
            or not isinstance(requests_per_minute, int)
            or requests_per_minute < 1
        ):
            raise ValueError("requests_per_minute must be a positive integer")
        self._http_client = http_client
        self._minimum_interval = 60.0 / requests_per_minute
        self._error_type = error_type
        self._clock = clock
        self._sleeper = sleeper
        self._next_request_at = clock()

    def get_bytes(self, url: str, limit: int, surface: str) -> bytes:
        delay = self._next_request_at - self._clock()
        if delay > 0:
            self._sleeper(delay)
        response = self._http_client.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=30,
            allow_redirects=False,
            stream=True,
        )
        try:
            status = getattr(response, "status_code", None)
            if status == 429:
                raise self._error_type(f"{surface} rate limited; resume later")
            if status != 200:
                raise self._error_type(f"{surface} returned HTTP {status}")
            chunks: list[bytes] = []
            size = 0
            iterator = getattr(response, "iter_content", None)
            if not callable(iterator):
                raise self._error_type(f"{surface} response is not streamable")
            for chunk in iterator(chunk_size=64 * 1024):
                if not isinstance(chunk, bytes):
                    raise self._error_type(f"{surface} stream is invalid")
                size += len(chunk)
                if size > limit:
                    raise self._error_type(f"{surface} body is too large")
                chunks.append(chunk)
            body = b"".join(chunks)
            if not body:
                raise self._error_type(f"{surface} body is missing")
            return body
        finally:
            self._next_request_at = self._clock() + self._minimum_interval
            close = getattr(response, "close", None)
            if callable(close):
                close()
