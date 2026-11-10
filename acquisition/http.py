"""HTTPS client that connects only to a caller-validated address."""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import certifi
import requests
import urllib3


class _PinnedResponse:
    def __init__(self, response: Any, pool: urllib3.HTTPSConnectionPool) -> None:
        self._response = response
        self._pool = pool
        self.status_code = response.status
        self.headers = response.headers

    def iter_content(self, chunk_size: int):
        yield from self._response.stream(chunk_size, decode_content=False)

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._pool.close()


class PinnedRequestsClient:
    """Use a validated IP while checking TLS against the URL hostname."""

    def get_pinned(
        self,
        url: str,
        *,
        resolved_addresses: tuple[str, ...],
        stream: bool,
        timeout: tuple[int, int],
        allow_redirects: bool,
    ) -> _PinnedResponse:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or parsed.fragment
        ):
            raise ValueError("pinned requests require a plain HTTPS URL")
        if allow_redirects:
            raise ValueError("pinned requests cannot follow redirects")
        if not resolved_addresses:
            raise ValueError("resolved_addresses must not be empty")
        try:
            addresses = tuple(str(ipaddress.ip_address(value)) for value in resolved_addresses)
        except ValueError as exc:
            raise ValueError("resolved_addresses must contain IP addresses") from exc
        if len(set(addresses)) != len(addresses):
            raise ValueError("resolved_addresses must not contain duplicates")

        hostname = parsed.hostname.encode("idna").decode("ascii")
        pool = urllib3.HTTPSConnectionPool(
            addresses[0],
            port=443,
            maxsize=1,
            block=True,
            cert_reqs="CERT_REQUIRED",
            ca_certs=certifi.where(),
            assert_hostname=hostname,
            server_hostname=hostname,
        )
        path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        try:
            response = pool.urlopen(
                "GET",
                path,
                headers={"Host": hostname},
                redirect=False,
                retries=False,
                preload_content=not stream,
                decode_content=False,
                timeout=urllib3.Timeout(connect=timeout[0], read=timeout[1]),
            )
        except (urllib3.exceptions.HTTPError, OSError) as exc:
            pool.close()
            raise requests.ConnectionError("pinned HTTPS request failed") from exc
        return _PinnedResponse(response, pool)
