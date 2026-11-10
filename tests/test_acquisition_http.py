from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from acquisition.http import PinnedRequestsClient


def _client(**overrides):
    values = {
        "requests_per_minute": 60,
        "clock": lambda: 0.0,
        "sleeper": lambda _seconds: None,
    }
    values.update(overrides)
    return PinnedRequestsClient(**values)


def test_pinned_client_connects_to_ip_and_validates_original_hostname(monkeypatch):
    response = MagicMock(status=200, headers={"Content-Type": "image/jpeg"})
    response.stream.return_value = iter((b"one", b"two"))
    pool = MagicMock()
    pool.urlopen.return_value = response
    create_pool = MagicMock(return_value=pool)
    monkeypatch.setattr("acquisition.http.urllib3.HTTPSConnectionPool", create_pool)

    result = _client().get_pinned(
        "https://upload.wikimedia.org/path/watch.jpg?width=1280",
        resolved_addresses=("203.0.113.8",),
        stream=True,
        timeout=(10, 30),
        allow_redirects=False,
    )

    assert create_pool.call_args.args == ("203.0.113.8",)
    assert create_pool.call_args.kwargs["assert_hostname"] == "upload.wikimedia.org"
    assert create_pool.call_args.kwargs["server_hostname"] == "upload.wikimedia.org"
    request = pool.urlopen.call_args
    assert request.args == ("GET", "/path/watch.jpg?width=1280")
    assert request.kwargs["headers"] == {
        "Host": "upload.wikimedia.org",
        "User-Agent": (
            "watch-authenticity/0.1 "
            "(https://github.com/luisroquette-labs/watch-authenticity)"
        ),
    }
    assert request.kwargs["redirect"] is False
    assert request.kwargs["retries"] is False
    assert list(result.iter_content(3)) == [b"one", b"two"]
    result.close()
    response.close.assert_called_once_with()
    pool.close.assert_called_once_with()


@pytest.mark.parametrize(
    "url,addresses,redirects",
    [
        ("https://upload.wikimedia.org/a.jpg", (), False),
        ("https://upload.wikimedia.org/a.jpg", ("not-an-ip",), False),
        ("https://upload.wikimedia.org/a.jpg", ("203.0.113.8", "203.0.113.8"), False),
        ("https://upload.wikimedia.org/a.jpg", ("203.0.113.8",), True),
        ("http://upload.wikimedia.org/a.jpg", ("203.0.113.8",), False),
    ],
)
def test_pinned_client_rejects_unpinned_or_redirecting_requests(
    monkeypatch, url, addresses, redirects
):
    create_pool = MagicMock()
    monkeypatch.setattr("acquisition.http.urllib3.HTTPSConnectionPool", create_pool)

    with pytest.raises(ValueError):
        _client().get_pinned(
            url,
            resolved_addresses=addresses,
            stream=True,
            timeout=(10, 30),
            allow_redirects=redirects,
        )

    create_pool.assert_not_called()


def test_every_attempt_and_redirect_call_is_rate_limited(monkeypatch):
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    response = MagicMock(status=200, headers={})
    pool = MagicMock()
    pool.urlopen.return_value = response
    monkeypatch.setattr(
        "acquisition.http.urllib3.HTTPSConnectionPool", MagicMock(return_value=pool)
    )
    client = _client(
        requests_per_minute=20,
        clock=lambda: now[0],
        sleeper=sleep,
    )

    first = client.get_pinned(
        "https://upload.wikimedia.org/one.jpg",
        resolved_addresses=("203.0.113.8",),
        stream=True,
        timeout=(10, 30),
        allow_redirects=False,
    )
    second = client.get_pinned(
        "https://upload.wikimedia.org/two.jpg",
        resolved_addresses=("203.0.113.8",),
        stream=True,
        timeout=(10, 30),
        allow_redirects=False,
    )

    assert sleeps == [3.0]
    first.close()
    second.close()


def test_redirect_and_retry_worst_case_cannot_exceed_reserved_minute(monkeypatch):
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    response = MagicMock(status=200, headers={})
    pool = MagicMock()
    pool.urlopen.return_value = response
    monkeypatch.setattr(
        "acquisition.http.urllib3.HTTPSConnectionPool", MagicMock(return_value=pool)
    )
    client = _client(
        requests_per_minute=19,
        clock=lambda: now[0],
        sleeper=sleep,
    )

    for attempt in range(20):
        result = client.get_pinned(
            f"https://upload.wikimedia.org/attempt-{attempt}.jpg",
            resolved_addresses=("203.0.113.8",),
            stream=True,
            timeout=(10, 30),
            allow_redirects=False,
        )
        result.close()

    assert len(sleeps) == 19
    assert now[0] == pytest.approx(60.0)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_rate_limit_requires_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        PinnedRequestsClient(requests_per_minute=value)
