from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from acquisition.http import PinnedRequestsClient


def test_pinned_client_connects_to_ip_and_validates_original_hostname(monkeypatch):
    response = MagicMock(status=200, headers={"Content-Type": "image/jpeg"})
    response.stream.return_value = iter((b"one", b"two"))
    pool = MagicMock()
    pool.urlopen.return_value = response
    create_pool = MagicMock(return_value=pool)
    monkeypatch.setattr("acquisition.http.urllib3.HTTPSConnectionPool", create_pool)

    result = PinnedRequestsClient().get_pinned(
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
    assert request.kwargs["headers"] == {"Host": "upload.wikimedia.org"}
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
        PinnedRequestsClient().get_pinned(
            url,
            resolved_addresses=addresses,
            stream=True,
            timeout=(10, 30),
            allow_redirects=redirects,
        )

    create_pool.assert_not_called()
