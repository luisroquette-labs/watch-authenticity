from __future__ import annotations

from types import SimpleNamespace

import ibm_boto3
import pytest

from acquisition.cos import AtomicCOSClient


class _Events:
    def __init__(self) -> None:
        self.handler = None
        self.registered: list[tuple[str, str]] = []
        self.unregistered: list[tuple[str, str]] = []

    def register_first(self, name, handler, *, unique_id):
        self.handler = handler
        self.registered.append((name, unique_id))

    def unregister(self, name, *, unique_id):
        self.unregistered.append((name, unique_id))
        self.handler = None


class _PreconditionFailed(Exception):
    def __init__(self) -> None:
        self.response = {
            "Error": {"Code": "PreconditionFailed"},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        }


class _COS:
    def __init__(self, *, conflict: bool = False, failure: Exception | None = None) -> None:
        self.meta = SimpleNamespace(events=_Events())
        self.conflict = conflict
        self.failure = failure
        self.request = None

    def put_object(self, **kwargs):
        params = {"headers": {}}
        assert self.meta.events.handler is not None
        self.meta.events.handler(params=params)
        self.request = (kwargs, params)
        if self.failure is not None:
            raise self.failure
        if self.conflict:
            raise _PreconditionFailed


def test_atomic_create_injects_condition_and_delegates_other_operations():
    raw = _COS()
    raw.head_object = lambda **kwargs: kwargs
    client = AtomicCOSClient(raw)

    assert client.put_object_if_absent(Bucket="private", Key="head", Body=b"one") is True
    assert raw.request[1]["headers"] == {"If-None-Match": "*"}
    assert client.head_object(Bucket="private", Key="head") == {
        "Bucket": "private",
        "Key": "head",
    }
    assert raw.meta.events.unregistered == raw.meta.events.registered


def test_atomic_create_returns_false_for_precondition_conflict():
    raw = _COS(conflict=True)

    assert AtomicCOSClient(raw).put_object_if_absent(
        Bucket="private", Key="head", Body=b"two"
    ) is False
    assert raw.meta.events.unregistered == raw.meta.events.registered


def test_atomic_create_propagates_other_failures_and_unregisters_handler():
    raw = _COS(failure=RuntimeError("network"))

    with pytest.raises(RuntimeError, match="network"):
        AtomicCOSClient(raw).put_object_if_absent(
            Bucket="private", Key="head", Body=b"two"
        )

    assert raw.meta.events.unregistered == raw.meta.events.registered


def test_atomic_create_reaches_real_ibm_sdk_request_as_if_none_match():
    class RequestCaptured(Exception):
        pass

    seen = {}
    raw = ibm_boto3.client(
        "s3",
        endpoint_url="https://s3.us-south.cloud-object-storage.appdomain.cloud",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    def capture(request, **_):
        seen["if_none_match"] = request.headers.get("If-None-Match")
        raise RequestCaptured

    raw.meta.events.register_first("before-send.s3.PutObject", capture)
    with pytest.raises(RequestCaptured):
        AtomicCOSClient(raw).put_object_if_absent(
            Bucket="valid-test-bucket-name",
            Key="head",
            Body=b"one",
        )

    assert seen == {"if_none_match": b"*"}
