"""Small IBM COS adapter for atomic, create-only ledger writes."""

from __future__ import annotations

from threading import Lock
from typing import Any


class AtomicCOSClient:
    """Delegate COS operations and add conditional create for ledger heads."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._conditional_put_lock = Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def put_object_if_absent(self, **kwargs: Any) -> bool:
        """Create an object only when its key does not already exist."""
        events = getattr(getattr(self._client, "meta", None), "events", None)
        if events is None:
            raise RuntimeError("COS client does not expose request events")

        event_name = "before-call.s3.PutObject"
        handler_id = f"watch-auth-if-none-match-{id(self)}"

        def add_condition(*, params: dict[str, Any], **_: Any) -> None:
            params.setdefault("headers", {})["If-None-Match"] = "*"

        with self._conditional_put_lock:
            events.register_first(event_name, add_condition, unique_id=handler_id)
            try:
                self._client.put_object(**kwargs)
            except Exception as exc:
                if _precondition_failed(exc):
                    return False
                raise
            finally:
                events.unregister(event_name, unique_id=handler_id)
        return True


def _precondition_failed(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    metadata = response.get("ResponseMetadata")
    error = response.get("Error")
    return (
        isinstance(metadata, dict)
        and metadata.get("HTTPStatusCode") == 412
    ) or (
        isinstance(error, dict)
        and error.get("Code") in {"412", "PreconditionFailed"}
    )
