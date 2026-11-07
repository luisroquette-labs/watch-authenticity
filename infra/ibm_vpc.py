"""Fail-closed IBM VPC instance lifecycle with deterministic reconciliation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import signal
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator
from urllib.parse import parse_qs, urlsplit

from infra.ibm_types import InstanceRef, RunSpec

if TYPE_CHECKING:
    from ibm_vpc import VpcV1


class AmbiguousProvisioning(RuntimeError):
    """Provider state cannot safely authorize creation or automatic cleanup."""


class PrototypeMismatch(AmbiguousProvisioning):
    """A deterministic name exists but its provider-owned fields differ."""


class MissingOwnershipTags(PrototypeMismatch):
    """A resource has no conflicting owner but lacks the current run tags."""


class ProvisioningFailed(RuntimeError):
    """The instance entered a terminal or unknown state before running."""


class CleanupFailures(RuntimeError):
    """Every independently observed teardown failure."""

    def __init__(self, errors: tuple[Exception, ...]) -> None:
        self.errors = errors
        summary = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"IBM VPC cleanup failed: {summary}")


class ResourceAbsent(RuntimeError):
    """The VPC resource GET itself returned 404."""


_TRANSITIONAL_STATES = frozenset({"pending", "starting", "restarting"})
_RECONCILE_ATTEMPTS = 4
_MAX_POLL_SECONDS = 30 * 60
_MAX_REQUEST_SECONDS = 5.0
_DEFAULT_CREATE_SECONDS = 60.0
_RECOVERY_DIR = Path.home() / ".local/state/watch-authenticity/ibm-vpc"
_PROVIDER_CALL_LOCK = threading.RLock()


@contextmanager
def _hard_deadline(deadline: float) -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        raise AmbiguousProvisioning(
            "provider calls require the main thread for a hard deadline"
        )
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise AmbiguousProvisioning("platform cannot enforce hard provider deadlines")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("provider call deadline expired")
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise AmbiguousProvisioning("an existing process alarm blocks provider calls")

    previous_handler = signal.getsignal(signal.SIGALRM)

    def expire(_signum: int, _frame: object) -> None:
        raise TimeoutError("provider call deadline expired")

    signal.signal(signal.SIGALRM, expire)
    interval = min(0.05, max(0.001, remaining))
    signal.setitimer(signal.ITIMER_REAL, remaining, interval)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


class _DeadlineClient:
    def __init__(self, client: Any, deadline: float) -> None:
        self._client = client
        self.deadline = deadline

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._client, name)
        if not callable(value):
            return value

        def bounded(*args: object, **kwargs: object) -> object:
            setter = getattr(self._client, "set_http_config", None)
            if not callable(setter):
                raise AmbiguousProvisioning(
                    "provider client cannot enforce request deadlines"
                )
            wait_seconds = self.deadline - time.monotonic()
            if wait_seconds <= 0 or not _PROVIDER_CALL_LOCK.acquire(
                timeout=wait_seconds
            ):
                raise TimeoutError("provider call deadline expired")
            try:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("provider call deadline expired")
                current = getattr(self._client, "http_config", None)
                previous = dict(current) if isinstance(current, dict) else {}
                active = dict(previous)
                active["timeout"] = min(_MAX_REQUEST_SECONDS, remaining)
                setter(active)
                try:
                    with _hard_deadline(self.deadline):
                        return value(*args, **kwargs)
                finally:
                    setter(previous)
            finally:
                _PROVIDER_CALL_LOCK.release()

        return bounded


def _deadline_client(client: Any, deadline: float) -> Any:
    if isinstance(client, _DeadlineClient):
        client.deadline = deadline
        return client
    return _DeadlineClient(client, deadline)


def build_instance_prototype(spec: RunSpec, user_data: str) -> dict[str, object]:
    """Build the exact SDK payload bound by the authorized RunSpec."""

    if not isinstance(user_data, str) or not user_data.strip():
        raise ValueError("user_data must be a non-empty string")
    if "\x00" in user_data or len(user_data.encode()) > 64 * 1024:
        raise ValueError("user_data must be NUL-free and at most 64 KiB")
    if hashlib.sha256(user_data.encode()).hexdigest() != spec.bootstrap_sha256:
        raise ValueError("user_data differs from the authorized bootstrap SHA-256")
    return {
        "name": f"watch-auth-{spec.run_id}",
        "resource_group": {"id": spec.resource_group_id},
        "vpc": {"id": spec.network_id},
        "zone": {"name": spec.zone},
        "image": {"id": spec.image_id},
        "profile": {"name": spec.profile},
        "primary_network_interface": {
            "subnet": {"id": spec.subnet_id},
            "primary_ip": {
                "address": spec.primary_ipv4_address,
                "auto_delete": False,
            },
            "security_groups": [{"id": spec.security_group_id}],
            "allow_ip_spoofing": False,
        },
        "boot_volume_attachment": {
            "delete_volume_on_instance_delete": True,
            "volume": {
                "capacity": int(spec.boot_volume_gb),
                "profile": {"name": "general-purpose"},
                "resource_group": {"id": spec.resource_group_id},
                "user_tags": ["project:watch-authenticity", f"run:{spec.run_id}"],
            },
        },
        "metadata_service": {
            "enabled": True,
            "protocol": "http",
            "response_hop_limit": 2,
        },
        "default_trusted_profile": {
            "auto_link": True,
            "target": {"id": spec.trusted_profile_id},
        },
        "user_data": user_data,
    }


def _result(response: object) -> dict[str, Any]:
    if isinstance(response, dict):
        result = response
    else:
        getter = getattr(response, "get_result", None)
        result = getter() if callable(getter) else None
    if not isinstance(result, dict):
        raise AmbiguousProvisioning("IBM VPC response is not a mapping")
    return result


def _field(record: dict[str, Any], *path: str) -> str:
    value: object = record
    for key in path:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return value if isinstance(value, str) else ""


def _resource_crn(
    record: dict[str, Any],
    spec: RunSpec,
    resource_type: str,
    resource_id: str,
) -> str:
    crn = _field(record, "crn")
    expected = _expected_resource_crn(spec, resource_type, resource_id)
    if crn != expected:
        raise PrototypeMismatch(f"{resource_type} CRN does not match the authorized resource")
    return crn


def _expected_resource_crn(
    spec: RunSpec,
    resource_type: str,
    resource_id: str,
) -> str:
    return (
        f"crn:v1:bluemix:public:is:{spec.zone}:a/{spec.account_id}"
        f"::{resource_type}:{resource_id}"
    )


def _instance_ref(record: dict[str, Any]) -> InstanceRef:
    network = record.get("primary_network_interface")
    if not isinstance(network, dict):
        attachment = record.get("primary_network_attachment")
        network = attachment if isinstance(attachment, dict) else {}
    return InstanceRef(
        id=_field(record, "id"),
        name=_field(record, "name"),
        profile=_field(record, "profile", "name"),
        zone=_field(record, "zone", "name"),
        vpc_id=_field(record, "vpc", "id"),
        subnet_id=_field(network, "subnet", "id"),
        image_id=_field(record, "image", "id"),
        status=_field(record, "status"),
    )


def _mismatches(record: dict[str, Any], spec: RunSpec) -> tuple[str, ...]:
    network = record.get("primary_network_interface")
    if not isinstance(network, dict):
        attachment = record.get("primary_network_attachment")
        network = attachment if isinstance(attachment, dict) else {}
    expected = {
        "name": f"watch-auth-{spec.run_id}",
        "resource_group": spec.resource_group_id,
        "profile": spec.profile,
        "zone": spec.zone,
        "vpc": spec.network_id,
        "subnet": spec.subnet_id,
        "primary_ip": spec.primary_ipv4_address,
        "image": spec.image_id,
    }
    actual = {
        "name": _field(record, "name"),
        "resource_group": _field(record, "resource_group", "id"),
        "profile": _field(record, "profile", "name"),
        "zone": _field(record, "zone", "name"),
        "vpc": _field(record, "vpc", "id"),
        "subnet": _field(network, "subnet", "id"),
        "primary_ip": _field(network, "primary_ip", "address"),
        "image": _field(record, "image", "id"),
    }
    mismatch = [field for field, value in expected.items() if actual[field] != value]
    metadata = record.get("metadata_service")
    if not isinstance(metadata, dict) or metadata != {
        "enabled": True,
        "protocol": "http",
        "response_hop_limit": 2,
    }:
        mismatch.append("metadata_service")
    return tuple(mismatch)


def _next_start(
    page: dict[str, Any],
    observed_count: int,
    *,
    context: str,
    expected_total_count: int | None,
) -> tuple[str | None, int]:
    total_count = page.get("total_count")
    if (
        not isinstance(total_count, int)
        or isinstance(total_count, bool)
        or total_count < observed_count
        or (
            expected_total_count is not None
            and total_count != expected_total_count
        )
    ):
        raise AmbiguousProvisioning(f"{context} total count is malformed")
    next_page = page.get("next")
    if observed_count == total_count:
        if next_page is not None:
            raise AmbiguousProvisioning(f"{context} has an unexpected next page")
        return None, total_count
    if not isinstance(next_page, dict):
        raise AmbiguousProvisioning(f"{context} ended before total count")
    direct = next_page.get("start")
    if isinstance(direct, str) and direct:
        return direct, total_count
    href = next_page.get("href")
    if not isinstance(href, str):
        raise AmbiguousProvisioning(f"{context} next page has no cursor")
    values = parse_qs(urlsplit(href).query).get("start")
    if not values or not values[0]:
        raise AmbiguousProvisioning(f"{context} next page has no cursor")
    return values[0], total_count


def _list_named(vpc: VpcV1, spec: RunSpec) -> tuple[dict[str, Any], ...]:
    name = f"watch-auth-{spec.run_id}"
    matches: list[dict[str, Any]] = []
    observed_count = 0
    start: str | None = None
    seen: set[str] = set()
    expected_total_count: int | None = None
    while True:
        kwargs: dict[str, object] = {
            "name": name,
            "resource_group_id": spec.resource_group_id,
        }
        if start is not None:
            kwargs["start"] = start
        page = _result(vpc.list_instances(**kwargs))
        instances = page.get("instances")
        if not isinstance(instances, list) or any(not isinstance(item, dict) for item in instances):
            raise AmbiguousProvisioning("IBM VPC instance list is malformed")
        observed_count += len(instances)
        matches.extend(item for item in instances if item.get("name") == name)
        start, expected_total_count = _next_start(
            page,
            observed_count,
            context="IBM VPC instance list",
            expected_total_count=expected_total_count,
        )
        if start is None:
            return tuple(matches)
        if start in seen:
            raise AmbiguousProvisioning("IBM VPC pagination repeated a cursor")
        seen.add(start)


def _one_match(vpc: VpcV1, spec: RunSpec) -> dict[str, Any] | None:
    matches = _list_named(vpc, spec)
    if len(matches) > 1:
        raise AmbiguousProvisioning("multiple instances use the deterministic run name")
    if not matches:
        return None
    mismatch = _mismatches(matches[0], spec)
    if mismatch:
        raise PrototypeMismatch("instance prototype mismatch: " + ", ".join(mismatch))
    return matches[0]


def _prototype_sha256(prototype: dict[str, object]) -> str:
    encoded = json.dumps(prototype, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_recovery_dir() -> None:
    missing: list[Path] = []
    current = _RECOVERY_DIR
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        _fsync_directory(directory.parent)


def _write_recovery(name: str, record: dict[str, object]) -> None:
    _ensure_recovery_dir()
    descriptor, temporary = tempfile.mkstemp(prefix=f"{name}-", dir=_RECOVERY_DIR)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(record, target, sort_keys=True, separators=(",", ":"))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, _RECOVERY_DIR / f"{name}.json")
        _fsync_directory(_RECOVERY_DIR)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _intent_record(
    spec: RunSpec,
    prototype: dict[str, object],
) -> dict[str, object]:
    return {
        "run_id": str(spec.run_id),
        "account_id": spec.account_id,
        "instance_name": prototype["name"],
        "prototype": prototype,
        "prototype_sha256": _prototype_sha256(prototype),
    }


def _persist_intent(spec: RunSpec, prototype: dict[str, object]) -> None:
    _write_recovery(f"{spec.run_id}-prototype", _intent_record(spec, prototype))


def _persist_accepted_instance(
    spec: RunSpec,
    prototype: dict[str, object],
    instance_id: str,
) -> None:
    record: dict[str, object] = {
        "run_id": str(spec.run_id),
        "account_id": spec.account_id,
        "instance_id": instance_id,
        "instance_name": prototype["name"],
        "prototype_sha256": _prototype_sha256(prototype),
        "subnet_id": spec.subnet_id,
        "public_gateway_id": spec.public_gateway_id,
        "public_gateway_floating_ip_id": spec.public_gateway_floating_ip_id,
    }
    path = _RECOVERY_DIR / f"{spec.run_id}-accepted.json"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _write_recovery(f"{spec.run_id}-accepted", record)
    except json.JSONDecodeError as exc:
        raise AmbiguousProvisioning("accepted instance record is unreadable") from exc
    else:
        if existing != record:
            raise AmbiguousProvisioning("accepted instance record differs from provider state")


def _persist_created_instance(
    spec: RunSpec,
    prototype: dict[str, object],
    instance: dict[str, Any],
) -> None:
    instance_id = _field(instance, "id")
    if not instance_id:
        raise AmbiguousProvisioning("created instance ID is missing")
    _persist_accepted_instance(spec, prototype, instance_id)
    crn = _resource_crn(instance, spec, "instance", instance_id)
    _persist_instance_crn(spec, instance_id, crn)
    candidate_volume_id = _field(instance, "boot_volume_attachment", "volume", "id")
    if not candidate_volume_id:
        raise PrototypeMismatch("instance boot volume attachment is incomplete")
    record = {
        "run_id": str(spec.run_id),
        "account_id": spec.account_id,
        "instance_id": instance_id,
        "instance_name": prototype["name"],
        "prototype_sha256": _prototype_sha256(prototype),
        "volume_id": candidate_volume_id,
        "subnet_id": spec.subnet_id,
        "public_gateway_id": spec.public_gateway_id,
        "public_gateway_floating_ip_id": spec.public_gateway_floating_ip_id,
    }
    path = _RECOVERY_DIR / f"{spec.run_id}-created.json"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _write_recovery(f"{spec.run_id}-created", record)
    except json.JSONDecodeError as exc:
        raise AmbiguousProvisioning("created instance record is unreadable") from exc
    else:
        if existing != record:
            raise AmbiguousProvisioning("created instance record differs from provider state")


def recover_created_instance_id(
    spec: RunSpec,
    prototype: dict[str, object],
) -> str:
    """Return the durably recorded post-create ID for fail-safe teardown."""

    record: object | None = None
    parse_error: Exception | None = None
    for path in (
        _RECOVERY_DIR / f"{spec.run_id}-accepted.json",
        _RECOVERY_DIR / f"{spec.run_id}-created.json",
    ):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            break
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as exc:
            parse_error = exc
            break
    if record is None:
        raise AmbiguousProvisioning(
            "created instance record is unavailable"
        ) from parse_error
    expected = {
        "run_id": str(spec.run_id),
        "account_id": spec.account_id,
        "instance_name": prototype["name"],
        "prototype_sha256": _prototype_sha256(prototype),
    }
    if not isinstance(record, dict) or any(record.get(key) != value for key, value in expected.items()):
        raise AmbiguousProvisioning("created instance record does not match the RunSpec")
    instance_id = record.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise AmbiguousProvisioning("created instance record has no instance ID")
    return instance_id


def _require_accepted_instance_id(spec: RunSpec, instance_id: str) -> None:
    path = _RECOVERY_DIR / f"{spec.run_id}-accepted.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise AmbiguousProvisioning("accepted instance record is unavailable") from exc
    expected = {
        "run_id": str(spec.run_id),
        "account_id": spec.account_id,
        "instance_id": instance_id,
        "subnet_id": spec.subnet_id,
        "public_gateway_id": spec.public_gateway_id,
        "public_gateway_floating_ip_id": spec.public_gateway_floating_ip_id,
    }
    if not isinstance(record, dict) or any(
        record.get(key) != value for key, value in expected.items()
    ):
        raise AmbiguousProvisioning(
            "accepted instance identity differs from provider state cleanup"
        )


def _persist_instance_crn(spec: RunSpec, instance_id: str, crn: str) -> None:
    _require_accepted_instance_id(spec, instance_id)
    record: dict[str, object] = {
        "run_id": str(spec.run_id),
        "account_id": spec.account_id,
        "instance_id": instance_id,
        "instance_crn": crn,
    }
    path = _RECOVERY_DIR / f"{spec.run_id}-instance-crn.json"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _write_recovery(f"{spec.run_id}-instance-crn", record)
    except json.JSONDecodeError as exc:
        raise AmbiguousProvisioning("instance CRN record is unreadable") from exc
    else:
        if existing != record:
            raise AmbiguousProvisioning("instance CRN record differs from provider state")


def _load_instance_crn(spec: RunSpec, instance_id: str) -> str:
    _require_accepted_instance_id(spec, instance_id)
    try:
        record = json.loads(
            (_RECOVERY_DIR / f"{spec.run_id}-instance-crn.json").read_text(
                encoding="utf-8"
            )
        )
    except FileNotFoundError:
        return _expected_resource_crn(spec, "instance", instance_id)
    except json.JSONDecodeError as exc:
        raise PrototypeMismatch("instance CRN is neither current nor durable") from exc
    crn = record.get("instance_crn") if isinstance(record, dict) else None
    if not isinstance(crn, str):
        raise PrototypeMismatch("instance CRN is neither current nor durable")
    _resource_crn({"crn": crn}, spec, "instance", instance_id)
    return crn


def _hydrate_created_instance(
    vpc: VpcV1,
    spec: RunSpec,
    prototype: dict[str, object],
    instance: dict[str, Any],
    *,
    deadline: float,
) -> dict[str, Any]:
    instance_id = _field(instance, "id")
    if not instance_id:
        raise AmbiguousProvisioning("created instance ID is missing")
    _persist_accepted_instance(spec, prototype, instance_id)
    has_initial_crn = "crn" in instance
    crn = (
        _resource_crn(instance, spec, "instance", instance_id)
        if has_initial_crn
        else None
    )
    if crn is not None:
        _persist_instance_crn(spec, instance_id, crn)
    if crn is not None and _field(instance, "boot_volume_attachment", "volume", "id"):
        return instance
    for attempt in range(_RECONCILE_ATTEMPTS):
        try:
            observed = _result(vpc.get_instance(instance_id))
        except Exception as exc:
            code = _status_code(exc)
            retryable = not isinstance(exc, AmbiguousProvisioning) and (
                code is None or code in {404, 408, 409, 429} or code >= 500
            )
            if not retryable:
                raise
        else:
            if _field(observed, "id") != instance_id:
                _persist_ambiguity(
                    "create", instance_id, "provider returned a different instance"
                )
                raise AmbiguousProvisioning("provider returned a different instance")
            has_observed_crn = "crn" in observed
            if has_observed_crn:
                crn = _resource_crn(observed, spec, "instance", instance_id)
                _persist_instance_crn(spec, instance_id, crn)
            if has_observed_crn and _field(
                observed, "boot_volume_attachment", "volume", "id"
            ):
                return observed
        if attempt + 1 < _RECONCILE_ATTEMPTS:
            _pause(attempt, deadline)
    _persist_ambiguity("create", instance_id, "accepted create has no cleanup context")
    raise AmbiguousProvisioning("accepted create cleanup context is unavailable")


def _intent_path(spec: RunSpec) -> Path:
    return _RECOVERY_DIR / f"{spec.run_id}-prototype.json"


def _clear_intent(spec: RunSpec) -> None:
    try:
        _intent_path(spec).unlink()
    except FileNotFoundError as exc:
        raise AmbiguousProvisioning("create intent disappeared unexpectedly") from exc
    _fsync_directory(_RECOVERY_DIR)


def _has_matching_intent(spec: RunSpec, prototype: dict[str, object]) -> bool:
    try:
        record = json.loads(_intent_path(spec).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except json.JSONDecodeError as exc:
        raise AmbiguousProvisioning("prior create intent is unreadable") from exc
    expected = _intent_record(spec, prototype)
    if record != expected:
        raise AmbiguousProvisioning("prior create intent differs from the requested prototype")
    return True


@contextmanager
def run_lifecycle_lease(
    spec: RunSpec,
    *,
    deadline: float | None = None,
) -> Iterator[None]:
    """Serialize all local-controller mutations for one immutable run."""

    _ensure_recovery_dir()
    lock_path = _RECOVERY_DIR / f"{spec.run_id}.lock"
    lock_existed = lock_path.exists()
    with lock_path.open("a+", encoding="utf-8") as lock:
        if not lock_existed:
            lock.flush()
            os.fsync(lock.fileno())
            _fsync_directory(_RECOVERY_DIR)
        if deadline is None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        else:
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("lifecycle lease deadline expired")
                try:
                    fcntl.flock(
                        lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    break
                except BlockingIOError:
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _persist_ambiguity(operation: str, resource_id: str, reason: str) -> None:
    key = hashlib.sha256(resource_id.encode()).hexdigest()[:16]
    _write_recovery(
        f"ambiguous-{operation}-{key}",
        {"operation": operation, "resource_id": resource_id, "reason": reason},
    )


def _persist_deletion(spec: RunSpec, resource_type: str, resource_id: str) -> None:
    suffix = ""
    if resource_type == "boot-volume":
        suffix = "-" + hashlib.sha256(resource_id.encode()).hexdigest()[:16]
    _write_recovery(
        f"{spec.run_id}-deleted-{resource_type}{suffix}",
        {
            "run_id": str(spec.run_id),
            "resource_type": resource_type,
            "resource_id": resource_id,
            "provider_state": "404",
        },
    )


def _load_boot_volume_candidates(spec: RunSpec) -> set[str]:
    path = _RECOVERY_DIR / f"{spec.run_id}-boot-volume-candidates.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except json.JSONDecodeError as exc:
        raise AmbiguousProvisioning("boot volume candidate record is unreadable") from exc
    volume_ids = record.get("volume_ids") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or record.get("run_id") != str(spec.run_id)
        or record.get("account_id") != spec.account_id
        or not isinstance(volume_ids, list)
        or any(not isinstance(item, str) or not item for item in volume_ids)
        or len(set(volume_ids)) != len(volume_ids)
    ):
        raise AmbiguousProvisioning("boot volume candidate record is invalid")
    return set(volume_ids)


def _quarantine_boot_volume_candidates(spec: RunSpec) -> None:
    path = _RECOVERY_DIR / f"{spec.run_id}-boot-volume-candidates.json"
    quarantined = path.with_name(
        f"{path.stem}.corrupt-{time.time_ns()}{path.suffix}"
    )
    os.replace(path, quarantined)
    _fsync_directory(_RECOVERY_DIR)


def _persist_boot_volume_candidates(spec: RunSpec, volume_ids: set[str]) -> None:
    _write_recovery(
        f"{spec.run_id}-boot-volume-candidates",
        {
            "run_id": str(spec.run_id),
            "account_id": spec.account_id,
            "volume_ids": sorted(volume_ids),
        },
    )


def _cleanup_path(spec: RunSpec) -> Path:
    return _RECOVERY_DIR / f"{spec.run_id}-cleanup.json"


def _persist_cleanup(
    spec: RunSpec,
    instance_id: str,
    volume_id: str,
    gateway_id: str,
    floating_ip_id: str,
) -> None:
    if (_RECOVERY_DIR / f"{spec.run_id}-accepted.json").exists():
        _require_accepted_instance_id(spec, instance_id)
    record: dict[str, object] = {
        "run_id": str(spec.run_id),
        "account_id": spec.account_id,
        "instance_id": instance_id,
        "volume_id": volume_id,
        "subnet_id": spec.subnet_id,
        "public_gateway_id": gateway_id,
        "public_gateway_floating_ip_id": floating_ip_id,
    }
    path = _cleanup_path(spec)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _write_recovery(f"{spec.run_id}-cleanup", record)
    except json.JSONDecodeError as exc:
        raise AmbiguousProvisioning("cleanup context is unreadable") from exc
    else:
        if existing != record:
            raise AmbiguousProvisioning("cleanup context differs from provider state")


def _load_cleanup(
    spec: RunSpec,
    instance_id: str,
    *,
    errors: list[Exception] | None = None,
) -> tuple[str | None, str, str]:
    def quarantine(path: Path) -> None:
        quarantined = path.with_name(
            f"{path.stem}.corrupt-{time.time_ns()}{path.suffix}"
        )
        os.replace(path, quarantined)
        _fsync_directory(_RECOVERY_DIR)

    accepted_path = _RECOVERY_DIR / f"{spec.run_id}-accepted.json"
    if accepted_path.exists():
        try:
            _require_accepted_instance_id(spec, instance_id)
        except AmbiguousProvisioning as exc:
            if errors is None:
                raise
            errors.append(exc)
            try:
                quarantine(accepted_path)
            except Exception as quarantine_error:
                errors.append(quarantine_error)
    expected = {
        "run_id": str(spec.run_id),
        "account_id": spec.account_id,
        "instance_id": instance_id,
        "subnet_id": spec.subnet_id,
        "public_gateway_id": spec.public_gateway_id,
        "public_gateway_floating_ip_id": spec.public_gateway_floating_ip_id,
    }
    last_error: Exception | None = None
    for path in (
        _cleanup_path(spec),
        _RECOVERY_DIR / f"{spec.run_id}-created.json",
        _RECOVERY_DIR / f"{spec.run_id}-accepted.json",
    ):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as exc:
            invalid = AmbiguousProvisioning("cleanup context is unreadable")
            invalid.__cause__ = exc
        else:
            if not isinstance(record, dict) or any(
                record.get(key) != value for key, value in expected.items()
            ):
                invalid = AmbiguousProvisioning(
                    "cleanup context does not match the RunSpec"
                )
            else:
                volume_id = record.get("volume_id")
                if volume_id is None or (
                    isinstance(volume_id, str) and volume_id
                ):
                    return (
                        volume_id,
                        spec.public_gateway_id,
                        spec.public_gateway_floating_ip_id,
                    )
                invalid = AmbiguousProvisioning(
                    "cleanup context has no boot volume ID"
                )
        last_error = invalid
        if errors is None:
            raise invalid
        errors.append(invalid)
        try:
            quarantine(path)
        except Exception as quarantine_error:
            errors.append(quarantine_error)
    raise AmbiguousProvisioning(
        "instance is absent but cleanup context is unavailable"
    ) from last_error


def _require_local_intent_ownership(spec: RunSpec, instance: dict[str, Any]) -> None:
    try:
        record = json.loads(_intent_path(spec).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise PrototypeMismatch("instance has neither run tags nor local create intent") from exc
    if not isinstance(record, dict):
        raise PrototypeMismatch("local create intent does not own the instance")
    prototype = record.get("prototype")
    user_data = prototype.get("user_data") if isinstance(prototype, dict) else None
    if not isinstance(prototype, dict) or not isinstance(user_data, str):
        raise PrototypeMismatch("local create intent does not own the instance")
    try:
        expected = build_instance_prototype(spec, user_data)
    except ValueError as exc:
        raise PrototypeMismatch("local create intent does not own the instance") from exc
    if (
        prototype != expected
        or record != _intent_record(spec, expected)
        or record.get("instance_name") != _field(instance, "name")
    ):
        raise PrototypeMismatch("local create intent does not own the instance")
    try:
        created_instance_id = recover_created_instance_id(spec, expected)
    except AmbiguousProvisioning as exc:
        raise PrototypeMismatch("local create intent does not own the instance") from exc
    if created_instance_id != _field(instance, "id"):
        raise PrototypeMismatch("local create intent does not own the instance")


def ensure_single_instance(
    vpc: VpcV1,
    spec: RunSpec,
    prototype: dict[str, object],
    *,
    tagger: Any,
    deadline: float | None = None,
) -> InstanceRef:
    """Adopt one exact instance or create once, never retrying creation."""

    expected = build_instance_prototype(spec, str(prototype.get("user_data", "")))
    if prototype != expected:
        raise ValueError("prototype differs from the authorized RunSpec")
    if deadline is None:
        deadline = time.monotonic() + _DEFAULT_CREATE_SECONDS
    _validate_deadline(deadline)
    with run_lifecycle_lease(spec, deadline=deadline):
        vpc = _deadline_client(vpc, deadline)
        tagger = _deadline_client(tagger, deadline)
        _validate_launch_dependencies(vpc, tagger, spec)
        matching_intent = _has_matching_intent(spec, prototype)
        existing = _one_match(vpc, spec)
        if existing is not None:
            if not matching_intent:
                raise AmbiguousProvisioning(
                    "instance exists without matching local create intent"
                )
            existing = _hydrate_created_instance(
                vpc, spec, prototype, existing, deadline=deadline
            )
            _persist_created_instance(spec, prototype, existing)
            return _finalize_adoption(vpc, tagger, spec, existing)
        if matching_intent:
            raise AmbiguousProvisioning(
                "prior create intent has no observable instance; do not create again"
            )
        _persist_intent(spec, prototype)
        try:
            created = _result(vpc.create_instance(prototype))
        except BaseException as exc:
            if isinstance(exc, Exception):
                code = _status_code(exc)
                if code is not None and code < 500 and code not in {408, 409, 429}:
                    _clear_intent(spec)
                    raise
                created = _reconcile_created_instance(
                    vpc, spec, exc, deadline=deadline
                )
            else:
                try:
                    _persist_interrupted_create(
                        vpc, spec, prototype, deadline=deadline
                    )
                except BaseException as recovery_error:
                    raise BaseExceptionGroup(
                        "create interruption and reconciliation both failed",
                        [exc, recovery_error],
                    )
                raise
        try:
            if not _field(created, "id"):
                created = _reconcile_created_instance(
                    vpc,
                    spec,
                    AmbiguousProvisioning("create response has no instance ID"),
                    deadline=deadline,
                )
            created = _hydrate_created_instance(
                vpc, spec, prototype, created, deadline=deadline
            )
            _persist_created_instance(spec, prototype, created)
        except BaseException as exc:
            if isinstance(exc, Exception):
                raise
            try:
                _persist_interrupted_create(
                    vpc, spec, prototype, deadline=deadline
                )
            except BaseException as recovery_error:
                raise BaseExceptionGroup(
                    "post-create interruption and reconciliation both failed",
                    [exc, recovery_error],
                )
            raise
        mismatch = _mismatches(created, spec)
        if mismatch:
            raise PrototypeMismatch(
                "created instance prototype mismatch: " + ", ".join(mismatch)
            )
        matches = _list_named(vpc, spec)
        observed_ids = {_field(item, "id") for item in matches} | {_field(created, "id")}
        if len(observed_ids) > 1:
            raise AmbiguousProvisioning("multiple instances appeared after create")
        return _finalize_adoption(vpc, tagger, spec, created)


def _persist_interrupted_create(
    vpc: VpcV1,
    spec: RunSpec,
    prototype: dict[str, object],
    *,
    deadline: float,
) -> None:
    interrupted = _reconcile_created_instance(
        vpc,
        spec,
        RuntimeError("create call was interrupted"),
        deadline=deadline,
    )
    interrupted = _hydrate_created_instance(
        vpc, spec, prototype, interrupted, deadline=deadline
    )
    _persist_created_instance(spec, prototype, interrupted)


def recover_accepted_instance(
    vpc: VpcV1,
    spec: RunSpec,
    prototype: dict[str, object],
    *,
    deadline: float,
) -> str:
    """Use cleanup-reserved time to discover one accepted create safely."""

    expected = build_instance_prototype(spec, str(prototype.get("user_data", "")))
    if prototype != expected:
        raise ValueError("prototype differs from the authorized RunSpec")
    _validate_deadline(deadline)
    with run_lifecycle_lease(spec, deadline=deadline):
        if not _has_matching_intent(spec, prototype):
            raise AmbiguousProvisioning("accepted create has no matching local intent")
        bounded = _deadline_client(vpc, deadline)
        created = _reconcile_created_instance(
            bounded,
            spec,
            RuntimeError("accepted create was not visible before create deadline"),
            deadline=deadline,
        )
        created = _hydrate_created_instance(
            bounded, spec, prototype, created, deadline=deadline
        )
        _persist_created_instance(spec, prototype, created)
        mismatch = _mismatches(created, spec)
        if mismatch:
            raise PrototypeMismatch(
                "recovered instance prototype mismatch: " + ", ".join(mismatch)
            )
        matches = _list_named(bounded, spec)
        observed_ids = {_field(item, "id") for item in matches} | {_field(created, "id")}
        if len(observed_ids) != 1:
            raise AmbiguousProvisioning("accepted create recovery is not unique")
        return _field(created, "id")


def _reconcile_created_instance(
    vpc: VpcV1,
    spec: RunSpec,
    cause: Exception,
    *,
    deadline: float,
) -> dict[str, Any]:
    for attempt in range(_RECONCILE_ATTEMPTS):
        try:
            adopted = _one_match(vpc, spec)
        except Exception as reconcile_exc:
            code = _status_code(reconcile_exc)
            retryable = not isinstance(reconcile_exc, AmbiguousProvisioning) and (
                code is None or code in {408, 409, 429} or code >= 500
            )
            if retryable and attempt + 1 < _RECONCILE_ATTEMPTS:
                _pause(attempt, deadline)
                continue
            _persist_ambiguity(
                "create", str(spec.run_id), "provider reconciliation failed"
            )
            raise AmbiguousProvisioning(
                "create outcome cannot be reconciled"
            ) from reconcile_exc
        if adopted is not None:
            return adopted
        if attempt + 1 < _RECONCILE_ATTEMPTS:
            _pause(attempt, deadline)
    _persist_ambiguity("create", str(spec.run_id), "provider outcome unknown")
    raise AmbiguousProvisioning(
        "create outcome is unknown; do not retry automatically"
    ) from cause


def _finalize_adoption(
    vpc: VpcV1,
    tagger: Any,
    spec: RunSpec,
    instance: dict[str, Any],
) -> InstanceRef:
    instance_id = _field(instance, "id")
    crn = _resource_crn(instance, spec, "instance", instance_id)
    _validate_trusted_profile(vpc, spec, instance_id, context="instance")
    _validate_instance_network(vpc, spec, instance)
    _validate_security_group_rules(vpc, spec)
    volume_id = _validate_instance_storage(vpc, spec, instance)
    gateway_id = _validate_subnet_ownership(vpc, tagger, spec)
    if gateway_id != spec.public_gateway_id:
        raise PrototypeMismatch("run subnet is not bound to its dedicated Public Gateway")
    floating_ip_id = _validate_gateway_and_floating_ip(vpc, tagger, spec)
    _validate_gateway_attachments(vpc, spec, {spec.subnet_id})
    _persist_cleanup(spec, instance_id, volume_id, gateway_id, floating_ip_id)
    try:
        _require_run_tags(tagger, crn, spec, "instance")
        return _instance_ref(instance)
    except MissingOwnershipTags:
        pass
    required_tags = ["project:watch-authenticity", f"run:{spec.run_id}"]
    attached = _result(tagger.attach_tag(
        tag_names=required_tags,
        resources=[{"resource_id": crn}],
        tag_type="user",
    ))
    results = attached.get("results")
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
        or results[0].get("resource_id") != crn
        or results[0].get("is_error") is True
    ):
        raise AmbiguousProvisioning("instance tag attachment was not confirmed")
    _require_run_tags(tagger, crn, spec, "instance")
    return _instance_ref(instance)


def _validate_trusted_profile(
    vpc: VpcV1,
    spec: RunSpec,
    instance_id: str,
    *,
    context: str,
) -> None:
    initialization = _result(vpc.get_instance_initialization(instance_id))
    trusted = initialization.get("default_trusted_profile")
    if (
        not isinstance(trusted, dict)
        or trusted.get("auto_link") is not True
        or _field(trusted, "target", "id") != spec.trusted_profile_id
    ):
        raise PrototypeMismatch(f"{context} trusted profile does not match RunSpec")


def _validate_launch_dependencies(vpc: VpcV1, tagger: Any, spec: RunSpec) -> None:
    image = _result(vpc.get_image(spec.image_id))
    operating_system = image.get("operating_system")
    if (
        _field(image, "id") != spec.image_id
        or image.get("status") != "available"
        or image.get("visibility") != "public"
        or image.get("user_data_format") != "cloud_init"
        or not isinstance(operating_system, dict)
        or operating_system.get("architecture") != "amd64"
        or "ubuntu" not in str(operating_system.get("family", "")).lower()
        or operating_system.get("name") != "ubuntu-22-04-amd64"
        or operating_system.get("version") != "22.04"
        or operating_system.get("user_data_format") != "cloud_init"
    ):
        raise PrototypeMismatch("image is not the authorized public Ubuntu 22.04 amd64 image")
    security_group = _result(vpc.get_security_group(spec.security_group_id))
    if (
        _field(security_group, "id") != spec.security_group_id
        or _field(security_group, "resource_group", "id") != spec.resource_group_id
        or _field(security_group, "vpc", "id") != spec.network_id
    ):
        raise PrototypeMismatch("security group differs from the authorized RunSpec")
    _validate_security_group_rules(vpc, spec)
    gateway_id = _validate_subnet_ownership(vpc, tagger, spec)
    if gateway_id != spec.public_gateway_id:
        raise PrototypeMismatch("run subnet is not bound to its dedicated Public Gateway")
    _validate_gateway_and_floating_ip(vpc, tagger, spec)
    _validate_gateway_attachments(vpc, spec, {spec.subnet_id})


def _validate_instance_storage(vpc: VpcV1, spec: RunSpec, instance: dict[str, Any]) -> str:
    instance_id = _field(instance, "id")
    attachment_id = _field(instance, "boot_volume_attachment", "id")
    volume_id = _field(instance, "boot_volume_attachment", "volume", "id")
    if not attachment_id or not volume_id:
        raise PrototypeMismatch("instance boot volume attachment is incomplete")
    attachment = _result(vpc.get_instance_volume_attachment(instance_id, attachment_id))
    if (
        attachment.get("delete_volume_on_instance_delete") is not True
        or _field(attachment, "volume", "id") != volume_id
    ):
        raise PrototypeMismatch("instance boot volume is not configured for safe deletion")
    _validate_volume(vpc, spec, volume_id)
    return volume_id


def _validate_volume(vpc: VpcV1, spec: RunSpec, volume_id: str) -> None:
    volume = _result(vpc.get_volume(volume_id))
    expected_tags = {"project:watch-authenticity", f"run:{spec.run_id}"}
    user_tags = volume.get("user_tags")
    actual_tags = set(user_tags) if isinstance(user_tags, list) else set()
    project_tags = {tag for tag in actual_tags if tag.startswith("project:")}
    run_tags = {tag for tag in actual_tags if tag.startswith("run:")}
    if (
        _field(volume, "id") != volume_id
        or volume.get("capacity") != int(spec.boot_volume_gb)
        or _field(volume, "profile", "name") != "general-purpose"
        or _field(volume, "resource_group", "id") != spec.resource_group_id
        or _field(volume, "zone", "name") != spec.zone
        or volume.get("encryption") != "provider_managed"
        or not expected_tags.issubset(actual_tags)
        or project_tags != {"project:watch-authenticity"}
        or run_tags != {f"run:{spec.run_id}"}
    ):
        raise PrototypeMismatch("instance boot volume differs from the authorized prototype")


def _revalidate_volume_delete(vpc: VpcV1, spec: RunSpec, volume_id: str) -> None:
    try:
        _validate_volume(vpc, spec, volume_id)
    except Exception as exc:
        if _status_code(exc) == 404:
            raise ResourceAbsent("boot volume is absent") from exc
        raise


def _discover_boot_volumes(
    vpc: VpcV1,
    spec: RunSpec,
    *,
    on_candidate: Callable[[str], None] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    volume_ids: list[str] = []
    listed_ids: set[str] = set()
    observed_count = 0
    start: str | None = None
    seen: set[str] = set()
    expected_total_count: int | None = None
    while True:
        kwargs: dict[str, object] = {
            "zone_name": spec.zone,
            "limit": 100,
        }
        if start is not None:
            kwargs["start"] = start
        page = _result(vpc.list_volumes(**kwargs))
        volumes = page.get("volumes")
        if not isinstance(volumes, list) or any(
            not isinstance(volume, dict) for volume in volumes
        ):
            raise AmbiguousProvisioning("IBM VPC volume list is malformed")
        observed_count += len(volumes)
        for volume in volumes:
            volume_id = _field(volume, "id")
            if not volume_id or volume_id in listed_ids:
                raise AmbiguousProvisioning("IBM VPC volume list has an invalid ID")
            listed_ids.add(volume_id)
            user_tags = volume.get("user_tags")
            if not isinstance(user_tags, list) or any(
                not isinstance(tag, str) for tag in user_tags
            ):
                raise AmbiguousProvisioning("IBM VPC volume list has malformed tags")
            if f"run:{spec.run_id}" in user_tags:
                volume_ids.append(volume_id)
                if on_candidate is not None:
                    on_candidate(volume_id)
        start, expected_total_count = _next_start(
            page,
            observed_count,
            context="IBM VPC volume list",
            expected_total_count=expected_total_count,
        )
        if start is None:
            run_inventory = tuple(sorted(volume_ids))
            return run_inventory, run_inventory
        if start in seen:
            raise AmbiguousProvisioning("IBM VPC volume pagination repeated a cursor")
        seen.add(start)


def _validate_instance_network(
    vpc: VpcV1,
    spec: RunSpec,
    instance: dict[str, Any],
) -> None:
    primary = instance.get("primary_network_interface")
    interfaces = instance.get("network_interfaces")
    attachments = instance.get("network_attachments")
    if (
        not isinstance(primary, dict)
        or not isinstance(interfaces, list)
        or len(interfaces) != 1
        or not isinstance(interfaces[0], dict)
        or not isinstance(attachments, list)
        or attachments
    ):
        raise PrototypeMismatch(
            "instance must have exactly one legacy network interface and no attachments"
        )
    interface_id = _field(primary, "id")
    if not interface_id or _field(interfaces[0], "id") != interface_id:
        raise PrototypeMismatch("instance primary network interface is inconsistent")
    interface = _result(
        vpc.get_instance_network_interface(_field(instance, "id"), interface_id)
    )
    security_groups = interface.get("security_groups")
    security_group_ids = {
        item.get("id")
        for item in security_groups
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(security_groups, list) else set()
    floating_ips = interface.get("floating_ips")
    if (
        _field(interface, "id") != interface_id
        or _field(interface, "subnet", "id") != spec.subnet_id
    ):
        raise PrototypeMismatch("instance network interface identity is inconsistent")
    if security_group_ids != {spec.security_group_id}:
        raise PrototypeMismatch("instance security_group is not authorized")
    if interface.get("allow_ip_spoofing") is not False:
        raise PrototypeMismatch("instance allow_ip_spoofing must be false")
    if not isinstance(floating_ips, list):
        raise AmbiguousProvisioning("instance floating IP response is malformed")
    if floating_ips:
        raise PrototypeMismatch("instance must not have a floating IP")
    start: str | None = None
    seen: set[str] = set()
    observed_count = 0
    expected_total_count: int | None = None
    while True:
        kwargs: dict[str, object] = {"target_id": interface_id}
        if start is not None:
            kwargs["start"] = start
        page = _result(vpc.list_floating_ips(**kwargs))
        floating_ips = page.get("floating_ips")
        if not isinstance(floating_ips, list):
            raise AmbiguousProvisioning("IBM VPC floating IP list is malformed")
        observed_count += len(floating_ips)
        if floating_ips:
            raise PrototypeMismatch("instance must not have a floating IP")
        start, expected_total_count = _next_start(
            page,
            observed_count,
            context="IBM VPC floating IP list",
            expected_total_count=expected_total_count,
        )
        if start is None:
            return
        if start in seen:
            raise AmbiguousProvisioning("IBM VPC floating IP pagination repeated a cursor")
        seen.add(start)


def _validate_security_group_rules(vpc: VpcV1, spec: RunSpec) -> None:
    outbound: set[tuple[str, int, int, str]] = set()
    result = _result(vpc.list_security_group_rules(spec.security_group_id))
    rules = result.get("rules")
    if not isinstance(rules, list) or any(not isinstance(rule, dict) for rule in rules):
        raise AmbiguousProvisioning("security group rules response is malformed")
    for rule in rules:
        direction = rule.get("direction")
        protocol = rule.get("protocol")
        if direction == "inbound":
            raise PrototypeMismatch("security group must deny all inbound traffic")
        if direction != "outbound" or protocol not in {"tcp", "udp"}:
            raise PrototypeMismatch("security group has an unauthorized outbound rule")
        port_min = rule.get("port_min")
        port_max = rule.get("port_max")
        remote = rule.get("remote")
        if (
            not isinstance(port_min, int)
            or not isinstance(port_max, int)
            or not isinstance(remote, dict)
            or set(remote) != {"cidr_block"}
            or not isinstance(remote.get("cidr_block"), str)
        ):
            raise PrototypeMismatch("security group outbound rule is not exact")
        outbound.add((protocol, port_min, port_max, remote["cidr_block"]))
    # The security group is the outer L4 layer. Task 5's hash-bound host bootstrap
    # denies direct container egress and enforces the FQDN allowlist through a proxy.
    required = {
        ("tcp", 443, 443, "0.0.0.0/0"),
        ("tcp", 53, 53, "0.0.0.0/0"),
        ("udp", 53, 53, "0.0.0.0/0"),
    }
    if outbound != required:
        raise PrototypeMismatch("security group outbound rules must be only DNS and HTTPS")


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        value = response.get("status_code") or response.get("StatusCode")
        if isinstance(value, int):
            return value
        error = response.get("Error")
        if isinstance(error, dict):
            raw = error.get("Code")
            if isinstance(raw, str) and raw.isdigit():
                return int(raw)
    return None


def _is_retryable_provider_error(exc: Exception) -> bool:
    if isinstance(exc, AmbiguousProvisioning):
        return False
    code = _status_code(exc)
    return code is None or code in {404, 408, 409, 429} or code >= 500


def _pause(attempt: int, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    delay = min(0.5 * (2 ** min(attempt, 4)), 5.0) + random.uniform(0, 0.1)
    time.sleep(min(delay, remaining))


def _validate_deadline(deadline: float) -> None:
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise TypeError("deadline must be a monotonic float")
    now = time.monotonic()
    if not math.isfinite(deadline) or deadline <= now:
        raise ValueError("deadline must be finite and in the future")
    if deadline - now > _MAX_POLL_SECONDS:
        raise ValueError("deadline must be at most 30 minutes away")


def wait_running(vpc: VpcV1, instance_id: str, *, deadline: float) -> InstanceRef:
    """Wait for running using a monotonic absolute deadline."""

    _validate_deadline(deadline)
    vpc = _deadline_client(vpc, deadline)
    attempt = 0
    read_failures = 0
    while time.monotonic() < deadline:
        try:
            record = _result(vpc.get_instance(instance_id))
        except Exception as exc:
            if not _is_retryable_provider_error(exc):
                _persist_ambiguity("wait-running", instance_id, "provider read failed")
                raise AmbiguousProvisioning("instance state is unknown") from exc
            read_failures += 1
            if read_failures >= _RECONCILE_ATTEMPTS:
                _persist_ambiguity("wait-running", instance_id, "rate limit or retry exhaustion")
                raise AmbiguousProvisioning(
                    "instance state is unknown after rate limit or retry exhaustion"
                ) from exc
            try:
                _pause(attempt, deadline)
            except TimeoutError as timeout:
                _persist_ambiguity("wait-running", instance_id, "deadline exceeded")
                raise TimeoutError("instance did not reach running before deadline") from timeout
            attempt += 1
            continue
        read_failures = 0
        ref = _instance_ref(record)
        if ref.id != instance_id:
            _persist_ambiguity("wait-running", instance_id, "provider returned a different instance")
            raise AmbiguousProvisioning("provider returned a different instance")
        if ref.status == "running":
            return ref
        if ref.status not in _TRANSITIONAL_STATES:
            if ref.status not in {"failed", "stopped", "deleting"}:
                _persist_ambiguity("wait-running", instance_id, f"unknown state: {ref.status}")
            raise ProvisioningFailed(f"instance entered terminal or unknown state: {ref.status}")
        try:
            _pause(attempt, deadline)
        except TimeoutError as exc:
            _persist_ambiguity("wait-running", instance_id, "deadline exceeded")
            raise TimeoutError("instance did not reach running before deadline") from exc
        attempt += 1
    _persist_ambiguity("wait-running", instance_id, "deadline exceeded")
    raise TimeoutError("instance did not reach running before deadline")


def _require_time_remaining(
    deadline: float,
    resource_id: str,
    description: str,
) -> None:
    if time.monotonic() >= deadline:
        _persist_ambiguity("delete", resource_id, f"{description}: deadline")
        raise TimeoutError(f"{description} deadline expired before mutation")


def _confirm_predelete_absence(
    reader: Callable[[str], object],
    resource_id: str,
    *,
    deadline: float,
    description: str,
) -> bool:
    consecutive_absent = 1
    attempt = 0
    while time.monotonic() < deadline:
        try:
            _pause(attempt, deadline)
        except TimeoutError as exc:
            raise TimeoutError(
                f"{description} absence was not confirmed"
            ) from exc
        try:
            reader(resource_id)
        except Exception as exc:
            if _status_code(exc) == 404:
                consecutive_absent += 1
                if consecutive_absent >= 2:
                    return True
            elif _is_retryable_provider_error(exc):
                consecutive_absent = 0
            else:
                raise
        else:
            return False
        attempt += 1
    raise TimeoutError(f"{description} absence was not confirmed")


def _delete_and_confirm(
    deleter: Callable[[str], object],
    reader: Callable[[str], object],
    resource_id: str,
    *,
    deadline: float,
    description: str,
    before_delete: Callable[[], None] | None = None,
    revalidate_first_attempt: bool = False,
) -> None:
    attempt = 0
    while time.monotonic() < deadline:
        if before_delete is not None and (attempt > 0 or revalidate_first_attempt):
            try:
                before_delete()
            except ResourceAbsent:
                if attempt > 0:
                    return
                if _confirm_predelete_absence(
                    reader,
                    resource_id,
                    deadline=deadline,
                    description=description,
                ):
                    return
                attempt += 1
                continue
            except Exception as exc:
                if not _is_retryable_provider_error(exc):
                    raise
                try:
                    _pause(attempt, deadline)
                except TimeoutError as timeout:
                    _persist_ambiguity(
                        "delete", resource_id, f"{description}: validation deadline"
                    )
                    raise TimeoutError(
                        f"{description} ownership was not revalidated"
                    ) from timeout
                attempt += 1
                continue
        _require_time_remaining(deadline, resource_id, description)
        try:
            deleter(resource_id)
        except Exception as exc:
            if _status_code(exc) != 404 and not _is_retryable_provider_error(exc):
                raise
        try:
            reader(resource_id)
        except Exception as exc:
            if _status_code(exc) == 404:
                return
            if not _is_retryable_provider_error(exc):
                _persist_ambiguity(
                    "delete", resource_id, f"{description}: provider read failed"
                )
                raise AmbiguousProvisioning(
                    f"{description} provider read failed"
                ) from exc
        try:
            _pause(attempt, deadline)
        except TimeoutError as exc:
            _persist_ambiguity("delete", resource_id, f"{description}: deadline")
            raise TimeoutError(f"{description} was not confirmed by 404") from exc
        attempt += 1
    _persist_ambiguity("delete", resource_id, f"{description}: deadline")
    raise TimeoutError(f"{description} was not confirmed by 404")


def _require_run_tags(
    tagger: Any,
    crn: str,
    spec: RunSpec,
    resource: str,
    *,
    stable: bool = False,
) -> None:
    def read_names() -> set[str]:
        names: set[str] = set()
        offset = 0
        expected_total_count: int | None = None
        while True:
            tags = _result(
                tagger.list_tags(
                    attached_to=crn,
                    tag_type="user",
                    providers=["ghost"],
                    offset=offset,
                    limit=100,
                )
            )
            items = tags.get("items")
            total_count = tags.get("total_count")
            page_offset = tags.get("offset")
            if (
                not isinstance(items, list)
                or any(not isinstance(item, dict) for item in items)
                or not isinstance(total_count, int)
                or isinstance(total_count, bool)
                or total_count < 0
                or (
                    expected_total_count is not None
                    and total_count != expected_total_count
                )
                or page_offset != offset
                or len(items) > 100
                or offset + len(items) > total_count
            ):
                raise AmbiguousProvisioning("tag ownership response is malformed")
            expected_total_count = total_count
            for item in items:
                name = item.get("name")
                if not isinstance(name, str) or not name or name in names:
                    raise AmbiguousProvisioning(
                        "tag ownership pagination has a missing or repeated name"
                    )
                names.add(name)
            offset += len(items)
            if offset == total_count:
                return names
            if not items:
                raise AmbiguousProvisioning(
                    "tag ownership pagination made no progress"
                )

    names = read_names()
    if stable and read_names() != names:
        raise AmbiguousProvisioning("tag ownership changed during validation")
    required = {"project:watch-authenticity", f"run:{spec.run_id}"}
    project_tags = {name for name in names if name.startswith("project:")}
    run_tags = {name for name in names if name.startswith("run:")}
    if (
        project_tags and project_tags != {"project:watch-authenticity"}
    ) or (run_tags and run_tags != {f"run:{spec.run_id}"}):
        raise PrototypeMismatch(f"{resource} has conflicting ownership tags")
    if not required.issubset(names):
        raise MissingOwnershipTags(f"{resource} is missing run ownership tags")


def _validate_subnet_ownership(
    vpc: VpcV1,
    tagger: Any,
    spec: RunSpec,
    *,
    stable_tags: bool = False,
) -> str | None:
    try:
        subnet = _result(vpc.get_subnet(spec.subnet_id))
    except Exception as exc:
        if _status_code(exc) == 404:
            raise ResourceAbsent("subnet is absent") from exc
        raise
    expected = {
        "id": spec.subnet_id,
        "resource_group": spec.resource_group_id,
        "vpc": spec.network_id,
        "zone": spec.zone,
    }
    actual = {
        "id": _field(subnet, "id"),
        "resource_group": _field(subnet, "resource_group", "id"),
        "vpc": _field(subnet, "vpc", "id"),
        "zone": _field(subnet, "zone", "name"),
    }
    if any(actual[field] != value for field, value in expected.items()):
        raise PrototypeMismatch("refusing cleanup from a subnet with mismatched ownership")
    crn = _resource_crn(subnet, spec, "subnet", spec.subnet_id)
    _require_run_tags(tagger, crn, spec, "subnet", stable=stable_tags)
    return _field(subnet, "public_gateway", "id") or None


def _validate_gateway_ownership(
    vpc: VpcV1,
    tagger: Any,
    spec: RunSpec,
    *,
    floating_ip_may_be_absent: bool = False,
    stable_tags: bool = False,
) -> str:
    try:
        gateway = _result(vpc.get_public_gateway(spec.public_gateway_id))
    except Exception as exc:
        if _status_code(exc) == 404:
            raise ResourceAbsent("Public Gateway is absent") from exc
        raise
    expected = {
        "id": spec.public_gateway_id,
        "resource_group": spec.resource_group_id,
        "vpc": spec.network_id,
        "zone": spec.zone,
    }
    actual = {
        "id": _field(gateway, "id"),
        "resource_group": _field(gateway, "resource_group", "id"),
        "vpc": _field(gateway, "vpc", "id"),
        "zone": _field(gateway, "zone", "name"),
    }
    mismatch = [field for field, value in expected.items() if actual[field] != value]
    if mismatch:
        raise PrototypeMismatch(
            "refusing cleanup of a Public Gateway with mismatched ownership fields"
        )
    crn = _resource_crn(
        gateway,
        spec,
        "public-gateway",
        spec.public_gateway_id,
    )
    _require_run_tags(
        tagger, crn, spec, "Public Gateway", stable=stable_tags
    )
    floating_ip_id = _field(gateway, "floating_ip", "id")
    if floating_ip_id != spec.public_gateway_floating_ip_id and not (
        floating_ip_may_be_absent and not floating_ip_id
    ):
        raise PrototypeMismatch("Public Gateway Floating IP differs from the RunSpec")
    return floating_ip_id or spec.public_gateway_floating_ip_id


def _validate_gateway_and_floating_ip(
    vpc: VpcV1,
    tagger: Any,
    spec: RunSpec,
) -> str:
    floating_ip_id = _validate_gateway_ownership(vpc, tagger, spec)
    _validate_floating_ip_ownership(vpc, tagger, spec, floating_ip_id)
    return floating_ip_id


def _validate_floating_ip_ownership(
    vpc: VpcV1,
    tagger: Any,
    spec: RunSpec,
    floating_ip_id: str,
    *,
    gateway_may_be_deleted: bool = False,
    stable_tags: bool = False,
) -> None:
    try:
        floating_ip = _result(vpc.get_floating_ip(floating_ip_id))
    except Exception as exc:
        if _status_code(exc) == 404:
            raise ResourceAbsent("Floating IP is absent") from exc
        raise
    expected = {
        "id": spec.public_gateway_floating_ip_id,
        "resource_group": spec.resource_group_id,
        "zone": spec.zone,
    }
    actual = {
        "id": _field(floating_ip, "id"),
        "resource_group": _field(floating_ip, "resource_group", "id"),
        "zone": _field(floating_ip, "zone", "name"),
    }
    if floating_ip_id != spec.public_gateway_floating_ip_id or actual != expected:
        raise PrototypeMismatch(
            "refusing cleanup of a Floating IP with mismatched ownership fields"
        )
    crn = _resource_crn(floating_ip, spec, "floating-ip", floating_ip_id)
    _require_run_tags(tagger, crn, spec, "Floating IP", stable=stable_tags)
    target = floating_ip.get("target")
    if target is None:
        if not gateway_may_be_deleted:
            raise PrototypeMismatch("Floating IP is not attached to the Public Gateway")
        return
    if not isinstance(target, dict):
        raise PrototypeMismatch("Floating IP target is malformed")
    if (
        target.get("resource_type") != "public_gateway"
        or _field(target, "id") != spec.public_gateway_id
    ):
        raise PrototypeMismatch("Floating IP target differs from the Public Gateway")
    _resource_crn(target, spec, "public-gateway", spec.public_gateway_id)
    if not gateway_may_be_deleted and target.get("deleted") is not None:
        raise PrototypeMismatch("Floating IP targets a deleted Public Gateway")


def _validate_gateway_attachments(
    vpc: VpcV1,
    spec: RunSpec,
    expected_subnets: set[str],
    *,
    stable: bool = False,
) -> None:
    def read_snapshot() -> tuple[set[str], set[str]]:
        attached_subnets: set[str] = set()
        listed_subnet_ids: set[str] = set()
        start: str | None = None
        seen: set[str] = set()
        observed_count = 0
        expected_total_count: int | None = None
        while True:
            kwargs: dict[str, object] = {"vpc_id": spec.network_id}
            if start is not None:
                kwargs["start"] = start
            page = _result(vpc.list_subnets(**kwargs))
            subnets = page.get("subnets")
            if not isinstance(subnets, list) or any(
                not isinstance(item, dict) for item in subnets
            ):
                raise AmbiguousProvisioning("IBM VPC subnet list is malformed")
            observed_count += len(subnets)
            for item in subnets:
                subnet_id = _field(item, "id")
                if not subnet_id or subnet_id in listed_subnet_ids:
                    raise AmbiguousProvisioning(
                        "IBM VPC subnet list has a missing or repeated ID"
                    )
                listed_subnet_ids.add(subnet_id)
                if _field(item, "public_gateway", "id") == spec.public_gateway_id:
                    attached_subnets.add(subnet_id)
            start, expected_total_count = _next_start(
                page,
                observed_count,
                context="IBM VPC subnet list",
                expected_total_count=expected_total_count,
            )
            if start is None:
                return attached_subnets, listed_subnet_ids
            if start in seen:
                raise AmbiguousProvisioning(
                    "IBM VPC subnet pagination repeated a cursor"
                )
            seen.add(start)

    attached_subnets, listed_subnet_ids = read_snapshot()
    if stable and read_snapshot() != (attached_subnets, listed_subnet_ids):
        raise AmbiguousProvisioning("IBM VPC subnet inventory changed")
    if attached_subnets != expected_subnets:
        raise PrototypeMismatch("Public Gateway has unauthorized subnet attachments")


def _cleanup_volumes(
    vpc: VpcV1,
    spec: RunSpec,
    volume_id: str | None,
    *,
    deadline: float,
) -> None:
    vpc = _deadline_client(vpc, deadline)
    discovered = volume_id is None
    candidate_ids: set[str] = set()
    candidate_ledger_errors: list[Exception] = []
    candidate_ledger_writable = True

    def persist_candidates() -> None:
        nonlocal candidate_ledger_writable
        if candidate_ledger_writable:
            try:
                _persist_boot_volume_candidates(spec, candidate_ids)
            except Exception as exc:
                candidate_ledger_errors.append(exc)
                candidate_ledger_writable = False

    def record_candidate(candidate_id: str) -> None:
        candidate_ids.add(candidate_id)
        persist_candidates()

    if discovered:
        try:
            candidate_ids = _load_boot_volume_candidates(spec)
        except Exception as exc:
            candidate_ledger_errors.append(exc)
            try:
                _quarantine_boot_volume_candidates(spec)
            except Exception as quarantine_error:
                candidate_ledger_errors.append(quarantine_error)
                candidate_ledger_writable = False
        discovered_ids, _ = _discover_boot_volumes(
            vpc, spec, on_candidate=record_candidate
        )
        candidate_ids.update(discovered_ids)
        persist_candidates()
        volume_ids = tuple(candidate_ids)
    else:
        assert volume_id is not None
        volume_ids = (volume_id,)
    cleanup_errors: dict[str, Exception] = {}
    known_volume_was_absent = False

    def cleanup_one(current_volume_id: str) -> bool:
        nonlocal known_volume_was_absent
        volume_absent = False
        try:
            _validate_volume(vpc, spec, current_volume_id)
        except Exception as exc:
            if _status_code(exc) != 404:
                cleanup_errors[current_volume_id] = exc
                return False
            try:
                volume_absent = _confirm_predelete_absence(
                    vpc.get_volume,
                    current_volume_id,
                    deadline=deadline,
                    description="boot volume",
                )
            except Exception as confirm_error:
                cleanup_errors[current_volume_id] = confirm_error
                return False
        if volume_absent:
            if not discovered and current_volume_id == volume_id:
                known_volume_was_absent = True
        else:
            try:
                _delete_and_confirm(
                    vpc.delete_volume,
                    vpc.get_volume,
                    current_volume_id,
                    deadline=deadline,
                    description="boot volume deletion",
                    before_delete=lambda: _revalidate_volume_delete(
                        vpc, spec, current_volume_id
                    ),
                    revalidate_first_attempt=True,
                )
            except Exception as exc:
                cleanup_errors[current_volume_id] = exc
                return False
        cleanup_errors.pop(current_volume_id, None)
        try:
            _persist_deletion(spec, "boot-volume", current_volume_id)
        except Exception as exc:
            cleanup_errors[current_volume_id] = exc
            return False
        return True

    for current_volume_id in volume_ids:
        assert current_volume_id is not None
        if cleanup_one(current_volume_id):
            candidate_ids.discard(current_volume_id)
    if discovered:
        persist_candidates()
        inventory_error: Exception | None = None
        inventory_attempt = 0
        previous_inventory: tuple[str, ...] | None = None
        while True:
            if time.monotonic() >= deadline:
                inventory_error = TimeoutError(
                    "run-owned boot volume inventory did not become empty"
                )
                break
            try:
                remaining_volume_ids, inventory = _discover_boot_volumes(
                    vpc, spec, on_candidate=record_candidate
                )
            except Exception as exc:
                inventory_error = exc
                break
            candidate_ids.update(remaining_volume_ids)
            persist_candidates()
            for remaining_volume_id in tuple(candidate_ids):
                if cleanup_one(remaining_volume_id):
                    candidate_ids.discard(remaining_volume_id)
            persist_candidates()
            for failed_volume_id in tuple(cleanup_errors):
                try:
                    vpc.get_volume(failed_volume_id)
                except Exception as exc:
                    if _status_code(exc) == 404:
                        try:
                            confirmed_absent = _confirm_predelete_absence(
                                vpc.get_volume,
                                failed_volume_id,
                                deadline=deadline,
                                description="boot volume",
                            )
                            if not confirmed_absent:
                                continue
                            _persist_deletion(spec, "boot-volume", failed_volume_id)
                        except Exception as receipt_error:
                            cleanup_errors[failed_volume_id] = receipt_error
                        else:
                            del cleanup_errors[failed_volume_id]
                            candidate_ids.discard(failed_volume_id)
            persist_candidates()
            if cleanup_errors or candidate_ids:
                if remaining_volume_ids:
                    inventory_error = AmbiguousProvisioning(
                        "run-owned boot volume inventory is not empty"
                    )
                break
            if (
                not remaining_volume_ids
                and previous_inventory == inventory
                and not candidate_ledger_errors
            ):
                _write_recovery(
                    f"{spec.run_id}-deleted-boot-volume-inventory",
                    {
                        "run_id": str(spec.run_id),
                        "resource_type": "boot-volume-inventory",
                        "provider_state": "authoritative-empty-list",
                    },
                )
                break
            if candidate_ledger_errors:
                break
            previous_inventory = None if remaining_volume_ids else inventory
            try:
                _pause(inventory_attempt, deadline)
            except TimeoutError:
                inventory_error = TimeoutError(
                    "run-owned boot volume inventory did not become empty"
                )
                break
            inventory_attempt += 1
        if inventory_error is not None:
            cleanup_errors["inventory"] = inventory_error
        if len(candidate_ledger_errors) == 1:
            cleanup_errors["candidate-ledger"] = candidate_ledger_errors[0]
        elif candidate_ledger_errors:
            cleanup_errors["candidate-ledger"] = CleanupFailures(
                tuple(candidate_ledger_errors)
            )
    elif known_volume_was_absent and not cleanup_errors:
        _cleanup_volumes(vpc, spec, None, deadline=deadline)
    errors = tuple(cleanup_errors.values())
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise CleanupFailures(errors)


def _revalidate_gateway_delete(
    vpc: VpcV1,
    tagger: Any,
    spec: RunSpec,
    *,
    floating_ip_absent: bool,
) -> None:
    _validate_gateway_ownership(
        vpc,
        tagger,
        spec,
        floating_ip_may_be_absent=floating_ip_absent,
        stable_tags=True,
    )
    _validate_gateway_attachments(vpc, spec, set(), stable=True)


def _revalidate_subnet_delete(
    vpc: VpcV1,
    tagger: Any,
    spec: RunSpec,
) -> None:
    attached_gateway = _validate_subnet_ownership(
        vpc, tagger, spec, stable_tags=True
    )
    if attached_gateway not in {None, spec.public_gateway_id}:
        raise PrototypeMismatch("refusing to delete a subnet bound to another gateway")


def _revalidate_instance_delete(
    vpc: VpcV1,
    tagger: Any,
    spec: RunSpec,
    instance_id: str,
    *,
    allow_local_intent: bool,
) -> None:
    try:
        instance = _result(vpc.get_instance(instance_id))
    except Exception as exc:
        if _status_code(exc) == 404:
            raise ResourceAbsent("instance is absent") from exc
        raise
    if _field(instance, "id") != instance_id:
        raise PrototypeMismatch("provider returned a different instance before delete")
    if "crn" in instance:
        crn = _resource_crn(instance, spec, "instance", instance_id)
    elif allow_local_intent:
        crn = _load_instance_crn(spec, instance_id)
    else:
        raise PrototypeMismatch("instance CRN is missing before delete")
    try:
        _require_run_tags(tagger, crn, spec, "instance", stable=True)
    except MissingOwnershipTags:
        if not allow_local_intent:
            raise
        _require_local_intent_ownership(spec, instance)


def _cleanup_gateway(
    vpc: VpcV1,
    tagger: Any,
    spec: RunSpec,
    gateway_id: str,
    *,
    deadline: float,
    floating_ip_absent: bool,
) -> None:
    now = time.monotonic()
    subnet_deadline = now + max(0.0, deadline - now) / 2
    vpc = _deadline_client(vpc, subnet_deadline)
    tagger = _deadline_client(tagger, subnet_deadline)
    receipt_errors: list[Exception] = []
    subnet_absent = False

    def persist_deletion(resource_type: str, resource_id: str) -> None:
        try:
            _persist_deletion(spec, resource_type, resource_id)
        except Exception as exc:
            receipt_errors.append(exc)

    try:
        attached_gateway = _validate_subnet_ownership(
            vpc, tagger, spec, stable_tags=True
        )
    except ResourceAbsent:
        if _confirm_predelete_absence(
            vpc.get_subnet,
            spec.subnet_id,
            deadline=subnet_deadline,
            description="run subnet",
        ):
            attached_gateway = None
            subnet_absent = True
        else:
            attached_gateway = _validate_subnet_ownership(
                vpc, tagger, spec, stable_tags=True
            )
    if attached_gateway not in {None, gateway_id}:
        raise PrototypeMismatch("refusing to delete a subnet bound to another gateway")
    if attached_gateway == gateway_id:
        try:
            _validate_gateway_ownership(
                vpc,
                tagger,
                spec,
                floating_ip_may_be_absent=floating_ip_absent,
            )
        except ResourceAbsent:
            gateway_exists = not _confirm_predelete_absence(
                vpc.get_public_gateway,
                gateway_id,
                deadline=subnet_deadline,
                description="Public Gateway",
            )
            if gateway_exists:
                _validate_gateway_ownership(
                    vpc,
                    tagger,
                    spec,
                    floating_ip_may_be_absent=floating_ip_absent,
                )
        else:
            gateway_exists = True
            _validate_gateway_attachments(
                vpc, spec, {spec.subnet_id}, stable=True
            )
    else:
        _validate_gateway_attachments(vpc, spec, set(), stable=True)
        try:
            _validate_gateway_ownership(
                vpc,
                tagger,
                spec,
                floating_ip_may_be_absent=floating_ip_absent,
            )
        except ResourceAbsent:
            gateway_exists = not _confirm_predelete_absence(
                vpc.get_public_gateway,
                gateway_id,
                deadline=subnet_deadline,
                description="Public Gateway",
            )
            if gateway_exists:
                _validate_gateway_ownership(
                    vpc,
                    tagger,
                    spec,
                    floating_ip_may_be_absent=floating_ip_absent,
                )
            else:
                persist_deletion("public-gateway", gateway_id)
        else:
            gateway_exists = True
    if not subnet_absent:
        try:
            _delete_and_confirm(
                vpc.delete_subnet,
                vpc.get_subnet,
                spec.subnet_id,
                deadline=subnet_deadline,
                description="run subnet deletion",
                before_delete=lambda: _revalidate_subnet_delete(vpc, tagger, spec),
                revalidate_first_attempt=True,
            )
        except ResourceAbsent:
            pass
    persist_deletion("subnet", spec.subnet_id)
    vpc = _deadline_client(vpc, deadline)
    tagger = _deadline_client(tagger, deadline)
    gateway_error: Exception | None = None
    try:
        if gateway_exists:
            _validate_gateway_attachments(vpc, spec, set(), stable=True)
            try:
                _validate_gateway_ownership(
                    vpc,
                    tagger,
                    spec,
                    floating_ip_may_be_absent=floating_ip_absent,
                )
            except ResourceAbsent:
                gateway_exists = not _confirm_predelete_absence(
                    vpc.get_public_gateway,
                    gateway_id,
                    deadline=deadline,
                    description="Public Gateway",
                )
                if gateway_exists:
                    _validate_gateway_ownership(
                        vpc,
                        tagger,
                        spec,
                        floating_ip_may_be_absent=floating_ip_absent,
                    )
        if gateway_exists:
            _delete_and_confirm(
                vpc.delete_public_gateway,
                vpc.get_public_gateway,
                gateway_id,
                deadline=deadline,
                description="Public Gateway deletion",
                before_delete=lambda: _revalidate_gateway_delete(
                    vpc,
                    tagger,
                    spec,
                    floating_ip_absent=floating_ip_absent,
                ),
                revalidate_first_attempt=True,
            )
    except Exception as exc:
        gateway_error = exc
    else:
        persist_deletion("public-gateway", gateway_id)
    if gateway_error is not None:
        receipt_errors.append(gateway_error)
    if len(receipt_errors) == 1:
        raise receipt_errors[0]
    if receipt_errors:
        raise CleanupFailures(tuple(receipt_errors))


def delete_and_wait(
    vpc: VpcV1,
    spec: RunSpec,
    instance_id: str,
    *,
    deadline: float,
    tagger: Any,
) -> None:
    """Delete only resources bound to the RunSpec and confirm each with 404."""

    _validate_deadline(deadline)
    with run_lifecycle_lease(spec, deadline=deadline):
        _delete_and_wait_locked(vpc, tagger, spec, instance_id, deadline=deadline)


def _read_instance_for_cleanup(
    vpc: VpcV1,
    instance_id: str,
    *,
    deadline: float,
) -> dict[str, Any] | None:
    attempt = 0
    consecutive_absent = 0
    while time.monotonic() < deadline:
        try:
            observed = _result(vpc.get_instance(instance_id))
        except Exception as exc:
            if _status_code(exc) == 404:
                consecutive_absent += 1
                if consecutive_absent >= 2:
                    return None
            elif not _is_retryable_provider_error(exc):
                raise
            else:
                consecutive_absent = 0
        else:
            return observed
        try:
            _pause(attempt, deadline)
        except TimeoutError as exc:
            raise TimeoutError("instance cleanup state is unknown") from exc
        attempt += 1
    raise TimeoutError("instance cleanup state is unknown")


def _delete_and_wait_locked(
    vpc: VpcV1,
    tagger: Any,
    spec: RunSpec,
    instance_id: str,
    *,
    deadline: float,
) -> None:

    now = time.monotonic()
    remaining = max(0.0, deadline - now)
    instance_deadline = now + remaining / 4
    volume_deadline = now + remaining / 2
    floating_ip_probe_deadline = now + remaining * 5 / 8
    gateway_deadline = now + remaining * 3 / 4
    floating_ip_deadline = deadline
    vpc = _deadline_client(vpc, instance_deadline)
    tagger = _deadline_client(tagger, instance_deadline)

    instance_exists = True
    instance_delete_error: Exception | None = None
    instance_receipt_error: Exception | None = None
    instance_read_error: Exception | None = None
    instance_confirmed_absent = False
    cleanup_ledger_errors: list[Exception] = []
    try:
        instance = _read_instance_for_cleanup(
            vpc, instance_id, deadline=instance_deadline
        )
    except Exception as exc:
        instance_read_error = exc
        volume_id, gateway_id, floating_ip_id = _load_cleanup(
            spec, instance_id, errors=cleanup_ledger_errors
        )
        instance_exists = False
    else:
        if instance is not None:
            pass
        else:
            instance_exists = False
            volume_id, gateway_id, floating_ip_id = _load_cleanup(
                spec, instance_id, errors=cleanup_ledger_errors
            )
    if not instance_exists and instance_read_error is None:
        instance_confirmed_absent = True
        try:
            _persist_deletion(spec, "instance", instance_id)
        except Exception as receipt_error:
            instance_receipt_error = receipt_error
    if instance_exists:
        assert instance is not None
        if _field(instance, "id") != instance_id:
            raise PrototypeMismatch(
                "refusing cleanup of a different instance than requested"
            )
        mismatch = _mismatches(instance, spec)
        local_intent_owns = False
        try:
            _require_local_intent_ownership(spec, instance)
        except PrototypeMismatch:
            pass
        else:
            local_intent_owns = True
        if "crn" in instance:
            crn = _resource_crn(instance, spec, "instance", instance_id)
        elif local_intent_owns:
            crn = _load_instance_crn(spec, instance_id)
        else:
            raise PrototypeMismatch("instance CRN is missing during cleanup")
        try:
            _require_run_tags(tagger, crn, spec, "instance")
        except MissingOwnershipTags:
            _require_local_intent_ownership(spec, instance)
            local_intent_owns = True
        if local_intent_owns:
            volume_id, gateway_id, floating_ip_id = _load_cleanup(
                spec, instance_id, errors=cleanup_ledger_errors
            )
            if volume_id is None:
                volume_id = _validate_instance_storage(vpc, spec, instance)
                _persist_cleanup(
                    spec, instance_id, volume_id, gateway_id, floating_ip_id
                )
        else:
            if mismatch:
                raise PrototypeMismatch(
                    "refusing cleanup of mismatched instance: " + ", ".join(mismatch)
                )
            _validate_trusted_profile(vpc, spec, instance_id, context="cleanup")
            _validate_instance_network(vpc, spec, instance)
            _validate_security_group_rules(vpc, spec)
            volume_id = _validate_instance_storage(vpc, spec, instance)
            current_gateway = _validate_subnet_ownership(vpc, tagger, spec)
            if current_gateway != spec.public_gateway_id:
                raise PrototypeMismatch("refusing cleanup of an unbound Public Gateway")
            gateway_id = current_gateway
            floating_ip_id = spec.public_gateway_floating_ip_id
            _persist_cleanup(
                spec, instance_id, volume_id, gateway_id, floating_ip_id
            )

    if instance_exists:
        try:
            _delete_and_confirm(
                vpc.delete_instance,
                vpc.get_instance,
                instance_id,
                deadline=instance_deadline,
                description="instance deletion",
                before_delete=lambda: _revalidate_instance_delete(
                    vpc,
                    tagger,
                    spec,
                    instance_id,
                    allow_local_intent=local_intent_owns,
                ),
                revalidate_first_attempt=True,
            )
        except Exception as exc:
            instance_delete_error = exc
        else:
            instance_confirmed_absent = True
            try:
                _persist_deletion(spec, "instance", instance_id)
            except Exception as receipt_error:
                instance_receipt_error = receipt_error
    if not instance_confirmed_absent:
        instance_errors = [
            error
            for error in (
                *cleanup_ledger_errors,
                instance_read_error,
                instance_delete_error,
                instance_receipt_error,
            )
            if error is not None
        ]
        if len(instance_errors) == 1:
            raise instance_errors[0]
        raise CleanupFailures(tuple(instance_errors))
    volume_error: Exception | None = None
    try:
        vpc = _deadline_client(vpc, volume_deadline)
        _cleanup_volumes(vpc, spec, volume_id, deadline=volume_deadline)
    except Exception as exc:
        volume_error = exc

    floating_ip_error: Exception | None = None
    floating_ip_exists = True
    try:
        vpc = _deadline_client(vpc, floating_ip_probe_deadline)
        tagger = _deadline_client(tagger, floating_ip_probe_deadline)
        _validate_floating_ip_ownership(
            vpc,
            tagger,
            spec,
            floating_ip_id,
            gateway_may_be_deleted=True,
        )
    except ResourceAbsent:
        try:
            floating_ip_exists = not _confirm_predelete_absence(
                vpc.get_floating_ip,
                floating_ip_id,
                deadline=floating_ip_probe_deadline,
                description="Floating IP",
            )
        except Exception as exc:
            floating_ip_error = exc
    except Exception as exc:
        if isinstance(exc, AmbiguousProvisioning):
            floating_ip_error = exc
        elif not _is_retryable_provider_error(exc):
            floating_ip_error = exc

    gateway_error: Exception | None = None
    try:
        _cleanup_gateway(
            vpc,
            tagger,
            spec,
            gateway_id,
            deadline=gateway_deadline,
            floating_ip_absent=not floating_ip_exists,
        )
    except Exception as exc:
        gateway_error = exc

    if floating_ip_error is None and floating_ip_exists:
        try:
            vpc = _deadline_client(vpc, floating_ip_deadline)
            tagger = _deadline_client(tagger, floating_ip_deadline)
            _delete_and_confirm(
                vpc.delete_floating_ip,
                vpc.get_floating_ip,
                floating_ip_id,
                deadline=floating_ip_deadline,
                description="Floating IP deletion",
                before_delete=lambda: _validate_floating_ip_ownership(
                    vpc,
                    tagger,
                    spec,
                    floating_ip_id,
                    gateway_may_be_deleted=True,
                    stable_tags=True,
                ),
                revalidate_first_attempt=True,
            )
        except Exception as exc:
            floating_ip_error = exc
    if floating_ip_error is None:
        try:
            _persist_deletion(spec, "floating-ip", floating_ip_id)
        except Exception as receipt_error:
            floating_ip_error = receipt_error
    cleanup_errors = [
        error
        for error in (
            *cleanup_ledger_errors,
            instance_read_error,
            instance_delete_error,
            instance_receipt_error,
            volume_error,
            gateway_error,
            floating_ip_error,
        )
        if error is not None
    ]
    if len(cleanup_errors) == 1:
        raise cleanup_errors[0]
    if cleanup_errors:
        raise CleanupFailures(tuple(cleanup_errors))
