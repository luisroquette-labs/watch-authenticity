"""Deterministic IBM VPC lifecycle tests; no provider calls leave this process."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from unittest.mock import MagicMock, call

import pytest
from ibm_cloud_sdk_core.authenticators import VPCInstanceAuthenticator
from ibm_vpc.vpc_v1 import InstanceMetadataServicePrototype
from requests import ReadTimeout

import infra.ibm_vpc as ibm_vpc
from infra.ibm_vpc import (
    AmbiguousProvisioning,
    CleanupFailures,
    PrototypeMismatch,
    ProvisioningFailed,
    build_instance_prototype,
    delete_and_wait,
    ensure_single_instance,
    recover_accepted_instance,
    recover_created_instance_id,
    wait_running,
)
from tests.conftest import safe_user_data, valid_run_spec, vpc_instance


class _Response:
    def __init__(self, result: object) -> None:
        self._result = result

    def get_result(self) -> object:
        return self._result


class _NotFound(Exception):
    def __init__(self) -> None:
        self.code = 404


class _TooManyRequests(Exception):
    def __init__(self) -> None:
        self.code = 429


class _Conflict(Exception):
    def __init__(self) -> None:
        self.code = 409


class _Forbidden(Exception):
    def __init__(self) -> None:
        self.code = 403


class _ServiceUnavailable(Exception):
    def __init__(self) -> None:
        self.code = 503


@pytest.fixture(autouse=True)
def _isolated_shared_state(monkeypatch, tmp_path):
    monkeypatch.setattr("infra.ibm_vpc._RECOVERY_DIR", tmp_path / "shared-state")
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)


def _page(*instances: dict[str, object]) -> _Response:
    return _Response({"instances": list(instances), "total_count": len(instances)})


def _instance_id() -> str:
    return str(vpc_instance()["id"])


def _outbound(protocol: str, port: int) -> dict[str, object]:
    return {
        "direction": "outbound",
        "protocol": protocol,
        "port_min": port,
        "port_max": port,
        "remote": {"cidr_block": "0.0.0.0/0"},
    }


def _network_interface(**overrides: object) -> dict[str, object]:
    interface: dict[str, object] = {
        "id": "0717-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "subnet": {"id": valid_run_spec().subnet_id},
        "primary_ip": {"address": valid_run_spec().primary_ipv4_address},
        "security_groups": [{"id": valid_run_spec().security_group_id}],
        "allow_ip_spoofing": False,
        "floating_ips": [],
    }
    interface.update(overrides)
    return interface


def _client() -> MagicMock:
    client = MagicMock()
    client.get_image.return_value = _Response(_image())
    client.get_security_group.return_value = _Response(_security_group())
    client.get_instance_initialization.return_value = _Response(
        {
            "default_trusted_profile": {
                "auto_link": True,
                "target": {
                    "id": valid_run_spec().trusted_profile_id,
                    "crn": "crn:v1:bluemix:public:iam-identity::profile:test",
                    "resource_type": "profile",
                },
            }
        }
    )
    client.list_subnets.return_value = _Response(
        {"subnets": [_subnet(valid_run_spec().public_gateway_id)], "total_count": 1}
    )
    client.get_subnet.return_value = _Response(
        _subnet(valid_run_spec().public_gateway_id)
    )
    client.get_public_gateway.return_value = _Response(_gateway())
    client.get_floating_ip.return_value = _Response(_floating_ip())
    client.get_instance_network_interface.return_value = _Response(
        _network_interface()
    )
    client.list_floating_ips.return_value = _Response(
        {"floating_ips": [], "total_count": 0}
    )
    client.get_instance_volume_attachment.return_value = _Response(_attachment())
    client.get_volume.return_value = _Response(_volume())
    client.list_volumes.return_value = _Response(
        {"volumes": [], "total_count": 0}
    )
    client.list_security_group_rules.return_value = _Response(
        {
            "rules": [
                _outbound("tcp", 443),
                _outbound("tcp", 53),
                _outbound("udp", 53),
            ]
        }
    )

    def delete_subnet(_subnet_id):
        client.get_subnet.side_effect = _NotFound()
        client.list_subnets.return_value = _Response(
            {"subnets": [], "total_count": 0}
        )

    client.delete_subnet.side_effect = delete_subnet

    def delete_floating_ip(_floating_ip_id):
        client.get_floating_ip.side_effect = _NotFound()

    client.delete_floating_ip.side_effect = delete_floating_ip

    def delete_gateway(_gateway_id):
        client.get_floating_ip.return_value = _Response(_floating_ip(target=None))

    client.delete_public_gateway.side_effect = delete_gateway
    return client


def _tagger() -> MagicMock:
    tagger = MagicMock()
    tagger.attach_tag.return_value = _Response(
        {"results": [{"resource_id": vpc_instance()["crn"], "is_error": False}]}
    )
    tagger.list_tags.return_value = _Response(
        {
            "total_count": 2,
            "offset": 0,
            "items": [
                {"name": "project:watch-authenticity"},
                {"name": f"run:{valid_run_spec().run_id}"},
            ]
        }
    )
    return tagger


def _gateway(**overrides: object) -> dict[str, object]:
    gateway: dict[str, object] = {
        "id": valid_run_spec().public_gateway_id,
        "crn": (
            "crn:v1:bluemix:public:is:us-south-1:a/account-1::public-gateway:"
            f"{valid_run_spec().public_gateway_id}"
        ),
        "resource_group": {"id": valid_run_spec().resource_group_id},
        "vpc": {"id": valid_run_spec().network_id},
        "zone": {"name": valid_run_spec().zone},
        "floating_ip": {"id": valid_run_spec().public_gateway_floating_ip_id},
    }
    gateway.update(overrides)
    return gateway


def _floating_ip(**overrides: object) -> dict[str, object]:
    floating_ip: dict[str, object] = {
        "id": valid_run_spec().public_gateway_floating_ip_id,
        "crn": (
            "crn:v1:bluemix:public:is:us-south-1:a/account-1::floating-ip:"
            f"{valid_run_spec().public_gateway_floating_ip_id}"
        ),
        "resource_group": {"id": valid_run_spec().resource_group_id},
        "zone": {"name": valid_run_spec().zone},
        "target": {
            "id": valid_run_spec().public_gateway_id,
            "crn": (
                "crn:v1:bluemix:public:is:us-south-1:a/account-1::public-gateway:"
                f"{valid_run_spec().public_gateway_id}"
            ),
            "resource_type": "public_gateway",
        },
    }
    floating_ip.update(overrides)
    return floating_ip


def _image(**overrides: object) -> dict[str, object]:
    image: dict[str, object] = {
        "id": valid_run_spec().image_id,
        "status": "available",
        "visibility": "public",
        "user_data_format": "cloud_init",
        "operating_system": {
            "architecture": "amd64",
            "family": "Ubuntu Linux",
            "name": "ubuntu-22-04-amd64",
            "user_data_format": "cloud_init",
            "version": "22.04",
        },
    }
    image.update(overrides)
    return image


def _security_group(**overrides: object) -> dict[str, object]:
    security_group: dict[str, object] = {
        "id": valid_run_spec().security_group_id,
        "resource_group": {"id": valid_run_spec().resource_group_id},
        "vpc": {"id": valid_run_spec().network_id},
    }
    security_group.update(overrides)
    return security_group


def _subnet(gateway_id: str | None = None) -> dict[str, object]:
    subnet: dict[str, object] = {
        "id": valid_run_spec().subnet_id,
        "crn": (
            "crn:v1:bluemix:public:is:us-south-1:a/account-1::subnet:"
            f"{valid_run_spec().subnet_id}"
        ),
        "resource_group": {"id": valid_run_spec().resource_group_id},
        "vpc": {"id": valid_run_spec().network_id},
        "zone": {"name": valid_run_spec().zone},
    }
    if gateway_id is not None:
        subnet["public_gateway"] = {"id": gateway_id}
    return subnet


def _attachment() -> dict[str, object]:
    return {
        "id": "r006-cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        "delete_volume_on_instance_delete": True,
        "volume": {"id": "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"},
    }


def _volume(**overrides: object) -> dict[str, object]:
    volume: dict[str, object] = {
        "id": "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        "capacity": 100,
        "profile": {"name": "general-purpose"},
        "resource_group": {"id": valid_run_spec().resource_group_id},
        "zone": {"name": valid_run_spec().zone},
        "encryption": "provider_managed",
        "user_tags": ["project:watch-authenticity", f"run:{valid_run_spec().run_id}"],
    }
    volume.update(overrides)
    return volume


def _deadline() -> float:
    return time.monotonic() + 60


def _seed_intent() -> None:
    spec = valid_run_spec()
    ibm_vpc._persist_intent(
        spec,
        build_instance_prototype(spec, safe_user_data()),
    )


def test_recovery_replace_is_followed_by_directory_fsync(monkeypatch):
    events: list[str] = []
    directory_fds: dict[int, str] = {}
    real_replace = ibm_vpc.os.replace
    real_open = ibm_vpc.os.open
    real_fsync = ibm_vpc.os.fsync

    def observed_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    def observed_open(path, flags, *args):
        descriptor = real_open(path, flags, *args)
        if str(path) == str(ibm_vpc._RECOVERY_DIR):
            directory_fds[descriptor] = "recovery"
            events.append("open-directory")
        elif str(path) == str(ibm_vpc._RECOVERY_DIR.parent):
            directory_fds[descriptor] = "parent"
            events.append("open-parent")
        return descriptor

    def observed_fsync(descriptor):
        if descriptor in directory_fds:
            events.append(f"fsync-{directory_fds[descriptor]}")
        return real_fsync(descriptor)

    monkeypatch.setattr(ibm_vpc.os, "replace", observed_replace)
    monkeypatch.setattr(ibm_vpc.os, "open", observed_open)
    monkeypatch.setattr(ibm_vpc.os, "fsync", observed_fsync)

    ibm_vpc._write_recovery("durable", {"state": "intent"})

    replace_index = events.index("replace")
    assert events.index("fsync-parent") < replace_index
    assert events.index("open-directory", replace_index) < events.index(
        "fsync-recovery", replace_index
    )


def test_build_instance_prototype_binds_every_authorized_resource():
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    assert prototype["name"] == f"watch-auth-{spec.run_id}"
    assert prototype["resource_group"] == {"id": spec.resource_group_id}
    assert prototype["vpc"] == {"id": spec.network_id}
    assert prototype["zone"] == {"name": spec.zone}
    assert prototype["image"] == {"id": spec.image_id}
    assert prototype["profile"] == {"name": "gx3-16x80x1l4"}
    assert prototype["primary_network_interface"] == {
        "subnet": {"id": spec.subnet_id},
        "primary_ip": {
            "address": spec.primary_ipv4_address,
            "auto_delete": False,
        },
        "security_groups": [{"id": spec.security_group_id}],
        "allow_ip_spoofing": False,
    }
    assert prototype["metadata_service"] == {
        "enabled": True,
        "protocol": "http",
        "response_hop_limit": 2,
    }
    assert prototype["default_trusted_profile"] == {
        "auto_link": True,
        "target": {"id": spec.trusted_profile_id},
    }
    assert prototype["boot_volume_attachment"] == {
        "delete_volume_on_instance_delete": True,
        "volume": {
            "capacity": 100,
            "profile": {"name": "general-purpose"},
            "resource_group": {"id": spec.resource_group_id},
            "user_tags": ["project:watch-authenticity", f"run:{spec.run_id}"],
        },
    }
    assert "floating" not in repr(prototype).lower()


def test_metadata_contract_matches_installed_trusted_profile_sdk():
    spec = valid_run_spec()
    metadata = build_instance_prototype(spec, safe_user_data())["metadata_service"]
    model = InstanceMetadataServicePrototype._from_dict(metadata)
    authenticator = VPCInstanceAuthenticator(iam_profile_id=spec.trusted_profile_id)

    assert authenticator.token_manager.url == "http://169.254.169.254"
    assert model.to_dict() == {
        "enabled": True,
        "protocol": "http",
        "response_hop_limit": 2,
    }


def test_build_instance_prototype_rejects_unbound_bootstrap():
    with pytest.raises(ValueError, match="bootstrap SHA-256"):
        build_instance_prototype(
            valid_run_spec(),
            safe_user_data() + "# changed\n",
        )


@pytest.mark.parametrize(
    "address", ("8.8.8.8", "127.0.0.1", "169.254.1.1", "0.0.0.0")
)
def test_run_spec_rejects_a_non_rfc1918_primary_ip(address):
    with pytest.raises(ValueError, match="RFC1918"):
        replace(valid_run_spec(), primary_ipv4_address=address)


def test_provider_reserved_ip_conflict_adopts_the_single_instance(
    monkeypatch, tmp_path
):
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    instance = vpc_instance()

    first = _client()
    first.list_instances.side_effect = [_page(), _page(instance)]
    first.create_instance.return_value = _Response(instance)
    monkeypatch.setattr("infra.ibm_vpc._RECOVERY_DIR", tmp_path / "controller-a")
    first_ref = ensure_single_instance(
        first, spec, prototype, tagger=_tagger(), deadline=_deadline()
    )

    second = _client()
    second.list_instances.side_effect = [_page(), _page(instance), _page(instance)]
    second.create_instance.side_effect = _Conflict()
    monkeypatch.setattr("infra.ibm_vpc._RECOVERY_DIR", tmp_path / "controller-b")
    second_ref = ensure_single_instance(
        second, spec, prototype, tagger=_tagger(), deadline=_deadline()
    )

    assert first_ref.id == second_ref.id
    first.create_instance.assert_called_once_with(prototype)
    second.create_instance.assert_called_once_with(prototype)


def test_create_path_caps_every_provider_call_to_its_deadline(monkeypatch):
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    monkeypatch.setattr("infra.ibm_vpc.time.monotonic", lambda: 1.0)

    ensure_single_instance(
        client,
        valid_run_spec(),
        build_instance_prototype(valid_run_spec(), safe_user_data()),
        tagger=_tagger(),
        deadline=3.0,
    )

    timeouts = [
        call_.args[0]["timeout"]
        for call_ in client.set_http_config.call_args_list
        if "timeout" in call_.args[0]
    ]
    assert timeouts
    assert all(0 < timeout <= 2.0 for timeout in timeouts)


def test_existing_exact_match_is_adopted_without_create():
    _seed_intent()
    client = _client()
    tagger = _tagger()
    client.list_instances.return_value = _page(vpc_instance())

    ref = ensure_single_instance(
        client,
        valid_run_spec(),
        build_instance_prototype(valid_run_spec(), safe_user_data()),
        tagger=tagger,
    )

    assert ref.id == "r006-77777777-7777-4777-8777-777777777777"
    client.get_instance_network_interface.assert_called_once_with(
        _instance_id(), "0717-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    )
    client.create_instance.assert_not_called()
    tagger.attach_tag.assert_not_called()
    prototype = build_instance_prototype(valid_run_spec(), safe_user_data())
    assert recover_created_instance_id(valid_run_spec(), prototype) == ref.id
    assert ibm_vpc._load_cleanup(valid_run_spec(), ref.id) == (
        "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        valid_run_spec().public_gateway_id,
        valid_run_spec().public_gateway_floating_ip_id,
    )


def test_cleanup_context_cannot_overwrite_the_original_instance():
    spec = valid_run_spec()
    original_id = _instance_id()
    volume_id = "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_accepted_instance(spec, prototype, original_id)
    ibm_vpc._persist_cleanup(
        spec,
        original_id,
        volume_id,
        spec.public_gateway_id,
        spec.public_gateway_floating_ip_id,
    )

    with pytest.raises(AmbiguousProvisioning, match="differs from provider state"):
        ibm_vpc._persist_cleanup(
            spec,
            "r006-replacement",
            volume_id,
            spec.public_gateway_id,
            spec.public_gateway_floating_ip_id,
        )

    assert ibm_vpc._load_cleanup(spec, original_id) == (
        volume_id,
        spec.public_gateway_id,
        spec.public_gateway_floating_ip_id,
    )


def test_cleanup_context_cannot_target_a_replacement_of_the_accepted_instance():
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_accepted_instance(spec, prototype, _instance_id())

    with pytest.raises(AmbiguousProvisioning, match="accepted instance identity"):
        ibm_vpc._persist_cleanup(
            spec,
            "r006-replacement",
            "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            spec.public_gateway_id,
            spec.public_gateway_floating_ip_id,
        )


def test_existing_instance_without_create_intent_is_not_adopted_or_tagged():
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    tagger = _tagger()

    with pytest.raises(AmbiguousProvisioning, match="without matching local create intent"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=tagger,
        )

    tagger.attach_tag.assert_not_called()
    client.create_instance.assert_not_called()


def test_existing_instance_cannot_overwrite_a_different_created_id():
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    _seed_intent()
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    replacement_id = "r006-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    replacement = vpc_instance(
        id=replacement_id,
        crn=(
            "crn:v1:bluemix:public:is:us-south-1:a/account-1::instance:"
            f"{replacement_id}"
        ),
    )
    client = _client()
    client.list_instances.return_value = _page(replacement)

    with pytest.raises(AmbiguousProvisioning, match="differs from provider state"):
        ensure_single_instance(client, spec, prototype, tagger=_tagger())

    assert recover_created_instance_id(spec, prototype) == _instance_id()
    client.create_instance.assert_not_called()


def test_tag_ownership_uses_ghost_provider_and_reads_every_page():
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    tagger = _tagger()

    def paged_tags(**kwargs):
        assert kwargs["providers"] == ["ghost"]
        assert kwargs["limit"] == 100
        offset = kwargs["offset"]
        item = (
            {"name": "project:watch-authenticity"}
            if offset == 0
            else {"name": f"run:{valid_run_spec().run_id}"}
        )
        return _Response({"total_count": 2, "offset": offset, "items": [item]})

    tagger.list_tags.side_effect = paged_tags

    ensure_single_instance(
        client,
        valid_run_spec(),
        build_instance_prototype(valid_run_spec(), safe_user_data()),
        tagger=tagger,
    )

    assert tagger.list_tags.call_count == 14


def test_tag_ownership_rejects_a_name_repeated_across_pages():
    spec = valid_run_spec()
    tagger = _tagger()
    tagger.list_tags.side_effect = lambda **kwargs: _Response(
        {
            "total_count": 2,
            "offset": kwargs["offset"],
            "items": [{"name": "project:watch-authenticity"}],
        }
    )

    with pytest.raises(AmbiguousProvisioning, match="repeated name"):
        ibm_vpc._require_run_tags(tagger, "crn:test", spec, "instance")


def test_tag_ownership_rejects_total_count_drift_between_pages():
    spec = valid_run_spec()
    tagger = _tagger()
    tagger.list_tags.side_effect = [
        _Response(
            {
                "total_count": 2,
                "offset": 0,
                "items": [{"name": "project:watch-authenticity"}],
            }
        ),
        _Response({"total_count": 1, "offset": 1, "items": []}),
    ]

    with pytest.raises(AmbiguousProvisioning, match="malformed"):
        ibm_vpc._require_run_tags(tagger, "crn:test", spec, "instance")


def test_destructive_tag_validation_requires_identical_snapshots():
    spec = valid_run_spec()
    tagger = _tagger()
    tagger.list_tags.side_effect = [
        _Response(
            {
                "total_count": 2,
                "offset": 0,
                "items": [
                    {"name": "project:watch-authenticity"},
                    {"name": f"run:{spec.run_id}"},
                ],
            }
        ),
        _Response(
            {
                "total_count": 2,
                "offset": 0,
                "items": [
                    {"name": "project:foreign"},
                    {"name": f"run:{spec.run_id}"},
                ],
            }
        ),
    ]

    with pytest.raises(AmbiguousProvisioning, match="changed"):
        ibm_vpc._require_run_tags(
            tagger, "crn:test", spec, "instance", stable=True
        )


def test_invalid_image_is_rejected_before_create():
    client = _client()
    client.get_image.return_value = _Response(_image(visibility="private"))

    with pytest.raises(PrototypeMismatch, match="public Ubuntu"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )

    client.create_instance.assert_not_called()


@pytest.mark.parametrize(
    "operating_system",
    (
        {
            "architecture": "amd64",
            "family": "Ubuntu Linux",
            "name": "ubuntu-24-04-amd64",
            "user_data_format": "cloud_init",
            "version": "24.04",
        },
        {
            "architecture": "amd64",
            "family": "Ubuntu Linux",
            "name": "ubuntu-22-04-amd64",
            "user_data_format": "cloud_init",
            "version": "24.04",
        },
    ),
)
def test_wrong_ubuntu_release_is_rejected_before_create(
    operating_system: dict[str, object],
):
    client = _client()
    client.get_image.return_value = _Response(_image(operating_system=operating_system))

    with pytest.raises(PrototypeMismatch, match="Ubuntu 22.04"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )

    client.create_instance.assert_not_called()


def test_inbound_rule_is_rejected_before_create():
    client = _client()
    client.list_security_group_rules.return_value = _Response(
        {
            "rules": [
                {"direction": "inbound", "protocol": "tcp", "port_min": 22, "port_max": 22}
            ]
        }
    )

    with pytest.raises(PrototypeMismatch, match="deny all inbound"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )

    client.create_instance.assert_not_called()


def test_unexpected_egress_destination_is_rejected_before_create():
    client = _client()
    https = _outbound("tcp", 443)
    https["remote"] = {"cidr_block": "10.0.0.0/8"}
    client.list_security_group_rules.return_value = _Response(
        {
            "rules": [https, _outbound("tcp", 53), _outbound("udp", 53)]
        }
    )

    with pytest.raises(PrototypeMismatch, match="only DNS and HTTPS"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )

    client.create_instance.assert_not_called()


def test_existing_instance_with_secondary_interface_is_not_adopted():
    _seed_intent()
    client = _client()
    instance = vpc_instance()
    instance["network_interfaces"] = [
        {"id": "0717-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
        {"id": "0717-secondary"},
    ]
    client.list_instances.return_value = _page(instance)

    with pytest.raises(PrototypeMismatch, match="exactly one legacy network interface"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )

    client.create_instance.assert_not_called()


def test_existing_instance_with_floating_ip_is_not_adopted():
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    client.list_floating_ips.return_value = _Response(
        {"floating_ips": [{"id": "r006-floating"}], "total_count": 1}
    )

    with pytest.raises(PrototypeMismatch, match="must not have a floating IP"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )

    client.create_instance.assert_not_called()


def test_foreign_run_tag_blocks_launch_before_create():
    client = _client()
    tagger = _tagger()
    tagger.list_tags.return_value = _Response(
        {
            "total_count": 3,
            "offset": 0,
            "items": [
                {"name": "project:watch-authenticity"},
                {"name": f"run:{valid_run_spec().run_id}"},
                {"name": "run:foreign"},
            ],
        }
    )

    with pytest.raises(PrototypeMismatch, match="ownership tags"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=tagger,
        )

    client.create_instance.assert_not_called()


def test_shared_gateway_is_rejected_before_create():
    client = _client()
    other = _subnet(valid_run_spec().public_gateway_id)
    other["id"] = "0717-other"
    client.list_subnets.return_value = _Response(
        {
            "subnets": [_subnet(valid_run_spec().public_gateway_id), other],
            "total_count": 2,
        }
    )

    with pytest.raises(PrototypeMismatch, match="unauthorized subnet"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )

    client.create_instance.assert_not_called()


def test_lost_create_response_reconciles_by_exact_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.list_instances.side_effect = [
        _page(),
        _page(vpc_instance()),
        _page(vpc_instance()),
    ]
    client.create_instance.side_effect = ReadTimeout()
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    assert ensure_single_instance(client, spec, prototype, tagger=_tagger()).id.startswith("r006-")
    assert client.create_instance.call_count == 1
    assert recover_created_instance_id(spec, prototype) == _instance_id()


def test_create_interruption_reconciles_and_persists_accepted_id(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.list_instances.side_effect = [
        _page(),
        _page(vpc_instance()),
    ]
    client.create_instance.side_effect = KeyboardInterrupt()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    with pytest.raises(KeyboardInterrupt):
        ensure_single_instance(client, spec, prototype, tagger=_tagger())

    assert recover_created_instance_id(spec, prototype) == _instance_id()


def test_cleanup_window_recovers_create_visible_after_create_timeout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    created = vpc_instance()
    client.list_instances.side_effect = [
        _page(),
        _page(),
        _page(),
        _page(),
        _page(),
        _page(created),
        _page(created),
    ]
    client.create_instance.side_effect = ReadTimeout()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    with pytest.raises(AmbiguousProvisioning, match="unknown"):
        ensure_single_instance(client, spec, prototype, tagger=_tagger())

    recovered = recover_accepted_instance(
        client,
        spec,
        prototype,
        deadline=time.monotonic() + 60,
    )

    assert recovered == _instance_id()
    assert recover_created_instance_id(spec, prototype) == _instance_id()


def test_partial_cleanup_hydration_keeps_durable_accepted_id(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _seed_intent()
    client = _client()
    incomplete = vpc_instance(boot_volume_attachment={})
    client.list_instances.return_value = _page(incomplete)
    client.get_instance.return_value = _Response(incomplete)
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    with pytest.raises(AmbiguousProvisioning, match="cleanup context"):
        recover_accepted_instance(
            client,
            spec,
            prototype,
            deadline=time.monotonic() + 60,
        )

    assert recover_created_instance_id(spec, prototype) == _instance_id()


def test_deterministic_create_4xx_clears_intent_and_allows_retry():
    client = _client()
    client.list_instances.return_value = _page()
    client.create_instance.side_effect = [_Forbidden(), _Response(vpc_instance())]
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    with pytest.raises(_Forbidden):
        ensure_single_instance(client, spec, prototype, tagger=_tagger())

    assert not ibm_vpc._intent_path(spec).exists()
    ref = ensure_single_instance(client, spec, prototype, tagger=_tagger())
    assert ref.id == _instance_id()
    assert client.create_instance.call_count == 2


def test_create_response_without_id_reconciles_and_persists(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    created = vpc_instance()
    client.list_instances.side_effect = [_page(), _page(created), _page(created)]
    client.create_instance.return_value = _Response(vpc_instance(id=""))
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    ref = ensure_single_instance(client, spec, prototype, tagger=_tagger())

    assert ref.id == created["id"]
    assert recover_created_instance_id(spec, prototype) == created["id"]


def test_create_response_with_id_hydrates_cleanup_context_after_id_is_persisted():
    client = _client()
    created = vpc_instance()
    incomplete = vpc_instance(boot_volume_attachment={"id": "attachment-only"})
    client.list_instances.return_value = _page()
    client.create_instance.return_value = _Response(incomplete)
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    def get_complete(instance_id):
        assert instance_id == created["id"]
        assert recover_created_instance_id(spec, prototype) == created["id"]
        return _Response(created)

    client.get_instance.side_effect = get_complete

    ref = ensure_single_instance(client, spec, prototype, tagger=_tagger())

    assert ref.id == created["id"]
    assert ibm_vpc._load_cleanup(spec, ref.id)[0] == (
        "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    )


def test_create_response_with_volume_waits_for_crn_before_returning():
    client = _client()
    created = vpc_instance()
    without_crn = vpc_instance()
    without_crn.pop("crn")
    client.list_instances.return_value = _page()
    client.create_instance.return_value = _Response(without_crn)
    client.get_instance.return_value = _Response(created)
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    ref = ensure_single_instance(client, spec, prototype, tagger=_tagger())

    assert ref.id == created["id"]
    client.get_instance.assert_called_once_with(created["id"])
    assert ibm_vpc._load_instance_crn(spec, ref.id) == created["crn"]


def test_accepted_create_keeps_recoverable_id_when_cleanup_context_never_appears(
    monkeypatch,
):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    incomplete = vpc_instance(boot_volume_attachment={"id": "attachment-only"})
    client.list_instances.return_value = _page()
    client.create_instance.return_value = _Response(incomplete)
    client.get_instance.return_value = _Response(incomplete)
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    with pytest.raises(AmbiguousProvisioning, match="cleanup context"):
        ensure_single_instance(client, spec, prototype, tagger=_tagger())

    assert recover_created_instance_id(spec, prototype) == incomplete["id"]
    assert client.create_instance.call_count == 1


def test_multiple_name_matches_quarantine_without_delete():
    client = _client()
    client.list_instances.return_value = _page(
        vpc_instance(id="r006-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        vpc_instance(id="r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
    )

    with pytest.raises(AmbiguousProvisioning, match="multiple"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )
    client.delete_instance.assert_not_called()


def test_instance_pagination_cannot_hide_a_duplicate():
    client = _client()
    first = vpc_instance(id="r006-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    second = vpc_instance(id="r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    client.list_instances.side_effect = [
        _Response(
            {
                "instances": [first],
                "total_count": 2,
                "next": {"start": "page-2"},
            }
        ),
        _Response({"instances": [second], "total_count": 2}),
    ]

    with pytest.raises(AmbiguousProvisioning, match="multiple"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )

    client.create_instance.assert_not_called()


def test_instance_pagination_missing_cursor_fails_closed():
    client = _client()
    client.list_instances.return_value = _Response(
        {"instances": [], "total_count": 1}
    )

    with pytest.raises(AmbiguousProvisioning, match="ended before total count"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )

    client.create_instance.assert_not_called()


def test_name_match_with_different_prototype_is_not_adopted_or_deleted():
    client = _client()
    client.list_instances.return_value = _page(
        vpc_instance(profile={"name": "bx2-2x8"})
    )

    with pytest.raises(PrototypeMismatch, match="profile"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )
    client.create_instance.assert_not_called()
    client.delete_instance.assert_not_called()


def test_prior_ambiguous_create_intent_blocks_a_second_create(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    client.list_instances.return_value = _page()
    client.create_instance.side_effect = ReadTimeout()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    with pytest.raises(AmbiguousProvisioning):
        ensure_single_instance(client, spec, prototype, tagger=_tagger())
    with pytest.raises(AmbiguousProvisioning, match="prior create intent"):
        ensure_single_instance(client, spec, prototype, tagger=_tagger())

    assert client.create_instance.call_count == 1


def test_existing_instance_does_not_bypass_conflicting_create_intent(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    first = _client()
    first.list_instances.return_value = _page()
    first.create_instance.side_effect = ReadTimeout()
    spec = valid_run_spec()

    with pytest.raises(AmbiguousProvisioning, match="outcome is unknown"):
        ensure_single_instance(
            first,
            spec,
            build_instance_prototype(spec, safe_user_data()),
            tagger=_tagger(),
        )

    second = _client()
    second.list_instances.return_value = _page(vpc_instance())
    changed_user_data = safe_user_data() + "\n# changed"
    changed_spec = replace(
        spec,
        bootstrap_sha256=hashlib.sha256(changed_user_data.encode()).hexdigest(),
    )
    with pytest.raises(AmbiguousProvisioning, match="intent differs"):
        ensure_single_instance(
            second,
            changed_spec,
            build_instance_prototype(changed_spec, changed_user_data),
            tagger=_tagger(),
        )

    second.create_instance.assert_not_called()


def test_lost_create_429_reconciliation_persists_ambiguity(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    client.list_instances.side_effect = [_page()] + [_TooManyRequests()] * 4
    client.create_instance.side_effect = ReadTimeout()

    with pytest.raises(AmbiguousProvisioning, match="reconciled"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )

    assert list((tmp_path / "shared-state").glob("ambiguous-create-*.json"))


def test_lost_create_retries_transient_reconciliation_read(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    instance = vpc_instance()
    client.list_instances.side_effect = [
        _page(),
        _ServiceUnavailable(),
        _page(instance),
        _page(instance),
    ]
    client.create_instance.side_effect = ReadTimeout()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())

    ref = ensure_single_instance(client, spec, prototype, tagger=_tagger())

    assert ref.id == instance["id"]
    assert recover_created_instance_id(spec, prototype) == instance["id"]


def test_security_group_mismatch_is_not_adopted():
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    client.get_instance_network_interface.return_value = _Response(
        _network_interface(security_groups=[{"id": "r006-unsafe"}])
    )

    with pytest.raises(PrototypeMismatch, match="security_group"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )


def test_inbound_security_group_rule_is_not_adopted():
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    client.list_security_group_rules.return_value = _Response(
        {
            "rules": [
                {"direction": "inbound", "protocol": "tcp", "port_min": 22, "port_max": 22},
                _outbound("tcp", 443),
                _outbound("tcp", 53),
                _outbound("udp", 53),
            ]
        }
    )

    with pytest.raises(PrototypeMismatch, match="deny all inbound"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )


def test_ip_spoofing_mismatch_is_not_adopted():
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    client.get_instance_network_interface.return_value = _Response(
        _network_interface(allow_ip_spoofing=True)
    )

    with pytest.raises(PrototypeMismatch, match="allow_ip_spoofing"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )


def test_boot_volume_mismatch_is_not_adopted():
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    client.get_volume.return_value = _Response(_volume(capacity=200))

    with pytest.raises(PrototypeMismatch, match="boot volume"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )


def test_user_managed_boot_volume_is_not_adopted():
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    client.get_volume.return_value = _Response(_volume(encryption="user_managed"))

    with pytest.raises(PrototypeMismatch, match="boot volume"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )


def test_instance_crn_must_bind_the_account_type_and_id():
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(
        vpc_instance(
            crn=(
                "crn:v1:bluemix:public:is:us-south-1:a/account-1::instance:"
                "r006-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
            )
        )
    )

    with pytest.raises(PrototypeMismatch, match="instance CRN"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )


@pytest.mark.parametrize("foreign_tag", ("run:foreign", "project:foreign"))
def test_boot_volume_with_foreign_ownership_tag_is_not_adopted(foreign_tag):
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    client.get_volume.return_value = _Response(
        _volume(
            user_tags=[
                "project:watch-authenticity",
                f"run:{valid_run_spec().run_id}",
                foreign_tag,
            ]
        )
    )

    with pytest.raises(PrototypeMismatch, match="boot volume"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )


def test_instance_tag_embedded_error_blocks_adoption():
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    tagger = _tagger()
    tagger.attach_tag.return_value = _Response(
        {"results": [{"resource_id": vpc_instance()["crn"], "is_error": True}]}
    )
    instance_crn = vpc_instance()["crn"]
    owned = tagger.list_tags.return_value
    tagger.list_tags.side_effect = lambda **kwargs: (
        _Response({"total_count": 0, "offset": 0, "items": []})
        if kwargs["attached_to"] == instance_crn
        else owned
    )

    with pytest.raises(AmbiguousProvisioning, match="tag attachment"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=tagger,
        )


def test_foreign_instance_tag_is_rejected_before_tag_mutation():
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    tagger = _tagger()
    instance_crn = vpc_instance()["crn"]
    owned = tagger.list_tags.return_value
    foreign = _Response(
        {
            "total_count": 1,
            "offset": 0,
            "items": [{"name": "run:foreign"}],
        }
    )
    tagger.list_tags.side_effect = lambda **kwargs: (
        foreign if kwargs["attached_to"] == instance_crn else owned
    )

    with pytest.raises(PrototypeMismatch, match="conflicting ownership tags"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=tagger,
        )

    tagger.attach_tag.assert_not_called()


def test_missing_instance_tags_are_attached_and_then_confirmed():
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    tagger = _tagger()
    instance_crn = vpc_instance()["crn"]
    owned = tagger.list_tags.return_value
    instance_reads = 0

    def tags(**kwargs):
        nonlocal instance_reads
        if kwargs["attached_to"] != instance_crn:
            return owned
        instance_reads += 1
        if instance_reads == 1:
            return _Response({"total_count": 0, "offset": 0, "items": []})
        return owned

    tagger.list_tags.side_effect = tags

    ensure_single_instance(
        client,
        valid_run_spec(),
        build_instance_prototype(valid_run_spec(), safe_user_data()),
        tagger=tagger,
    )

    tagger.attach_tag.assert_called_once()
    assert instance_reads == 2


def test_instance_is_validated_before_any_tag_mutation():
    _seed_intent()
    events: list[str] = []
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    client.get_instance_initialization.side_effect = lambda *_args: (
        events.append("profile") or _Response({})
    )
    tagger = _tagger()
    original_attach = tagger.attach_tag.return_value
    tagger.attach_tag.side_effect = lambda **_kwargs: (
        events.append("tag") or original_attach
    )

    with pytest.raises(PrototypeMismatch, match="trusted profile"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=tagger,
        )

    assert events == ["profile"]
    tagger.attach_tag.assert_not_called()


def test_trusted_profile_mismatch_is_not_adopted():
    _seed_intent()
    client = _client()
    client.list_instances.return_value = _page(vpc_instance())
    client.get_instance_initialization.return_value = _Response(
        {"default_trusted_profile": {"auto_link": True, "target": {"id": "wrong"}}}
    )

    with pytest.raises(PrototypeMismatch, match="trusted profile"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )


def test_duplicate_discovered_after_create_is_quarantined(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    created = vpc_instance()
    duplicate = vpc_instance(id="r006-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    client.list_instances.side_effect = [_page(), _page(created, duplicate)]
    client.create_instance.return_value = _Response(created)

    with pytest.raises(AmbiguousProvisioning, match="multiple instances"):
        ensure_single_instance(
            client,
            valid_run_spec(),
            build_instance_prototype(valid_run_spec(), safe_user_data()),
            tagger=_tagger(),
        )
    client.delete_instance.assert_not_called()
    created_record = json.loads(
        (ibm_vpc._RECOVERY_DIR / f"{valid_run_spec().run_id}-created.json").read_text()
    )
    assert created_record["instance_id"] == created["id"]
    assert created_record["instance_name"] == created["name"]
    assert ibm_vpc._load_cleanup(valid_run_spec(), str(created["id"])) == (
        "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        valid_run_spec().public_gateway_id,
        valid_run_spec().public_gateway_floating_ip_id,
    )
    assert recover_created_instance_id(
        valid_run_spec(),
        build_instance_prototype(valid_run_spec(), safe_user_data()),
    ) == created["id"]


def test_created_instance_recovery_rejects_a_different_prototype():
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    changed_user_data = safe_user_data() + "# changed\n"
    changed_spec = replace(
        spec,
        bootstrap_sha256=hashlib.sha256(changed_user_data.encode()).hexdigest(),
    )

    with pytest.raises(AmbiguousProvisioning, match="does not match"):
        recover_created_instance_id(
            changed_spec,
            build_instance_prototype(changed_spec, changed_user_data),
        )


def test_wait_running_uses_bounded_polling(monkeypatch):
    client = MagicMock()
    client.get_instance.side_effect = [
        _Response(vpc_instance(status="pending")),
        _Response(vpc_instance(status="starting")),
        _Response(vpc_instance(status="running")),
    ]
    monkeypatch.setattr("infra.ibm_vpc.time.monotonic", lambda: 1.0)
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)

    ref = wait_running(client, _instance_id(), deadline=10.0)

    assert ref.status == "running"
    assert client.get_instance.call_count == 3
    timeouts = [
        call_.args[0]["timeout"]
        for call_ in client.set_http_config.call_args_list
        if "timeout" in call_.args[0]
    ]
    assert len(timeouts) == 3
    assert all(0 < timeout <= 5.0 for timeout in timeouts)


def test_deadline_client_preserves_and_restores_http_config(monkeypatch):
    class Client:
        def __init__(self):
            self.http_config = {"proxies": {"https": "proxy"}}
            self.observed = None

        def set_http_config(self, config):
            self.http_config = config

        def read(self):
            self.observed = dict(self.http_config)
            return "ok"

    client = Client()
    monkeypatch.setattr("infra.ibm_vpc.time.monotonic", lambda: 1.0)

    assert ibm_vpc._deadline_client(client, 3.0).read() == "ok"
    assert client.observed == {
        "proxies": {"https": "proxy"},
        "timeout": 2.0,
    }
    assert client.http_config == {"proxies": {"https": "proxy"}}


def test_deadline_client_interrupts_a_call_that_keeps_making_progress():
    class Client:
        http_config = {}

        def set_http_config(self, config):
            self.http_config = config

        def slow_trickle(self):
            while True:
                time.sleep(0.005)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="deadline expired"):
        ibm_vpc._deadline_client(Client(), started + 0.03).slow_trickle()

    assert time.monotonic() - started < 0.2


def test_wait_running_rejects_response_for_another_instance():
    client = MagicMock()
    client.get_instance.return_value = _Response(vpc_instance())

    with pytest.raises(AmbiguousProvisioning, match="different instance"):
        wait_running(
            client,
            "r006-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            deadline=_deadline(),
        )


def test_wait_running_retries_initial_provider_404(monkeypatch):
    client = MagicMock()
    client.get_instance.side_effect = [
        _NotFound(),
        _Response(vpc_instance(status="running")),
    ]
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)

    ref = wait_running(client, _instance_id(), deadline=_deadline())

    assert ref.status == "running"
    assert client.get_instance.call_count == 2


@pytest.mark.parametrize("state", ("failed", "stopped", "mystery"))
def test_wait_running_fails_closed_on_terminal_or_unknown_state(state):
    client = _client()
    client.get_instance.return_value = _Response(vpc_instance(status=state))

    with pytest.raises(ProvisioningFailed, match=state):
        wait_running(client, _instance_id(), deadline=_deadline())


def test_wait_running_quarantines_after_429_exhaustion(monkeypatch):
    client = _client()
    client.get_instance.side_effect = _TooManyRequests()
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("infra.ibm_vpc.time.monotonic", lambda: 1.0)

    with pytest.raises(AmbiguousProvisioning, match="rate limit"):
        wait_running(client, _instance_id(), deadline=10.0)


def test_polling_rejects_infinite_deadline():
    client = _client()
    with pytest.raises(ValueError, match="finite"):
        wait_running(client, _instance_id(), deadline=float("inf"))
    client.get_instance.assert_not_called()


def test_delete_waits_for_vsi_volume_and_gateway_404(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_subnet.return_value = _Response(_subnet(valid_run_spec().public_gateway_id))
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)

    delete_and_wait(
        client, valid_run_spec(), _instance_id(), deadline=_deadline(), tagger=_tagger()
    )

    client.delete_instance.assert_called_once_with(_instance_id())
    assert client.get_volume.call_count == 4
    assert client.mock_calls.index(call.delete_subnet(valid_run_spec().subnet_id)) < client.mock_calls.index(call.delete_public_gateway(valid_run_spec().public_gateway_id))
    assert client.get_public_gateway.call_count == 4
    client.delete_floating_ip.assert_called_once_with(
        valid_run_spec().public_gateway_floating_ip_id
    )
    assert client.get_floating_ip.call_count == 3


def test_instance_receipt_failure_does_not_block_other_cleanup(monkeypatch):
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]
    persist_deletion = ibm_vpc._persist_deletion

    def fail_instance_receipt(spec, resource_type, resource_id):
        if resource_type == "instance":
            raise OSError("receipt fsync failed")
        persist_deletion(spec, resource_type, resource_id)

    monkeypatch.setattr(
        "infra.ibm_vpc._persist_deletion", fail_instance_receipt
    )

    with pytest.raises(OSError, match="receipt fsync failed"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_volume.assert_called_once()
    client.delete_public_gateway.assert_called_once()
    client.delete_floating_ip.assert_called_once()


def test_instance_delete_failure_blocks_dependent_cleanup():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
    ]
    client.delete_instance.side_effect = _Forbidden()
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    with pytest.raises(_Forbidden):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_volume.assert_not_called()
    client.delete_subnet.assert_not_called()
    client.delete_public_gateway.assert_not_called()
    client.delete_floating_ip.assert_not_called()


def test_cleanup_rejects_provider_record_for_a_different_instance(tmp_path):
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
        _NotFound(),
    ]
    requested_id = "r006-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    with pytest.raises(PrototypeMismatch, match="different instance"):
        delete_and_wait(
            client,
            valid_run_spec(),
            requested_id,
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_instance.assert_not_called()


def test_expired_deadline_blocks_the_next_resource_delete(monkeypatch):
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.return_value = _Response(_volume())
    monkeypatch.setattr(
        "infra.ibm_vpc.time.monotonic",
        lambda: 2.0 if client.get_instance.call_count >= 3 else 1.0,
    )

    with pytest.raises(CleanupFailures) as caught:
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=1.5,
            tagger=_tagger(),
        )

    assert [type(error) for error in caught.value.errors] == [
        TimeoutError,
        TimeoutError,
        TimeoutError,
    ]
    client.delete_instance.assert_called_once_with(_instance_id())
    client.delete_volume.assert_not_called()
    client.delete_subnet.assert_not_called()


def test_read_failure_after_delete_is_persisted_as_ambiguous(tmp_path):
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _Forbidden(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = _NotFound()
    client.get_floating_ip.side_effect = _NotFound()

    with pytest.raises(AmbiguousProvisioning, match="provider read failed"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    assert list((tmp_path / "shared-state").glob("ambiguous-delete-*.json"))
    client.delete_volume.assert_not_called()


def test_lost_delete_response_still_polls_for_404(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_subnet.return_value = _Response(_subnet(valid_run_spec().public_gateway_id))
    client.delete_instance.side_effect = ReadTimeout()
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client, valid_run_spec(), _instance_id(), deadline=_deadline(), tagger=_tagger()
    )

    assert client.get_instance.call_count == 3


def test_delete_conflict_still_polls_and_cleans_metered_resources(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance(status="deleting")),
        _Response(vpc_instance(status="deleting")),
        _NotFound(),
    ]
    client.delete_instance.side_effect = _Conflict()
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        valid_run_spec(),
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_volume.assert_called_once()
    client.delete_public_gateway.assert_called_once()


def test_delete_conflict_retries_delete_while_resource_still_exists(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance(status="deleting")),
        _Response(vpc_instance(status="deleting")),
        _Response(vpc_instance(status="deleting")),
        _Response(vpc_instance(status="deleting")),
        _NotFound(),
    ]
    client.delete_instance.side_effect = [_Conflict(), None]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        valid_run_spec(),
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    assert client.delete_instance.call_count == 2


def test_instance_delete_retry_rechecks_ownership_tags(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_accepted_instance(spec, prototype, _instance_id())
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
    ]
    client.delete_instance.side_effect = _Conflict()
    tagger = _tagger()
    tagger.list_tags.return_value = _Response(
        {
            "total_count": 2,
            "offset": 0,
            "items": [
                {"name": "project:watch-authenticity"},
                {"name": "run:foreign"},
            ],
        }
    )

    with pytest.raises(PrototypeMismatch, match="conflicting ownership"):
        ibm_vpc._delete_and_confirm(
            client.delete_instance,
            client.get_instance,
            _instance_id(),
            deadline=_deadline(),
            description="instance deletion",
            before_delete=lambda: ibm_vpc._revalidate_instance_delete(
                client,
                tagger,
                spec,
                _instance_id(),
                allow_local_intent=True,
            ),
        )

    client.delete_instance.assert_called_once_with(_instance_id())


def test_instance_delete_retry_rejects_present_conflicting_crn(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_accepted_instance(spec, prototype, _instance_id())
    client = _client()
    foreign = vpc_instance(
        crn=(
            "crn:v1:bluemix:public:is:us-south-1:a/account-2::instance:"
            f"{_instance_id()}"
        )
    )
    client.get_instance.side_effect = [_Response(foreign), _Response(foreign)]
    client.delete_instance.side_effect = _Conflict()

    with pytest.raises(PrototypeMismatch, match="instance CRN"):
        ibm_vpc._delete_and_confirm(
            client.delete_instance,
            client.get_instance,
            _instance_id(),
            deadline=_deadline(),
            description="instance deletion",
            before_delete=lambda: ibm_vpc._revalidate_instance_delete(
                client,
                _tagger(),
                spec,
                _instance_id(),
                allow_local_intent=True,
            ),
        )

    client.delete_instance.assert_called_once_with(_instance_id())


@pytest.mark.parametrize("bad_crn", ("", None, 7))
def test_instance_delete_retry_rejects_present_malformed_crn(
    monkeypatch, bad_crn
):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_accepted_instance(spec, prototype, _instance_id())
    client = _client()
    malformed = vpc_instance(crn=bad_crn)
    client.get_instance.side_effect = [_Response(malformed), _Response(malformed)]
    client.delete_instance.side_effect = _Conflict()

    with pytest.raises(PrototypeMismatch, match="instance CRN"):
        ibm_vpc._delete_and_confirm(
            client.delete_instance,
            client.get_instance,
            _instance_id(),
            deadline=_deadline(),
            description="instance deletion",
            before_delete=lambda: ibm_vpc._revalidate_instance_delete(
                client,
                _tagger(),
                spec,
                _instance_id(),
                allow_local_intent=True,
            ),
        )

    client.delete_instance.assert_called_once_with(_instance_id())


def test_volume_delete_retry_rechecks_provider_ownership(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    spec = valid_run_spec()
    client = _client()
    foreign = _volume(user_tags=["project:foreign", f"run:{spec.run_id}"])
    client.get_volume.side_effect = [
        _Response(foreign),
        _Response(foreign),
    ]
    client.delete_volume.side_effect = _Conflict()
    volume_id = "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"

    with pytest.raises(PrototypeMismatch, match="boot volume"):
        ibm_vpc._delete_and_confirm(
            client.delete_volume,
            client.get_volume,
            volume_id,
            deadline=_deadline(),
            description="boot volume deletion",
            before_delete=lambda: ibm_vpc._revalidate_volume_delete(
                client, spec, volume_id
            ),
        )

    client.delete_volume.assert_called_once_with(volume_id)


def test_gateway_delete_retry_rechecks_empty_attachments(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    spec = valid_run_spec()
    client = _client()
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
    ]
    other = _subnet(spec.public_gateway_id)
    other["id"] = "0717-foreign"
    client.list_subnets.return_value = _Response(
        {"subnets": [other], "total_count": 1}
    )
    client.delete_public_gateway.side_effect = _Conflict()

    with pytest.raises(PrototypeMismatch, match="unauthorized subnet"):
        ibm_vpc._delete_and_confirm(
            client.delete_public_gateway,
            client.get_public_gateway,
            spec.public_gateway_id,
            deadline=_deadline(),
            description="Public Gateway deletion",
            before_delete=lambda: ibm_vpc._revalidate_gateway_delete(
                client,
                _tagger(),
                spec,
                floating_ip_absent=False,
            ),
        )

    client.delete_public_gateway.assert_called_once_with(spec.public_gateway_id)


def test_gateway_attachments_reject_a_subnet_repeated_across_pages():
    spec = valid_run_spec()
    client = _client()
    subnet = _subnet(spec.public_gateway_id)
    client.list_subnets.side_effect = [
        _Response(
            {
                "subnets": [subnet],
                "total_count": 2,
                "next": {"start": "second-page"},
            }
        ),
        _Response({"subnets": [subnet], "total_count": 2}),
    ]

    with pytest.raises(AmbiguousProvisioning, match="repeated ID"):
        ibm_vpc._validate_gateway_attachments(client, spec, {spec.subnet_id})


def test_gateway_attachments_reject_total_count_drift_between_pages():
    spec = valid_run_spec()
    client = _client()
    client.list_subnets.side_effect = [
        _Response(
            {
                "subnets": [_subnet(spec.public_gateway_id)],
                "total_count": 2,
                "next": {"start": "second-page"},
            }
        ),
        _Response({"subnets": [], "total_count": 1}),
    ]

    with pytest.raises(AmbiguousProvisioning, match="total count"):
        ibm_vpc._validate_gateway_attachments(client, spec, {spec.subnet_id})


def test_subnet_delete_conflict_retries_until_absent(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.delete_subnet.side_effect = [_Conflict(), None]
    subnet_reads = 0

    def subnet_state(_subnet_id):
        nonlocal subnet_reads
        subnet_reads += 1
        if subnet_reads < 5:
            return _Response(_subnet(valid_run_spec().public_gateway_id))
        client.list_subnets.return_value = _Response(
            {"subnets": [], "total_count": 0}
        )
        raise _NotFound()

    client.get_subnet.side_effect = subnet_state
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        valid_run_spec(),
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    assert subnet_reads == 5
    client.delete_public_gateway.assert_called_once()


def test_delete_404_still_requires_get_404_confirmation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _NotFound(),
        _NotFound(),
    ]
    client.delete_instance.side_effect = _NotFound()
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        valid_run_spec(),
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    assert client.get_instance.call_count == 3


def test_existing_boot_volume_is_deleted_and_confirmed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_subnet.return_value = _Response(_subnet(valid_run_spec().public_gateway_id))
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client, valid_run_spec(), _instance_id(), deadline=_deadline(), tagger=_tagger()
    )

    client.delete_volume.assert_called_once_with(
        "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    )
    assert client.get_volume.call_count == 4


def test_cleanup_rejects_mismatched_trusted_profile(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.get_instance.return_value = _Response(vpc_instance())
    client.get_instance_initialization.return_value = _Response(
        {"default_trusted_profile": {"auto_link": True, "target": {"id": "other"}}}
    )

    with pytest.raises(PrototypeMismatch, match="trusted profile"):
        delete_and_wait(
            client, valid_run_spec(), _instance_id(), deadline=_deadline(), tagger=_tagger()
        )
    client.delete_instance.assert_not_called()
    client.delete_subnet.assert_not_called()


def test_cleanup_deletes_locally_created_vsi_despite_trusted_profile_mismatch():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_instance_initialization.return_value = _Response(
        {"default_trusted_profile": {"auto_link": True, "target": {"id": "other"}}}
    )
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.get_instance_initialization.assert_not_called()
    client.delete_instance.assert_called_once_with(_instance_id())
    client.delete_volume.assert_called_once()
    client.delete_public_gateway.assert_called_once()
    client.delete_floating_ip.assert_called_once()


def test_cleanup_uses_durable_crn_when_later_instance_read_omits_it():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    without_crn = vpc_instance()
    without_crn.pop("crn")
    client.get_instance.side_effect = [
        _Response(without_crn),
        _Response(without_crn),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_instance.assert_called_once_with(_instance_id())


def test_cleanup_rejects_present_conflicting_crn_despite_local_intent():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.return_value = _Response(
        vpc_instance(
            crn=(
                "crn:v1:bluemix:public:is:us-south-1:a/account-2::instance:"
                f"{_instance_id()}"
            )
        )
    )

    with pytest.raises(PrototypeMismatch, match="instance CRN"):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_instance.assert_not_called()


@pytest.mark.parametrize("bad_crn", ("", None, 7))
def test_cleanup_rejects_present_malformed_crn_despite_local_intent(bad_crn):
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.return_value = _Response(vpc_instance(crn=bad_crn))

    with pytest.raises(PrototypeMismatch, match="instance CRN"):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_instance.assert_not_called()


def test_accepted_identity_derives_crn_when_provider_never_returns_it():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_accepted_instance(spec, prototype, _instance_id())
    client = _client()
    without_crn = vpc_instance()
    without_crn.pop("crn")
    client.get_instance.side_effect = [
        _Response(without_crn),
        _Response(without_crn),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_instance.assert_called_once_with(_instance_id())


def test_accepted_identity_rejects_account_switch():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_accepted_instance(spec, prototype, _instance_id())
    other_account = replace(
        spec,
        account_id="account-2",
        cos_service_instance_id=spec.cos_service_instance_id.replace(
            "a/account-1:", "a/account-2:"
        ),
    )

    with pytest.raises(AmbiguousProvisioning, match="accepted instance identity"):
        ibm_vpc._load_instance_crn(other_account, _instance_id())


def test_cleanup_rejects_instance_without_run_tags(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.get_instance.return_value = _Response(vpc_instance())
    tagger = _tagger()
    tagger.list_tags.return_value = _Response(
        {"total_count": 0, "offset": 0, "items": []}
    )

    with pytest.raises(PrototypeMismatch, match="neither run tags nor local create intent"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=tagger,
        )

    client.delete_instance.assert_not_called()


def test_cleanup_accepts_matching_durable_intent_when_instance_tags_are_missing(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    _seed_intent()
    ibm_vpc._persist_created_instance(
        valid_run_spec(),
        build_instance_prototype(valid_run_spec(), safe_user_data()),
        vpc_instance(),
    )
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]
    tagger = _tagger()
    instance_crn = vpc_instance()["crn"]
    owned = tagger.list_tags.return_value
    tagger.list_tags.side_effect = lambda **kwargs: (
        _Response({"total_count": 0, "offset": 0, "items": []})
        if kwargs["attached_to"] == instance_crn
        else owned
    )

    delete_and_wait(
        client,
        valid_run_spec(),
        _instance_id(),
        deadline=_deadline(),
        tagger=tagger,
    )

    client.delete_instance.assert_called_once_with(_instance_id())


def test_cleanup_deletes_locally_created_instance_after_post_create_mismatch():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    created = vpc_instance()
    ibm_vpc._persist_created_instance(spec, prototype, created)
    client = _client()
    mismatched = vpc_instance(profile={"name": "changed-after-create"})
    client.get_instance.side_effect = [
        _Response(mismatched),
        _Response(mismatched),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_instance.assert_called_once_with(_instance_id())
    client.delete_volume.assert_called_once()
    client.delete_public_gateway.assert_called_once()
    client.delete_floating_ip.assert_called_once_with(
        spec.public_gateway_floating_ip_id
    )


def test_cleanup_rejects_replacement_id_even_with_matching_name_and_intent():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    replacement_id = "r006-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    replacement = vpc_instance(
        id=replacement_id,
        crn=(
            "crn:v1:bluemix:public:is:us-south-1:a/account-1::instance:"
            f"{replacement_id}"
        ),
    )
    client = _client()
    client.get_instance.return_value = _Response(replacement)
    tagger = _tagger()
    replacement_crn = replacement["crn"]
    owned = tagger.list_tags.return_value
    tagger.list_tags.side_effect = lambda **kwargs: (
        _Response({"total_count": 0, "offset": 0, "items": []})
        if kwargs["attached_to"] == replacement_crn
        else owned
    )

    with pytest.raises(PrototypeMismatch, match="intent does not own"):
        delete_and_wait(
            client,
            spec,
            replacement_id,
            deadline=_deadline(),
            tagger=tagger,
        )

    client.delete_instance.assert_not_called()


def test_cleanup_rejects_tampered_local_intent_digest(tmp_path):
    _seed_intent()
    intent_path = (
        tmp_path
        / "shared-state"
        / f"{valid_run_spec().run_id}-prototype.json"
    )
    record = json.loads(intent_path.read_text(encoding="utf-8"))
    record["prototype_sha256"] = "f" * 64
    intent_path.write_text(json.dumps(record), encoding="utf-8")
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
        _NotFound(),
    ]
    tagger = _tagger()
    instance_crn = vpc_instance()["crn"]
    owned = tagger.list_tags.return_value
    tagger.list_tags.side_effect = lambda **kwargs: (
        _Response({"total_count": 0, "offset": 0, "items": []})
        if kwargs["attached_to"] == instance_crn
        else owned
    )

    with pytest.raises(PrototypeMismatch, match="intent does not own"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=tagger,
        )

    client.delete_instance.assert_not_called()


@pytest.mark.parametrize("foreign_tag", ("run:foreign", "project:foreign"))
def test_cleanup_does_not_use_intent_to_override_conflicting_ownership_tag(
    monkeypatch, tmp_path, foreign_tag
):
    monkeypatch.chdir(tmp_path)
    _seed_intent()
    client = _client()
    client.get_instance.return_value = _Response(vpc_instance())
    tagger = _tagger()
    tagger.list_tags.return_value = _Response(
        {
            "total_count": 3,
            "offset": 0,
            "items": [
                {"name": "project:watch-authenticity"},
                {"name": f"run:{valid_run_spec().run_id}"},
                {"name": foreign_tag},
            ],
        }
    )

    with pytest.raises(PrototypeMismatch, match="conflicting ownership tags"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=tagger,
        )

    client.delete_instance.assert_not_called()


def test_cleanup_rejects_gateway_without_provider_ownership(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_subnet.return_value = _Response(_subnet(valid_run_spec().public_gateway_id))
    client.get_public_gateway.return_value = _Response(
        _gateway(vpc={"id": "r006-foreign"})
    )

    with pytest.raises(PrototypeMismatch, match="ownership"):
        delete_and_wait(
            client, valid_run_spec(), _instance_id(), deadline=_deadline(), tagger=_tagger()
        )
    client.delete_public_gateway.assert_not_called()


def test_cleanup_rejects_gateway_crn_for_another_resource(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.return_value = _Response(
        _gateway(
            crn=(
                "crn:v1:bluemix:public:is:us-south-1:a/account-1::public-gateway:"
                "r006-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
            )
        )
    )

    with pytest.raises(PrototypeMismatch, match="public-gateway CRN"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_instance.assert_called_once()
    client.delete_subnet.assert_not_called()


def test_cleanup_rejects_foreign_gateway_floating_ip():
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.return_value = _Response(
        _gateway(floating_ip={"id": "r006-foreign"})
    )

    with pytest.raises(PrototypeMismatch, match="Floating IP differs"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_public_gateway.assert_not_called()
    client.delete_floating_ip.assert_called_once_with(
        valid_run_spec().public_gateway_floating_ip_id
    )


def test_cleanup_rejects_floating_ip_reassigned_to_another_target():
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_floating_ip.side_effect = [
        _Response(_floating_ip(
            target={
                "id": "r006-foreign",
                "crn": (
                    "crn:v1:bluemix:public:is:us-south-1:a/account-1::public-gateway:"
                    "r006-foreign"
                ),
                "resource_type": "public_gateway",
                "deleted": {},
            },
        )),
    ]

    with pytest.raises(PrototypeMismatch, match="target differs"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_floating_ip.assert_not_called()


def test_cleanup_rechecks_floating_ip_target_immediately_before_delete():
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_floating_ip.side_effect = [
        _Response(_floating_ip()),
        _Response(
            _floating_ip(
                target={
                    "id": "r006-foreign",
                    "crn": (
                        "crn:v1:bluemix:public:is:us-south-1:a/account-1::"
                        "public-gateway:r006-foreign"
                    ),
                    "resource_type": "public_gateway",
                }
            )
        ),
    ]

    with pytest.raises(PrototypeMismatch, match="target differs"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_floating_ip.assert_not_called()


def test_cleanup_rechecks_floating_ip_target_before_delete_retry(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
        _NotFound(),
    ]
    foreign = _floating_ip(
        target={
            "id": "r006-foreign",
            "crn": (
                "crn:v1:bluemix:public:is:us-south-1:a/account-1::"
                "public-gateway:r006-foreign"
            ),
            "resource_type": "public_gateway",
        }
    )
    client.get_floating_ip.side_effect = [
        _Response(_floating_ip()),
        _Response(_floating_ip()),
        _Response(foreign),
        _Response(foreign),
    ]
    client.delete_floating_ip.side_effect = _Conflict()

    with pytest.raises(PrototypeMismatch, match="target differs"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_floating_ip.assert_called_once_with(
        valid_run_spec().public_gateway_floating_ip_id
    )


def test_cleanup_retries_transient_floating_ip_predelete_validation(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_floating_ip.side_effect = [
        _Response(_floating_ip()),
        ReadTimeout(),
        _Response(_floating_ip(target=None)),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        valid_run_spec(),
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_floating_ip.assert_called_once_with(
        valid_run_spec().public_gateway_floating_ip_id
    )


def test_tag_api_404_does_not_prove_floating_ip_absent(monkeypatch):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    client = _client()
    client.get_floating_ip.side_effect = [
        _Response(_floating_ip()),
        _Response(_floating_ip()),
        _NotFound(),
    ]
    tagger = _tagger()
    valid_tags = tagger.list_tags.return_value
    tagger.list_tags.side_effect = [_NotFound(), valid_tags]
    spec = valid_run_spec()

    ibm_vpc._delete_and_confirm(
        client.delete_floating_ip,
        client.get_floating_ip,
        spec.public_gateway_floating_ip_id,
        deadline=_deadline(),
        description="Floating IP deletion",
        before_delete=lambda: ibm_vpc._validate_floating_ip_ownership(
            client,
            tagger,
            spec,
            spec.public_gateway_floating_ip_id,
            gateway_may_be_deleted=True,
        ),
    )

    client.delete_floating_ip.assert_called_once_with(
        spec.public_gateway_floating_ip_id
    )


def test_cleanup_rejects_floating_ip_crn_for_another_resource():
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]
    client.get_floating_ip.return_value = _Response(
        _floating_ip(
            crn=(
                "crn:v1:bluemix:public:is:us-south-1:a/account-1::floating-ip:"
                "r006-foreign"
            )
        )
    )

    with pytest.raises(PrototypeMismatch, match="floating-ip CRN"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_public_gateway.assert_called_once_with(
        valid_run_spec().public_gateway_id
    )
    client.delete_floating_ip.assert_not_called()


def test_cleanup_rejects_gateway_attached_to_another_subnet(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_subnet.return_value = _Response(_subnet(valid_run_spec().public_gateway_id))
    client.get_public_gateway.return_value = _Response(_gateway())
    other = _subnet(valid_run_spec().public_gateway_id)
    other["id"] = "0717-other"
    client.list_subnets.return_value = _Response(
        {
            "subnets": [_subnet(valid_run_spec().public_gateway_id), other],
            "total_count": 2,
        }
    )

    with pytest.raises(PrototypeMismatch, match="unauthorized subnet"):
        delete_and_wait(
            client, valid_run_spec(), _instance_id(), deadline=_deadline(), tagger=_tagger()
        )
    client.delete_subnet.assert_not_called()


def test_cleanup_rechecks_gateway_attachments_after_detach(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _NotFound(),
        _NotFound(),
    ]
    client.get_public_gateway.return_value = _Response(_gateway())
    other = _subnet(valid_run_spec().public_gateway_id)
    other["id"] = "0717-concurrent"
    client.list_subnets.side_effect = [
        _Response(
            {
                "subnets": [_subnet(valid_run_spec().public_gateway_id)],
                "total_count": 1,
            }
        ),
        _Response({"subnets": [other], "total_count": 1}),
    ]

    with pytest.raises(AmbiguousProvisioning, match="subnet inventory changed"):
        delete_and_wait(
            client,
            valid_run_spec(),
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_subnet.assert_not_called()
    client.delete_public_gateway.assert_not_called()


def test_delete_does_not_report_success_before_vsi_404(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _client()
    client.get_instance.return_value = _Response(vpc_instance(status="deleting"))
    client.get_subnet.return_value = _Response(_subnet(valid_run_spec().public_gateway_id))
    monkeypatch.setattr(
        "infra.ibm_vpc.time.monotonic",
        lambda: 2.0 if client.get_instance.call_count >= 2 else 1.0,
    )
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError):
        delete_and_wait(
            client, valid_run_spec(), _instance_id(), deadline=1.5, tagger=_tagger()
        )
    client.delete_public_gateway.assert_not_called()


def test_delete_resumes_from_persisted_context_after_lost_response(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first = _client()
    first.get_instance.return_value = _Response(vpc_instance(status="deleting"))
    first.get_subnet.return_value = _Response(_subnet(valid_run_spec().public_gateway_id))
    monkeypatch.setattr(
        "infra.ibm_vpc.time.monotonic",
        lambda: 2.0 if first.get_instance.call_count >= 2 else 1.0,
    )
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    with pytest.raises(TimeoutError):
        delete_and_wait(
            first, valid_run_spec(), _instance_id(), deadline=1.5, tagger=_tagger()
        )

    second = MagicMock()
    second.list_volumes.return_value = _Response(
        {"volumes": [], "total_count": 0}
    )
    second.get_instance.side_effect = _NotFound()
    second.get_volume.side_effect = _NotFound()
    second.get_subnet.return_value = _Response(_subnet(valid_run_spec().public_gateway_id))
    second.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]
    second.get_floating_ip.side_effect = [
        _Response(_floating_ip(target=None)),
        _Response(_floating_ip(target=None)),
        _NotFound(),
    ]
    second.list_subnets.return_value = _Response(
        {"subnets": [_subnet(valid_run_spec().public_gateway_id)], "total_count": 1}
    )
    second.delete_subnet.side_effect = lambda _subnet_id: (
        setattr(second.get_subnet, "side_effect", _NotFound()),
        setattr(
            second.list_subnets,
            "return_value",
            _Response({"subnets": [], "total_count": 0}),
        ),
    )
    monkeypatch.setattr("infra.ibm_vpc.time.monotonic", lambda: 1.0)
    delete_and_wait(
        second, valid_run_spec(), _instance_id(), deadline=61.0, tagger=_tagger()
    )

    second.delete_instance.assert_not_called()
    assert second.get_volume.call_args_list == [
        call("r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
        call("r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
    ]
    second.delete_public_gateway.assert_called_once_with(valid_run_spec().public_gateway_id)
    assert (
        tmp_path
        / "shared-state"
        / f"{valid_run_spec().run_id}-deleted-instance.json"
    ).exists()


def test_resume_revalidates_volume_before_delete(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first = _client()
    first.get_instance.return_value = _Response(vpc_instance(status="deleting"))
    monkeypatch.setattr(
        "infra.ibm_vpc.time.monotonic",
        lambda: 2.0 if first.get_instance.call_count >= 2 else 1.0,
    )
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    with pytest.raises(TimeoutError):
        delete_and_wait(
            first,
            valid_run_spec(),
            _instance_id(),
            deadline=1.5,
            tagger=_tagger(),
        )

    second = _client()
    second.get_instance.side_effect = _NotFound()
    second.get_volume.return_value = _Response(
        _volume(resource_group={"id": "r006-foreign"})
    )
    second.get_subnet.return_value = _Response(_subnet())
    second.list_subnets.return_value = _Response(
        {"subnets": [_subnet()], "total_count": 1}
    )
    second.delete_subnet.side_effect = lambda _subnet_id: (
        setattr(second.get_subnet, "side_effect", _NotFound()),
        setattr(
            second.list_subnets,
            "return_value",
            _Response({"subnets": [], "total_count": 0}),
        ),
    )
    second.get_public_gateway.side_effect = _NotFound()
    second.get_floating_ip.side_effect = _NotFound()
    monkeypatch.setattr("infra.ibm_vpc.time.monotonic", lambda: 1.0)

    with pytest.raises(PrototypeMismatch, match="boot volume"):
        delete_and_wait(
            second,
            valid_run_spec(),
            _instance_id(),
            deadline=61.0,
            tagger=_tagger(),
        )

    second.delete_volume.assert_not_called()


def test_resume_accepts_gateway_already_deleted_after_lost_response(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first = _client()
    first.get_instance.return_value = _Response(vpc_instance(status="deleting"))
    first.get_subnet.return_value = _Response(_subnet(valid_run_spec().public_gateway_id))
    monkeypatch.setattr(
        "infra.ibm_vpc.time.monotonic",
        lambda: 2.0 if first.get_instance.call_count >= 2 else 1.0,
    )
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    with pytest.raises(TimeoutError):
        delete_and_wait(
            first, valid_run_spec(), _instance_id(), deadline=1.5, tagger=_tagger()
        )

    second = MagicMock()
    second.list_volumes.return_value = _Response(
        {"volumes": [], "total_count": 0}
    )
    second.get_instance.side_effect = _NotFound()
    second.get_volume.side_effect = _NotFound()
    second.get_subnet.return_value = _Response(_subnet())
    second.list_subnets.return_value = _Response(
        {"subnets": [_subnet()], "total_count": 1}
    )
    second.delete_subnet.side_effect = lambda _subnet_id: (
        setattr(second.get_subnet, "side_effect", _NotFound()),
        setattr(
            second.list_subnets,
            "return_value",
            _Response({"subnets": [], "total_count": 0}),
        ),
    )
    second.get_public_gateway.side_effect = _NotFound()
    second.get_floating_ip.side_effect = _NotFound()
    monkeypatch.setattr("infra.ibm_vpc.time.monotonic", lambda: 1.0)

    delete_and_wait(
        second, valid_run_spec(), _instance_id(), deadline=61.0, tagger=_tagger()
    )

    second.delete_public_gateway.assert_not_called()
    assert (tmp_path / "shared-state" / f"{valid_run_spec().run_id}-deleted-public-gateway.json").exists()


def test_corrupt_cleanup_record_falls_back_and_reports_after_cleanup(tmp_path):
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    ibm_vpc._persist_cleanup(
        spec,
        _instance_id(),
        "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        spec.public_gateway_id,
        spec.public_gateway_floating_ip_id,
    )
    cleanup_path = tmp_path / "shared-state" / f"{spec.run_id}-cleanup.json"
    cleanup_path.write_text("{", encoding="utf-8")
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.side_effect = _NotFound()
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    with pytest.raises(AmbiguousProvisioning, match="cleanup context"):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_public_gateway.assert_called_once()
    assert list(
        (tmp_path / "shared-state").glob(
            f"{spec.run_id}-cleanup.corrupt-*.json"
        )
    )


def test_semantically_invalid_cleanup_record_falls_back_and_reports(tmp_path):
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    ibm_vpc._persist_cleanup(
        spec,
        _instance_id(),
        "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        spec.public_gateway_id,
        spec.public_gateway_floating_ip_id,
    )
    cleanup_path = tmp_path / "shared-state" / f"{spec.run_id}-cleanup.json"
    record = json.loads(cleanup_path.read_text(encoding="utf-8"))
    record["public_gateway_id"] = "r006-replacement"
    cleanup_path.write_text(json.dumps(record), encoding="utf-8")
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.side_effect = _NotFound()
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    with pytest.raises(AmbiguousProvisioning, match="does not match"):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_subnet.assert_called_once_with(spec.subnet_id)
    client.delete_public_gateway.assert_called_once_with(spec.public_gateway_id)
    assert list(
        (tmp_path / "shared-state").glob(
            f"{spec.run_id}-cleanup.corrupt-*.json"
        )
    )


def test_corrupt_accepted_record_falls_back_and_reports_after_cleanup(tmp_path):
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    accepted_path = tmp_path / "shared-state" / f"{spec.run_id}-accepted.json"
    accepted_path.write_text("{", encoding="utf-8")
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.side_effect = _NotFound()
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    with pytest.raises(AmbiguousProvisioning, match="accepted instance"):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_public_gateway.assert_called_once()
    assert list(
        (tmp_path / "shared-state").glob(
            f"{spec.run_id}-accepted.corrupt-*.json"
        )
    )


def test_missing_floating_ip_does_not_hide_existing_detached_gateway():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_subnet.return_value = _Response(_subnet())
    client.list_subnets.return_value = _Response(
        {"subnets": [_subnet()], "total_count": 1}
    )
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]
    client.get_floating_ip.side_effect = _NotFound()

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_public_gateway.assert_called_once_with(
        spec.public_gateway_id
    )
    client.delete_floating_ip.assert_not_called()


def test_stale_subnet_gateway_reference_does_not_block_floating_ip_cleanup():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.side_effect = _NotFound()
    client.get_subnet.return_value = _Response(_subnet(spec.public_gateway_id))
    client.get_public_gateway.side_effect = _NotFound()

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_subnet.assert_called_once_with(spec.subnet_id)
    client.delete_public_gateway.assert_not_called()
    client.delete_floating_ip.assert_called_once_with(
        spec.public_gateway_floating_ip_id
    )


def test_missing_subnet_does_not_block_gateway_and_floating_ip_cleanup():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.side_effect = _NotFound()
    client.get_subnet.side_effect = _NotFound()
    client.list_subnets.return_value = _Response({"subnets": [], "total_count": 0})
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_public_gateway.assert_called_once_with(spec.public_gateway_id)
    client.delete_floating_ip.assert_called_once_with(
        spec.public_gateway_floating_ip_id
    )
    assert client.get_subnet.call_count == 2


def test_floating_ip_probe_error_does_not_block_gateway_cleanup():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.side_effect = _NotFound()
    client.get_floating_ip.side_effect = [_NotFound(), _Forbidden()]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    with pytest.raises(_Forbidden):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_subnet.assert_called_once_with(spec.subnet_id)
    client.delete_public_gateway.assert_called_once_with(spec.public_gateway_id)
    client.delete_floating_ip.assert_not_called()


def test_subnet_receipt_failure_does_not_block_gateway_cleanup(monkeypatch):
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.side_effect = _NotFound()
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]
    persist_deletion = ibm_vpc._persist_deletion

    def fail_subnet_receipt(spec, resource_type, resource_id):
        if resource_type == "subnet":
            raise OSError("subnet receipt fsync failed")
        persist_deletion(spec, resource_type, resource_id)

    monkeypatch.setattr("infra.ibm_vpc._persist_deletion", fail_subnet_receipt)

    with pytest.raises(OSError, match="subnet receipt fsync failed"):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_public_gateway.assert_called_once_with(spec.public_gateway_id)


def test_cleanup_resolves_boot_volume_before_deleting_accepted_instance():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_accepted_instance(spec, prototype, _instance_id())
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.get_instance_volume_attachment.assert_called_once()
    client.delete_volume.assert_called_once_with(
        "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    )
    cleanup = json.loads(
        (ibm_vpc._RECOVERY_DIR / f"{spec.run_id}-cleanup.json").read_text()
    )
    assert cleanup["volume_id"] == "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def test_resume_deletes_floating_ip_after_gateway_is_already_deleted(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    second = MagicMock()
    second.list_volumes.return_value = _Response(
        {"volumes": [], "total_count": 0}
    )
    second.get_instance.side_effect = _NotFound()
    second.get_volume.side_effect = _NotFound()
    second.get_subnet.return_value = _Response(_subnet())
    second.list_subnets.return_value = _Response(
        {"subnets": [_subnet()], "total_count": 1}
    )
    second.delete_subnet.side_effect = lambda _subnet_id: (
        setattr(second.get_subnet, "side_effect", _NotFound()),
        setattr(
            second.list_subnets,
            "return_value",
            _Response({"subnets": [], "total_count": 0}),
        ),
    )
    second.get_public_gateway.side_effect = _NotFound()
    second.get_floating_ip.side_effect = [
        _Response(_floating_ip(target=None)),
        _Response(_floating_ip(target=None)),
        _NotFound(),
    ]
    monkeypatch.setattr("infra.ibm_vpc.time.monotonic", lambda: 1.0)

    delete_and_wait(
        second,
        spec,
        _instance_id(),
        deadline=61.0,
        tagger=_tagger(),
    )

    second.delete_public_gateway.assert_not_called()
    second.delete_floating_ip.assert_called_once_with(
        spec.public_gateway_floating_ip_id
    )


def test_resume_deletes_gateway_after_floating_ip_is_already_absent():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.side_effect = _NotFound()
    client.get_floating_ip.side_effect = _NotFound()
    gateway_without_ip = _gateway(floating_ip={})
    client.get_public_gateway.side_effect = [
        _Response(gateway_without_ip),
        _Response(gateway_without_ip),
        _Response(gateway_without_ip),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_public_gateway.assert_called_once_with(spec.public_gateway_id)


def test_partial_create_recovery_cleans_known_gateway_and_floating_ip(
    monkeypatch,
):
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    incomplete = vpc_instance(boot_volume_attachment={"id": "attachment-only"})
    first = _client()
    first.list_instances.return_value = _page()
    first.create_instance.return_value = _Response(incomplete)
    first.get_instance.return_value = _Response(incomplete)
    with pytest.raises(AmbiguousProvisioning, match="cleanup context"):
        ensure_single_instance(first, spec, prototype, tagger=_tagger())

    second = _client()
    second.get_instance.side_effect = _NotFound()
    second.list_volumes.side_effect = [
        _Response({"volumes": [_volume()], "total_count": 1}),
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [], "total_count": 0}),
    ]
    second.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    second.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        second,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    second.delete_instance.assert_not_called()
    second.delete_volume.assert_called_once_with(
        "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    )
    second.delete_public_gateway.assert_called_once_with(spec.public_gateway_id)
    second.delete_floating_ip.assert_called_once_with(
        spec.public_gateway_floating_ip_id
    )


def test_stale_volume_inventory_404_does_not_block_network_cleanup():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_accepted_instance(spec, prototype, _instance_id())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.list_volumes.side_effect = [
        _Response({"volumes": [_volume()], "total_count": 1}),
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [], "total_count": 0}),
    ]
    client.get_volume.side_effect = _NotFound()
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_volume.assert_not_called()
    client.delete_public_gateway.assert_called_once_with(spec.public_gateway_id)
    client.delete_floating_ip.assert_called_once_with(
        spec.public_gateway_floating_ip_id
    )


def test_known_volume_initial_404_is_rechecked_before_delete():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.side_effect = [
        _NotFound(),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_volume.assert_called_once_with(
        "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    )


def test_floating_ip_initial_404_is_rechecked_before_delete():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.side_effect = _NotFound()
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]
    client.get_floating_ip.side_effect = [
        _NotFound(),
        _Response(_floating_ip()),
        _Response(_floating_ip()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_floating_ip.assert_called_once_with(
        spec.public_gateway_floating_ip_id
    )


def test_ambiguous_volume_inventory_reports_after_network_cleanup():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_accepted_instance(spec, prototype, _instance_id())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.list_volumes.return_value = _Response({"volumes": "malformed", "total_count": 1})
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    with pytest.raises(AmbiguousProvisioning, match="volume list"):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_public_gateway.assert_called_once_with(spec.public_gateway_id)
    client.delete_floating_ip.assert_called_once_with(
        spec.public_gateway_floating_ip_id
    )


def test_known_volume_mismatch_reports_after_network_cleanup():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.return_value = _Response(_volume(capacity=999))
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    with pytest.raises(PrototypeMismatch, match="boot volume"):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_public_gateway.assert_called_once_with(spec.public_gateway_id)
    client.delete_floating_ip.assert_called_once_with(
        spec.public_gateway_floating_ip_id
    )


def test_transient_floating_ip_read_retries_after_gateway_cleanup():
    client = _client()
    client.get_instance.side_effect = [
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.side_effect = [
        _Response(_volume()),
        _Response(_volume()),
        _Response(_volume()),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]
    client.get_floating_ip.side_effect = [
        ReadTimeout(),
        _Response(_floating_ip(target=None)),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        valid_run_spec(),
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_public_gateway.assert_called_once()
    client.delete_floating_ip.assert_called_once()


def test_transient_initial_instance_read_reconciles_before_cleanup():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = [ReadTimeout(), _NotFound(), _NotFound()]
    client.get_volume.side_effect = _NotFound()
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    assert client.get_instance.call_count == 3
    client.delete_public_gateway.assert_called_once()


def test_initial_404_does_not_hide_a_late_visible_instance():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = [
        _NotFound(),
        _Response(vpc_instance()),
        _Response(vpc_instance()),
        _NotFound(),
    ]
    client.get_volume.side_effect = _NotFound()
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]

    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_instance.assert_called_once_with(_instance_id())


def test_exhausted_initial_instance_read_blocks_dependent_cleanup(monkeypatch):
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_volume.side_effect = _NotFound()
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]
    reader = MagicMock(side_effect=TimeoutError("unknown"))
    monkeypatch.setattr("infra.ibm_vpc._read_instance_for_cleanup", reader)

    with pytest.raises(TimeoutError, match="unknown"):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    assert reader.call_count == 1
    client.delete_volume.assert_not_called()
    client.delete_subnet.assert_not_called()
    client.delete_public_gateway.assert_not_called()


def test_next_cleanup_attempt_rediscovers_boot_volumes(monkeypatch):
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_accepted_instance(spec, prototype, _instance_id())
    volume_id = "r006-volume-visible-after-vsi"
    client = _client()
    client.list_volumes.side_effect = [
        _Response({"volumes": [_volume(id=volume_id)], "total_count": 1}),
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [], "total_count": 0}),
    ]
    client.get_volume.side_effect = [
        _Response(_volume(id=volume_id)),
        _Response(_volume(id=volume_id)),
        _NotFound(),
    ]
    client.get_public_gateway.side_effect = [
        _Response(_gateway()),
        _Response(_gateway()),
        _Response(_gateway()),
        _NotFound(),
    ]
    reader = MagicMock(side_effect=[TimeoutError("unknown"), None])
    monkeypatch.setattr("infra.ibm_vpc._read_instance_for_cleanup", reader)

    with pytest.raises(TimeoutError, match="unknown"):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )
    delete_and_wait(
        client,
        spec,
        _instance_id(),
        deadline=_deadline(),
        tagger=_tagger(),
    )

    client.delete_volume.assert_called_once_with(volume_id)
    assert client.list_volumes.call_count == 3


def test_gateway_read_failure_does_not_block_owned_floating_ip_cleanup():
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_volume.side_effect = _NotFound()
    client.get_public_gateway.side_effect = ReadTimeout()
    client.get_floating_ip.side_effect = [
        _Response(_floating_ip()),
        _Response(_floating_ip()),
        _NotFound(),
    ]

    with pytest.raises(ReadTimeout):
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    client.delete_public_gateway.assert_not_called()
    client.delete_floating_ip.assert_called_once_with(
        spec.public_gateway_floating_ip_id
    )


def test_cleanup_reports_every_independent_resource_failure(monkeypatch):
    _seed_intent()
    spec = valid_run_spec()
    prototype = build_instance_prototype(spec, safe_user_data())
    ibm_vpc._persist_created_instance(spec, prototype, vpc_instance())
    client = _client()
    client.get_instance.side_effect = _NotFound()
    client.get_floating_ip.side_effect = [
        _Response(_floating_ip()),
        _Response(_floating_ip()),
    ]
    client.delete_floating_ip.side_effect = _Forbidden()
    monkeypatch.setattr(
        "infra.ibm_vpc._cleanup_volumes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("volume")),
    )
    monkeypatch.setattr(
        "infra.ibm_vpc._cleanup_gateway",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("gateway")),
    )

    with pytest.raises(CleanupFailures) as caught:
        delete_and_wait(
            client,
            spec,
            _instance_id(),
            deadline=_deadline(),
            tagger=_tagger(),
        )

    assert [str(error) for error in caught.value.errors] == [
        "volume",
        "gateway",
        "",
    ]


def test_predelete_validation_cannot_mutate_after_deadline(monkeypatch):
    ticks = iter((1.0, 2.0))
    monkeypatch.setattr("infra.ibm_vpc.time.monotonic", lambda: next(ticks))
    deleter = MagicMock()

    with pytest.raises(TimeoutError, match="deadline expired"):
        ibm_vpc._delete_and_confirm(
            deleter,
            MagicMock(),
            "resource-id",
            deadline=1.5,
            description="resource deletion",
            before_delete=lambda: None,
            revalidate_first_attempt=True,
        )

    deleter.assert_not_called()


def test_discovered_volume_failure_does_not_block_later_volume():
    spec = valid_run_spec()
    first = "r006-volume-1"
    second = "r006-volume-2"
    client = _client()
    client.list_volumes.side_effect = [
        _Response(
            {
                "volumes": [_volume(id=first), _volume(id=second)],
                "total_count": 2,
            }
        ),
        _Response({"volumes": [], "total_count": 0}),
    ]
    first_mismatch = _volume(id=first, user_tags=["project:foreign", f"run:{spec.run_id}"])
    second_reads = 0

    def get_volume(volume_id):
        nonlocal second_reads
        if volume_id == first:
            return _Response(first_mismatch)
        second_reads += 1
        if second_reads < 3:
            return _Response(_volume(id=second))
        raise _NotFound()

    client.get_volume.side_effect = get_volume

    with pytest.raises(PrototypeMismatch, match="boot volume"):
        ibm_vpc._cleanup_volumes(client, spec, None, deadline=_deadline())

    assert client.delete_volume.call_args_list == [call(second)]


def test_volume_receipt_failure_does_not_block_later_volume(monkeypatch):
    spec = valid_run_spec()
    first = "r006-volume-receipt-failed"
    second = "r006-volume-still-cleaned"
    client = _client()
    client.list_volumes.side_effect = [
        _Response(
            {
                "volumes": [_volume(id=first), _volume(id=second)],
                "total_count": 2,
            }
        ),
        _Response({"volumes": [], "total_count": 0}),
    ]
    reads = {first: 0, second: 0}

    def get_volume(volume_id):
        reads[volume_id] += 1
        if reads[volume_id] < 3:
            return _Response(_volume(id=volume_id))
        raise _NotFound()

    client.get_volume.side_effect = get_volume
    persist_deletion = ibm_vpc._persist_deletion

    def fail_first_receipt(spec, resource_type, resource_id):
        if resource_type == "boot-volume" and resource_id == first:
            raise OSError("volume receipt fsync failed")
        persist_deletion(spec, resource_type, resource_id)

    monkeypatch.setattr("infra.ibm_vpc._persist_deletion", fail_first_receipt)

    with pytest.raises(OSError, match="volume receipt fsync failed"):
        ibm_vpc._cleanup_volumes(client, spec, None, deadline=_deadline())

    assert set(call_.args[0] for call_ in client.delete_volume.call_args_list) == {
        first,
        second,
    }


def test_discovered_volume_inventory_waits_for_an_authoritative_empty_list(
    monkeypatch, tmp_path
):
    spec = valid_run_spec()
    volume_id = "r006-volume-stale"
    receipt = (
        tmp_path
        / "shared-state"
        / f"{spec.run_id}-deleted-boot-volume-inventory.json"
    )
    client = _client()
    pages = iter(
        (
            _Response({"volumes": [_volume(id=volume_id)], "total_count": 1}),
            _Response({"volumes": [_volume(id=volume_id)], "total_count": 1}),
            _Response({"volumes": [], "total_count": 0}),
            _Response({"volumes": [], "total_count": 0}),
        )
    )

    def list_volumes(**_kwargs):
        assert not receipt.exists()
        return next(pages)

    client.list_volumes.side_effect = list_volumes
    client.get_volume.side_effect = _NotFound()
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)

    ibm_vpc._cleanup_volumes(client, spec, None, deadline=_deadline())

    assert client.list_volumes.call_count == 4
    assert receipt.exists()


def test_discovered_volume_candidate_survives_a_failed_cleanup(tmp_path):
    spec = valid_run_spec()
    volume_id = "r006-volume-durable"
    first = _client()

    def first_list(**kwargs):
        assert "tag" not in kwargs
        return _Response(
            {"volumes": [_volume(id=volume_id)], "total_count": 1}
        )

    first.list_volumes.side_effect = first_list
    first.get_volume.return_value = _Response(
        _volume(id=volume_id, capacity=999)
    )

    with pytest.raises(CleanupFailures) as caught:
        ibm_vpc._cleanup_volumes(first, spec, None, deadline=_deadline())

    assert any(isinstance(error, PrototypeMismatch) for error in caught.value.errors)

    candidate_path = (
        tmp_path
        / "shared-state"
        / f"{spec.run_id}-boot-volume-candidates.json"
    )
    assert json.loads(candidate_path.read_text())["volume_ids"] == [volume_id]

    second = _client()
    second.list_volumes.return_value = _Response(
        {"volumes": [], "total_count": 0}
    )
    second.get_volume.side_effect = _NotFound()

    ibm_vpc._cleanup_volumes(second, spec, None, deadline=_deadline())

    assert json.loads(candidate_path.read_text())["volume_ids"] == []


def test_volume_candidate_is_persisted_before_the_next_page_fails(tmp_path):
    spec = valid_run_spec()
    volume_id = "r006-volume-first-page"
    client = _client()
    client.list_volumes.side_effect = [
        _Response(
            {
                "volumes": [_volume(id=volume_id)],
                "total_count": 2,
                "next": {"start": "second-page"},
            }
        ),
        ReadTimeout(),
    ]

    with pytest.raises(ReadTimeout):
        ibm_vpc._cleanup_volumes(client, spec, None, deadline=_deadline())

    candidate_path = (
        tmp_path
        / "shared-state"
        / f"{spec.run_id}-boot-volume-candidates.json"
    )
    assert json.loads(candidate_path.read_text())["volume_ids"] == [volume_id]


def test_corrupt_candidate_ledger_does_not_block_provider_cleanup(tmp_path):
    spec = valid_run_spec()
    volume_id = "r006-volume-provider-recovered"
    candidate_path = (
        tmp_path
        / "shared-state"
        / f"{spec.run_id}-boot-volume-candidates.json"
    )
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text("{", encoding="utf-8")
    client = _client()
    client.list_volumes.side_effect = [
        _Response({"volumes": [_volume(id=volume_id)], "total_count": 1}),
        _Response({"volumes": [], "total_count": 0}),
    ]
    client.get_volume.side_effect = [
        _Response(_volume(id=volume_id)),
        _Response(_volume(id=volume_id)),
        _NotFound(),
    ]

    with pytest.raises(AmbiguousProvisioning, match="unreadable"):
        ibm_vpc._cleanup_volumes(client, spec, None, deadline=_deadline())

    client.delete_volume.assert_called_once_with(volume_id)
    assert not (
        tmp_path
        / "shared-state"
        / f"{spec.run_id}-deleted-boot-volume-inventory.json"
    ).exists()


def test_candidate_ledger_write_failure_does_not_block_volume_delete(monkeypatch):
    spec = valid_run_spec()
    volume_id = "r006-volume-ledger-write-failed"
    client = _client()
    client.list_volumes.side_effect = [
        _Response({"volumes": [_volume(id=volume_id)], "total_count": 1}),
        _Response({"volumes": [], "total_count": 0}),
    ]
    client.get_volume.side_effect = [
        _Response(_volume(id=volume_id)),
        _Response(_volume(id=volume_id)),
        _NotFound(),
    ]
    monkeypatch.setattr(
        "infra.ibm_vpc._persist_boot_volume_candidates",
        lambda *_args: (_ for _ in ()).throw(OSError("candidate fsync failed")),
    )

    with pytest.raises(OSError, match="candidate fsync failed"):
        ibm_vpc._cleanup_volumes(client, spec, None, deadline=_deadline())

    client.delete_volume.assert_called_once_with(volume_id)


def test_empty_run_volume_inventory_ignores_unrelated_zone_churn(tmp_path):
    spec = valid_run_spec()
    first = _volume(id="r006-unrelated-a", user_tags=[])
    second = _volume(id="r006-unrelated-b", user_tags=[])
    client = _client()
    client.list_volumes.side_effect = [
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [first], "total_count": 1}),
        _Response({"volumes": [second], "total_count": 1}),
    ]

    ibm_vpc._cleanup_volumes(client, spec, None, deadline=_deadline())

    assert client.list_volumes.call_count == 3
    assert (
        tmp_path
        / "shared-state"
        / f"{spec.run_id}-deleted-boot-volume-inventory.json"
    ).exists()


def test_first_volume_delete_revalidates_ownership():
    spec = valid_run_spec()
    volume_id = "r006-volume-raced"
    client = _client()
    client.get_volume.side_effect = [
        _Response(_volume(id=volume_id)),
        _Response(_volume(id=volume_id, capacity=999)),
    ]

    with pytest.raises(PrototypeMismatch, match="boot volume"):
        ibm_vpc._cleanup_volumes(
            client, spec, volume_id, deadline=_deadline()
        )

    client.delete_volume.assert_not_called()


def test_final_inventory_deletes_a_valid_volume_missed_by_initial_discovery(
    monkeypatch
):
    spec = valid_run_spec()
    volume_id = "r006-volume-late"
    client = _client()
    client.list_volumes.side_effect = [
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [_volume(id=volume_id)], "total_count": 1}),
        _Response({"volumes": [], "total_count": 0}),
        _Response({"volumes": [], "total_count": 0}),
    ]
    client.get_volume.side_effect = [
        _Response(_volume(id=volume_id)),
        _Response(_volume(id=volume_id)),
        _NotFound(),
    ]
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)

    ibm_vpc._cleanup_volumes(client, spec, None, deadline=_deadline())

    client.delete_volume.assert_called_once_with(volume_id)
    assert client.list_volumes.call_count == 4


def test_resume_refuses_to_delete_subnet_bound_to_replaced_gateway(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first = _client()
    first.get_instance.return_value = _Response(vpc_instance(status="deleting"))
    first.get_subnet.return_value = _Response(_subnet(valid_run_spec().public_gateway_id))
    monkeypatch.setattr(
        "infra.ibm_vpc.time.monotonic",
        lambda: 2.0 if first.get_instance.call_count >= 2 else 1.0,
    )
    monkeypatch.setattr("infra.ibm_vpc.time.sleep", lambda _seconds: None)
    with pytest.raises(TimeoutError):
        delete_and_wait(
            first, valid_run_spec(), _instance_id(), deadline=1.5, tagger=_tagger()
        )

    second = MagicMock()
    second.list_volumes.return_value = _Response(
        {"volumes": [], "total_count": 0}
    )
    second.get_instance.side_effect = _NotFound()
    second.get_volume.side_effect = _NotFound()
    second.get_subnet.return_value = _Response(_subnet("r006-replacement"))
    second.get_floating_ip.side_effect = _NotFound()
    monkeypatch.setattr("infra.ibm_vpc.time.monotonic", lambda: 1.0)
    with pytest.raises(PrototypeMismatch, match="another gateway"):
        delete_and_wait(
            second, valid_run_spec(), _instance_id(), deadline=61.0, tagger=_tagger()
        )

    second.delete_subnet.assert_not_called()
