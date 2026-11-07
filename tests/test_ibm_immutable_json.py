"""Immutable documents use the pinned IBM SDK's actual PutObject contract."""

import hashlib
from importlib.metadata import version

import pytest
from ibm_botocore.session import get_session
from ibm_botocore.validate import validate_parameters

from infra.ibm_cos import IntegrityError, load_immutable_json, put_immutable_json
from tests.conftest import valid_run_spec
from tests.test_ibm_cos import FakeCOS


def test_content_addressed_write_obeys_pinned_sdk_and_is_idempotent():
    assert version("ibm-cos-sdk") == "2.16.2"
    model = get_session().get_service_model("s3").operation_model("PutObject").input_shape
    assert "IfNoneMatch" not in model.members
    spec = valid_run_spec()

    class ContractCOS(FakeCOS):
        writes = 0

        def put_object(self, **kwargs):
            validate_parameters(kwargs, model)
            self.writes += 1
            return super().put_object(**kwargs)

    cos = ContractCOS(bucket=spec.bucket, prefix=spec.prefix)
    body = b'{"value":1}'
    key = put_immutable_json(cos, spec, f"{spec.prefix}/receipt.json", body)
    assert key.endswith(f"-{hashlib.sha256(body).hexdigest()}.json")
    assert put_immutable_json(cos, spec, f"{spec.prefix}/receipt.json", body) == key
    assert cos.writes == 1
    assert load_immutable_json(cos, spec, f"{spec.prefix}/receipt.json") == {"value": 1}


@pytest.mark.parametrize("existing", [True, False])
def test_immutable_json_detects_collision_before_write_or_in_readback(existing):
    spec = valid_run_spec()
    body = b'{"value":1}'
    key = f"{spec.prefix}/receipt-{hashlib.sha256(body).hexdigest()}.json"

    class CorruptCOS(FakeCOS):
        def put_object(self, **kwargs):
            kwargs["Body"] = b'{"value":2}'
            return super().put_object(**kwargs)

    cos = CorruptCOS(bucket=spec.bucket, prefix=spec.prefix)
    if existing:
        cos.objects[key] = (b'{"value":2}', {})
    with pytest.raises(IntegrityError, match="differs"):
        put_immutable_json(cos, spec, key, body)


def test_logical_immutable_document_rejects_two_distinct_content_keys():
    spec = valid_run_spec()
    cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    for body in (b'{"value":1}', b'{"value":2}'):
        put_immutable_json(cos, spec, f"{spec.prefix}/receipt.json", body)
    with pytest.raises(IntegrityError, match="conflicting"):
        load_immutable_json(cos, spec, f"{spec.prefix}/receipt.json")
