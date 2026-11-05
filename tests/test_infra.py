"""Tests for `infra/aws_train.py` and `infra/s3_sync.py`.

Covers this step's Success Criteria and the CK-20 Test Cases to Cover:
`confirm=False` refuses to provision, `confirm=True` proceeds. boto3 is
mocked throughout — per this step's Blockers & Risks, no test may ever
touch real AWS.
"""

from unittest.mock import MagicMock, patch

import pytest

from infra.aws_train import DEFAULT_INSTANCE_TYPE, estimate_cost, launch_training
from infra.s3_sync import download_photo, upload_photo

# --- CK-20: launch_training confirm gate ------------------------------------


def test_launch_training_refuses_without_confirm():
    with pytest.raises(PermissionError):
        launch_training(instance_type="g5.xlarge", confirm=False)


@patch("infra.aws_train.boto3.client")
def test_launch_training_never_touches_boto3_without_confirm(mock_boto_client):
    with pytest.raises(PermissionError):
        launch_training(instance_type="g5.xlarge", confirm=False)

    mock_boto_client.assert_not_called()


@patch("infra.aws_train.boto3.client")
def test_launch_training_proceeds_with_confirm(mock_boto_client):
    mock_ec2 = MagicMock()
    mock_boto_client.return_value = mock_ec2

    launch_training(instance_type="g5.xlarge", confirm=True)

    mock_boto_client.assert_called_once_with("ec2")
    mock_ec2.run_instances.assert_called_once()


# --- AD-2: g6.xlarge default + g5-vs-g6 divergence surfaced, not silent ----


def test_default_instance_type_is_g6_xlarge():
    assert DEFAULT_INSTANCE_TYPE == "g6.xlarge"


def test_cost_estimate_flags_g6_as_diverging_from_spec():
    assert estimate_cost("g6.xlarge").is_spec_default is False


def test_cost_estimate_flags_g5_as_matching_spec():
    assert estimate_cost("g5.xlarge").is_spec_default is True


def test_refusal_message_surfaces_instance_type_and_cost():
    with pytest.raises(PermissionError) as exc_info:
        launch_training(instance_type="g6.xlarge", confirm=False)

    message = str(exc_info.value)
    assert "g6.xlarge" in message
    assert "$" in message  # cost is disclosed, never resolved silently


# --- s3_sync: boto3 S3 photo upload/download (mockable, no real spend) -----


@patch("infra.s3_sync.boto3.client")
def test_upload_photo_calls_s3_upload_file(mock_boto_client):
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    upload_photo(bucket="watch-photos", local_path="/tmp/dial.jpg", key="dial.jpg")

    mock_boto_client.assert_called_once_with("s3")
    mock_s3.upload_file.assert_called_once_with("/tmp/dial.jpg", "watch-photos", "dial.jpg")


@patch("infra.s3_sync.boto3.client")
def test_download_photo_calls_s3_download_file(mock_boto_client):
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    download_photo(bucket="watch-photos", key="dial.jpg", local_path="/tmp/dial.jpg")

    mock_boto_client.assert_called_once_with("s3")
    mock_s3.download_file.assert_called_once_with("watch-photos", "dial.jpg", "/tmp/dial.jpg")
