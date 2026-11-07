"""Read destination probe records before mutating the watchdog deployment."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import requests


def verify_routing_probes(
    proof: object,
    read: Callable[[str], object],
    *,
    account: str,
    region: str,
    logs_crn: str,
    monitoring_crn: str,
    project: str,
    service_account: str,
    now: datetime,
) -> None:
    """Local input locates records; only authenticated destination reads prove delivery."""

    if not isinstance(proof, dict) or set(proof) != {"probe_id", "logs", "monitoring"}:
        raise ValueError("provider-read routing probe references are required; local flags are invalid")
    probe_id = proof["probe_id"]
    if not isinstance(probe_id, str) or not probe_id:
        raise ValueError("routing probe ID is missing")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("routing observation time must be timezone aware")
    for kind, crn in (("logs", logs_crn), ("monitoring", monitoring_crn)):
        parts = crn.split(":")
        if (
            len(parts) != 10 or parts[5] != region or parts[6] != f"a/{account}"
            or not parts[7] or parts[4] != ("logs" if kind == "logs" else "sysdig-monitor")
        ):
            raise ValueError("routing destination CRN differs from selected service/account/region")
        reference = proof[kind]
        if not isinstance(reference, dict) or set(reference) != {"record_id", "record_url"}:
            raise ValueError("routing probe reference must contain only provider record ID and URL")
        record_id, url = reference["record_id"], reference["record_url"]
        if not isinstance(record_id, str) or not record_id or not isinstance(url, str):
            raise ValueError("routing probe record ID or URL is missing")
        parsed = urlsplit(url)
        host = (
            f"{parts[7]}.api.{region}.logs.cloud.ibm.com" if kind == "logs"
            else f"{region}.monitoring.cloud.ibm.com"
        )
        if (
            parsed.scheme != "https" or parsed.hostname != host
            or parsed.port not in (None, 443) or parsed.username or parsed.password
            or parsed.fragment or not parsed.path.startswith(("/api/", "/v1/"))
        ):
            raise ValueError("routing probe URL is not an authenticated IBM destination")
        # shortcut: only provider record exports with this schema are supported; add an
        # official query-response adapter when its destination contract is confirmed.
        response = read(url)
        if not isinstance(response, dict) or not isinstance(response.get("records"), list):
            raise ValueError("unrecognized provider routing record response")
        matches = [item for item in response["records"] if isinstance(item, dict) and item.get("id") == record_id]
        if len(matches) != 1:
            raise ValueError("provider routing record is missing or ambiguous")
        record = matches[0]
        if (
            record.get("destination_crn") != crn or record.get("probe_id") != probe_id
            or record.get("account_id") != account or record.get("region") != region
            or record.get("project") != project or record.get("service_account") != service_account
            or record.get("authentication") != "trusted_profile"
        ):
            raise ValueError("provider routing probe identity or destination differs")
        try:
            observed = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise ValueError("provider routing probe timestamp is invalid") from exc
        if observed.tzinfo is None or not timedelta(0) <= now - observed <= timedelta(minutes=15):
            raise ValueError("provider routing probe is stale or future dated")


def main() -> None:
    proof = json.loads(Path(os.environ["WATCH_AUTH_PLATFORM_ROUTING_PROOF"]).read_text())
    tokens = json.loads(subprocess.run(
        ["ibmcloud", "iam", "oauth-tokens", "--output", "json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    token = tokens.get("iam_token")
    if not isinstance(token, str) or not token.startswith("Bearer "):
        raise ValueError("short-lived IBM IAM token is unavailable")

    def read(url: str) -> Any:
        response = requests.get(
            url, headers={"Authorization": token}, timeout=30, allow_redirects=False,
        )
        if response.status_code != 200:
            raise ValueError("provider routing record read failed")
        return response.json()

    verify_routing_probes(
        proof, read,
        account=os.environ["WATCH_AUTH_ACCOUNT_ID"],
        region=os.environ["WATCH_AUTH_REGION"],
        logs_crn=os.environ["WATCH_AUTH_PLATFORM_LOGS_CRN"],
        monitoring_crn=os.environ["WATCH_AUTH_PLATFORM_MONITORING_CRN"],
        project=os.environ["WATCH_AUTH_CE_PROJECT"],
        service_account=os.environ["WATCH_AUTH_WATCHDOG_SERVICE_ACCOUNT"],
        now=datetime.now(timezone.utc),
    )


if __name__ == "__main__":
    main()
