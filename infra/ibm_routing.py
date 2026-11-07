"""Block deployment until credentialless routing has a supported provider proof."""

from typing import NoReturn


class RoutingProofUnavailable(RuntimeError):
    """The documented destination readbacks cannot establish source provenance."""


def verify_routing_probes(_evidence: object = None) -> NoReturn:
    """Reject local flags and destination records as credentialless-routing proof."""

    # Logs /v1/query returns SSE, not a records envelope:
    # https://cloud.ibm.com/docs/apis/logs-service-api
    # Archive metadata/labels/user_data do not attest producer credentials:
    # https://github.com/ibm-cloud-docs/cloud-logs/blob/master/query-archive-data-bucket.md
    # Monitoring POST /api/data requires IBMInstanceID and returns data[].d/t:
    # https://cloud.ibm.com/docs/monitoring?topic=monitoring-metrics_api
    raise RoutingProofUnavailable(
        "IBM routing verification blocked: documented Logs query/archive and "
        "Monitoring /api/data readbacks do not attest producer credential mode "
        "or service-account binding. Provider-verified Code Engine integration/IAM "
        "provenance bound to the destination CRNs, source identity and fresh "
        "delivery records is required (PENDENTE E2E)."
    )


def main() -> NoReturn:
    try:
        verify_routing_probes()
    except RoutingProofUnavailable as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
