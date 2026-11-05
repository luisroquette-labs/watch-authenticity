"""AWS GPU training launch — confirm-gated to prevent accidental spend (CK-20).

Real money is on the line here: $5-20 per training round (Requirements
item 9), irreversible once an instance is running. Every non-trivial
provisioning call MUST pass through `launch_training(..., confirm=True)`
first — `confirm=False` (the default a caller would reach for accidentally)
refuses before boto3 is ever touched.

This module also owns AD-2, the g5-vs-g6 divergence from the original spec:
the default here is the cheaper `g6.xlarge`, and that choice is surfaced in
the cost-estimate string shown at the same confirmation boundary, never
resolved silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import boto3

# AD-2: default diverges from the spec's g5.xlarge (same 24GB VRAM, ~20%
# cheaper — SKILL.md:110). The divergence is surfaced via CostEstimate below,
# not just baked silently into this constant.
DEFAULT_INSTANCE_TYPE = "g6.xlarge"

# SKILL.md:110 — 3rd-party-tracker-sourced hourly rates, not AWS's live
# price-list API. Confirm current pricing at deploy time before spending.
_HOURLY_RATE_ESTIMATE_USD = {
    "g5.xlarge": 1.006,  # spec default (A10G, 24GB)
    "g6.xlarge": 0.805,  # AD-2 default (L4, 24GB)
}

# Requirements item 9: a training round takes 5-20h depending on dataset size.
_ESTIMATED_TRAINING_HOURS_RANGE = (5, 20)


@dataclass(frozen=True)
class CostEstimate:
    """Owner-facing cost disclosure shown at the confirmation boundary."""

    instance_type: str
    is_spec_default: bool  # True only for g5.xlarge, the task's original pick
    hourly_rate_usd: float
    estimated_low_usd: float
    estimated_high_usd: float

    def __str__(self) -> str:
        divergence = (
            "matches the spec'd g5.xlarge"
            if self.is_spec_default
            else "diverges from the spec'd g5.xlarge (AD-2: ~20% cheaper, same VRAM)"
        )
        return (
            f"{self.instance_type} @ ${self.hourly_rate_usd}/hr, {divergence} — "
            f"estimated ${self.estimated_low_usd}-${self.estimated_high_usd} "
            f"for a 5-20h training round"
        )


def estimate_cost(instance_type: str) -> CostEstimate:
    """Builds the g5-vs-g6 + cost disclosure every confirmation prompt shows.

    Kept separate from `launch_training` so a caller (CLI, notebook) can
    display the estimate to the owner before deciding whether to pass
    `confirm=True`, instead of only learning the cost from a refusal.
    """
    rate = _HOURLY_RATE_ESTIMATE_USD.get(instance_type)
    if rate is None:
        raise ValueError(
            f"No cost estimate known for instance type '{instance_type}'; "
            f"supported: {sorted(_HOURLY_RATE_ESTIMATE_USD)}"
        )

    low_hours, high_hours = _ESTIMATED_TRAINING_HOURS_RANGE
    return CostEstimate(
        instance_type=instance_type,
        is_spec_default=(instance_type == "g5.xlarge"),
        hourly_rate_usd=rate,
        estimated_low_usd=round(rate * low_hours, 2),
        estimated_high_usd=round(rate * high_hours, 2),
    )


def launch_training(instance_type: str = DEFAULT_INSTANCE_TYPE, *, confirm: bool) -> Any:
    """Provisions a GPU training instance — refuses without explicit confirm (CK-20).

    The cost estimate is computed before the confirm check so it can be
    surfaced in the refusal message itself: an owner who calls this without
    `confirm=True` still sees what the spend would have been, rather than a
    bare "refused" with no number attached.
    """
    estimate = estimate_cost(instance_type)
    if not confirm:
        raise PermissionError(f"Refusing to provision without confirm=True. Cost estimate: {estimate}")

    # ponytail: AMI/security-group/key-pair/startup-script wiring is deferred
    # to whatever calls this with confirm=True — this module's job is the
    # confirm gate + cost disclosure (CK-20), not the full launch config.
    ec2 = boto3.client("ec2")
    return ec2.run_instances(InstanceType=instance_type, MinCount=1, MaxCount=1)
