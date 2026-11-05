"""Every safety-critical, rubric-tested rule the task's credibility rests on.

This module is L1 — framework-free, stdlib-only (plus `core.types`, which is
itself stdlib-only). Every later stage (scrape, manifest, train, calibrate,
evaluate, publish) MUST import these functions rather than re-implementing
any of them: duplicating a rule here across stages is a direct CK-22
violation, and the single-source-of-truth guarantee is the reason this
module exists at all.
"""

from __future__ import annotations

from core.types import Grade, Label

# Confidence is structurally forbidden from reading as an absolute verdict
# (Requirements item 2): clamping away from the 0.0/1.0 extremes is what
# makes "0.1%-99.9%" true for every reported score, not just the typical one.
CONFIDENCE_FLOOR = 0.001
CONFIDENCE_CEILING = 0.999

# ponytail: a fixed policy value from the owner's brief ("~60%"), not a
# figure derived from measured superclone-detection accuracy. Revisit once
# real held-out-factory evaluation data exists for grade-4 samples.
GRADE4_GENUINE_CEILING = 0.60

# Marketplaces where a seller's own listing text decides genuine-vs-replica;
# an "original" claim here is the single pitfall CK-13/CK-18 exist to block.
_MARKETPLACE_SOURCES = frozenset({"olx", "mercadolivre"})

# Owner-forum sources treated as routine genuine photos (Requirements item 5).
_GENUINE_FORUM_SOURCES = frozenset({"chrono24", "watchuseek"})

# Dedicated replica-forum / replica-site sources — a named factory or site
# that sells/discusses replicas, as opposed to an ambiguous marketplace ad.
_REPLICA_SOURCES = frozenset({"reptime", "chinese_replica"})

_REDDIT_SOURCES = frozenset({"reddit"})

# Named superclone factories (Requirements item 3: "Clean, VSF — clone até
# do movimento"). ponytail: extend this set as new factories surface; any
# other named factory defaults to HIGH_GRADE_REP, the conservative choice.
_SUPERCLONE_FACTORIES = frozenset({"clean", "vsf"})

# Below this listing price, an anonymous marketplace fake is assumed to be
# street-fake territory (quartz movement, no-magnification lupa).
PRICE_GRADE_1_CEILING = 50.0


def clip_confidence(p: float) -> float:
    """Clamps a raw probability into the reportable confidence range (CK-1).

    A reported score of exactly 0% or 100% would read as an absolute
    verdict, which the task's design explicitly forbids — this clamp is
    what keeps every reported confidence inside the open interval.
    """
    return max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, p))


def apply_grade4_ceiling(raw_genuine: float, grade: int) -> float:
    """Caps genuine-confidence for grade-4 (superclone) output (CK-3).

    A superclone is not reliably detectable from a phone photo, so the
    pipeline is structurally forbidden from reporting high genuine-
    confidence for one — this cap applies regardless of the model's raw
    internal score.
    """
    if grade == Grade.SUPERCLONE:
        return min(raw_genuine, GRADE4_GENUINE_CEILING)
    return raw_genuine


def grade_map(*, factory: str | None = None, price: float | None = None) -> int:
    """Maps a named replica factory or a listing price to a fake grade.

    Single source of truth for factory/price -> grade (CK-22): scrapers
    and the manifest stage call this instead of each re-deriving their own
    mapping. `factory` wins when both are given — a named factory
    identifies a specific grade directly, while `price` is only a fallback
    proxy for anonymous marketplace listings that name no factory.
    """
    if factory is not None:
        is_superclone = factory.strip().lower() in _SUPERCLONE_FACTORIES
        return int(Grade.SUPERCLONE if is_superclone else Grade.HIGH_GRADE_REP)

    if price is not None:
        return int(Grade.STREET_FAKE if price < PRICE_GRADE_1_CEILING else Grade.COMMON_REPLICA)

    raise ValueError("grade_map requires factory or price")


def source_label_rule(
    source: str | None = None,
    *,
    subreddit: str | None = None,
    marketplace: str | None = None,
    listing_text: str | None = None,
    permission_on_file: bool = False,
) -> Label:
    """Derives a training label from WHERE a photo was sourced (CK-13/27).

    The label comes from the source's trustworthiness, never from someone
    eyeballing the watch (Requirements item 5). EXCLUDED checks run first
    and unconditionally, before any GENUINE/FAKE branch, so neither a
    seller's "original" lie nor an unpermitted Reddit scrape can ever fall
    through to a genuine label (the CK-18/CK-29 pitfalls).
    """
    is_marketplace_ad = (marketplace or source or "").strip().lower() in _MARKETPLACE_SOURCES
    if is_marketplace_ad and listing_text and "original" in listing_text.lower():
        return Label.EXCLUDED  # seller claims genuine — poisons training (CK-13)

    is_reddit_source = (source or "").strip().lower() in _REDDIT_SOURCES or bool(subreddit)
    if is_reddit_source:
        return Label.GENUINE if permission_on_file else Label.EXCLUDED  # CK-27

    if is_marketplace_ad:
        return Label.FAKE  # declared réplica/inspirado listing

    normalized_source = (source or "").strip().lower()
    if normalized_source in _GENUINE_FORUM_SOURCES:
        return Label.GENUINE
    if normalized_source in _REPLICA_SOURCES:
        return Label.FAKE

    return Label.EXCLUDED  # unrecognized source — never fabricate a label


def aggregate_angles(scores_per_angle: dict[str, float]) -> dict[str, float]:
    """Combines per-guided-angle genuine-probability scores into one score.

    A plain arithmetic mean — zero trainable parameters by design (CK-19).
    This function being a non-learned rule is itself the invariant under
    test: no later stage may replace this with a second trained model.
    """
    if not scores_per_angle:
        raise ValueError("aggregate_angles requires at least one angle score")

    raw_genuine = sum(scores_per_angle.values()) / len(scores_per_angle)
    return {"raw_genuine": raw_genuine}


def publish_gate(standard_acc: float, cross_source_acc: float, max_delta: float) -> bool:
    """Returns whether a weight version may publish, source-bias-wise (CK-8).

    Blocks when standard-test accuracy outpaces cross-source-test accuracy
    by more than `max_delta` — the signature of a model that learned "which
    site this photo came from" instead of "is this watch fake" (the task's
    top-named risk, Requirements item 13).
    """
    return (standard_acc - cross_source_acc) <= max_delta


def goldset_regression_gate(current_acc: float, baseline_acc: float) -> bool:
    """Returns whether a weight version may publish against the gold set (CK-12).

    Blocks whenever accuracy regresses below the previously published
    baseline, even marginally — the frozen gold set exists specifically to
    make regression a publish-blocking fact, not a warning.
    """
    return current_acc >= baseline_acc
