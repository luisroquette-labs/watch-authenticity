"""Reference-specific technical knowledge for authenticity assessment."""

from reference_intelligence.compare import assess_reference
from reference_intelligence.knowledge import load_knowledge
from reference_intelligence.types import (
    AssessmentStatus,
    CheckStatus,
    ReferenceAssessment,
    SourceClass,
    TechnicalClaim,
    VisualObservation,
)

__all__ = [
    "AssessmentStatus",
    "CheckStatus",
    "ReferenceAssessment",
    "SourceClass",
    "TechnicalClaim",
    "VisualObservation",
    "assess_reference",
    "load_knowledge",
]
