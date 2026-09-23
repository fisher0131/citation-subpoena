"""Subpoena: make every citation testify."""

from .claims import Citation, Claim, parse_citations, split_claims
from .evidence import FetchOutcome, OpenAccessResolver, StaticFetcher, classify_fetch_failure
from .labels import (
    GROUNDING_LABELS,
    HALLUCINATION_LABELS,
    ClaimVerdict,
    label_from_states,
    summarize_verdicts,
)
from .pipeline import AuditConfig, AuditReport, OfflineCorpus, audit, fetch_evidence, verify_claims
from .stats import wilson_proportion_interval
from .verify import STATES, rule_check, validate_detailed

__all__ = [
    "AuditConfig",
    "AuditReport",
    "Citation",
    "Claim",
    "ClaimVerdict",
    "FetchOutcome",
    "GROUNDING_LABELS",
    "HALLUCINATION_LABELS",
    "OfflineCorpus",
    "OpenAccessResolver",
    "STATES",
    "StaticFetcher",
    "audit",
    "classify_fetch_failure",
    "fetch_evidence",
    "label_from_states",
    "parse_citations",
    "rule_check",
    "split_claims",
    "summarize_verdicts",
    "validate_detailed",
    "verify_claims",
    "wilson_proportion_interval",
]
