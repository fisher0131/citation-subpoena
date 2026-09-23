"""Claim-vs-source verification with fail-closed row validation.

核心移植自 citation_entropy 的 claim_verification.py：每条判定都必须通过结构校验，
quote 必须在被声明来源的快照里逐字存在；任何一行校验失败，该行降级为
unverifiable，绝不让坏行冒充 support。

与原项目的差别在于判定单位。原项目对"一条检索分支"出一次状态；本框架对
"一个来源"出一次状态，因为幻觉检验的关键恰恰是区分"绑错来源"（G1）与
"根本没有来源"（G2）——只有逐来源状态能把这两者分开。
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Sequence

STATES = {"support", "contradict", "not_found", "unverifiable"}

SYSTEM_PROMPT = (
    "Check every candidate factual proposition against EVERY supplied source, separately. "
    'Return JSON {"checks":[{"claim_id":str,"evidence_url":str,"state":'
    '"support|contradict|not_found|unverifiable","quote":str|null}]}. Emit exactly one check '
    "per (claim_id, evidence_url) pair. support means an exact quoted span from THAT source "
    "states the proposition; contradict means an exact quoted span from THAT source states an "
    "incompatible fact; not_found means THAT source has been fully read and states neither. "
    "Use unverifiable when the source text is missing or insufficient to decide. Quotes must be "
    "copied exactly from that source's snippet or page text. Do not use outside knowledge and "
    "never reuse one source's quote for another source's row."
)

VERSION = "audit-verify-v1"


def _unverifiable(claim_id: str, url: str | None, reason: str) -> dict:
    return {
        "claim_id": claim_id,
        "evidence_url": url,
        "state": "unverifiable",
        "quote": None,
        "validation_error": reason,
    }


def _normalize_text(value: str) -> str:
    """Collapse whitespace so a quote spanning a reflowed line still matches."""

    return re.sub(r"\s+", " ", (value or "")).strip()


def _validate_check(check: object, claim_id: str, evidence: dict) -> tuple[dict, str | None]:
    """Validate one (claim, source) row; a bad row can never establish a state."""

    if not isinstance(check, dict):
        return _unverifiable(claim_id, None, "invalid_check"), "invalid_check"
    url, quote, state = check.get("evidence_url"), check.get("quote"), check.get("state")
    if not isinstance(url, str) or url not in evidence:
        return _unverifiable(claim_id, url, "unknown_evidence_url"), "unknown_evidence_url"
    if not isinstance(state, str) or state not in STATES:
        return _unverifiable(claim_id, url, "invalid_state"), "invalid_state"
    item = evidence[url]
    if state in {"support", "contradict"}:
        if not isinstance(quote, str) or not quote.strip():
            return _unverifiable(claim_id, url, "missing_evidence_span"), "missing_evidence_span"
        haystack = _normalize_text(item.get("text") or "")
        snippet = _normalize_text(item.get("snippet") or "")
        if _normalize_text(quote) not in haystack and _normalize_text(quote) not in snippet:
            return _unverifiable(claim_id, url, "quote_not_in_snapshot"), "quote_not_in_snapshot"
    elif quote is not None:
        # Do not trust a not_found whose attached evidence contradicts its shape.
        return _unverifiable(claim_id, url, "unexpected_evidence_span"), "unexpected_evidence_span"
    elif state == "not_found" and not (_normalize_text(item.get("text") or "")):
        return _unverifiable(claim_id, url, "incomplete_evidence_for_absence"), "incomplete_evidence_for_absence"
    return {"claim_id": claim_id, "evidence_url": url, "state": state, "quote": quote}, None


def _expected_pairs(candidates: list[dict], evidence: Sequence[dict]) -> list[tuple[str, str]]:
    """One row per (claim, source): the unit the audit decision needs."""

    return [
        (str(row["claim_id"]), str(item["url"]))
        for row in candidates
        for item in evidence
    ]


def validate_detailed(
    raw: object, candidates: list[dict], evidence: Sequence[dict]
) -> tuple[list[dict], str | None, dict]:
    """Salvage valid rows while conservatively degrading bad rows to unknown."""

    expected = _expected_pairs(candidates, evidence)
    expected_set = set(expected)
    checks = raw.get("checks") if isinstance(raw, dict) else None
    if not isinstance(checks, list):
        return [], "invalid_checks_wrapper", {
            "status": "invalid",
            "error_counts": {"invalid_checks_wrapper": 1},
            "expected_count": len(expected),
            "valid_count": 0,
            "unverifiable_count": len(expected),
            "ignored_count": 0,
        }
    evidence_by_url = {item["url"]: item for item in evidence}
    grouped: dict[tuple[str, str], list[object]] = {pair: [] for pair in expected}
    first_pair_for_claim: dict[str, tuple[str, str]] = {}
    for pair in expected:
        first_pair_for_claim.setdefault(pair[0], pair)
    errors = Counter()
    ignored = 0
    for check in checks:
        if not isinstance(check, dict):
            errors["invalid_check"] += 1
            ignored += 1
            continue
        key = (str(check.get("claim_id", "")), str(check.get("evidence_url", "")))
        if key not in expected_set:
            fallback = first_pair_for_claim.get(key[0])
            if fallback is None:
                errors["unexpected_or_invalid_pair"] += 1
                ignored += 1
                continue
            # A row naming a source the audit never supplied can never establish
            # a state. Route it onto the claim's row so it is failed closed as
            # unknown_evidence_url instead of vanishing into the ignored count.
            grouped[fallback].append(check)
            continue
        grouped[key].append(check)

    result = []
    for claim_id, url in expected:
        rows = grouped[(claim_id, url)]
        if not rows:
            errors["missing_check"] += 1
            result.append(_unverifiable(claim_id, url, "missing_check"))
            continue
        validated = [_validate_check(row, claim_id, evidence_by_url) for row in rows]
        if len(validated) > 1:
            canonical = {
                (row["state"], row.get("evidence_url"), row.get("quote"), error)
                for row, error in validated
            }
            if len(canonical) != 1:
                errors["duplicate_conflict"] += 1
                result.append(_unverifiable(claim_id, url, "duplicate_conflict"))
                continue
            errors["duplicate_consistent"] += len(validated) - 1
        row, error = validated[0]
        if error:
            errors[error] += 1
        result.append(row)

    invalid = sum(1 for row in result if row["state"] == "unverifiable")
    diagnostics = {
        "status": "partial" if errors else "strict",
        "error_counts": dict(sorted(errors.items())),
        "expected_count": len(expected),
        "valid_count": len(result) - invalid,
        "unverifiable_count": invalid,
        "ignored_count": ignored,
    }
    material_errors = [
        name for name in errors
        if name not in {"duplicate_consistent", "incomplete_evidence_for_absence"}
    ]
    primary_error = None
    if material_errors:
        primary_error = material_errors[0] if len(material_errors) == 1 else "partial_checks"
    return result, primary_error, diagnostics


def messages(candidates: list[dict], evidence: Sequence[dict]) -> list[dict]:
    """Build the verifier prompt for one claim set against one evidence view."""

    payload = [
        {"url": item["url"], "snippet": item.get("snippet") or "", "text": item.get("text") or ""}
        for item in evidence
    ]
    user = json.dumps(
        {
            "claims": [{"claim_id": row["claim_id"], "claim_text": row["claim_text"]} for row in candidates],
            "evidence": payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def input_hash(messages: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Rule-based verifier (no model calls; string grounding with light stemming)
# --------------------------------------------------------------------------- #

#: Units and connectives that appear in almost any numeric sentence, so they
#: carry no signal about which fact is being stated. Excluded from overlap.
_GENERIC_TOKENS = frozenset(
    {
        "million", "billion", "trillion", "dollars", "dollar", "usd", "eur",
        "per", "cent", "percent", "percentage", "total", "totalled",
        "during", "in", "the", "and", "for", "from", "with", "that", "this",
        "its", "their", "was", "were", "has", "have", "had", "been",
        "year", "period", "quarter", "end", "about", "into", "than",
    }
)

_SUFFIX_RULES = (
    ("ations", "at"),
    ("ation", "at"),
    ("tions", "t"),
    ("tion", "t"),
    ("ments", "ment"),
    ("ing", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)


def _stem(token: str) -> str:
    """A tiny stemmer: enough to match ``generated`` against ``generation``.

    A real stemmer is not needed because the verifier only compares a claim
    against sentences already retrieved for it; the goal is to stop
    ``...generation reached 520...`` from missing ``...generated 480...``.
    """

    lowered = token.casefold()
    for suffix, replacement in _SUFFIX_RULES:
        if len(lowered) > len(suffix) + 1 and lowered.endswith(suffix):
            return lowered[: -len(suffix)] + replacement
    return lowered


def _content_tokens(text: str) -> set[str]:
    raw = re.findall(r"(?u)\b\w+\b", text or "")
    return {_stem(token) for token in raw if len(token) > 2 and token.casefold() not in _GENERIC_TOKENS}


def _numeric_values(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", text or "")


def _sentence_spans(haystack: str) -> list[str]:
    return [part.strip() for part in haystack.split(".") if part.strip()]


def _one_source(claim_text: str, item: dict) -> tuple[str, str | None]:
    """Decide one claim against one source, returning (state, quote).

    The order is exact span, then paraphrase-with-numeric-agreement, then
    contradiction, then absence. Only the exact-span path is proof-quality; the
    paraphrase path additionally requires the claim's first numeric value to
    appear in the same sentence, which is what stops ``480`` from matching a
    sentence that merely discusses capacity in general.
    """

    text = _normalize_text(item.get("text") or "")
    snippet = _normalize_text(item.get("snippet") or "")
    haystack = text or snippet
    if not haystack:
        return "unverifiable", None
    claim_norm = _normalize_text(claim_text)
    if claim_norm and claim_norm in haystack:
        return "support", claim_norm

    claim_tokens = _content_tokens(claim_norm)
    claim_values = _numeric_values(claim_norm)
    if not claim_tokens:
        return "unverifiable", None

    best_sentence = None
    best_overlap = 0.0
    for sentence in _sentence_spans(haystack):
        sentence_tokens = _content_tokens(sentence)
        if not sentence_tokens:
            continue
        overlap = len(claim_tokens & sentence_tokens) / len(claim_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence
    # Paraphrase support: enough of the claim survives, and its salient number
    # is actually present rather than merely implied by the topic. The numeric
    # requirement is what makes this safe: a sentence that does not carry the
    # claim's value cannot ground it, however topical it is.
    if best_sentence is not None and best_overlap >= 0.6 and claim_values:
        sentence_values = _numeric_values(best_sentence)
        if claim_values[0] in sentence_values:
            return "support", best_sentence
    # Contradiction is far narrower than absence. It requires the sentence to
    # be about the *same measured fact*: the claim's numeric value must be one
    # the sentence plausibly restates, which we approximate by requiring the
    # value's unit token to be shared. Without this, any sentence about the
    # same entity that mentions a different number would count as a
    # contradiction -- "received 2.3 million in subsidies" against "generated
    # 480 gigawatt-hours" is two facts about one company, not one contradiction.
    if best_sentence is not None and best_overlap >= 0.5 and claim_values:
        sentence_values = _numeric_values(best_sentence)
        if (
            sentence_values
            and claim_values[0] not in sentence_values
            and _shares_unit_token(claim_norm, best_sentence)
        ):
            return "contradict", best_sentence
    if not text:
        return "unverifiable", None
    return "not_found", None


#: Tokens that must agree for two numeric sentences to be about the same fact.
_UNIT_HINTS = frozenset(
    {
        "gigawatt", "gigawatthour", "megawatt", "megawatthour", "kilowatt",
        "tonne", "tonnes", "kilotonne", "million", "billion",
        "percent", "dollar", "dollars", "euro", "euros", "yuan", "yen",
        "staff", "employee", "employees", "site", "sites", "array", "arrays",
        "offset", "offsets", "subsidie", "subsidies", "grant", "grants",
        "loan", "loans", "round", "financing", "emission", "emissions",
        "metre", "metres", "met", "litre", "litres", "hectare", "hectares",
    }
)


def _shares_unit_token(left: str, right: str) -> bool:
    """Do these two sentences name the same unit for their numbers?"""

    left_units = _content_tokens(left) & _UNIT_HINTS
    right_units = _content_tokens(right) & _UNIT_HINTS
    return bool(left_units and left_units & right_units)


def rule_check(claim_text: str, evidence: Sequence[dict]) -> dict:
    """Deterministic single-source stand-in for the model verifier.

    Kept for backwards compatibility with the single-row contract; the pipeline
    uses :func:`rule_check_per_source` because binding errors are per-source.
    """

    if not evidence:
        return _unverifiable("rule", None, "no_evidence_supplied")
    state, quote = _one_source(claim_text, evidence[0])
    return {"claim_id": "rule", "evidence_url": evidence[0]["url"], "state": state, "quote": quote}


def rule_check_per_source(claim_text: str, evidence: Sequence[dict]) -> list[dict]:
    """One row per source, the same contract the model verifier must satisfy."""

    return [
        {"claim_id": "rule", "evidence_url": item["url"], "state": state, "quote": quote}
        for item in evidence
        for state, quote in [_one_source(claim_text, item)]
    ]


def summarize(rows: Sequence[dict]) -> dict:
    """Aggregate per-source rows into one state map plus diagnostics."""

    states: dict[str, str] = {}
    errors: Counter = Counter()
    for row in rows:
        url = str(row.get("evidence_url") or "")
        state = str(row.get("state") or "unverifiable")
        if url in states and states[url] != state:
            errors["conflicting_state_for_source"] += 1
            states[url] = "unverifiable"
        else:
            states[url] = state
        if row.get("validation_error"):
            errors[row["validation_error"]] += 1
    counts = Counter(states.values())
    return {
        "status": "ok" if not errors else "partial",
        "state_counts": dict(sorted(counts.items())),
        "error_counts": dict(sorted(errors.items())),
        "source_count": len(states),
    }
