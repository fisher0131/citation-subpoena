"""Tests for the fail-closed verifier and the grounding decision rules."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import unittest

from subpoena import labels as labels_module
from subpoena.labels import (
    GROUNDING_LABELS,
    HALLUCINATION_LABELS,
    ClaimVerdict,
    decide,
    summarize_verdicts,
)
from subpoena.verify import (
    STATES,
    _one_source,
    rule_check_per_source,
    validate_detailed,
)


def source(url: str, text: str, *, snippet: str = "") -> dict:
    return {"url": url, "snippet": snippet, "text": text}


ANNUAL = source(
    "https://annual",
    "The company generated 480 gigawatt-hours in 2024, up from 455 in 2023. "
    "It employed 84 full-time staff. The report makes no statement about subsidies.",
)
BRIEF = source(
    "https://brief",
    "The company holds an estimated 19 percent share of the regional segment. "
    "Series B closed at 22 million dollars in November 2024.",
)


class OneSourceTests(unittest.TestCase):
    def test_exact_span_is_support(self):
        state, quote = _one_source("The company generated 480 gigawatt-hours in 2024, up from 455 in 2023.", ANNUAL)
        self.assertEqual(state, "support")
        self.assertIn("480", quote)

    def test_paraphrase_with_matching_number_is_support(self):
        state, quote = _one_source("Generation reached 480 gigawatt-hours in 2024.", ANNUAL)
        self.assertEqual(state, "support")
        self.assertIn("480", quote)

    def test_wrong_number_for_the_same_fact_is_contradict(self):
        state, quote = _one_source("Generation reached 520 gigawatt-hours in 2024.", ANNUAL)
        self.assertEqual(state, "contradict")

    def test_different_fact_same_entity_is_not_contradiction(self):
        # "2.3 million in subsidies" vs "480 gigawatt-hours" are two facts about
        # one company. A false contradiction here was the key regression.
        state, _ = _one_source("The company received 2.3 million dollars in subsidies in 2024.", ANNUAL)
        self.assertEqual(state, "not_found")

    def test_absent_fact_is_not_found(self):
        state, _ = _one_source("The company operates a pilot programme in Portugal.", ANNUAL)
        self.assertEqual(state, "not_found")

    def test_missing_text_is_unverifiable_never_not_found(self):
        state, _ = _one_source("The company generated 480 gigawatt-hours.", source("https://x", ""))
        self.assertEqual(state, "unverifiable")

    def test_snippet_only_absence_is_unverifiable(self):
        # Absence cannot be established from material that was never read.
        state, _ = _one_source("The company generated 999 gigawatt-hours.", source("https://x", "", snippet="A short note."))
        self.assertEqual(state, "unverifiable")

    def test_rule_check_per_source_emits_one_row_per_url(self):
        rows = rule_check_per_source("Generation reached 480 gigawatt-hours in 2024.", [ANNUAL, BRIEF])
        self.assertEqual([row["evidence_url"] for row in rows], ["https://annual", "https://brief"])
        self.assertEqual([row["state"] for row in rows], ["support", "not_found"])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [{"claim_id": "c1", "claim_text": "The company generated 480 gigawatt-hours in 2024."}]
        self.evidence = [ANNUAL]

    def test_support_with_invented_quote_fails_closed(self):
        raw = {"checks": [{"claim_id": "c1", "evidence_url": "https://annual", "state": "support", "quote": "pure invention"}]}
        rows, error, diagnostics = validate_detailed(raw, self.candidates, self.evidence)
        self.assertEqual(rows[0]["state"], "unverifiable")
        self.assertEqual(rows[0]["validation_error"], "quote_not_in_snapshot")
        self.assertEqual(error, "quote_not_in_snapshot")

    def test_support_with_unknown_url_fails_closed(self):
        raw = {"checks": [{"claim_id": "c1", "evidence_url": "https://nowhere", "state": "support", "quote": "x"}]}
        rows, error, _ = validate_detailed(raw, self.candidates, self.evidence)
        self.assertEqual(rows[0]["state"], "unverifiable")
        # A row naming a source the audit never supplied can never establish
        # support for the claim it claims to quote.
        self.assertIn("unknown_evidence_url", rows[0]["validation_error"])

    def test_bad_row_does_not_discard_a_valid_neighbour(self):
        candidates = self.candidates + [{"claim_id": "c2", "claim_text": "It employed 84 staff."}]
        raw = {"checks": [
            {"claim_id": "c1", "evidence_url": "https://annual", "state": "support", "quote": "pure invention"},
            {"claim_id": "c2", "evidence_url": "https://annual", "state": "support", "quote": "It employed 84 full-time staff."},
        ]}
        rows, error, diagnostics = validate_detailed(raw, candidates, self.evidence)
        self.assertEqual([row["state"] for row in rows], ["unverifiable", "support"])
        self.assertEqual(diagnostics["valid_count"], 1)

    def test_missing_pair_is_unverifiable_not_ignored(self):
        raw = {"checks": []}
        rows, error, diagnostics = validate_detailed(raw, self.candidates, self.evidence)
        self.assertEqual(rows[0]["state"], "unverifiable")
        self.assertEqual(rows[0]["validation_error"], "missing_check")
        self.assertEqual(diagnostics["unverifiable_count"], 1)

    def test_not_found_carrying_a_quote_is_rejected(self):
        raw = {"checks": [{"claim_id": "c1", "evidence_url": "https://annual", "state": "not_found", "quote": "The company generated 480 gigawatt-hours in 2024."}]}
        rows, error, _ = validate_detailed(raw, self.candidates, self.evidence)
        self.assertEqual(rows[0]["state"], "unverifiable")
        self.assertEqual(error, "unexpected_evidence_span")

    def test_conflicting_duplicates_degrade_to_unknown(self):
        good = {"claim_id": "c1", "evidence_url": "https://annual", "state": "support", "quote": "The company generated 480 gigawatt-hours in 2024, up from 455 in 2023."}
        bad = dict(good, state="contradict")
        rows, error, _ = validate_detailed({"checks": [good, bad]}, self.candidates, self.evidence)
        self.assertEqual(rows[0]["state"], "unverifiable")
        self.assertEqual(error, "duplicate_conflict")

    def test_wrapper_shape_error(self):
        rows, error, diagnostics = validate_detailed({"nope": 1}, self.candidates, self.evidence)
        self.assertEqual(error, "invalid_checks_wrapper")
        self.assertEqual(diagnostics["unverifiable_count"], 1)


class DecisionTests(unittest.TestCase):
    def test_supported_when_cited_source_supports(self):
        label, reason, url, _ = decide({"https://annual": "support"}, {}, cited_url="https://annual")
        self.assertEqual(label, "SUPPORTED")
        self.assertEqual(url, "https://annual")

    def test_g1_when_citation_is_silent_but_pool_supports(self):
        label, reason, url, _ = decide(
            {"https://annual": "not_found"},
            {"https://brief": "support"},
            cited_url="https://annual",
        )
        self.assertEqual(label, "G1_CITATION_BINDING")
        self.assertEqual(url, "https://brief")
        self.assertIn("does not cite", reason)

    def test_g2_when_no_source_anywhere_supports(self):
        label, _, _, _ = decide(
            {"https://annual": "not_found"},
            {"https://brief": "not_found"},
            cited_url="https://annual",
        )
        self.assertEqual(label, "G2_EVIDENCE_GAP")

    def test_g3_when_cited_source_contradicts(self):
        label, reason, _, _ = decide({"https://annual": "contradict"}, {}, cited_url="https://annual")
        self.assertEqual(label, "G3_CONTRADICTED")
        self.assertIn("incompatible", reason)

    def test_g3_outranks_pool_support(self):
        # A source that actively disagrees is worse than a misbound citation.
        label, _, _, _ = decide(
            {"https://annual": "contradict"},
            {"https://brief": "support"},
            cited_url="https://annual",
        )
        self.assertEqual(label, "G3_CONTRADICTED")

    def test_unverifiable_when_every_row_failed(self):
        label, _, _, _ = decide({"https://annual": "unverifiable"}, {}, cited_url="https://annual")
        self.assertEqual(label, "UNVERIFIABLE")

    def test_empty_states_is_gap(self):
        label, _, _, _ = decide({}, {}, cited_url="https://annual")
        self.assertEqual(label, "G2_EVIDENCE_GAP")


class SummaryTests(unittest.TestCase):
    def _verdict(self, label: str) -> ClaimVerdict:
        return ClaimVerdict(claim_id="c", claim_text="t", grounding_label=label)

    def test_rate_excludes_undecidable_labels(self):
        summary = summarize_verdicts([
            self._verdict("SUPPORTED"),
            self._verdict("G2_EVIDENCE_GAP"),
            self._verdict("FETCH_FAILURE"),
            self._verdict("NOT_APPLICABLE"),
            self._verdict("UNVERIFIABLE"),
        ])
        self.assertEqual(summary["decidable_claim_count"], 2)
        self.assertEqual(summary["hallucination_count"], 1)
        self.assertEqual(summary["hallucination_rate"], 0.5)
        self.assertEqual(len(summary["hallucination_rate_wilson_95"]), 2)

    def test_no_decidable_claims_gives_null_rate(self):
        summary = summarize_verdicts([self._verdict("NOT_APPLICABLE")])
        self.assertIsNone(summary["hallucination_rate"])
        self.assertIsNone(summary["hallucination_rate_wilson_95"])

    def test_counts_cover_every_label(self):
        summary = summarize_verdicts([self._verdict("G1_CITATION_BINDING")])
        self.assertEqual(summary["grounding_label_counts"]["G1_CITATION_BINDING"], 1)
        self.assertIn("G1_CITATION_BINDING", summary["verdicts_by_label"])

    def test_taxonomy_partitions_correctly(self):
        self.assertEqual(len(GROUNDING_LABELS), 7)
        self.assertEqual(set(HALLUCINATION_LABELS), {"G1_CITATION_BINDING", "G2_EVIDENCE_GAP", "G3_CONTRADICTED"})
        self.assertNotIn("UNVERIFIABLE", HALLUCINATION_LABELS)
        for state in STATES:
            self.assertIn(state, {"support", "contradict", "not_found", "unverifiable"})


if __name__ == "__main__":
    unittest.main()
