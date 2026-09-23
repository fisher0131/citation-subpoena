"""Tests for citation-marker parsing and claim splitting."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import unittest

from subpoena.claims import (
    Citation,
    parse_citations,
    parse_footnote_definitions,
    split_claims,
)


class FootnoteDefinitionTests(unittest.TestCase):
    def test_definitions_are_extracted(self):
        text = "Some claim.[^1]\n\n[^1]: https://example.org/a\n[^2]: https://example.org/b\n"
        definitions = parse_footnote_definitions(text)
        self.assertEqual(definitions, {"1": "https://example.org/a", "2": "https://example.org/b"})

    def test_definitions_are_not_parsed_as_citations(self):
        text = "Some claim.[^1]\n\n[^1]: https://example.org/a\n"
        citations = parse_citations(text, url_map={"1": "https://example.org/a"})
        # One citation from the body; the definition line must not double it.
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].url, "https://example.org/a")


class CitationParsingTests(unittest.TestCase):
    def test_bracket_numeric_resolves_via_url_map(self):
        text = "A claim with one marker.[^3] And another.[^4]"
        citations = parse_citations(text, url_map={"3": "https://a", "4": "https://b"})
        self.assertEqual([citation.url for citation in citations], ["https://a", "https://b"])

    def test_comma_compound_marker_binds_to_first_definition(self):
        text = "A claim.[^1,2]"
        citations = parse_citations(text, url_map={"1": "https://a", "2": "https://b"})
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].url, "https://a")

    def test_unresolved_marker_keeps_citation_with_null_url(self):
        # An unresolvable marker is a finding, not something to drop silently.
        text = "A claim.[^99]"
        citations = parse_citations(text, url_map={})
        self.assertEqual(len(citations), 1)
        self.assertIsNone(citations[0].url)
        self.assertFalse(citations[0].resolved())

    def test_markdown_inline_link_carries_its_own_url(self):
        text = "See [the annual report](https://example.org/report) for details."
        citations = parse_citations(text)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].url, "https://example.org/report")
        self.assertEqual(citations[0].style, "markdown_link")

    def test_author_year_marker(self):
        text = "Series B closed in November (Helios Analytics, 2024)."
        citations = parse_citations(text, url_map={"(Helios Analytics, 2024)": "https://a"})
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].url, "https://a")

    def test_spans_are_ordered_and_disjoint(self):
        text = "First.[^1] Second.[^2]"
        citations = parse_citations(text, url_map={"1": "https://a", "2": "https://b"})
        starts = [citation.span_start for citation in citations]
        self.assertEqual(starts, sorted(starts))
        for left, right in zip(citations, citations[1:]):
            self.assertLess(left.span_end, right.span_start)


class ClaimSplitTests(unittest.TestCase):
    def test_inline_marker_binds_to_its_own_sentence(self):
        # The regression this guards: a marker inside a sentence used to bind
        # to the NEXT sentence because the current one had not ended yet.
        text = (
            "In 2024 the company generated 480 gigawatt-hours.[^1]\n"
            "It employed 84 staff.[^2]"
        )
        citations = parse_citations(text, url_map={"1": "https://a", "2": "https://b"})
        claims = split_claims(text, citations)
        self.assertEqual(len(claims), 2)
        self.assertEqual([citation.marker for citation in claims[0].citations], ["[^1]"])
        self.assertEqual([citation.marker for citation in claims[1].citations], ["[^2]"])

    def test_line_boundaries_split_claims(self):
        text = "First claim on one line.\nSecond claim on the next line."
        claims = split_claims(text)
        self.assertEqual(len(claims), 2)
        self.assertTrue(claims[0].claim_text.startswith("First claim"))
        self.assertTrue(claims[1].claim_text.startswith("Second claim"))

    def test_footnote_lines_are_not_claims(self):
        text = "A claim.[^1]\n\n[^1]: https://example.org/a\n"
        claims = split_claims(text, parse_citations(text, url_map={"1": "https://a"}))
        self.assertEqual(len(claims), 1)
        self.assertNotIn("[^1]:", claims[0].claim_text)

    def test_rhetorical_sentence_is_non_verifiable(self):
        text = "The company delivered results.\nOverall, the outlook remains positive."
        claims = split_claims(text)
        self.assertFalse(claims[0].non_verifiable)
        self.assertTrue(claims[1].non_verifiable)

    def test_verification_text_strips_markers(self):
        text = "The company generated 480 gigawatt-hours in 2024.[^1]"
        claims = split_claims(text, parse_citations(text, url_map={"1": "https://a"}))
        self.assertNotIn("[^1]", claims[0].verification_text)
        self.assertIn("480 gigawatt-hours", claims[0].verification_text)

    def test_span_offsets_are_byte_accurate(self):
        text = "First claim.[^1] Second claim."
        claims = split_claims(text)
        for claim in claims:
            self.assertEqual(text[claim.span_start:claim.span_end].strip(), claim.claim_text)


if __name__ == "__main__":
    unittest.main()
