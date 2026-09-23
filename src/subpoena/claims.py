"""Claim splitting and citation-marker binding for a finished report.

拆分规则移植自 annotation_manual_v1.md 第 1.1 节；引用标记解析支持 markdown
脚注 [^n]、行内 [n]、(Author, Year) 与 [source](url) 四种常见写法。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Citation marker parsing
# --------------------------------------------------------------------------- #

#: Numeric bracket markers: [1], [2,3], [^1], [^1,2]
_BRACKET_NUMERIC = re.compile(r"\[\^?([0-9]+(?:\s*,\s*[0-9]+)*)\]")
#: Author-year markers: (Smith, 2020) / (Smith and Jones, 2020) / (Smith et al., 2020; Jones, 2019)
_AUTHOR_YEAR = re.compile(
    r"\(([A-Z][A-Za-z'\-]+(?:\s+(?:and|et al\.?,?|&)?\s?[A-Z]?[A-Za-z'\-]*)*),"
    r"(?:\s*([0-9]{4}))[a-z]?(?:;[^\)]*)?\)"
)
#: Markdown inline links: [text](url)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)")
#: Markdown footnote definitions: [^1]: url
_FOOTNOTE_DEF = re.compile(r"^\[\^([0-9]+)\]:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Citation:
    """One citation occurrence in the report."""

    citation_id: str
    marker: str  # the raw marker text, e.g. "[1]" or "(Smith, 2020)"
    style: str  # bracket_numeric | author_year | markdown_link | footnote
    url: str | None = None
    span_start: int = 0
    span_end: int = 0

    def resolved(self) -> bool:
        return bool(self.url)


def _stable_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(material).hexdigest()[:12]}"


def parse_footnote_definitions(text: str) -> dict[str, str]:
    """Return {marker number: url/text} for [^n]: ... definition lines."""

    definitions: dict[str, str] = {}
    for match in _FOOTNOTE_DEF.finditer(text or ""):
        definitions[str(match.group(1))] = match.group(2).strip()
    return definitions


def parse_citations(text: str, *, url_map: dict[str, str] | None = None) -> list[Citation]:
    """Find every citation marker in the report and resolve it to a URL.

    ``url_map`` supplies an explicit marker->URL mapping (the ``--citations``
    file) for markers that carry no URL inline. A citation that stays
    unresolved is kept, because an unresolvable marker is itself a finding.
    """

    text = text or ""
    url_map = url_map or {}
    citations: list[Citation] = []
    seen: set[tuple[int, int]] = set()
    # Footnote definitions are report apparatus: the marker inside them would
    # otherwise be parsed as a second, duplicate citation for the same source.
    apparatus_spans = [
        (match.start(), match.end()) for match in _FOOTNOTE_DEF.finditer(text)
    ]

    def _is_apparatus(span: tuple[int, int]) -> bool:
        return any(start <= span[0] and span[1] <= end for start, end in apparatus_spans)

    for match in _MARKDOWN_LINK.finditer(text):
        key = (match.start(), match.end())
        if key in seen or _is_apparatus(key):
            continue
        seen.add(key)
        citations.append(
            Citation(
                citation_id=_stable_id("cit", match.start(), match.group(1)),
                marker=match.group(0),
                style="markdown_link",
                url=match.group(2),
                span_start=match.start(),
                span_end=match.end(),
            )
        )

    for match in _BRACKET_NUMERIC.finditer(text):
        key = (match.start(), match.end())
        if key in seen or _is_apparatus(key):
            continue
        seen.add(key)
        numbers = [part.strip() for part in match.group(1).split(",")]
        # A multi-number marker resolves to the first definition; the audit
        # records one citation per marker, not per number.
        first = numbers[0]
        url = url_map.get(first) or url_map.get(match.group(0))
        citations.append(
            Citation(
                citation_id=_stable_id("cit", match.start(), first),
                marker=match.group(0),
                style="bracket_numeric",
                url=url,
                span_start=match.start(),
                span_end=match.end(),
            )
        )

    for match in _AUTHOR_YEAR.finditer(text):
        key = (match.start(), match.end())
        if key in seen or _is_apparatus(key):
            continue
        seen.add(key)
        marker_key = match.group(0)
        url = url_map.get(marker_key)
        citations.append(
            Citation(
                citation_id=_stable_id("cit", match.start(), marker_key),
                marker=marker_key,
                style="author_year",
                url=url,
                span_start=match.start(),
                span_end=match.end(),
            )
        )

    citations.sort(key=lambda citation: citation.span_start)
    return citations


# --------------------------------------------------------------------------- #
# Claim splitting
# --------------------------------------------------------------------------- #

#: Sentence splitter: splits on line boundaries and sentence-final punctuation.
_LINE_OR_SENTENCE = re.compile(r"\n+|(?<=[.!?;])\s+(?=[A-Z0-9\"'(\[])")
#: Rhetorical / non-verifiable filler, per annotation_manual_v1.md rule 4.
_NON_VERIFIABLE_HINTS = (
    "it is worth noting",
    "in conclusion",
    "overall,",
    "as shown above",
    "needless to say",
    "importantly,",
)
#: Footnote definition lines are report apparatus, not claims.
_FOOTNOTE_LINE = re.compile(r"^\s*\[\^[0-9]+\]:")
#: Headings and list markers carry no verifiable proposition of their own.
_HEADING_LINE = re.compile(r"^\s*#{1,6}\s*")


@dataclass(frozen=True)
class Claim:
    """One minimal verifiable claim and the citations bound to it."""

    claim_id: str
    claim_text: str
    span_start: int
    span_end: int
    citations: tuple[Citation, ...] = ()
    non_verifiable: bool = False

    @property
    def citation_urls(self) -> tuple[str, ...]:
        return tuple(citation.url for citation in self.citations if citation.url)

    @property
    def verification_text(self) -> str:
        """The claim with citation markers stripped, for source comparison.

        The verifier compares this text against a source snapshot. Leaving the
        marker in would break the exact-span match: ``... in 2023.[^1]`` never
        appears in any source, so every inline-cited claim would look
        unsupported even when the source states it verbatim.
        """

        text = self.claim_text
        for citation in self.citations:
            text = text.replace(citation.marker, "")
        return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Return (sentence, start, end) with byte-accurate offsets.

    Line boundaries are treated as sentence boundaries: a markdown report is a
    sequence of lines, and treating a newline as whitespace continuation would
    merge the first sentence of the report with everything after it.
    """

    text = text or ""
    pieces: list[tuple[int, int]] = []
    last = 0
    for match in _LINE_OR_SENTENCE.finditer(text):
        if match.start() > last:
            pieces.append((last, match.start()))
        last = match.end()
    if last < len(text):
        pieces.append((last, len(text)))
    sentences: list[tuple[str, int, int]] = []
    for start, end in pieces:
        chunk = text[start:end].strip()
        if not chunk:
            continue
        if _FOOTNOTE_LINE.match(chunk) or _HEADING_LINE.match(chunk):
            continue
        sentences.append((chunk, start, end))
    return sentences


CLAIM_SPLIT_VERSION = "rule-based-claim-split-v2"


def _is_non_verifiable(sentence: str) -> bool:
    lowered = sentence.strip().lower()
    if not lowered:
        return True
    if lowered.endswith("?"):
        return True
    return any(hint in lowered for hint in _NON_VERIFIABLE_HINTS)


def split_claims(text: str, citations: list[Citation] | None = None) -> list[Claim]:
    """Split a report into minimal verifiable claims and bind citations.

    The binding rule is proximity: a citation follows the claim it supports.
    A citation always binds to the nearest claim whose span ends at or before
    the citation, which is the convention in numbered-reference reports.
    """

    text = text or ""
    citations = list(citations or parse_citations(text))
    claims: list[Claim] = []
    for index, (sentence, start, end) in enumerate(_split_sentences(text)):
        claim_id = _stable_id("clm", index, sentence)
        claims.append(
            Claim(
                claim_id=claim_id,
                claim_text=sentence.strip(),
                span_start=start,
                span_end=end,
                non_verifiable=_is_non_verifiable(sentence),
            )
        )
    if not claims:
        return claims

    # Bind each citation to the claim that contains it, falling back to the
    # nearest preceding claim for markers that sit between sentences. A marker
    # inside a sentence (`... in 2023.[^1]`) must bind to that sentence, not to
    # the next one, so `span_end <= anchor` alone is wrong: it misses every
    # inline marker because the sentence has not ended yet at the marker.
    for citation in citations:
        anchor_start, anchor_end = citation.span_start, citation.span_end
        bound_index = 0
        for index, claim in enumerate(claims):
            if claim.span_start <= anchor_start and anchor_end <= claim.span_end:
                bound_index = index
                break
            if claim.span_start > anchor_start:
                break
            bound_index = index
        bound = claims[bound_index]
        claims[bound_index] = Claim(
            claim_id=bound.claim_id,
            claim_text=bound.claim_text,
            span_start=bound.span_start,
            span_end=bound.span_end,
            citations=tuple(sorted(set(bound.citations + (citation,)), key=lambda c: c.span_start)),
            non_verifiable=bound.non_verifiable,
        )
    return claims
