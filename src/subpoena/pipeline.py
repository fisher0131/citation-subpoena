"""End-to-end audit pipeline: fetch citations, verify claims, label grounding.

这是整个框架的编排层。它把 evidence.py（抓取）、claims.py（拆分与绑定）、
verify.py（逐字核验）和 labels.py（双轴判定）组装成一次审计运行。

设计上沿用原项目的信息边界原则：核验只看已抓取的来源快照，不看报告之外
的知识；抓取失败永远单列，不与"来源不支持"混为一谈。
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

from . import claims as claims_module
from . import verify as verify_module
from .evidence import (
    CachedFetcher,
    ChatClient,
    FetchOutcome,
    OpenAccessResolver,
    StaticFetcher,
    now_iso,
    parse_json_object,
)
from . import labels as labels_module
from .labels import ClaimVerdict, HALLUCINATION_LABELS, label_from_states, summarize_verdicts
from .verify import STATES

AUDIT_VERSION = "subpoena-v1"


@dataclass(frozen=True)
class EvidenceItem:
    """One fetched source as the verifier will see it."""

    url: str
    title: str
    snippet: str
    text: str
    fetch_status: str  # fetched | empty | failed
    failure_class: str | None
    evidence_grade: str  # full_text | snippet_only | unavailable
    content_sha256: str | None
    final_url: str
    from_cache: bool

    def to_view(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "text": self.text,
            "fetch_status": "fetched" if self.evidence_grade == "full_text" else "failed",
        }


@dataclass
class AuditConfig:
    fetch_mode: str = "offline"  # offline | live
    workers: int = 6
    fetch_timeout: float = 15.0
    fetch_retries: int = 2
    max_fetch_chars: int = 20000
    open_access_enabled: bool = True
    open_access_email: str = ""
    verify_mode: str = "rule"  # rule | model
    model: str = ""
    api_key: str = ""
    api_base_url: str = "https://api.openai.com/v1"
    model_max_tokens: int = 4096
    corpus_path: str | None = None


@dataclass
class AuditReport:
    version: str
    started_at: str
    finished_at: str
    config: dict
    claim_count: int
    citation_count: int
    fetched_source_count: int
    fetch_failures: list[dict] = field(default_factory=list)
    verdicts: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)
    model_calls: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": self.config,
            "claim_count": self.claim_count,
            "citation_count": self.citation_count,
            "fetched_source_count": self.fetched_source_count,
            "fetch_failures": self.fetch_failures,
            "verdicts": self.verdicts,
            "summary": self.summary,
            "evidence": self.evidence,
            "model_calls": self.model_calls,
            "error": self.error,
        }


class OfflineCorpus:
    """A local stand-in for the network, so the audit can run without a key.

    It serves pre-recorded page text for URLs it knows and returns a
    ``DEAD_LINK`` outcome for everything else. This is the same contract
    ``StaticFetcher`` offers, so the pipeline above never branches on offline
    versus live after the fetcher is built.
    """

    def __init__(self, records: Sequence[Mapping[str, object]]):
        self._store: dict[str, dict] = {}
        for record in records:
            url = str(record.get("url", "")).strip()
            if url:
                self._store[url] = dict(record)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "OfflineCorpus":
        records = []
        path = Path(path)
        if path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        return cls(records)

    def fetch(self, url: str):
        record = self._store.get(url)
        if record is None:
            from .evidence import classify_fetch_failure

            return FetchOutcome(
                url, url, "failed", 404, "", None, "http_404", now_iso(), 0,
                classify_fetch_failure(404, "http_404"), None, False,
            )
        text = str(record.get("text", ""))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
        return FetchOutcome(
            url, url, "fetched" if text else "empty", 200,
            text, digest, None, now_iso(), 0, None,
            str(record.get("content_type", "text/html")), False,
        )

    @property
    def urls(self) -> list[str]:
        return sorted(self._store)


def _build_fetcher(config: AuditConfig) -> object:
    if config.fetch_mode == "offline":
        if not config.corpus_path:
            raise ValueError("offline fetch_mode requires a corpus_path")
        return OfflineCorpus.from_jsonl(config.corpus_path)
    return CachedFetcher(
        StaticFetcher(
            timeout=config.fetch_timeout,
            max_retries=config.fetch_retries,
            max_chars=config.max_fetch_chars,
        ),
        cache_path=Path("results") / "fetch_cache.json",
    )


def _to_evidence_item(url: str, outcome, title: str, snippet: str, config: AuditConfig) -> EvidenceItem:
    grade = "full_text"
    if outcome.status != "fetched" or not outcome.text:
        grade = "snippet_only" if snippet else "unavailable"
    return EvidenceItem(
        url=url,
        title=title,
        snippet=snippet,
        text=outcome.text[: config.max_fetch_chars] if outcome.status == "fetched" else "",
        fetch_status=outcome.status,
        failure_class=outcome.failure_class,
        evidence_grade=grade,
        content_sha256=outcome.content_sha256,
        final_url=outcome.final_url,
        from_cache=bool(outcome.from_cache),
    )


def fetch_evidence(urls: Sequence[str], config: AuditConfig, *, titles: Mapping[str, str] | None = None) -> tuple[list[EvidenceItem], list[dict]]:
    """Fetch every cited URL once; blocked scholarly sources get an OA retry."""

    fetcher = _build_fetcher(config)
    titles = titles or {}
    oa_resolver: OpenAccessResolver | None = None
    if config.fetch_mode != "offline" and config.open_access_enabled:
        oa_resolver = OpenAccessResolver(email=config.open_access_email)

    def _one(url: str) -> tuple[EvidenceItem, dict]:
        outcome = fetcher.fetch(url)
        attempt = {
            "url": url,
            "status": outcome.status,
            "http_status": outcome.http_status,
            "failure_class": outcome.failure_class,
            "latency_ms": outcome.latency_ms,
            "open_access_attempted": False,
        }
        if outcome.status != "fetched" and oa_resolver is not None:
            attempt["open_access_attempted"] = True
            locations = oa_resolver.resolve(url, titles.get(url))
            attempt["open_access_locations"] = len(locations)
            for location in locations:
                recovered = fetcher.fetch(location.url)
                if recovered.status == "fetched" and recovered.text:
                    outcome = recovered
                    attempt["open_access_recovered_via"] = location.provider
                    attempt["open_access_url"] = location.url
                    break
        item = _to_evidence_item(url, outcome, titles.get(url, ""), "", config)
        return item, attempt

    unique_urls = sorted(set(str(url) for url in urls if url))
    if not unique_urls and config.fetch_mode == "offline" and config.corpus_path:
        # A report whose citations resolve to nothing fetchable is still
        # auditable: offline, the corpus is the whole evidence pool, so a claim
        # with no citation of its own is checked against every source the
        # auditor holds instead of being scored against an empty pool.
        unique_urls = OfflineCorpus.from_jsonl(config.corpus_path).urls
    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
        results = list(pool.map(_one, unique_urls))
    items = [item for item, _ in results]
    attempts = [attempt for _, attempt in results]
    return items, attempts


def _views_for(claim: claims_module.Claim, pool: list[EvidenceItem]) -> tuple[list[dict], list[dict]]:
    """The cited view and the pool view for one claim.

    The cited view contains only the sources the claim binds; the pool view
    contains every other fetched source. Verifying them separately is what
    separates a misbound citation (supported, but by the wrong source) from a
    genuine evidence gap (supported nowhere). A claim with no resolvable
    citation has an empty cited view and falls back to the pool alone, which
    is reported as an unresolved-citation condition rather than a silent pass.
    """

    urls = set(claim.citation_urls)
    cited = [item.to_view() for item in pool if item.url in urls]
    others = [item.to_view() for item in pool if item.url not in urls]
    return cited, others


def _verify_view(
    candidates: list[dict], view: list[dict], client, config: AuditConfig
) -> tuple[list[dict], str | None, list[dict]]:
    """Run one verifier pass over one evidence view; returns per-source rows."""

    if not view:
        return [], None, []
    if config.verify_mode == "model" and client is not None:
        request = verify_module.messages(candidates, view)
        response = client.chat(request, temperature=0.0, max_tokens=config.model_max_tokens)
        call = {
            "claim_ids": [row["claim_id"] for row in candidates],
            "view_urls": [item["url"] for item in view],
            "input_sha256": verify_module.input_hash(request),
            "status": response.status,
            "error_code": response.error_code,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cost_usd": response.cost_usd,
            "raw_output": response.text,
        }
        if response.status != "ok":
            return [], response.error_code or "verifier_call_failed", [call]
        try:
            rows, error, _ = verify_module.validate_detailed(
                parse_json_object(response.text), candidates, view
            )
        except ValueError:
            rows, error = [], "unparseable_verification_json"
        return rows, error, [call]
    rows: list[dict] = []
    for candidate in candidates:
        rows.extend(verify_module.rule_check_per_source(candidate["claim_text"], view))
    return rows, None, []


def verify_claims(
    report_claims: Sequence[claims_module.Claim],
    pool: list[EvidenceItem],
    config: AuditConfig,
) -> tuple[list[ClaimVerdict], list[dict]]:
    """Two-stage verification: the claim's own citations first, then the pool.

    The order matters. A claim whose citation is silent but which a pool source
    supports is a binding error (G1); the same claim with no pool support is an
    evidence gap (G2). Checking only the pool would merge those two cases, and
    checking only the citation would report every G1 as G2.
    """

    verdicts: list[ClaimVerdict] = []
    calls: list[dict] = []
    client = None
    if config.verify_mode == "model":
        if not config.api_key:
            raise ValueError("verify_mode=model requires an api_key")
        client = ChatClient(config.api_key, config.model, base_url=config.api_base_url)

    for claim in report_claims:
        if claim.non_verifiable:
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    grounding_label="NOT_APPLICABLE",
                    reason="rhetorical or non-verifiable sentence",
                )
            )
            continue
        cited_view, pool_view = _views_for(claim, pool)
        cited_url = claim.citation_urls[0] if claim.citation_urls else None
        unresolved = any(not citation.resolved() for citation in claim.citations)
        if not cited_view and not pool_view:
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    grounding_label="FETCH_FAILURE",
                    evidence_url=cited_url,
                    reason="no cited source could be resolved or fetched",
                )
            )
            continue
        if unresolved and not cited_view and claim.citations:
            # A citation marker that maps to no URL is a report defect of its
            # own kind: the reader cannot reach the source at all. It is not
            # evidence that the claim is false, so it never counts as a
            # hallucination in either direction.
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    grounding_label="FETCH_FAILURE",
                    evidence_url=cited_url,
                    reason="citation marker could not be resolved to any URL",
                )
            )
            continue

        candidates = [{"claim_id": claim.claim_id, "claim_text": claim.verification_text}]
        cited_rows, cited_error, cited_calls = _verify_view(candidates, cited_view, client, config)
        calls.extend(cited_calls)
        if cited_error or not any(row.get("state") == "support" for row in cited_rows):
            # Only when the claim's own citations fail to support it does the
            # wider pool decide between a misbound citation and a real gap.
            pool_rows, pool_error, pool_calls = _verify_view(candidates, pool_view, client, config)
            calls.extend(pool_calls)
            if pool_error:
                pool_rows = []
        else:
            pool_rows = []
            pool_error = None

        cited_states = _row_states(cited_rows, cited_error)
        pool_states = _row_states(pool_rows, pool_error)
        label, reason, evidence_url, _ = labels_module.decide(
            cited_states,
            pool_states,
            cited_url=cited_url,
            has_citation=bool(claim.citations),
        )
        rows = cited_rows or pool_rows
        quote = next((row.get("quote") for row in rows if isinstance(row, dict) and row.get("quote")), None)
        url_hint = evidence_url or next(
            (row.get("evidence_url") for row in rows if isinstance(row, dict) and row.get("evidence_url")), None
        )
        verdicts.append(
            ClaimVerdict(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                grounding_label=label,
                evidence_url=url_hint or cited_url,
                quote=quote,
                reason=reason,
                cited_states=tuple(sorted(cited_states.items())),
                pool_states=tuple(sorted(pool_states.items())),
            )
        )
    return verdicts, calls


def _row_states(rows: list[dict], error: str | None) -> dict[str, str]:
    """Turn per-source rows into a state map, fail-closed on a bad pass."""

    states: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = str(row.get("evidence_url") or "")
        if not url:
            continue
        state = str(row.get("state") or "unverifiable")
        if state not in verify_module.STATES:
            state = "unverifiable"
        states[url] = state
    if error:
        # A failed verifier pass cannot establish support for any source.
        states = {url: "unverifiable" for url in states}
    return states


def audit(report_text: str, config: AuditConfig, *, url_map: dict[str, str] | None = None) -> AuditReport:
    """Run a full audit of one finished report."""

    started = now_iso()
    started_perf = time.perf_counter()
    # Footnote definitions resolve markers inline, so a report that carries its
    # own reference block needs no external citation file. An explicit url_map
    # wins because it is the audited artefact, not the report's own claim about
    # what its sources are.
    combined_map = dict(claims_module.parse_footnote_definitions(report_text))
    combined_map.update(url_map or {})
    citations = claims_module.parse_citations(report_text, url_map=combined_map or None)
    report_claims = claims_module.split_claims(report_text, citations)
    pool, attempts = fetch_evidence(
        [citation.url for citation in citations if citation.url],
        config,
        titles={citation.url: citation.marker for citation in citations if citation.url},
    )
    fetch_failures = [
        {
            "url": attempt["url"],
            "status": attempt["status"],
            "http_status": attempt["http_status"],
            "failure_class": attempt["failure_class"],
            "open_access_attempted": attempt.get("open_access_attempted", False),
            "open_access_recovered_via": attempt.get("open_access_recovered_via"),
        }
        for attempt in attempts
        if attempt["status"] != "fetched"
    ]
    verdicts, calls = verify_claims(report_claims, pool, config)
    summary = summarize_verdicts(verdicts)
    summary["elapsed_seconds"] = round(time.perf_counter() - started_perf, 3)
    summary["unresolved_citation_count"] = sum(1 for citation in citations if not citation.resolved())
    finished = now_iso()
    return AuditReport(
        version=AUDIT_VERSION,
        started_at=started,
        finished_at=finished,
        config={
            "fetch_mode": config.fetch_mode,
            "verify_mode": config.verify_mode,
            "model": config.model or None,
            "corpus_path": config.corpus_path,
            "open_access_enabled": config.open_access_enabled and config.fetch_mode != "offline",
            "workers": config.workers,
        },
        claim_count=len(report_claims),
        citation_count=len(citations),
        fetched_source_count=len(pool),
        fetch_failures=fetch_failures,
        verdicts=[verdict.__dict__ for verdict in verdicts],
        summary=summary,
        evidence=[item.__dict__ for item in pool],
        model_calls=calls,
    )
