"""Fetching cited sources with explicit failure classes.

移植自 citation_entropy 的 backends.py，保留 StaticFetcher / CachedFetcher /
OpenAccessResolver 与失败分类，删掉了与本任务无关的搜索后端。
抓取失败永远不得被当成"证据不存在"。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import requests

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}

FETCH_FAILURE_CLASSES = (
    "BLOCKED_PERMANENT",
    "BLOCKED_TRANSIENT",
    "FORMAT_UNSUPPORTED",
    "DEAD_LINK",
    "TIMEOUT",
    "EMPTY_CONTENT",
    "OTHER",
)

_NETWORK_ERRORS = {
    "Timeout",
    "ReadTimeout",
    "ConnectTimeout",
    "ConnectionError",
    "SSLError",
    "TooManyRedirects",
    "ChunkedEncodingError",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def classify_fetch_failure(http_status: int | None, error_code: str | None) -> str | None:
    """Map a transport error onto the frozen failure taxonomy.

    The taxonomy is deliberately separate from the grounding label
    ``FETCH_FAILURE``: it records *why* a source could not be obtained, so a
    permanent WAF block is never silently treated as evidence absence.
    """

    if not error_code:
        return None
    if error_code == "pdf_not_supported":
        return "FORMAT_UNSUPPORTED"
    if error_code == "empty_content":
        return "EMPTY_CONTENT"
    if error_code in _NETWORK_ERRORS or "Timeout" in error_code:
        return "TIMEOUT"
    if http_status in (401, 403, 451):
        return "BLOCKED_PERMANENT"
    if http_status in (408, 425, 429) or (http_status is not None and http_status >= 500):
        return "BLOCKED_TRANSIENT"
    if http_status in (404, 410):
        return "DEAD_LINK"
    return "OTHER"


@dataclass(frozen=True)
class FetchOutcome:
    url: str
    final_url: str
    status: str  # fetched | empty | failed
    http_status: int | None
    text: str
    content_sha256: str | None
    error_code: str | None
    fetched_at: str
    latency_ms: int
    failure_class: str | None = None
    content_type: str | None = None
    from_cache: bool = False


_FETCH_OUTCOME_FIELDS = tuple(item.name for item in fields(FetchOutcome))


class _TextExtractor:
    """Minimal HTML text extractor (from html.parser.HTMLParser in the original)."""

    _SKIP = {"script", "style", "nav", "footer", "header", "noscript", "svg", "form", "iframe", "button"}

    def __init__(self) -> None:
        from html.parser import HTMLParser

        self._parser = HTMLParser(convert_charrefs=True)
        self._parser.handle_starttag = self._handle_starttag
        self._parser.handle_endtag = self._handle_endtag
        self._parser.handle_data = self._handle_data
        self._skip_depth = 0
        self.parts: list[str] = []

    def _handle_starttag(self, tag, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def _handle_endtag(self, tag) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def _handle_data(self, data) -> None:
        if not self._skip_depth:
            text = data.strip()
            if text:
                self.parts.append(text)

    def feed(self, markup: str) -> None:
        self._parser.feed(markup)


class StaticFetcher:
    """Synchronous HTML/PDF/XML fetch with content hashing and failure states."""

    _RETRYABLE_STATUS = (403, 408, 425, 429, 500, 502, 503, 504)

    def __init__(
        self,
        timeout: float = 15.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        max_chars: int = 20000,
        *,
        pdf_support: bool = True,
        backoff_factor: float = 2.0,
        max_pdf_pages: int = 50,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_chars = max_chars
        self.pdf_support = pdf_support
        self.backoff_factor = backoff_factor
        self.max_pdf_pages = max_pdf_pages

    def _delay(self, attempt: int) -> float:
        return self.retry_delay * (self.backoff_factor**attempt)

    def _extract_xml(self, content: bytes) -> str:
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return ""
        parts: list[str] = []
        for section in ("abstract", "body"):
            node = root.find(f".//{section}")
            if node is not None:
                parts.extend(text.strip() for text in node.itertext() if text.strip())
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def _extract_pdf(self, content: bytes) -> str:
        if not self.pdf_support or PdfReader is None:
            return ""
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception:
            return ""
        parts = []
        for page in reader.pages[: self.max_pdf_pages]:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                parts.append(text)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def fetch(self, url: str) -> FetchOutcome:
        started = time.perf_counter()
        last_error = None
        http_status = None
        final_url = url
        text = ""
        content_type = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(
                    url, headers=DEFAULT_HEADERS, timeout=self.timeout, allow_redirects=True
                )
                http_status = response.status_code
                final_url = str(response.url)
                content_type = response.headers.get("content-type", "").lower()
                if http_status >= 400:
                    last_error = f"http_{http_status}"
                    if http_status in self._RETRYABLE_STATUS and attempt < self.max_retries:
                        time.sleep(self._delay(attempt))
                        continue
                    break
                if "application/pdf" in content_type:
                    if not self.pdf_support or PdfReader is None:
                        last_error = "pdf_not_supported"
                        break
                    text = self._extract_pdf(response.content)[: self.max_chars]
                    if not text.strip():
                        last_error = "empty_content"
                        break
                elif "xml" in content_type:
                    text = self._extract_xml(response.content)[: self.max_chars]
                    if not text.strip():
                        last_error = "empty_content"
                        break
                else:
                    parser = _TextExtractor()
                    parser.feed(response.text)
                    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()[: self.max_chars]
                    if not text.strip():
                        last_error = "empty_content"
                        break
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                return FetchOutcome(
                    url,
                    final_url,
                    "fetched",
                    http_status,
                    text,
                    digest,
                    None,
                    now_iso(),
                    int((time.perf_counter() - started) * 1000),
                    None,
                    content_type or None,
                )
            except requests.RequestException as exc:
                last_error = type(exc).__name__
                if attempt < self.max_retries:
                    time.sleep(self._delay(attempt))
                    continue
                break
        status = "empty" if last_error == "empty_content" else "failed"
        return FetchOutcome(
            url,
            final_url,
            status,
            http_status,
            "",
            None,
            last_error,
            now_iso(),
            int((time.perf_counter() - started) * 1000),
            classify_fetch_failure(http_status, last_error),
            content_type or None,
        )


class CachedFetcher:
    """Thread-safe URL-level cache; page content is not the randomized variable."""

    def __init__(self, inner, cache_path=None):
        self.inner = inner
        self.cache_path = Path(cache_path) if cache_path else None
        self._lock = threading.Lock()
        self._store = self._load()
        self.hits = 0
        self.misses = 0

    def _load(self) -> dict:
        if not self.cache_path or not self.cache_path.is_file():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _persist(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._store, ensure_ascii=False, sort_keys=True)
        temporary = self.cache_path.with_suffix(
            f"{self.cache_path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        for attempt in range(6):
            try:
                temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, self.cache_path)
                return
            except PermissionError:
                time.sleep(0.1 * (attempt + 1))
        try:
            temporary.unlink()
        except OSError:
            pass

    def fetch(self, url: str) -> FetchOutcome:
        with self._lock:
            cached = self._store.get(url)
        if isinstance(cached, dict):
            with self._lock:
                self.hits += 1
            filtered = {k: v for k, v in cached.items() if k in _FETCH_OUTCOME_FIELDS}
            return replace(FetchOutcome(**filtered), from_cache=True)
        outcome = self.inner.fetch(url)
        with self._lock:
            self._store[url] = asdict(outcome)
            self.misses += 1
            self._persist()
        return outcome

    @property
    def stats(self) -> dict:
        return {"cache_hits": self.hits, "cache_misses": self.misses, "cached_urls": len(self._store)}


# --------------------------------------------------------------------------- #
# Open-access fallback for blocked scholarly sources
# --------------------------------------------------------------------------- #

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)
_PMCID_RE = re.compile(r"(PMC\d{6,9})", re.I)


def extract_doi(value: str) -> str | None:
    match = _DOI_RE.search(unquote(value or ""))
    return match.group(0).rstrip(").,;") if match else None


def extract_pmcid(value: str) -> str | None:
    match = _PMCID_RE.search(value or "")
    return match.group(1).upper() if match else None


def title_query(title: str, *, max_words: int = 10) -> str:
    return " ".join(re.sub(r"[^A-Za-z0-9 ]", " ", title or "").split()[:max_words])


@dataclass(frozen=True)
class OpenAccessLocation:
    provider: str  # ncbi_pmc | openalex | unpaywall
    url: str
    kind: str  # pmc_fulltext_xml | pdf | landing


class OpenAccessResolver:
    """Resolve a blocked scholarly source through public, documented APIs.

    It only asks metadata services for a copy that is already openly available.
    It never solves a bot check, rotates identity, or bypasses a paywall.
    """

    EUtils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    OpenAlex = "https://api.openalex.org/works"
    Unpaywall = "https://api.unpaywall.org/v2"

    def __init__(
        self,
        *,
        email: str = "",
        timeout: float = 20.0,
        user_agent: str = "subpoena/0.1",
        enable_ncbi_pmc: bool = True,
        enable_openalex: bool = True,
        enable_unpaywall: bool = True,
    ):
        self.email = (email or "").strip()
        self.timeout = timeout
        self.enable_ncbi_pmc = enable_ncbi_pmc
        self.enable_openalex = enable_openalex
        self.enable_unpaywall = enable_unpaywall and bool(self.email)
        contact = f" (mailto:{self.email})" if self.email else ""
        self._headers = {"User-Agent": f"{user_agent}{contact}", "Accept": "application/json"}

    @property
    def enabled(self) -> bool:
        return self.enable_ncbi_pmc or self.enable_openalex or self.enable_unpaywall

    @staticmethod
    def _openalex_locations(work: dict) -> list[OpenAccessLocation]:
        out: list[OpenAccessLocation] = []
        locations = [work.get("best_oa_location")] + list(work.get("locations") or [])
        for location in locations:
            if not isinstance(location, dict) or not location.get("is_oa"):
                continue
            for kind, url in (
                ("pdf", location.get("pdf_url")),
                ("landing", location.get("landing_page_url")),
            ):
                if url:
                    out.append(OpenAccessLocation("openalex", url, kind))
        return out

    @staticmethod
    def _unpaywall_locations(payload: dict) -> list[OpenAccessLocation]:
        out: list[OpenAccessLocation] = []
        locations = [payload.get("best_oa_location")] + list(payload.get("oa_locations") or [])
        for location in locations:
            if not isinstance(location, dict):
                continue
            for kind, url in (
                ("pdf", location.get("url_for_pdf")),
                ("landing", location.get("url_for_landing_page") or location.get("url")),
            ):
                if url:
                    out.append(OpenAccessLocation("unpaywall", url, kind))
        return out

    def _get_json(self, url: str, params: dict | None = None) -> dict:
        try:
            response = requests.get(url, params=params, headers=self._headers, timeout=self.timeout)
            if response.status_code != 200:
                return {}
            payload = response.json()
        except (requests.RequestException, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _doi_locations(self, doi: str) -> list[OpenAccessLocation]:
        out: list[OpenAccessLocation] = []
        if self.enable_openalex:
            work = self._get_json(f"{self.OpenAlex}/doi:{doi}")
            if work:
                out.extend(self._openalex_locations(work))
        if self.enable_unpaywall:
            payload = self._get_json(f"{self.Unpaywall}/{doi}", {"email": self.email})
            if payload:
                out.extend(self._unpaywall_locations(payload))
        return out

    def resolve(self, url: str, title: str | None = None) -> list[OpenAccessLocation]:
        """Return official open-access copies for a blocked source, best first."""

        if not self.enabled:
            return []
        found: list[OpenAccessLocation] = []
        pmcid = extract_pmcid(url)
        if pmcid and self.enable_ncbi_pmc:
            query = f"?db=pmc&id={pmcid}&rettype=xml&retmode=xml"
            found.append(OpenAccessLocation("ncbi_pmc", f"{self.EUtils}{query}", "pmc_fulltext_xml"))
        doi = extract_doi(url)
        if doi:
            found.extend(self._doi_locations(doi))
        seen: set[str] = set()
        unique: list[OpenAccessLocation] = []
        for location in found:
            if location.url and location.url not in seen:
                seen.add(location.url)
                unique.append(location)
        return unique


def parse_json_object(text: str):
    """Extract the first JSON object/array from a response body."""

    import json as _json

    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return _json.loads(text[start : end + 1])
            except _json.JSONDecodeError:
                continue
    raise ValueError("response did not contain a JSON object or array")


@dataclass
class ChatResult:
    text: str
    status: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    error_code: str | None = None
    raw_usage: dict = field(default_factory=dict)


class ChatClient:
    """Minimal OpenAI-compatible chat client (requests-only, no SDK dependency)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 90.0,
        max_retries: int = 3,
        retry_delay: float = 3.0,
    ):
        if not api_key:
            raise ValueError("an API key is required for live verification")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(
        self,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    body = response.json()
                    usage = body.get("usage") or {}
                    text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
                    return ChatResult(
                        text,
                        "ok",
                        int(usage.get("prompt_tokens", 0)),
                        int(usage.get("completion_tokens", 0)),
                        float(usage.get("cost", 0.0) or 0.0),
                        int((time.perf_counter() - started) * 1000),
                        None,
                        usage,
                    )
                last_error = f"http_{response.status_code}"
                if response.status_code in (401, 403, 402):
                    break
            except requests.RequestException as exc:
                last_error = type(exc).__name__
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (attempt + 1))
        return ChatResult("", "failed", 0, 0, 0.0, int((time.perf_counter() - started) * 1000), last_error)
