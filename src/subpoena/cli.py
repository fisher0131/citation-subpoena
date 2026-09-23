"""Command-line entry point for Subpoena."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from subpoena import AuditConfig
from subpoena.claims import parse_footnote_definitions
from subpoena.pipeline import audit


def _load_url_map(path: Path | None, footnote_defs: dict[str, str]) -> dict[str, str]:
    """Resolve citation markers to URLs from a JSON file plus footnote defs."""

    url_map: dict[str, str] = {}
    if path and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for key, value in payload.items():
            url_map[str(key)] = str(value)
    for number, target in footnote_defs.items():
        url_map.setdefault(number, target)
    return url_map


def _format_verdict(row: dict, index: int) -> str:
    label = row["grounding_label"]
    marker = "HALLUCINATION" if label in {"G1_CITATION_BINDING", "G2_EVIDENCE_GAP", "G3_CONTRADICTED"} else "ok"
    quote = f"  quote: {row['quote']}" if row.get("quote") else ""
    return (
        f"[{index + 1}] ({marker}) {label}\n    {row['claim_text']}\n"
        f"    evidence: {row.get('evidence_url')}{quote}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether a finished report's citations support its claims."
    )
    parser.add_argument("report", type=Path, help="markdown report to audit")
    parser.add_argument("--citations", type=Path, help="JSON mapping citation markers to URLs")
    parser.add_argument("--corpus", type=Path, help="offline corpus jsonl (one {url,text,...} per line)")
    parser.add_argument(
        "--fetch",
        choices=["offline", "live"],
        default="offline",
        help="offline uses --corpus; live fetches the real web",
    )
    parser.add_argument(
        "--verify",
        choices=["rule", "model"],
        default="rule",
        help="rule = deterministic string grounding; model = LLM verifier (needs --api-key)",
    )
    parser.add_argument("--model", default="", help="model id for the live verifier")
    parser.add_argument("--api-key", default="", help="API key (or set SUBPOENA_API_KEY)")
    parser.add_argument("--api-base-url", default="", help="chat completions base URL")
    parser.add_argument("--open-access-email", default="", help="contact email for Unpaywall/OpenAlex")
    parser.add_argument("--no-open-access", action="store_true", help="disable the OA fallback")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, help="write the full audit JSON here")
    parser.add_argument("--fail-on-hallucination", type=float, default=None,
                        help="exit 1 if the hallucination rate exceeds this threshold")
    args = parser.parse_args(argv)

    report_text = args.report.read_text(encoding="utf-8-sig")
    url_map = _load_url_map(args.citations, parse_footnote_definitions(report_text))
    corpus_path = str(args.corpus) if args.corpus else None
    if args.fetch == "offline" and not corpus_path:
        parser.error("--fetch offline requires --corpus")

    api_key = args.api_key or os.environ.get("SUBPOENA_API_KEY", "")
    config = AuditConfig(
        fetch_mode=args.fetch,
        workers=args.workers,
        verify_mode=args.verify,
        model=args.model,
        api_key=api_key,
        api_base_url=args.api_base_url or "https://api.openai.com/v1",
        open_access_enabled=not args.no_open_access,
        open_access_email=args.open_access_email,
        corpus_path=corpus_path,
    )
    report = audit(report_text, config, url_map=url_map or None)

    print(f"audit {report.version}  ({report.started_at} -> {report.finished_at})")
    print(f"claims: {report.claim_count}   citations: {report.citation_count}   sources: {report.fetched_source_count}")
    summary = report.summary
    rate = summary.get("hallucination_rate")
    if rate is None:
        print("hallucination rate: n/a (no decidable claims)")
    else:
        lower, upper = summary["hallucination_rate_wilson_95"]
        print(f"hallucination rate: {rate:.1%}  wilson95 [{lower:.1%}, {upper:.1%}]")
    print(f"labels: {summary['grounding_label_counts']}")
    if report.fetch_failures:
        print(f"fetch failures ({len(report.fetch_failures)}):")
        for failure in report.fetch_failures:
            recovered = f" -> recovered via {failure['open_access_recovered_via']}" if failure.get("open_access_recovered_via") else ""
            print(f"  {failure['url']}  {failure['status']}/{failure['failure_class']}{recovered}")
    print("\nverdicts:")
    for index, row in enumerate(report.verdicts):
        print(_format_verdict(row, index))
        print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"full audit written to {args.output}")

    if args.fail_on_hallucination is not None and rate is not None and rate > args.fail_on_hallucination:
        print(f"hallucination rate {rate:.1%} exceeds threshold {args.fail_on_hallucination:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
