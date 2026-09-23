"""End-to-end pipeline tests against the offline demo corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import unittest

from audit_cli import main as cli_main
from subpoena import AuditConfig
from subpoena import pipeline
from subpoena.labels import HALLUCINATION_LABELS
from subpoena.pipeline import OfflineCorpus, audit

CORPUS = ROOT / "corpus" / "demo" / "sources.jsonl"
REPORT = ROOT / "corpus" / "demo" / "report.md"
CITATIONS = ROOT / "corpus" / "demo" / "citations.json"


def _config(**overrides) -> AuditConfig:
    settings = {
        "fetch_mode": "offline",
        "corpus_path": str(CORPUS),
        "verify_mode": "rule",
    }
    settings.update(overrides)
    return AuditConfig(**settings)


class OfflineCorpusTests(unittest.TestCase):
    def test_known_url_serves_full_text(self):
        corpus = OfflineCorpus.from_jsonl(CORPUS)
        outcome = corpus.fetch("https://example.org/meridian/annual-report-2024")
        self.assertEqual(outcome.status, "fetched")
        self.assertIn("480 gigawatt-hours", outcome.text)

    def test_unknown_url_is_a_dead_link(self):
        corpus = OfflineCorpus.from_jsonl(CORPUS)
        outcome = corpus.fetch("https://example.org/does-not-exist")
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_class, "DEAD_LINK")

    def test_empty_corpus_file_is_not_an_error(self):
        corpus = OfflineCorpus.from_jsonl(ROOT / "corpus" / "demo" / "absent.jsonl")
        self.assertEqual(corpus.urls, [])


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.report_text = REPORT.read_text(encoding="utf-8-sig")
        self.url_map = json.loads(CITATIONS.read_text(encoding="utf-8-sig"))

    def test_demo_audit_reproduces_expected_labels(self):
        # This is the framework's regression contract: each demo sentence is
        # built to exercise exactly one grounding label, so a change here means
        # a decision rule changed.
        result = audit(self.report_text, _config(), url_map=self.url_map)
        labels = {
            verdict["claim_text"]: verdict["grounding_label"]
            for verdict in result.verdicts
        }
        self.assertEqual(labels["In 2024 Meridian Solar generated 480 gigawatt-hours of electricity from its Northfield and Helios arrays, up from 455 gigawatt-hours in 2023.[^1]"], "SUPPORTED")
        self.assertEqual(labels["The company closed its Series B financing at 22 million dollars in November 2024.[^2]"], "SUPPORTED")
        self.assertEqual(labels["Meridian Solar holds an estimated 19 percent share of the regional utility-scale segment.[^1]"], "G1_CITATION_BINDING")
        self.assertEqual(labels["The company's generation reached 520 gigawatt-hours in 2024.[^1]"], "G3_CONTRADICTED")
        self.assertEqual(labels["Meridian Solar received 2.3 million dollars in government subsidies during 2024.[^1]"], "G2_EVIDENCE_GAP")
        self.assertEqual(labels["The company operates a battery storage pilot programme in Portugal.[^5]"], "FETCH_FAILURE")
        self.assertEqual(labels["Overall, Meridian Solar remains committed to a sustainable energy future."], "NOT_APPLICABLE")

    def test_hallucination_rate_is_over_the_wilson_interval(self):
        result = audit(self.report_text, _config(), url_map=self.url_map)
        summary = result.summary
        decidable = summary["decidable_claim_count"]
        hallucinations = summary["hallucination_count"]
        self.assertEqual(decidable, 7)
        self.assertEqual(hallucinations, 3)
        lower, upper = summary["hallucination_rate_wilson_95"]
        self.assertLessEqual(lower, summary["hallucination_rate"])
        self.assertGreaterEqual(upper, summary["hallucination_rate"])

    def test_quote_is_recorded_for_supported_claims(self):
        result = audit(self.report_text, _config(), url_map=self.url_map)
        supported = next(v for v in result.verdicts if v["grounding_label"] == "SUPPORTED")
        self.assertIn("480", supported["quote"])

    def test_every_fetched_source_has_a_snapshot_hash(self):
        result = audit(self.report_text, _config(), url_map=self.url_map)
        self.assertEqual(len(result.evidence), 4)
        for item in result.evidence:
            self.assertTrue(item["content_sha256"])

    def test_footnote_definitions_supply_the_url_map(self):
        # With no explicit --citations file, the footnote block alone must
        # resolve every marker the report actually uses.
        result = audit(self.report_text, _config())
        labels = {verdict["claim_text"]: verdict["grounding_label"] for verdict in result.verdicts}
        self.assertEqual(
            labels["The company closed its Series B financing at 22 million dollars in November 2024.[^2]"],
            "SUPPORTED",
        )

    def test_claims_without_citations_fall_back_to_the_pool(self):
        result = audit("The company holds an estimated 19 percent share of the regional segment.", _config())
        self.assertEqual(len(result.verdicts), 1)
        self.assertEqual(result.verdicts[0]["grounding_label"], "SUPPORTED")

    def test_unresolvable_citation_is_fetch_failure_not_hallucination(self):
        result = audit("The company employs 120 people in Portugal.[^42]", _config())
        self.assertEqual(result.verdicts[0]["grounding_label"], "FETCH_FAILURE")
        self.assertNotIn(result.verdicts[0]["grounding_label"], HALLUCINATION_LABELS)

    def test_corpus_gap_is_g2_even_when_the_pool_is_large(self):
        result = audit("The company received a 9 million dollar grant from the region.", _config())
        self.assertEqual(result.verdicts[0]["grounding_label"], "G2_EVIDENCE_GAP")

    def test_model_mode_without_key_is_refused(self):
        with self.assertRaises(ValueError):
            audit(self.report_text, _config(verify_mode="model"), url_map=self.url_map)

    def test_offline_without_corpus_is_refused(self):
        with self.assertRaises(ValueError):
            audit(self.report_text, AuditConfig(fetch_mode="offline"))


class ModelVerifierTests(unittest.TestCase):
    """The model path reuses the same validation contract as the rule path.

    A stub client replays canned responses so the plumbing is exercised without
    network access; the fail-closed behaviour is what is under test, not the
    model's accuracy.
    """

    class _StubClient:
        model = "stub"
        reasoning_effort = None

        def __init__(self, payloads):
            self.payloads = list(payloads)
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append(messages)
            payload = self.payloads.pop(0) if self.payloads else {"checks": []}
            from subpoena.evidence import ChatResult

            return ChatResult(
                text=json.dumps(payload),
                status="ok",
                prompt_tokens=1,
                completion_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
                error_code=None,
                raw_usage={},
            )

    def setUp(self):
        self.report_text = REPORT.read_text(encoding="utf-8-sig")
        self.url_map = json.loads(CITATIONS.read_text(encoding="utf-8-sig"))
        self.source_url = "https://example.org/meridian/annual-report-2024"
        self.text = OfflineCorpus.from_jsonl(CORPUS)._store[self.source_url]["text"]
        # A real quote must be copied from the snapshot; pick the sentence the
        # claim restates so the model path can genuinely validate it.
        self.quote = next(s for s in self.text.split(".") if "480 gigawatt-hours" in s).strip()

    def _install_client(self, client):
        original = pipeline.ChatClient
        pipeline.ChatClient = lambda *args, **kwargs: client
        return original

    def _single_claim_evidence(self):
        from subpoena.claims import parse_citations, split_claims

        text = "The company generated 480 gigawatt-hours in 2024.[^1]"
        citations = parse_citations(text, url_map={"1": self.source_url})
        claims = split_claims(text, citations)
        evidence = [
            pipeline.EvidenceItem(
                url=self.source_url, title="t", snippet="", text=self.text,
                fetch_status="fetched", failure_class=None, evidence_grade="full_text",
                content_sha256="x", final_url=self.source_url, from_cache=False,
            )
        ]
        return claims, evidence

    def test_supporting_model_row_is_accepted(self):
        claims, evidence = self._single_claim_evidence()
        client = self._StubClient([
            {"checks": [{"claim_id": claims[0].claim_id, "evidence_url": self.source_url, "state": "support", "quote": self.quote}]}
        ])
        original = self._install_client(client)
        try:
            verdicts, calls = pipeline.verify_claims(
                claims, evidence, _config(verify_mode="model", api_key="stub")
            )
        finally:
            pipeline.ChatClient = original
        self.assertEqual(verdicts[0].grounding_label, "SUPPORTED")
        self.assertEqual(len(calls), 1)

    def test_invented_model_quote_fails_closed(self):
        claims, evidence = self._single_claim_evidence()
        claim_id = claims[0].claim_id
        source_url = self.source_url

        class _Liar:
            model = "stub"
            reasoning_effort = None

            def chat(self, messages, **kwargs):
                from subpoena.evidence import ChatResult

                return ChatResult(
                    text=json.dumps({"checks": [
                        {"claim_id": claim_id, "evidence_url": source_url,
                         "state": "support", "quote": "this sentence does not exist in the snapshot"}
                    ]}),
                    status="ok", prompt_tokens=1, completion_tokens=1, cost_usd=0.0,
                    latency_ms=1, error_code=None, raw_usage={},
                )

        original = self._install_client(_Liar())
        try:
            verdicts, _ = pipeline.verify_claims(
                claims, evidence, _config(verify_mode="model", api_key="stub")
            )
        finally:
            pipeline.ChatClient = original
        self.assertNotEqual(verdicts[0].grounding_label, "SUPPORTED")


class CliTests(unittest.TestCase):
    def test_cli_writes_an_audit_artifact(self):
        output = ROOT / "results" / "test_cli_audit.json"
        if output.exists():
            output.unlink()
        exit_code = cli_main([
            str(REPORT),
            "--corpus", str(CORPUS),
            "--citations", str(CITATIONS),
            "--fetch", "offline",
            "--output", str(output),
        ])
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], "subpoena-v1")
        self.assertEqual(payload["claim_count"], 9)

    def test_cli_exit_code_on_hallucination_threshold(self):
        # The demo hallucination rate is 3/7 = 43%; a 20% gate must fail.
        exit_code = cli_main([
            str(REPORT),
            "--corpus", str(CORPUS),
            "--fetch", "offline",
            "--fail-on-hallucination", "0.2",
            "--output", str(ROOT / "results" / "test_cli_threshold.json"),
        ])
        self.assertEqual(exit_code, 1)

    def test_cli_passes_when_threshold_is_above_the_rate(self):
        exit_code = cli_main([
            str(REPORT),
            "--corpus", str(CORPUS),
            "--fetch", "offline",
            "--fail-on-hallucination", "0.9",
            "--output", str(ROOT / "results" / "test_cli_pass.json"),
        ])
        self.assertEqual(exit_code, 0)

    def test_cli_offline_without_corpus_errors(self):
        with self.assertRaises(SystemExit):
            cli_main([str(REPORT), "--fetch", "offline"])


if __name__ == "__main__":
    unittest.main()
