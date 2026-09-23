"""Grounding labels and the fail-closed decision rules.

Label taxonomy 移植自 citation_entropy 的标注规范 (annotation_manual_v1.md)
与生产 judge 提示 (run_real_pilot.JUDGE_SYS)。双轴必须独立：引用存在不等于
事实正确，结论正确不等于绑定正确。

判定顺序固定且最严重优先：来源直接反驳 > 绑定沉默但别处支持 > 哪都没有。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

GROUNDING_LABELS = (
    "SUPPORTED",
    "G1_CITATION_BINDING",
    "G2_EVIDENCE_GAP",
    "G3_CONTRADICTED",
    "FETCH_FAILURE",
    "NOT_APPLICABLE",
    "UNVERIFIABLE",
)

#: 幻觉标签：这三类说明引用没有真正支持陈述。
HALLUCINATION_LABELS = frozenset(
    {"G1_CITATION_BINDING", "G2_EVIDENCE_GAP", "G3_CONTRADICTED"}
)

#: 计入幻觉率分母的标签：能明确判定的只有 SUPPORTED 与三类幻觉。
DECIDABLE_LABELS = HALLUCINATION_LABELS | {"SUPPORTED"}

#: 证据状态，与 verify.STATES 对齐。
STATES = ("support", "contradict", "not_found", "unverifiable")

VERSION = "audit-labels-v1"


@dataclass(frozen=True)
class ClaimVerdict:
    """One claim's grounding verdict against the assembled evidence pool."""

    claim_id: str
    claim_text: str
    grounding_label: str
    evidence_url: str | None = None
    quote: str | None = None
    reason: str = ""
    cited_states: tuple[tuple[str, str], ...] = ()
    pool_states: tuple[tuple[str, str], ...] = ()

    @property
    def is_hallucination(self) -> bool:
        return self.grounding_label in HALLUCINATION_LABELS

    @property
    def is_decidable(self) -> bool:
        return self.grounding_label in DECIDABLE_LABELS


def _states_map(pairs: Mapping[str, str] | tuple[tuple[str, str], ...]) -> dict[str, str]:
    if isinstance(pairs, Mapping):
        return dict(pairs)
    return {str(url): str(state) for url, state in pairs}


def decide(
    cited_states: Mapping[str, str] | tuple[tuple[str, str], ...],
    pool_states: Mapping[str, str] | tuple[tuple[str, str], ...] = (),
    *,
    cited_url: str | None = None,
    has_citation: bool = True,
) -> tuple[str, str, str | None, str | None]:
    """Apply the frozen decision order to one claim's per-source states.

    Returns ``(label, reason, evidence_url, quote)``. ``cited_states`` are the
    states for the citations the claim actually binds; ``pool_states`` cover
    every other fetched source and are what separates a misbound citation from
    a genuine evidence gap. ``has_citation`` tells the decision order whether
    the claim binds any citation at all: pool support for a claim that cites
    nothing is plain support, while pool support for a claim that cites a
    silent source is a binding error.
    """

    cited = _states_map(cited_states)
    pool = _states_map(pool_states)
    all_states = {**pool, **cited}

    if not all_states:
        return "G2_EVIDENCE_GAP", "no source could be checked for this claim", cited_url, None

    # Decision order is fixed and most-severe-first. A source that actively
    # disagrees outranks a misbound citation, because a claim bound to a
    # contradicting source is wrong on its own terms, while G1 only says the
    # right fact was attached to the wrong source.
    cited_support = next(
        (url for url, state in cited.items() if state == "support"), None
    )
    cited_contradict = next(
        (url for url, state in cited.items() if state == "contradict"), None
    )
    if cited_contradict is not None:
        return (
            "G3_CONTRADICTED",
            f"source {cited_contradict} states an incompatible value for this claim",
            cited_contradict,
            None,
        )
    if cited_support is not None:
        return "SUPPORTED", "a cited source states the claim", cited_support, None

    # The claim's own citations are silent or unreadable. Support elsewhere in
    # the pool means the failure is binding, not absence -- but only when the
    # claim actually binds a citation. A claim that cites nothing and is stated
    # by a pool source is simply grounded, so it is not scored as a binding
    # error against a citation that does not exist.
    pool_support = next((url for url, state in pool.items() if state == "support"), None)
    if pool_support is not None:
        if has_citation:
            return (
                "G1_CITATION_BINDING",
                "the claim is stated by a source it does not cite; its own citation is silent",
                pool_support,
                None,
            )
        return "SUPPORTED", "a source in the evidence pool states the claim", pool_support, None
    pool_contradict = next((url for url, state in pool.items() if state == "contradict"), None)
    if pool_contradict is not None:
        return (
            "G3_CONTRADICTED",
            f"source {pool_contradict} states an incompatible value for this claim",
            pool_contradict,
            None,
        )
    if any(state == "not_found" for state in all_states.values()):
        return (
            "G2_EVIDENCE_GAP",
            "no fetched source states the claim",
            cited_url,
            None,
        )
    return "UNVERIFIABLE", "every source row failed validation", cited_url, None


def label_from_states(states: Mapping[str, str], cited_url: str | None = None) -> tuple[str, str]:
    """Single-view convenience wrapper around :func:`decide`."""

    label, reason, _, _ = decide(states, {}, cited_url=cited_url)
    return label, reason


def summarize_verdicts(verdicts: list[ClaimVerdict]) -> dict:
    """Report-level summary with a Wilson interval on the hallucination rate."""

    from .stats import wilson_proportion_interval

    counts = {label: 0 for label in GROUNDING_LABELS}
    for verdict in verdicts:
        counts[verdict.grounding_label] = counts.get(verdict.grounding_label, 0) + 1
    decidable = [verdict for verdict in verdicts if verdict.is_decidable]
    hallucinations = sum(verdict.is_hallucination for verdict in decidable)
    if decidable:
        lower, upper = wilson_proportion_interval(hallucinations, len(decidable))
        rate = hallucinations / len(decidable)
    else:
        lower, upper, rate = None, None, None
    by_label = {
        label: {
            "count": sum(1 for verdict in verdicts if verdict.grounding_label == label),
            "claim_ids": [verdict.claim_id for verdict in verdicts if verdict.grounding_label == label],
        }
        for label in GROUNDING_LABELS
        if any(verdict.grounding_label == label for verdict in verdicts)
    }
    return {
        "label_version": VERSION,
        "claim_count": len(verdicts),
        "decidable_claim_count": len(decidable),
        "hallucination_count": hallucinations,
        "hallucination_rate": rate,
        "hallucination_rate_wilson_95": [lower, upper] if decidable else None,
        "grounding_label_counts": {
            label: detail["count"] for label, detail in by_label.items()
        },
        "verdicts_by_label": by_label,
        "fetch_failure_count": counts.get("FETCH_FAILURE", 0),
    }
