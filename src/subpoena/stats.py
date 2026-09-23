"""Wilson score interval and bootstrap helpers.

移植自 citation_entropy 的 statistics.py，只保留审计报告需要的部分。
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Hashable, Mapping, Sequence


def wilson_proportion_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""

    if not isinstance(successes, int) or not isinstance(total, int):
        raise TypeError("successes and total must be integers")
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total and total > 0")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sample")
    position = probability * (len(sorted_values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_summary(
    values: Mapping[Hashable, Sequence[float]] | Sequence[float],
    family_ids: Sequence[Hashable] | None = None,
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap over clusters (e.g. claims grouped by report section)."""

    from collections import defaultdict
    from random import Random
    from statistics import fmean

    grouped: dict[Hashable, list[float]] = defaultdict(list)
    if isinstance(values, Mapping):
        for key, observations in values.items():
            grouped[key].extend(float(value) for value in observations)
    else:
        if family_ids is None or len(values) != len(family_ids):
            raise ValueError("flat values require one family_id per observation")
        for key, value in zip(family_ids, values):
            grouped[key].append(float(value))
    if not grouped or any(not observations for observations in grouped.values()):
        raise ValueError("each of at least one cluster must contain an observation")

    keys = sorted(grouped, key=lambda key: (type(key).__name__, repr(key)))
    observed = [value for key in keys for value in grouped[key]]
    rng = Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled = [rng.choice(keys) for _ in keys]
        estimates.append(fmean(value for key in sampled for value in grouped[key]))
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        "method": "cluster_percentile_bootstrap",
        "estimate": fmean(observed),
        "lower": _percentile(estimates, tail),
        "upper": _percentile(estimates, 1.0 - tail),
        "confidence": confidence,
        "resamples": resamples,
        "cluster_count": len(keys),
        "observation_count": len(observed),
    }
