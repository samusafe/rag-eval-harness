"""
Quality gates over an aggregated scorecard.

Pure logic, no heavy imports: run_eval.py turns these failures into a non-zero
exit code so CI (or the qlora-8gb-pipeline post-export step) can block a model
that regressed. Missing metrics FAIL their gate rather than pass silently - a
gated metric that produced no data is a broken eval set, not a green light.
"""

from __future__ import annotations

from typing import Any

# Keys must match aggregate() in services/eval_metrics.py (pinned by tests/test_gates.py).
MIN_GATES = (
    "retrieval_hit_rate",
    "exact_retrieval_hit_rate",
    "mean_reciprocal_rank",
    "answer_keyword_recall",
    "refusal_accuracy",
    "answerability_accuracy",
    "citation_coverage",
    "citation_validity",
)
MAX_GATES = ("median_latency_s", "p95_latency_s")


def check_gates(
    agg: dict[str, Any],
    minimums: dict[str, float] | None = None,
    maximums: dict[str, float] | None = None,
) -> list[str]:
    """Return human-readable failure strings; empty list = every gate passed.

    minimums: metric must be >= threshold (quality axes).
    maximums: metric must be <= threshold (latency).
    """
    failures: list[str] = []
    for metric, threshold in sorted((minimums or {}).items()):
        if metric not in MIN_GATES:
            raise ValueError(f"Unknown min-gate metric: {metric!r} (valid: {MIN_GATES})")
        value = agg.get(metric)
        if value is None:
            failures.append(f"{metric}: gate >= {threshold:g}, but the eval set produced no data for it")
        elif value < threshold:
            failures.append(f"{metric}: {value:.3f} < required {threshold:g}")
    for metric, threshold in sorted((maximums or {}).items()):
        if metric not in MAX_GATES:
            raise ValueError(f"Unknown max-gate metric: {metric!r} (valid: {MAX_GATES})")
        value = agg.get(metric)
        if value is None:
            failures.append(f"{metric}: gate <= {threshold:g}, but the eval set produced no data for it")
        elif value > threshold:
            failures.append(f"{metric}: {value:.2f}s > allowed {threshold:g}s")
    return failures


# What an eval set must contain for each gated metric to exist at all.
_METRIC_NEEDS = {
    "retrieval_hit_rate": "rows with expected_sources",
    "exact_retrieval_hit_rate": "rows with expected_source_ids",
    "mean_reciprocal_rank": "rows with expected_source_ids",
    "answer_keyword_recall": "rows with expected_keywords",
    "refusal_accuracy": "must_refuse rows",
    "answerability_accuracy": "answerable (must_refuse: false) rows",
    "citation_coverage": "answerable (must_refuse: false) rows",
    "citation_validity": "answerable (must_refuse: false) rows",
}


def unsatisfiable_gates(
    minimums: dict[str, float],
    maximums: dict[str, float],
    supported: set[str],
) -> list[str]:
    """Gates on metrics this eval set can never produce. Checked BEFORE the
    run, so a `--gate-mrr` against a set with no expected_source_ids fails in
    milliseconds with a reason, not after every LLM call with "no data"."""
    return [
        f"{metric}: the eval set has no {_METRIC_NEEDS.get(metric, 'rows')} so this gate can never pass"
        for metric in sorted(set(minimums) | set(maximums))
        if metric not in supported
    ]
