"""Which gates an eval set can satisfy at all -- decided before the run."""
from services.eval_set import EvalCase, metrics_supported_by


def _row(**overrides):
    base = {
        "id": "q", "question": "?", "expected_sources": ["a"], "expected_keywords": [],
        "must_refuse": False,
    }
    return EvalCase(**{**base, **overrides})


def test_latency_metrics_are_always_supported():
    assert {"median_latency_s", "p95_latency_s"} <= metrics_supported_by([_row()])


def test_exact_hit_and_mrr_need_expected_source_ids():
    without = metrics_supported_by([_row()])
    assert "exact_retrieval_hit_rate" not in without
    assert "mean_reciprocal_rank" not in without
    with_ids = metrics_supported_by([_row(expected_source_ids=["c1"])])
    assert {"exact_retrieval_hit_rate", "mean_reciprocal_rank"} <= with_ids


def test_refusal_needs_a_refusal_row_and_answerability_needs_an_answerable_row():
    answerable_only = metrics_supported_by([_row()])
    assert "refusal_accuracy" not in answerable_only
    assert {"answerability_accuracy", "citation_coverage", "citation_validity"} <= answerable_only
    refusal_only = metrics_supported_by([_row(id="r", expected_sources=[], must_refuse=True)])
    assert "refusal_accuracy" in refusal_only
    assert "answerability_accuracy" not in refusal_only
    assert "retrieval_hit_rate" not in refusal_only
