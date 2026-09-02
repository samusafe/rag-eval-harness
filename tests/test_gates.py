"""Unit tests for services.gates - pure, no LangChain/torch needed."""

import pytest

from services.gates import MAX_GATES, MIN_GATES, check_gates, unsatisfiable_gates

GOOD_AGG = {
    "n": 10,
    "retrieval_hit_rate": 0.9,
    "answer_keyword_recall": 0.75,
    "refusal_accuracy": 1.0,
    "median_latency_s": 4.2,
}


def test_no_gates_no_failures():
    assert check_gates(GOOD_AGG) == []


def test_all_gates_pass():
    failures = check_gates(
        GOOD_AGG,
        minimums={"retrieval_hit_rate": 0.8, "answer_keyword_recall": 0.6, "refusal_accuracy": 0.9},
        maximums={"median_latency_s": 8.0},
    )
    assert failures == []


def test_min_gate_failure_names_metric_and_values():
    failures = check_gates(GOOD_AGG, minimums={"answer_keyword_recall": 0.8})
    assert len(failures) == 1
    assert "answer_keyword_recall" in failures[0]
    assert "0.750" in failures[0]
    assert "0.8" in failures[0]


def test_max_gate_failure():
    failures = check_gates(GOOD_AGG, maximums={"median_latency_s": 3.0})
    assert len(failures) == 1
    assert "median_latency_s" in failures[0]


def test_boundary_values_pass():
    agg = dict(GOOD_AGG, retrieval_hit_rate=0.8, median_latency_s=8.0)
    assert check_gates(agg, minimums={"retrieval_hit_rate": 0.8}, maximums={"median_latency_s": 8.0}) == []


def test_missing_metric_fails_its_gate():
    agg = dict(GOOD_AGG, refusal_accuracy=None)
    failures = check_gates(agg, minimums={"refusal_accuracy": 0.9})
    assert len(failures) == 1
    assert "no data" in failures[0]


def test_multiple_failures_all_reported():
    agg = dict(GOOD_AGG, retrieval_hit_rate=0.1, refusal_accuracy=0.2)
    failures = check_gates(
        agg, minimums={"retrieval_hit_rate": 0.8, "refusal_accuracy": 0.9}
    )
    assert len(failures) == 2


def test_unknown_min_metric_raises():
    with pytest.raises(ValueError, match="Unknown min-gate"):
        check_gates(GOOD_AGG, minimums={"median_latency_s": 1.0})


def test_unknown_max_metric_raises():
    with pytest.raises(ValueError, match="Unknown max-gate"):
        check_gates(GOOD_AGG, maximums={"retrieval_hit_rate": 0.5})


def test_p95_latency_is_a_max_gate():
    assert "p95_latency_s" in MAX_GATES
    agg = dict(GOOD_AGG, p95_latency_s=9.0)
    assert check_gates(agg, maximums={"p95_latency_s": 8.0})


@pytest.mark.parametrize("metric", MIN_GATES)
def test_every_min_gate_key_is_produced_by_aggregate(metric):
    from services.eval_metrics import aggregate

    assert metric in aggregate([])


@pytest.mark.parametrize("metric", MAX_GATES)
def test_every_max_gate_key_is_produced_by_aggregate(metric):
    from services.eval_metrics import aggregate

    assert metric in aggregate([])


def test_unsatisfiable_gates_names_metrics_the_eval_set_cannot_produce():
    failures = unsatisfiable_gates(
        minimums={"mean_reciprocal_rank": 0.5, "refusal_accuracy": 0.9},
        maximums={"median_latency_s": 8.0},
        supported={"refusal_accuracy", "median_latency_s"},
    )
    assert len(failures) == 1
    assert "mean_reciprocal_rank" in failures[0]
    assert "expected_source_ids" in failures[0]


def test_unsatisfiable_gates_is_empty_when_everything_is_supported():
    assert unsatisfiable_gates({"refusal_accuracy": 0.9}, {}, {"refusal_accuracy"}) == []
