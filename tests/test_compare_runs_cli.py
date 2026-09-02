"""Unit tests for the pure helpers in eval/compare_runs.py — no live services,
no MLflow, no rendered rich console output (only the markup strings the
helpers build are asserted)."""
import json

import pytest

from eval import compare_runs
from eval.compare_runs import (
    PQ_METRICS,
    SUMMARY_METRICS,
    _delta,
    _fmt_summary,
    _pq_fmt,
    load_run,
    per_question_diff,
)

# --------------------------------------------------------------------------
# load_run
# --------------------------------------------------------------------------


def test_load_run_rejects_unparseable_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_run(path)


def test_load_run_rejects_non_object(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(SystemExit):
        load_run(path)


def test_load_run_rejects_missing_summary(tmp_path):
    path = tmp_path / "no_summary.json"
    path.write_text(json.dumps({"results": []}), encoding="utf-8")
    with pytest.raises(SystemExit):
        load_run(path)


def test_load_run_rejects_missing_results(tmp_path):
    path = tmp_path / "no_results.json"
    path.write_text(json.dumps({"summary": {}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        load_run(path)


def test_load_run_returns_label_and_file(tmp_path):
    path = tmp_path / "eval_test.json"
    path.write_text(
        json.dumps(
            {"summary": {}, "results": [], "model": "my-model", "timestamp": "20260101T000000_000000Z"}
        ),
        encoding="utf-8",
    )
    data = load_run(path)
    assert "my-model" in data["_label"]
    assert data["_file"] == path.name


def test_load_run_escapes_rich_markup_in_model_name(tmp_path):
    path = tmp_path / "eval_hostile.json"
    path.write_text(
        json.dumps({"summary": {}, "results": [], "model": "weird[red]model", "timestamp": "t"}),
        encoding="utf-8",
    )
    data = load_run(path)
    # A raw "[red]" would be interpreted as rich markup; escape() turns it
    # into a literal "\[red]" so the model name can never forge a style tag.
    assert "\\[red]" in data["_label"]


# --------------------------------------------------------------------------
# _fmt_summary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ("median_latency_s", 4.2, "4.20s"),
        ("p95_latency_s", 9.0, "9.00s"),
        ("mean_reciprocal_rank", 0.5, "0.500"),
        ("refusal_accuracy", 0.9, "90%"),
    ],
)
def test_fmt_summary_formats_by_key(key, value, expected):
    assert _fmt_summary(key, value) == expected


def test_fmt_summary_none_is_dash():
    assert _fmt_summary("anything", None) == "-"


# --------------------------------------------------------------------------
# _delta
# --------------------------------------------------------------------------


def test_delta_latency_increase_is_red_and_seconds():
    # Higher latency is worse (higher_better=False).
    result = _delta("median_latency_s", 4.0, 5.0, False)
    assert "red" in result
    assert "s" in result


def test_delta_recall_increase_is_green_and_pts():
    result = _delta("answer_keyword_recall", 0.5, 0.6, True)
    assert "green" in result
    assert "pts" in result


def test_delta_mrr_uses_three_decimals():
    result = _delta("mean_reciprocal_rank", 0.5, 0.6, True)
    assert "0.100" in result


def test_delta_equal_values_is_dim_equals():
    assert _delta("answer_keyword_recall", 0.5, 0.5, True) == "[dim]=[/dim]"


def test_delta_none_either_side_is_na():
    assert _delta("answer_keyword_recall", None, 0.5, True) == "[dim]n/a[/dim]"
    assert _delta("answer_keyword_recall", 0.5, None, True) == "[dim]n/a[/dim]"


# --------------------------------------------------------------------------
# SUMMARY_METRICS / PQ_METRICS
# --------------------------------------------------------------------------


def test_summary_metrics_contains_p95_latency():
    assert ("p95_latency_s", "P95 latency (s)", False) in SUMMARY_METRICS


def test_pq_metrics_keys_include_expected():
    keys = {key for key, _, _ in PQ_METRICS}
    assert {
        "exact_retrieval_hit",
        "reciprocal_rank",
        "answerability_ok",
        "citation_present",
        "citation_validity",
    } <= keys


# --------------------------------------------------------------------------
# per_question_diff
# --------------------------------------------------------------------------


def test_per_question_diff_sorts_regressions_first(monkeypatch):
    monkeypatch.setattr(compare_runs.settings, "COMPARE_LATENCY_NOISE_S", 0.5)
    base = {
        "results": [
            {"id": "q1", "retrieval_hit": True},
            {"id": "q2", "retrieval_hit": False},
        ]
    }
    cand = {
        "results": [
            {"id": "q1", "retrieval_hit": False},  # regression
            {"id": "q2", "retrieval_hit": True},  # improvement
        ]
    }
    rows, _, _ = per_question_diff(base, cand)
    assert [r[0] for r in rows] == [False, True]


def test_per_question_diff_ignores_latency_below_noise_floor(monkeypatch):
    monkeypatch.setattr(compare_runs.settings, "COMPARE_LATENCY_NOISE_S", 0.5)
    base = {"results": [{"id": "q1", "latency_s": 1.0}]}

    cand_small_delta = {"results": [{"id": "q1", "latency_s": 1.2}]}  # |delta| 0.2 < 0.5
    rows, _, _ = per_question_diff(base, cand_small_delta)
    assert rows == []

    cand_large_delta = {"results": [{"id": "q1", "latency_s": 1.6}]}  # |delta| 0.6 > 0.5
    rows, _, _ = per_question_diff(base, cand_large_delta)
    assert len(rows) == 1
    assert rows[0][2] == "latency_s"


def test_per_question_diff_bool_flip_reports_improved_correctly():
    base = {"results": [{"id": "q1", "refusal_ok": False}]}
    cand = {"results": [{"id": "q1", "refusal_ok": True}]}
    rows, _, _ = per_question_diff(base, cand)
    assert rows == [(True, "q1", "refusal_ok", "refusal", False, True)]


def test_per_question_diff_reports_only_in_base_and_candidate():
    base = {"results": [{"id": "q1", "retrieval_hit": True}, {"id": "q2", "retrieval_hit": True}]}
    cand = {"results": [{"id": "q1", "retrieval_hit": True}, {"id": "q3", "retrieval_hit": True}]}
    rows, only_base, only_cand = per_question_diff(base, cand)
    assert only_base == ["q2"]
    assert only_cand == ["q3"]


def test_per_question_diff_skips_none_on_either_side():
    base = {"results": [{"id": "q1", "retrieval_hit": None, "keyword_recall": 0.5}]}
    cand = {"results": [{"id": "q1", "retrieval_hit": True, "keyword_recall": None}]}
    rows, _, _ = per_question_diff(base, cand)
    assert rows == []


# --------------------------------------------------------------------------
# _pq_fmt
# --------------------------------------------------------------------------


def test_pq_fmt_bool():
    assert _pq_fmt("retrieval_hit", True) == "OK"
    assert _pq_fmt("retrieval_hit", False) == "X"


def test_pq_fmt_latency():
    assert _pq_fmt("latency_s", 1.5) == "1.50s"


def test_pq_fmt_float():
    assert _pq_fmt("keyword_recall", 0.5) == "0.50"


def test_pq_fmt_none():
    assert _pq_fmt("keyword_recall", None) == "-"
