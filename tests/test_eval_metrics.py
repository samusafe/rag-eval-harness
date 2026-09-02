"""Pure scorers and the aggregate: plain dicts and fake docs, no ML stack."""
import pytest

from services.eval_metrics import (
    REFUSAL,
    aggregate,
    answerability_ok,
    citation_scores,
    document_id,
    normalize_refusal,
    percentile,
    refusal_ok,
)


class _Doc:
    def __init__(self, metadata=None, doc_id=None):
        self.metadata = metadata or {}
        self.page_content = "text"
        if doc_id is not None:
            self.id = doc_id


# --- refusal normalization ----------------------------------------------------


def test_refusal_accepts_typographic_apostrophe_and_markdown_emphasis():
    curly = REFUSAL.replace("'", "’")
    assert refusal_ok(curly, True) is True
    assert refusal_ok(f"**{REFUSAL}**", True) is True
    assert refusal_ok(f"_{REFUSAL}_", True) is True


def test_answerability_treats_typographic_refusal_as_a_refusal():
    curly = REFUSAL.replace("'", "’")
    assert answerability_ok(curly, False) is False
    assert answerability_ok("The policy grants 20 days.", False) is True
    assert answerability_ok(REFUSAL, True) is None


def test_refusal_still_rejects_added_words():
    assert refusal_ok(f"{REFUSAL} Sorry!", True) is False
    assert normalize_refusal("  A  b ") == "a b"


# --- citations ----------------------------------------------------------------


def test_uncited_answer_has_no_validity_score():
    context = "[Source 1: policy.pdf, Page 1]\n..."
    assert citation_scores("No citation", context, False) == (False, None)


def test_citation_validity_is_the_fraction_of_real_headers():
    context = "[Source 1: policy.pdf, Page 1]\npolicy"
    present, validity = citation_scores(
        "See [Source 1: policy.pdf, Page 1] and [Source 7: made-up.pdf, Page 3].", context, False
    )
    assert present is True
    assert validity == 0.5


# --- aggregate ----------------------------------------------------------------


def test_aggregate_tolerates_rows_missing_keys():
    agg = aggregate([{"latency_s": 1.0}, {"latency_s": 3.0, "keyword_recall": 0.5}])
    assert agg["n"] == 2
    assert agg["retrieval_hit_rate"] is None
    assert agg["answer_keyword_recall"] == 0.5
    assert agg["median_latency_s"] == 2.0
    assert agg["p95_latency_s"] == pytest.approx(2.9)


def test_aggregate_does_not_round_a_rate_past_a_gate_threshold():
    # 3999 hits out of 5000 is 0.7998; rounding to 3 places would report 0.8
    # and let `--gate-hit-rate 0.8` pass a run that missed it.
    results = [{"retrieval_hit": True}] * 3999 + [{"retrieval_hit": False}] * 1001
    assert aggregate(results)["retrieval_hit_rate"] < 0.8


def test_aggregate_reports_every_summary_key():
    results = [
        {
            "retrieval_hit": True, "exact_retrieval_hit": False, "reciprocal_rank": 0.5,
            "keyword_recall": 1.0, "refusal_ok": None, "answerability_ok": True,
            "citation_present": True, "citation_validity": 1.0, "latency_s": 2.0,
            "retrieval_latency_s": 0.5, "generation_latency_s": 1.5,
            "vector_retrieval_latency_s": 0.2, "rerank_latency_s": 0.3,
        },
        {
            "retrieval_hit": None, "exact_retrieval_hit": None, "reciprocal_rank": None,
            "keyword_recall": None, "refusal_ok": True, "answerability_ok": None,
            "citation_present": None, "citation_validity": None, "latency_s": 4.0,
            "retrieval_latency_s": 1.0, "generation_latency_s": 3.0,
            "vector_retrieval_latency_s": 0.4, "rerank_latency_s": 0.6,
        },
    ]
    assert aggregate(results) == {
        "n": 2,
        "retrieval_hit_rate": 1.0,
        "answer_keyword_recall": 1.0,
        "refusal_accuracy": 1.0,
        "answerability_accuracy": 1.0,
        "citation_coverage": 1.0,
        "citation_validity": 1.0,
        "exact_retrieval_hit_rate": 0.0,
        "mean_reciprocal_rank": 0.5,
        "median_latency_s": 3.0,
        "p95_latency_s": 3.9,
        "median_retrieval_latency_s": 0.75,
        "median_generation_latency_s": 2.25,
        "median_vector_retrieval_latency_s": 0.3,
        "median_rerank_latency_s": 0.45,
    }


def test_percentile_interpolates_between_ranks():
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    assert percentile([7.0], 0.95) == 7.0


# --- document ids -------------------------------------------------------------


@pytest.mark.parametrize("key", ["source_id", "chunk_id", "document_id", "id"])
def test_document_id_accepts_each_ingester_key(key):
    assert document_id(_Doc({key: " c-1 "})) == "c-1"


def test_document_id_falls_back_to_the_doc_attribute_and_never_fabricates():
    assert document_id(_Doc({}, doc_id="attr-1")) == "attr-1"
    assert document_id(_Doc({"source_id": "  "})) is None
    assert document_id(_Doc({"source_id": "meta"}, doc_id="attr")) == "meta"
