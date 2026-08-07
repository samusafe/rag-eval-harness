"""Unit tests for the pure scoring/aggregation/persistence helpers in
services/eval_service.py — no retriever, LLM, or MLflow involved."""
import hashlib
import json

from services.eval_service import (
    REFUSAL,
    aggregate,
    keyword_recall,
    refusal_ok,
    retrieval_hit,
    write_results,
)


class _FakeDoc:
    def __init__(self, source_file: str):
        self.metadata = {"source_file": source_file}


def test_retrieval_hit_true_on_substring_match():
    docs = [_FakeDoc("employee-handbook.pdf")]
    assert retrieval_hit(docs, ["handbook"]) is True


def test_retrieval_hit_false_when_no_match():
    docs = [_FakeDoc("expense-policy.pdf")]
    assert retrieval_hit(docs, ["handbook"]) is False


def test_retrieval_hit_none_when_not_scored():
    assert retrieval_hit([_FakeDoc("anything.pdf")], []) is None


def test_keyword_recall_fraction():
    assert keyword_recall("The vacation policy grants 20 days.", ["vacation", "days"]) == 1.0
    assert keyword_recall("Nothing relevant here.", ["vacation", "days"]) == 0.0
    assert keyword_recall("Only vacation is mentioned.", ["vacation", "sick leave"]) == 0.5


def test_keyword_recall_none_when_not_scored():
    assert keyword_recall("anything", []) is None


def test_keyword_recall_is_lexical_not_factual():
    # Documents the known limitation: a negated/wrong answer that still
    # contains the expected string scores a full 1.0. Keyword recall is a
    # content-regression signal, not a correctness judgement.
    assert keyword_recall("The policy is NOT 20 days.", ["20 days"]) == 1.0


def test_refusal_ok_accepts_only_formatting_differences():
    # Normalization is limited to trim, whitespace collapsing and case.
    assert refusal_ok(REFUSAL, True) is True
    assert refusal_ok(REFUSAL.upper(), True) is True
    assert refusal_ok(f"  \n{REFUSAL}\t ", True) is True
    assert refusal_ok(REFUSAL.replace(" ", "  "), True) is True


def test_refusal_ok_rejects_answer_plus_refusal():
    # The whole point: a hallucinated answer that also contains the refusal
    # sentence must NOT count as a correct refusal.
    assert refusal_ok(f"The capital is Canberra. {REFUSAL}", True) is False
    assert refusal_ok(f"{REFUSAL} But here is what I think anyway.", True) is False
    assert refusal_ok(f'"{REFUSAL}"', True) is False
    assert refusal_ok("I know the answer!", True) is False


def test_refusal_ok_none_when_not_a_refusal_row():
    assert refusal_ok("anything", False) is None


def test_aggregate_ignores_none_values_per_metric():
    results = [
        {"retrieval_hit": None, "keyword_recall": None, "refusal_ok": True, "latency_s": 1.2},
        {"retrieval_hit": True, "keyword_recall": 0.5, "refusal_ok": None, "latency_s": 0.8},
    ]
    agg = aggregate(results)
    assert agg["n"] == 2
    assert agg["retrieval_hit_rate"] == 1.0
    assert agg["answer_keyword_recall"] == 0.5
    assert agg["refusal_accuracy"] == 1.0
    assert agg["median_latency_s"] == 1.0


def test_aggregate_handles_all_none_metric():
    results = [{"retrieval_hit": None, "keyword_recall": None, "refusal_ok": None, "latency_s": 0.5}]
    agg = aggregate(results)
    assert agg["retrieval_hit_rate"] is None
    assert agg["answer_keyword_recall"] is None
    assert agg["refusal_accuracy"] is None


def test_write_results_roundtrip(tmp_path):
    results = [
        {"id": "q1", "retrieval_hit": True, "keyword_recall": 1.0, "refusal_ok": None, "latency_s": 1.0}
    ]
    agg = {
        "n": 1,
        "retrieval_hit_rate": 1.0,
        "answer_keyword_recall": 1.0,
        "refusal_accuracy": None,
        "median_latency_s": 1.0,
    }
    out_path = write_results(results, agg, tmp_path, "test-model")

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["model"] == "test-model"
    assert data["summary"] == agg
    assert data["results"] == results
    # Pre-existing keys are unchanged; run_manifest is purely additive so old
    # scorecards stay diffable against new ones.
    assert {"timestamp", "model", "embed_model", "summary", "results"} <= data.keys()
    assert data["run_manifest"]["chat_model"] == "test-model"
    # No eval_set_path passed -> explicit null, never a fabricated hash.
    assert data["run_manifest"]["eval_set_sha256"] is None


def test_write_results_records_the_eval_set_hash(tmp_path):
    eval_set = tmp_path / "eval_set.jsonl"
    eval_set.write_bytes(b"{}\n")
    out_path = write_results([], {"n": 0}, tmp_path, "test-model", eval_set_path=eval_set)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["run_manifest"]["eval_set_sha256"] == hashlib.sha256(b"{}\n").hexdigest()
