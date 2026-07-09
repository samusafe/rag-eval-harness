"""Validate the shipped example eval set matches the schema run_eval.py and
eval_service.py expect. No Ollama/MLflow/Postgres required — this only reads
and shape-checks the JSONL file."""
from pathlib import Path

from services.eval_service import load_eval_set

EVAL_SET = Path(__file__).resolve().parent.parent / "eval" / "eval_set.example.jsonl"

REQUIRED_KEYS = {"id", "question", "expected_sources", "expected_keywords", "must_refuse"}


def test_eval_set_loads_expected_row_count():
    rows = load_eval_set(EVAL_SET)
    assert 8 <= len(rows) <= 10


def test_eval_set_rows_match_schema():
    rows = load_eval_set(EVAL_SET)
    seen_ids = set()
    for row in rows:
        assert REQUIRED_KEYS <= row.keys(), f"row {row.get('id')} missing keys"
        assert isinstance(row["id"], str) and row["id"], "id must be a non-empty string"
        assert row["id"] not in seen_ids, f"duplicate id: {row['id']}"
        seen_ids.add(row["id"])
        assert isinstance(row["question"], str) and row["question"]
        assert isinstance(row["expected_sources"], list)
        assert all(isinstance(s, str) for s in row["expected_sources"])
        assert isinstance(row["expected_keywords"], list)
        assert all(isinstance(k, str) for k in row["expected_keywords"])
        assert isinstance(row["must_refuse"], bool)


def test_eval_set_has_refusal_rows_for_anti_hallucination_coverage():
    rows = load_eval_set(EVAL_SET)
    refusals = [r for r in rows if r["must_refuse"]]
    assert len(refusals) >= 2, "need at least a few must_refuse rows to catch hallucination"
    # Refusal rows shouldn't also carry expected_sources/keywords — they're
    # out-of-KB by construction and nothing should be scored against them.
    for r in refusals:
        assert r["expected_sources"] == []
        assert r["expected_keywords"] == []


def test_eval_set_ignores_comment_lines():
    # The file ships with leading `//` comment lines above the JSON rows —
    # confirm the loader actually skips them rather than erroring.
    raw_lines = EVAL_SET.read_text(encoding="utf-8").splitlines()
    assert any(line.strip().startswith("//") for line in raw_lines)
    load_eval_set(EVAL_SET)  # would raise on a malformed/comment line if unhandled
