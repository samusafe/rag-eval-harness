"""The eval-set loader is the harness's fail-closed gate: a bad row must stop
the run, not quietly shrink the question set. These tests cover both the
shipped example set and every rejection path, using temp files — no
Ollama/MLflow/Postgres required."""
import json
from pathlib import Path

import pytest

from services.eval_set import EvalSetValidationError, load_eval_set

EVAL_SET = Path(__file__).resolve().parent.parent / "eval" / "eval_set.example.jsonl"

VALID_ROW = {
    "id": "vacation-days",
    "question": "How many vacation days?",
    "expected_sources": ["employee-handbook"],
    "expected_keywords": ["20 days"],
    "must_refuse": False,
}


def _write_set(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "eval_set.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _row(**overrides) -> str:
    return json.dumps({**VALID_ROW, **overrides})


# --------------------------------------------------------------------------
# The shipped example set
# --------------------------------------------------------------------------


def test_eval_set_loads_expected_row_count():
    rows = load_eval_set(EVAL_SET)
    assert len(rows) == 10


def test_eval_set_rows_match_schema():
    # Strict validation happens inside load_eval_set, so reaching here at all
    # means the shipped set is well-formed; assert the typed shape it produced.
    rows = load_eval_set(EVAL_SET)
    ids = [row.id for row in rows]
    assert len(ids) == len(set(ids)), "ids must be unique"
    for row in rows:
        assert row.id and row.question
        assert all(isinstance(source, str) for source in row.expected_sources)
        assert all(isinstance(keyword, str) for keyword in row.expected_keywords)
        assert isinstance(row.must_refuse, bool)
        # Reserved for the future exact-ID retrieval metric; unset today.
        assert row.expected_source_ids is None


def test_eval_set_has_refusal_rows_for_anti_hallucination_coverage():
    rows = load_eval_set(EVAL_SET)
    refusals = [row for row in rows if row.must_refuse]
    assert len(refusals) == 3, "need at least a few must_refuse rows to catch hallucination"
    # Refusal rows shouldn't also carry expected_sources/keywords — they're
    # out-of-KB by construction and nothing should be scored against them.
    for row in refusals:
        assert row.expected_sources == []
        assert row.expected_keywords == []


def test_eval_set_ignores_comment_lines():
    # The file ships with leading `//` comment lines above the JSON rows —
    # confirm the loader skips them rather than failing validation on them.
    raw_lines = EVAL_SET.read_text(encoding="utf-8").splitlines()
    assert any(line.strip().startswith("//") for line in raw_lines)
    load_eval_set(EVAL_SET)


def test_comments_and_blank_lines_are_skipped_not_counted(tmp_path):
    path = _write_set(tmp_path, "// a comment", "", _row(), "   ", "// trailing note")
    assert len(load_eval_set(path)) == 1


# --------------------------------------------------------------------------
# Strict mode (the default) rejects bad data loudly
# --------------------------------------------------------------------------


def test_strict_rejects_malformed_json(tmp_path):
    path = _write_set(tmp_path, _row(), "{not json")
    with pytest.raises(EvalSetValidationError) as excinfo:
        load_eval_set(path)
    assert "line 2" in str(excinfo.value)
    assert "invalid JSON" in str(excinfo.value)


def test_strict_rejects_missing_field(tmp_path):
    incomplete = {key: value for key, value in VALID_ROW.items() if key != "must_refuse"}
    path = _write_set(tmp_path, json.dumps(incomplete))
    with pytest.raises(EvalSetValidationError) as excinfo:
        load_eval_set(path)
    assert "line 1" in str(excinfo.value)
    assert "must_refuse" in str(excinfo.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"must_refuse": "true"},  # string, not bool — no coercion in strict mode
        {"expected_keywords": "20 days"},  # bare string, not a list
        {"expected_sources": [1, 2]},  # list of non-strings
        {"id": 7},  # non-string id
    ],
)
def test_strict_rejects_wrong_types(tmp_path, overrides):
    path = _write_set(tmp_path, _row(**overrides))
    with pytest.raises(EvalSetValidationError):
        load_eval_set(path)


@pytest.mark.parametrize("overrides", [{"id": "   "}, {"question": ""}])
def test_strict_rejects_blank_id_and_question(tmp_path, overrides):
    path = _write_set(tmp_path, _row(**overrides))
    with pytest.raises(EvalSetValidationError):
        load_eval_set(path)


def test_strict_rejects_unknown_field(tmp_path):
    # A typo'd key would otherwise be silently ignored and score nothing.
    path = _write_set(tmp_path, _row(expected_keyword=["20 days"]))
    with pytest.raises(EvalSetValidationError):
        load_eval_set(path)


def test_strict_rejects_duplicate_ids(tmp_path):
    path = _write_set(tmp_path, _row(), _row(question="A different question?"))
    with pytest.raises(EvalSetValidationError) as excinfo:
        load_eval_set(path)
    assert "duplicate id" in str(excinfo.value)
    assert "line 2" in str(excinfo.value)


def test_strict_accepts_expected_source_ids_field(tmp_path):
    path = _write_set(tmp_path, _row(expected_source_ids=["doc-1#chunk-3"]))
    assert load_eval_set(path)[0].expected_source_ids == ["doc-1#chunk-3"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_keywords": [""]},
        {"expected_sources": ["handbook", "HANDBOOK"]},
        {"must_refuse": True, "expected_sources": ["handbook"]},
        {"expected_sources": [], "expected_keywords": []},
    ],
)
def test_strict_rejects_incoherent_expectations(tmp_path, overrides):
    path = _write_set(tmp_path, _row(**overrides))
    with pytest.raises(EvalSetValidationError):
        load_eval_set(path)


# --------------------------------------------------------------------------
# Permissive mode is opt-in only, and still refuses to produce nothing
# --------------------------------------------------------------------------


def test_permissive_skips_invalid_rows_and_logs(tmp_path, caplog):
    path = _write_set(tmp_path, _row(), "{not json", _row(id="second"))
    with caplog.at_level("WARNING"):
        rows = load_eval_set(path, strict=False)
    assert [row.id for row in rows] == ["vacation-days", "second"]
    assert "line 2" in caplog.text


def test_permissive_still_raises_when_nothing_valid_remains(tmp_path):
    path = _write_set(tmp_path, "{not json")
    with pytest.raises(ValueError, match="No eval rows loaded"):
        load_eval_set(path, strict=False)


def test_empty_file_raises(tmp_path):
    path = _write_set(tmp_path, "// only a comment")
    with pytest.raises(ValueError, match="No eval rows loaded"):
        load_eval_set(path)
