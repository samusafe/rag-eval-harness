"""
Eval set — the typed JSONL contract and its fail-closed loader.

FAIL-CLOSED BY DESIGN
    An eval set is a measuring instrument. A malformed row is a broken
    instrument, not a row to quietly drop — a silently skipped question
    changes every rate in the scorecard while the run still looks green. So
    `load_eval_set` raises by default, naming the file, line number and cause,
    and only degrades to log-and-skip when a caller explicitly opts in
    (`run_eval.py --permissive-eval-set`, for investigating a bad file).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)


class EvalCase(BaseModel):
    """One validated eval question — the typed contract for a JSONL row.

    `strict=True` means no type coercion (`"true"` is not a bool, `1` is not a
    string) and `extra="forbid"` means an unexpected key is an error, not a
    typo that silently does nothing. Both exist so a hand-edited eval set
    can't drift away from what the scorer actually reads.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    question: str
    expected_sources: list[str]
    expected_keywords: list[str]
    must_refuse: bool
    # Optional stable chunk/document IDs for the exact hit-rate / MRR metrics.
    # Optional, so every existing dataset keeps loading unchanged; `retrieval_hit`
    # deliberately still scores on `expected_sources` substrings.
    expected_source_ids: list[str] | None = None

    @field_validator("id", "question")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("expected_sources", "expected_keywords", "expected_source_ids")
    @classmethod
    def clean_unique_lists(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("items must not be blank")
        folded = [item.casefold() for item in cleaned]
        if len(folded) != len(set(folded)):
            raise ValueError("items must be unique (case-insensitive)")
        return cleaned

    @model_validator(mode="after")
    def coherent_expectations(self) -> EvalCase:
        if self.must_refuse and (
            self.expected_sources or self.expected_keywords or self.expected_source_ids
        ):
            raise ValueError("refusal rows cannot contain positive expectations")
        if not self.must_refuse and not (
            self.expected_sources or self.expected_keywords or self.expected_source_ids
        ):
            raise ValueError("answerable rows need at least one expected source, source id, or keyword")
        return self


class EvalSetValidationError(ValueError):
    """An eval set row is invalid, so the run's numbers can't be trusted."""


def _line_error(path: Path, line_num: int, message: str) -> EvalSetValidationError:
    return EvalSetValidationError(f"Invalid eval set {path}, line {line_num}: {message}")


def load_eval_set(path: Path, *, strict: bool = True) -> list[EvalCase]:
    """Load JSONL eval rows, failing closed on invalid data.

    Blank lines and lines starting with `//` are comments. Every other line
    must parse as JSON, validate against `EvalCase`, and carry an `id` unseen
    so far in the file. `strict=False` is the investigation-only escape hatch:
    it logs each rejected row (with line number and cause) and continues.
    """
    rows: list[EvalCase] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as eval_file:
        for line_num, line in enumerate(eval_file, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                issue = _line_error(path, line_num, f"invalid JSON: {error.msg}")
                if strict:
                    raise issue from error
                logger.warning("%s", issue)
                continue
            try:
                row = EvalCase.model_validate(payload)
            except ValidationError as error:
                issue = _line_error(path, line_num, f"schema validation failed: {error}")
                if strict:
                    raise issue from error
                logger.warning("%s", issue)
                continue
            # Duplicate ids would double-count one question and break the
            # per-question diff in compare_runs.py (which keys results by id).
            if row.id in seen_ids:
                issue = _line_error(path, line_num, f"duplicate id: {row.id!r}")
                if strict:
                    raise issue
                logger.warning("%s", issue)
                continue
            seen_ids.add(row.id)
            rows.append(row)
    if not rows:
        raise ValueError(f"No eval rows loaded from {path}")
    return rows


def metrics_supported_by(rows: list[EvalCase]) -> set[str]:
    """Which scorecard metrics this eval set can produce at all.

    A gate on a metric outside this set can never pass — it would fail after
    the full run with "produced no data". `run_eval.py` uses this to refuse
    such a gate before spending a single LLM call."""
    supported: set[str] = {"median_latency_s", "p95_latency_s"}
    if any(row.expected_sources for row in rows):
        supported.add("retrieval_hit_rate")
    if any(row.expected_source_ids for row in rows):
        supported.update({"exact_retrieval_hit_rate", "mean_reciprocal_rank"})
    if any(row.expected_keywords for row in rows):
        supported.add("answer_keyword_recall")
    if any(row.must_refuse for row in rows):
        supported.add("refusal_accuracy")
    if any(not row.must_refuse for row in rows):
        supported.update({"answerability_accuracy", "citation_coverage", "citation_validity"})
    return supported
