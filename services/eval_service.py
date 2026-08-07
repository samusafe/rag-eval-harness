"""
RAG eval — shared scoring core.
======================================================================
The reusable, dependency-light heart of the eval harness: load and validate
the eval set, score one question through the real chain, aggregate, persist
(with a reproducibility manifest), and optionally log to MLflow.
`eval/run_eval.py` is just the presentation layer on top of this (rich tables,
argparse) — kept separate so this module stays importable and unit-testable
without pulling in `rich`.

FAIL-CLOSED BY DESIGN
    An eval set is a measuring instrument. A malformed row is a broken
    instrument, not a row to quietly drop — a silently skipped question
    changes every rate in the scorecard while the run still looks green. So
    `load_eval_set` raises by default, naming the file, line number and cause,
    and only degrades to log-and-skip when a caller explicitly opts in
    (`run_eval.py --permissive-eval-set`, for investigating a bad file).
======================================================================
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import statistics
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from config import settings
from services.rag_chain import RAG_PROMPT_TEMPLATE, _format_docs, _format_history

logger = logging.getLogger(__name__)

# Repo root, so default paths resolve the same way regardless of the caller's cwd.
APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_SET = APP_ROOT / "eval" / "eval_set.example.jsonl"
DEFAULT_OUT_DIR = APP_ROOT / "eval" / "results"

# The exact refusal sentence the prompt instructs the model to produce for
# out-of-KB questions (services/rag_chain.py RAG_PROMPT_TEMPLATE).
REFUSAL = "I don't have enough information in the knowledge base to answer this question."


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
    # Forward-compatible slot for an exact chunk/document-ID retrieval metric.
    # Optional, so every existing dataset keeps loading unchanged; `retrieval_hit`
    # deliberately still scores on `expected_sources` substrings until a real
    # second consumer exists for IDs.
    expected_source_ids: list[str] | None = None

    @field_validator("id", "question")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


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


def retrieval_hit(docs: Sequence[Any], expected_sources: list[str]) -> bool | None:
    """True if any expected source substring appears in any retrieved doc's
    source_file. None when the row lists no expected_sources (e.g. refusal
    rows) — excluded from the rate rather than counted as a miss.

    Substring matching over file names, not exact chunk identity: it tolerates
    the path/extension noise real ingestion pipelines produce. See
    `EvalCase.expected_source_ids` for the exact-match successor."""
    if not expected_sources:
        return None
    retrieved = [str(doc.metadata.get("source_file", "")).lower() for doc in docs]
    return any(expected.lower() in source for expected in expected_sources for source in retrieved)


def keyword_recall(answer: str, expected_keywords: list[str]) -> float | None:
    """Fraction of expected keywords present in the answer (case-insensitive
    substring match). None if none specified for this row.

    LEXICAL, NOT FACTUAL: this measures whether the expected strings appear,
    not whether the answer is true. "The policy is not 20 days" scores 1.0
    against `["20 days"]`. Treat it as a cheap regression signal on answer
    content, never as a correctness or hallucination score — refusal accuracy
    is the anti-hallucination metric."""
    if not expected_keywords:
        return None
    normalized_answer = answer.lower()
    return sum(keyword.lower() in normalized_answer for keyword in expected_keywords) / len(
        expected_keywords
    )


def _normalize_refusal(text: str) -> str:
    """Trim, collapse internal whitespace, casefold — and nothing else.

    Deliberately small: it forgives formatting the model can't control
    (a trailing newline, a double space, capitalization) without forgiving
    any added words."""
    return " ".join(text.split()).casefold()


def refusal_ok(answer: str, must_refuse: bool) -> bool | None:
    """For out-of-KB questions, correct == the WHOLE answer is the refusal
    sentence. None for every other row.

    Full-string equality after `_normalize_refusal`, not a substring test: a
    substring test would score "The capital is Canberra. I don't have enough
    information in the knowledge base to answer this question." as a correct
    refusal, i.e. it would pass a hallucination as anti-hallucination."""
    if not must_refuse:
        return None
    return _normalize_refusal(answer) == _normalize_refusal(REFUSAL)


def aggregate(results: list[dict[str, Any]]) -> dict[str, int | float | None]:
    """Roll per-question results into the scorecard summary. Each metric only
    averages over rows where it applies (None-filtered) — a mixed eval set
    (some rows scoring retrieval, some refusal) never dilutes either metric."""
    retrieval = [row["retrieval_hit"] for row in results if row["retrieval_hit"] is not None]
    keywords = [row["keyword_recall"] for row in results if row["keyword_recall"] is not None]
    refusals = [row["refusal_ok"] for row in results if row["refusal_ok"] is not None]
    latency = [row["latency_s"] for row in results]

    def percentage(values: list[float | bool]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "n": len(results),
        "retrieval_hit_rate": percentage(retrieval),
        "answer_keyword_recall": percentage(keywords),
        "refusal_accuracy": percentage(refusals),
        "median_latency_s": round(statistics.median(latency), 2) if latency else None,
    }


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git_metadata() -> dict[str, str | bool | None]:
    """Best-effort git provenance. Never fatal: the harness must run from a
    tarball, a Docker image, or a checkout without git installed, so every
    failure mode (no git binary, not a repo, ownership check, hang) resolves
    to an explicit None rather than an exception or a fabricated value."""

    def run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-c", f"safe.directory={APP_ROOT}", *args],
                cwd=APP_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit_sha = run_git("rev-parse", "HEAD")
    porcelain = run_git("status", "--porcelain")
    # `dirty` is None (unknown), not False, when git couldn't answer — the
    # difference matters when you're deciding whether a scorecard is citable.
    return {"commit_sha": commit_sha, "dirty": None if porcelain is None else bool(porcelain)}


def build_run_manifest(
    eval_set_path: Path | None,
    model: str,
    *,
    timestamp: datetime | None = None,
    git_metadata: dict[str, str | bool | None] | None = None,
) -> dict[str, Any]:
    """Everything needed to answer "what exactly produced this scorecard?".

    A score is only evidence if you can say what was measured: which commit,
    which eval set bytes, which models and retrieval knobs, which prompt. All
    of that is captured here and embedded in the result JSON.

    `timestamp` / `git_metadata` are injectable purely so unit tests can build
    a deterministic manifest without a clock or a git repo."""
    moment = timestamp or datetime.now(UTC)
    moment = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)

    retrieval = {"top_k": settings.RETRIEVAL_TOP_K, "rerank_top_n": settings.RERANK_TOP_N}
    config_fingerprint = {
        "chat_model": model,
        "embedding_model": settings.OLLAMA_EMBED_MODEL,
        "reranker_model": settings.RERANK_MODEL,
        "retrieval": retrieval,
    }
    git = git_metadata if git_metadata is not None else _git_metadata()
    return {
        "schema_version": 1,
        "timestamp_utc": moment.isoformat().replace("+00:00", "Z"),
        "git": {"commit_sha": git.get("commit_sha"), "dirty": git.get("dirty")},
        "eval_set_sha256": _sha256(eval_set_path.read_bytes()) if eval_set_path else None,
        "chat_model": model,
        "embedding_model": settings.OLLAMA_EMBED_MODEL,
        "reranker_model": settings.RERANK_MODEL,
        "retrieval": retrieval,
        "python_version": platform.python_version(),
        "prompt_sha256": _sha256(RAG_PROMPT_TEMPLATE.encode("utf-8")),
        "config_sha256": _sha256(
            json.dumps(config_fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def write_results(
    results: list[dict[str, Any]],
    agg: dict[str, Any],
    out_dir: Path,
    model: str,
    *,
    eval_set_path: Path | None = None,
) -> Path:
    """Persist one run's full scorecard as JSON — the artifact `compare_runs.py`
    diffs. Filename embeds a sortable timestamp so `sorted(glob(...))` is
    chronological with no extra bookkeeping. Every pre-existing top-level key
    is unchanged; `run_manifest` is purely additive, so older scorecards still
    compare against newer ones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace(":", "_").replace("/", "_")
    out_path = out_dir / f"eval_{safe_model}_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "model": model,
        "embed_model": settings.OLLAMA_EMBED_MODEL,
        "summary": agg,
        "results": results,
        "run_manifest": build_run_manifest(eval_set_path, model),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def log_mlflow(agg: dict[str, Any], model: str, out_path: Path) -> None:
    """Optional: log this eval run (params + summary metrics + scorecard
    artifact, which carries the run manifest) to MLflow so different
    models/prompts/adapters sit side-by-side. Never fatal — observability must
    never break the eval itself."""
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"eval-{model}"):
            mlflow.log_params(
                {
                    "chat_model": model,
                    "embed_model": settings.OLLAMA_EMBED_MODEL,
                    "n_questions": agg["n"],
                }
            )
            for key in (
                "retrieval_hit_rate",
                "answer_keyword_recall",
                "refusal_accuracy",
                "median_latency_s",
            ):
                value = agg.get(key)
                if value is not None:
                    mlflow.log_metric(key, float(value))
            mlflow.log_artifact(str(out_path))
        logger.info("Eval logged to MLflow (%s)", settings.MLFLOW_EXPERIMENT_NAME)
    except Exception as error:  # noqa: BLE001 - tracking must never break the eval
        logger.warning("MLflow logging skipped: %s", error)


def _eval_one(row: EvalCase, retriever: Any, chain: Any) -> dict[str, Any]:
    """Score ONE eval row through the real pipeline: time retrieval +
    generation, then compute the per-question metrics. Blocking (LLM + DB
    calls) — callers on an event loop should offload this to a thread."""
    question = row.question
    started = time.perf_counter()
    docs = retriever.invoke(question)
    answer = chain.invoke(
        {
            "chat_history": _format_history([]),
            "context": _format_docs(docs),
            "question": question,
        }
    )
    latency = round(time.perf_counter() - started, 2)
    return {
        "id": row.id,
        "question": question,
        "retrieval_hit": retrieval_hit(docs, row.expected_sources),
        "keyword_recall": keyword_recall(answer, row.expected_keywords),
        "refusal_ok": refusal_ok(answer, row.must_refuse),
        "latency_s": latency,
        "retrieved_sources": [doc.metadata.get("source_file", "?") for doc in docs],
        "answer": answer,
    }
