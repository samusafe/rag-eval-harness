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
import importlib.metadata
import json
import logging
import os
import platform
import re
import statistics
import subprocess
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from config import settings
from services.rag_chain import (
    RAG_PROMPT_TEMPLATE,
    _format_docs_with_stats,
    _format_history,
)

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


def _document_id(doc: Any) -> str | None:
    """Return a stable ingester-provided ID without fabricating one."""
    metadata = getattr(doc, "metadata", {})
    for key in ("source_id", "chunk_id", "document_id", "id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    value = getattr(doc, "id", None)
    return str(value).strip() if value is not None and str(value).strip() else None


def exact_retrieval_hit(docs: Sequence[Any], expected_ids: list[str] | None) -> bool | None:
    if not expected_ids:
        return None
    expected = {value.casefold() for value in expected_ids}
    return any((doc_id := _document_id(doc)) is not None and doc_id.casefold() in expected for doc in docs)


def reciprocal_rank(docs: Sequence[Any], expected_ids: list[str] | None) -> float | None:
    if not expected_ids:
        return None
    expected = {value.casefold() for value in expected_ids}
    for rank, doc in enumerate(docs, 1):
        doc_id = _document_id(doc)
        if doc_id is not None and doc_id.casefold() in expected:
            return round(1 / rank, 4)
    return 0.0


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


_CITATION_RE = re.compile(r"\[Source \d+: [^\]\r\n]+\]")


def citation_scores(answer: str, context: str, must_refuse: bool) -> tuple[bool | None, float | None]:
    """Deterministic citation presence and validity against emitted headers."""
    if must_refuse:
        return None, None
    citations = _CITATION_RE.findall(answer)
    valid = set(_CITATION_RE.findall(context))
    return bool(citations), (
        sum(citation in valid for citation in citations) / len(citations) if citations else 0.0
    )


def aggregate(results: list[dict[str, Any]]) -> dict[str, int | float | None]:
    """Roll per-question results into the scorecard summary. Each metric only
    averages over rows where it applies (None-filtered) — a mixed eval set
    (some rows scoring retrieval, some refusal) never dilutes either metric."""
    retrieval = [row["retrieval_hit"] for row in results if row["retrieval_hit"] is not None]
    keywords = [row["keyword_recall"] for row in results if row["keyword_recall"] is not None]
    refusals = [row["refusal_ok"] for row in results if row["refusal_ok"] is not None]
    answerability = [
        cast(float | bool, row.get("answerability_ok"))
        for row in results
        if row.get("answerability_ok") is not None
    ]
    citation_presence = [
        cast(float | bool, row.get("citation_present"))
        for row in results
        if row.get("citation_present") is not None
    ]
    citation_validity = [
        cast(float | bool, row.get("citation_validity"))
        for row in results
        if row.get("citation_validity") is not None
    ]
    exact_hits = [
        cast(float | bool, row.get("exact_retrieval_hit"))
        for row in results
        if row.get("exact_retrieval_hit") is not None
    ]
    reciprocal_ranks = [
        cast(float, row.get("reciprocal_rank"))
        for row in results
        if row.get("reciprocal_rank") is not None
    ]
    latency = [row["latency_s"] for row in results]
    retrieval_latency = [row["retrieval_latency_s"] for row in results if row.get("retrieval_latency_s") is not None]
    generation_latency = [row["generation_latency_s"] for row in results if row.get("generation_latency_s") is not None]
    vector_latency = [
        cast(float, row.get("vector_retrieval_latency_s"))
        for row in results
        if row.get("vector_retrieval_latency_s") is not None
    ]
    rerank_latency = [
        cast(float, row.get("rerank_latency_s"))
        for row in results
        if row.get("rerank_latency_s") is not None
    ]

    def percentage(values: list[float | bool]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "n": len(results),
        "retrieval_hit_rate": percentage(retrieval),
        "answer_keyword_recall": percentage(keywords),
        "refusal_accuracy": percentage(refusals),
        "answerability_accuracy": percentage(answerability),
        "citation_coverage": percentage(citation_presence),
        "citation_validity": percentage(citation_validity),
        "exact_retrieval_hit_rate": percentage(exact_hits),
        "mean_reciprocal_rank": round(statistics.fmean(reciprocal_ranks), 3) if reciprocal_ranks else None,
        "median_latency_s": round(statistics.median(latency), 2) if latency else None,
        "p95_latency_s": round(_percentile(latency, 0.95), 2) if latency else None,
        "median_retrieval_latency_s": round(statistics.median(retrieval_latency), 3) if retrieval_latency else None,
        "median_generation_latency_s": round(statistics.median(generation_latency), 3) if generation_latency else None,
        "median_vector_retrieval_latency_s": round(statistics.median(vector_latency), 3) if vector_latency else None,
        "median_rerank_latency_s": round(statistics.median(rerank_latency), 3) if rerank_latency else None,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except (ImportError, OSError):
        return None


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
    collection_id: str | None = None,
    runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to answer "what exactly produced this scorecard?".

    A score is only evidence if you can say what was measured: which commit,
    which eval set bytes, which models and retrieval knobs, which prompt. All
    of that is captured here and embedded in the result JSON.

    `timestamp` / `git_metadata` are injectable purely so unit tests can build
    a deterministic manifest without a clock or a git repo."""
    moment = timestamp or datetime.now(UTC)
    moment = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)

    retrieval = {
        "top_k": settings.RETRIEVAL_TOP_K,
        "rerank_top_n": settings.RERANK_TOP_N,
        "distance_strategy": settings.PGVECTOR_DISTANCE_STRATEGY,
        "collection_name": "rag_documents",
        "collection_id_filter": collection_id,
        "corpus_revision": settings.CORPUS_REVISION,
    }
    config_fingerprint = {
        "chat_model": model,
        "embedding_model": settings.OLLAMA_EMBED_MODEL,
        "reranker_model": settings.RERANK_MODEL,
        "retrieval": retrieval,
        "temperature": settings.OLLAMA_TEMPERATURE,
        "num_ctx": settings.OLLAMA_NUM_CTX,
        "num_predict": settings.OLLAMA_NUM_PREDICT,
        "context_reserve_tokens": settings.RAG_CONTEXT_RESERVE_TOKENS,
        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
        "request_timeout": settings.OLLAMA_REQUEST_TIMEOUT,
        "reranker_device": settings.RERANK_DEVICE,
        "reranker_revision": settings.RERANK_MODEL_REVISION,
        "reranker_max_length": settings.RERANK_MAX_LENGTH,
        "reranker_batch_size": settings.RERANK_BATCH_SIZE,
    }
    git = git_metadata if git_metadata is not None else _git_metadata()
    return {
        "schema_version": 2,
        "timestamp_utc": moment.isoformat().replace("+00:00", "Z"),
        "git": {"commit_sha": git.get("commit_sha"), "dirty": git.get("dirty")},
        "eval_set_sha256": _sha256(eval_set_path.read_bytes()) if eval_set_path else None,
        "chat_model": model,
        "embedding_model": settings.OLLAMA_EMBED_MODEL,
        "reranker_model": settings.RERANK_MODEL,
        "retrieval": retrieval,
        "generation": {
            "temperature": settings.OLLAMA_TEMPERATURE,
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "num_predict": settings.OLLAMA_NUM_PREDICT,
            "context_reserve_tokens": settings.RAG_CONTEXT_RESERVE_TOKENS,
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
        },
        "reranker": {
            "device": settings.RERANK_DEVICE,
            "revision": settings.RERANK_MODEL_REVISION,
            "max_length": settings.RERANK_MAX_LENGTH,
            "batch_size": settings.RERANK_BATCH_SIZE,
        },
        "python_version": platform.python_version(),
        "runtime": runtime_metadata if runtime_metadata is not None else collect_runtime_metadata(),
        "prompt_sha256": _sha256(RAG_PROMPT_TEMPLATE.encode("utf-8")),
        "config_sha256": _sha256(
            json.dumps(config_fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def collect_runtime_metadata() -> dict[str, Any]:
    """Best-effort, non-secret host/package provenance; never requires a GPU."""
    packages: dict[str, str | None] = {}
    for name in (
        "langchain-core", "langchain-classic", "langchain-ollama",
        "langchain-postgres", "sentence-transformers", "torch", "mlflow-skinny", "pydantic",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    runtime: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "packages": packages,
        "ollama_host_flags": {
            key: os.environ.get(key)
            for key in ("OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_NUM_PARALLEL")
        },
    }
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        runtime["gpus"] = gpu.stdout.strip().splitlines() if gpu.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        runtime["gpus"] = []
    return runtime


def write_results(
    results: list[dict[str, Any]],
    agg: dict[str, Any],
    out_dir: Path,
    model: str,
    *,
    eval_set_path: Path | None = None,
    collection_id: str | None = None,
    runtime_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist one run's full scorecard as JSON — the artifact `compare_runs.py`
    diffs. Filename embeds a sortable timestamp so `sorted(glob(...))` is
    chronological with no extra bookkeeping. Every pre-existing top-level key
    is unchanged; `run_manifest` is purely additive, so older scorecards still
    compare against newer ones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-") or "model"
    safe_model = safe_model[:64]
    model_hash = _sha256(model.encode("utf-8"))[:8]
    out_dir = out_dir.resolve()
    out_path = (out_dir / f"eval_{safe_model}_{model_hash}_{stamp}.json").resolve()
    if out_dir not in out_path.parents:
        raise ValueError("Result path escaped the configured output directory")
    persisted_results = _redact_results(results) if settings.RESULT_CONTENT_MODE == "redacted" else results
    payload = {
        "schema_version": 2,
        "timestamp": stamp,
        "model": model,
        "embed_model": settings.OLLAMA_EMBED_MODEL,
        "summary": agg,
        "results": persisted_results,
        "run_manifest": build_run_manifest(
            eval_set_path, model, collection_id=collection_id, runtime_metadata=runtime_metadata
        ),
    }
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    fd, temp_name = tempfile.mkstemp(prefix=f".{out_path.name}.", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as temp_file:
            temp_file.write(serialized)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        Path(temp_name).replace(out_path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return out_path


def _redact_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    redacted = []
    for row in results:
        copy = dict(row)
        for key in ("question", "answer"):
            value = copy.pop(key, None)
            if value is not None:
                copy[f"{key}_sha256"] = _sha256(str(value).encode("utf-8"))
        redacted.append(copy)
    return redacted


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
                "exact_retrieval_hit_rate",
                "mean_reciprocal_rank",
                "answer_keyword_recall",
                "refusal_accuracy",
                "answerability_accuracy",
                "citation_coverage",
                "citation_validity",
                "median_latency_s",
                "p95_latency_s",
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
    rss_samples = [_process_rss_bytes()]
    started = time.perf_counter()
    docs = retriever.invoke(question)
    retrieval_stage_timings = getattr(retriever, "last_timings", {})
    rss_samples.append(_process_rss_bytes())
    retrieval_finished = time.perf_counter()
    context, context_stats = _format_docs_with_stats(docs)
    response = chain.invoke(
        {
            "chat_history": _format_history([]),
            "context": context,
            "question": question,
        }
    )
    finished = time.perf_counter()
    rss_samples.append(_process_rss_bytes())
    answer = response if isinstance(response, str) else str(getattr(response, "content", response))
    response_metadata = getattr(response, "response_metadata", {}) or {}
    usage_metadata = getattr(response, "usage_metadata", {}) or {}
    citation_present, citation_validity = citation_scores(answer, context, row.must_refuse)
    return {
        "id": row.id,
        "question": question,
        "retrieval_hit": retrieval_hit(docs, row.expected_sources),
        "exact_retrieval_hit": exact_retrieval_hit(docs, row.expected_source_ids),
        "reciprocal_rank": reciprocal_rank(docs, row.expected_source_ids),
        "keyword_recall": keyword_recall(answer, row.expected_keywords),
        "refusal_ok": refusal_ok(answer, row.must_refuse),
        "answerability_ok": None if row.must_refuse else _normalize_refusal(answer) != _normalize_refusal(REFUSAL),
        "citation_present": citation_present,
        "citation_validity": citation_validity,
        "latency_s": finished - started,
        "retrieval_latency_s": retrieval_finished - started,
        "vector_retrieval_latency_s": retrieval_stage_timings.get("vector_retrieval_latency_s"),
        "rerank_latency_s": retrieval_stage_timings.get("rerank_latency_s"),
        "generation_latency_s": finished - retrieval_finished,
        "retrieved_sources": [doc.metadata.get("source_file", "?") for doc in docs],
        "retrieved_source_ids": [_document_id(doc) for doc in docs],
        "ollama": {"response_metadata": response_metadata, "usage_metadata": usage_metadata},
        "context": context_stats,
        "process_rss_max_observed_bytes": max(
            (value for value in rss_samples if value is not None), default=None
        ),
        "answer": answer,
    }
