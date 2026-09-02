"""
RAG eval — orchestration.
======================================================================
Score one question through the real chain, persist a run's scorecard (with
its reproducibility manifest), and optionally log it to MLflow. The pieces
it composes each live in their own module:

  services/eval_set.py       typed JSONL contract + fail-closed loader
  services/eval_metrics.py   pure, dependency-free scorers and aggregate
  services/eval_manifest.py  provenance (git, host, packages, config hashes)
  services/rag_prompt.py     prompt template, refusal sentence, context format

`eval/run_eval.py` is just the presentation layer on top of this (rich tables,
argparse). The public names of the four modules above are re-exported here so
existing `from services.eval_service import ...` callers keep working.
======================================================================
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import settings
from services.eval_manifest import (
    APP_ROOT,
    MANIFEST_SCHEMA_VERSION,
    build_run_manifest,
    collect_runtime_metadata,
    git_metadata,
    process_rss_bytes,
    sha256_hex,
)
from services.eval_metrics import (
    REFUSAL,
    aggregate,
    answerability_ok,
    citation_scores,
    document_id,
    exact_retrieval_hit,
    keyword_recall,
    normalize_refusal,
    percentile,
    reciprocal_rank,
    refusal_ok,
    retrieval_hit,
)
from services.eval_set import EvalCase, EvalSetValidationError, load_eval_set, metrics_supported_by
from services.rag_prompt import RAG_PROMPT_TEMPLATE, format_context_with_stats, format_history

__all__ = [
    "APP_ROOT",
    "DEFAULT_EVAL_SET",
    "DEFAULT_OUT_DIR",
    "MANIFEST_SCHEMA_VERSION",
    "RAG_PROMPT_TEMPLATE",
    "REFUSAL",
    "EvalCase",
    "EvalSetValidationError",
    "aggregate",
    "answerability_ok",
    "build_run_manifest",
    "citation_scores",
    "collect_runtime_metadata",
    "document_id",
    "eval_one",
    "exact_retrieval_hit",
    "format_context_with_stats",
    "format_history",
    "git_metadata",
    "keyword_recall",
    "load_eval_set",
    "log_mlflow",
    "metrics_supported_by",
    "normalize_refusal",
    "percentile",
    "process_rss_bytes",
    "reciprocal_rank",
    "refusal_ok",
    "retrieval_hit",
    "sha256_hex",
    "write_results",
]

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SET = APP_ROOT / "eval" / "eval_set.example.jsonl"
DEFAULT_OUT_DIR = APP_ROOT / "eval" / "results"

# Top-level scorecard payload layout. v2: `run_manifest` added. Bump only when a
# TOP-LEVEL key changes; the manifest has its own MANIFEST_SCHEMA_VERSION.
RESULT_SCHEMA_VERSION = 2

# Longest model-name fragment kept in a result filename (the 8-char model hash
# after it keeps distinct names distinct).
_FILENAME_MODEL_CHARS = 64

# Manifest fields worth a tag in MLflow so a run's comparability can be judged
# from the tracking UI alone, without opening the artifact.
_MLFLOW_TAG_PATHS: tuple[tuple[str, ...], ...] = (
    ("eval_set_sha256",),
    ("eval_set_name",),
    ("eval_set_strict",),
    ("prompt_sha256",),
    ("config_sha256",),
    ("retrieval", "corpus_revision"),
    ("git", "commit_sha"),
    ("git", "dirty"),
    ("embedding_model",),
    ("reranker_model",),
    ("result_content_mode",),
)


def eval_one(row: EvalCase, retriever: Any, chain: Any) -> dict[str, Any]:
    """Score ONE eval row through the real pipeline: time retrieval +
    generation, then compute the per-question metrics. Blocking (LLM + DB
    calls) — callers on an event loop should offload this to a thread.

    RSS is sampled only OUTSIDE the timed regions, so the telemetry never
    perturbs the latencies it sits next to."""
    question = row.question
    rss_samples = [process_rss_bytes()]
    started = time.perf_counter()
    docs = retriever.invoke(question)
    retrieval_finished = time.perf_counter()
    retrieval_stage_timings = getattr(retriever, "last_timings", {})
    context, context_stats = format_context_with_stats(docs)
    generation_started = time.perf_counter()
    response = chain.invoke(
        {
            "chat_history": format_history([]),
            "context": context,
            "question": question,
        }
    )
    finished = time.perf_counter()
    rss_samples.append(process_rss_bytes())
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
        "answerability_ok": answerability_ok(answer, row.must_refuse),
        "citation_present": citation_present,
        "citation_validity": citation_validity,
        "latency_s": finished - started,
        "retrieval_latency_s": retrieval_finished - started,
        "vector_retrieval_latency_s": retrieval_stage_timings.get("vector_retrieval_latency_s"),
        "rerank_latency_s": retrieval_stage_timings.get("rerank_latency_s"),
        "generation_latency_s": finished - generation_started,
        "retrieved_sources": [doc.metadata.get("source_file", "?") for doc in docs],
        "retrieved_source_ids": [document_id(doc) for doc in docs],
        "ollama": {"response_metadata": response_metadata, "usage_metadata": usage_metadata},
        "context": context_stats,
        "process_rss_max_observed_bytes": max(
            (value for value in rss_samples if value is not None), default=None
        ),
        "answer": answer,
    }


# Backward-compatible alias for downstream callers (README's HTTP-target recipe).
_eval_one = eval_one


def write_results(
    results: list[dict[str, Any]],
    agg: dict[str, Any],
    out_dir: Path,
    model: str,
    *,
    eval_set_path: Path | None = None,
    collection_id: str | None = None,
    runtime_metadata: dict[str, Any] | None = None,
    eval_set_strict: bool | None = None,
    gates: dict[str, float] | None = None,
) -> Path:
    """Persist one run's full scorecard as JSON — the artifact `compare_runs.py`
    diffs. The filename is `eval_<model>_<hash>_<stamp>.json`: model-first, so
    a name sort is NOT chronological — `compare_runs.pick_latest` orders by
    `run_manifest.timestamp_utc` instead. Every pre-existing top-level key is
    unchanged; `run_manifest` is additive, so older scorecards still load."""
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-") or "model"
    safe_model = safe_model[:_FILENAME_MODEL_CHARS]
    model_hash = sha256_hex(model.encode("utf-8"))[:8]
    out_dir = out_dir.resolve()
    out_path = (out_dir / f"eval_{safe_model}_{model_hash}_{stamp}.json").resolve()
    if out_dir not in out_path.parents:
        raise ValueError("Result path escaped the configured output directory")
    persisted_results = _redact_results(results) if settings.RESULT_CONTENT_MODE == "redacted" else results
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp": stamp,
        "model": model,
        "embed_model": settings.OLLAMA_EMBED_MODEL,
        "summary": agg,
        "results": persisted_results,
        "run_manifest": build_run_manifest(
            eval_set_path,
            model,
            collection_id=collection_id,
            runtime_metadata=runtime_metadata,
            eval_set_strict=eval_set_strict,
            gates=gates,
        ),
    }
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    # mkstemp creates the file 0o600; the atomic replace carries that mode over,
    # so a scorecard with verbatim Q/A is never world-readable on a shared host.
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
    """Replace every free-text field with its SHA-256: question and answer,
    plus the retrieved source names and IDs (also corpus-identifying)."""
    redacted = []
    for row in results:
        copy = dict(row)
        for key in ("question", "answer"):
            value = copy.pop(key, None)
            if value is not None:
                copy[f"{key}_sha256"] = sha256_hex(str(value).encode("utf-8"))
        for key in ("retrieved_sources", "retrieved_source_ids"):
            values = copy.pop(key, None)
            if values is not None:
                copy[f"{key}_sha256"] = [
                    None if value is None else sha256_hex(str(value).encode("utf-8")) for value in values
                ]
        redacted.append(copy)
    return redacted


def _manifest_tags(manifest: dict[str, Any]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for path in _MLFLOW_TAG_PATHS:
        current: Any = manifest
        for key in path:
            current = current.get(key) if isinstance(current, dict) else None
        if current is not None:
            tags[".".join(path)] = str(current)
    return tags


def log_mlflow(agg: dict[str, Any], model: str, out_path: Path) -> None:
    """Optional: log this eval run (params + summary metrics + provenance tags
    + the scorecard artifact, which carries the run manifest) to MLflow so
    different models/prompts/adapters sit side-by-side and their comparability
    can be judged from the tracking UI. Never fatal — observability must never
    break the eval itself. Where the artifact goes is governed by
    `MLFLOW_TRACKING_URI` / `MLFLOW_ALLOW_REMOTE` in config.py."""
    try:
        import mlflow

        manifest = json.loads(out_path.read_text(encoding="utf-8")).get("run_manifest") or {}
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
            mlflow.set_tags(_manifest_tags(manifest))
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
