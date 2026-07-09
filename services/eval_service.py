"""
RAG eval — shared scoring core.
======================================================================
The reusable, dependency-light heart of the eval harness: load the eval set,
score one question through the real chain, aggregate, persist, and (optionally)
log to MLflow. `eval/run_eval.py` is just the presentation layer on top of this
(rich tables, argparse) — kept separate so this module stays importable and
unit-testable without pulling in `rich`.
======================================================================
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path

from config import settings
from services.rag_chain import _format_docs, _format_history

logger = logging.getLogger(__name__)

# Repo root, so default paths resolve the same way regardless of the caller's cwd.
APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_SET = APP_ROOT / "eval" / "eval_set.example.jsonl"
DEFAULT_OUT_DIR = APP_ROOT / "eval" / "results"

# The exact refusal sentence the prompt instructs the model to produce for
# out-of-KB questions (services/rag_chain.py RAG_PROMPT_TEMPLATE).
REFUSAL = "I don't have enough information in the knowledge base to answer this question."


def load_eval_set(path: Path) -> list[dict]:
    """Load JSONL eval rows. Lines starting with // are comments and skipped.
    A malformed line is logged and dropped — never silently ignored, never
    fatal for the rest of the set."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipped malformed eval line %d: %s", line_num, e)
    if not rows:
        raise ValueError(f"No eval rows loaded from {path}")
    return rows


def retrieval_hit(docs, expected_sources: list[str]) -> bool | None:
    """True if any expected source substring appears in any retrieved doc's
    source_file. None when the row lists no expected_sources (e.g. refusal
    rows) — excluded from the rate rather than counted as a miss."""
    if not expected_sources:
        return None
    retrieved = [str(d.metadata.get("source_file", "")).lower() for d in docs]
    return any(exp.lower() in src for exp in expected_sources for src in retrieved)


def keyword_recall(answer: str, expected_keywords: list[str]) -> float | None:
    """Fraction of expected keywords present in the answer (case-insensitive).
    None if none specified for this row."""
    if not expected_keywords:
        return None
    a = answer.lower()
    return sum(1 for kw in expected_keywords if kw.lower() in a) / len(expected_keywords)


def refusal_ok(answer: str, must_refuse: bool) -> bool | None:
    """For out-of-KB questions, correct == produced the exact refusal
    sentence. None for every other row."""
    if not must_refuse:
        return None
    return REFUSAL.lower() in answer.lower()


def aggregate(results: list[dict]) -> dict:
    """Roll per-question results into the scorecard summary. Each metric only
    averages over rows where it applies (None-filtered) — a mixed eval set
    (some rows scoring retrieval, some refusal) never dilutes either metric."""
    rh = [r["retrieval_hit"] for r in results if r["retrieval_hit"] is not None]
    kr = [r["keyword_recall"] for r in results if r["keyword_recall"] is not None]
    rf = [r["refusal_ok"] for r in results if r["refusal_ok"] is not None]
    lat = [r["latency_s"] for r in results]
    pct = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    return {
        "n": len(results),
        "retrieval_hit_rate": pct(rh),
        "answer_keyword_recall": pct(kr),
        "refusal_accuracy": pct(rf),
        "median_latency_s": round(statistics.median(lat), 2) if lat else None,
    }


def write_results(results: list[dict], agg: dict, out_dir: Path, model: str) -> Path:
    """Persist one run's full scorecard as JSON — the artifact `compare_runs.py`
    diffs. Filename embeds a sortable timestamp so `sorted(glob(...))` is
    chronological with no extra bookkeeping."""
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
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def log_mlflow(agg: dict, model: str, out_path: Path) -> None:
    """Optional: log this eval run (params + summary metrics + scorecard
    artifact) to MLflow so different models/prompts/adapters sit side-by-side.
    Never fatal — observability must never break the eval itself."""
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
                v = agg.get(key)
                if v is not None:
                    mlflow.log_metric(key, float(v))
            mlflow.log_artifact(str(out_path))
        logger.info("Eval logged to MLflow (%s)", settings.MLFLOW_EXPERIMENT_NAME)
    except Exception as e:  # noqa: BLE001 — tracking must never break the eval
        logger.warning("MLflow logging skipped: %s", e)


def _eval_one(row: dict, retriever, chain) -> dict:
    """Score ONE eval row through the real pipeline: time retrieval +
    generation, then compute the per-question metrics. Blocking (LLM + DB
    calls) — callers on an event loop should offload this to a thread."""
    q = row["question"]
    t0 = time.perf_counter()
    docs = retriever.invoke(q)
    answer = chain.invoke(
        {
            "chat_history": _format_history([]),
            "context": _format_docs(docs),
            "question": q,
        }
    )
    latency = round(time.perf_counter() - t0, 2)

    return {
        "id": row.get("id", q[:40]),
        "question": q,
        "retrieval_hit": retrieval_hit(docs, row.get("expected_sources", [])),
        "keyword_recall": keyword_recall(answer, row.get("expected_keywords", [])),
        "refusal_ok": refusal_ok(answer, row.get("must_refuse", False)),
        "latency_s": latency,
        "retrieved_sources": [d.metadata.get("source_file", "?") for d in docs],
        "answer": answer,
    }
