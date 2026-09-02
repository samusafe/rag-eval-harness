"""
Run provenance — everything needed to answer "what exactly produced this
scorecard?", plus the best-effort host probes (git, nvidia-smi, process RSS)
that feed it.

A score is only evidence if you can say what was measured: which commit,
which eval set bytes, which models and retrieval knobs, which prompt. All of
that is captured here and embedded in the result JSON by
`services/eval_service.write_results`. No ML stack is imported: the prompt
hash comes from `services/rag_prompt`, not from the chain.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import settings
from services.rag_prompt import RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Repo root, so default paths resolve the same way regardless of the caller's cwd.
APP_ROOT = Path(__file__).resolve().parent.parent

# Bump when a field is ADDED to the manifest. v3: eval_set_name, eval_set_strict,
# gates, result_content_mode (comparability inputs that v2 could not record).
MANIFEST_SCHEMA_VERSION = 3


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def process_rss_bytes() -> int | None:
    """Current resident set size, or None when it cannot be sampled.

    Best-effort and never fatal: memory telemetry must not break an eval run,
    so an unavailable sample is logged at debug level and reported as None."""
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except (ImportError, OSError) as error:
        logger.debug("RSS sampling unavailable: %s", error)
        return None


def _probe(*argv: str) -> str | None:
    """Run a host probe with a fixed argv and an absolute executable path.

    Resolving via `shutil.which` means a stray `git.exe` / `nvidia-smi.exe`
    dropped into the checkout is never picked up ahead of PATH (Windows
    searches the cwd first). Returns stdout on success, None on any failure."""
    executable = shutil.which(argv[0])
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, *argv[1:]],
            cwd=APP_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.HOST_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_metadata() -> dict[str, str | bool | None]:
    """Best-effort git provenance. Never fatal: the harness must run from a
    tarball, a Docker image, or a checkout without git installed, so every
    failure mode (no git binary, not a repo, ownership check, hang) resolves
    to an explicit None rather than an exception or a fabricated value."""
    commit_sha = _probe("git", "-c", f"safe.directory={APP_ROOT}", "rev-parse", "HEAD")
    porcelain = _probe("git", "-c", f"safe.directory={APP_ROOT}", "status", "--porcelain")
    # `dirty` is None (unknown), not False, when git couldn't answer — the
    # difference matters when you're deciding whether a scorecard is citable.
    return {"commit_sha": commit_sha, "dirty": None if porcelain is None else bool(porcelain)}


def collect_runtime_metadata() -> dict[str, Any]:
    """Best-effort, non-secret host/package provenance; never requires a GPU.

    `gpus` is `None` when nvidia-smi could not be queried (unknown) and `[]`
    when it answered that there are none — the same unknown-vs-false
    distinction `git_metadata` makes for `dirty`."""
    packages: dict[str, str | None] = {}
    for name in (
        "langchain-core", "langchain-classic", "langchain-ollama",
        "langchain-postgres", "sentence-transformers", "torch", "mlflow-skinny", "pydantic",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu_query = _probe("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "packages": packages,
        "ollama_host_flags": {
            "OLLAMA_FLASH_ATTENTION": settings.OLLAMA_FLASH_ATTENTION,
            "OLLAMA_KV_CACHE_TYPE": settings.OLLAMA_KV_CACHE_TYPE,
            "OLLAMA_NUM_PARALLEL": settings.OLLAMA_NUM_PARALLEL,
        },
        "gpus": None if gpu_query is None else gpu_query.splitlines(),
    }


def build_run_manifest(
    eval_set_path: Path | None,
    model: str,
    *,
    timestamp: datetime | None = None,
    git_metadata: dict[str, str | bool | None] | None = None,
    collection_id: str | None = None,
    runtime_metadata: dict[str, Any] | None = None,
    eval_set_strict: bool | None = None,
    gates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Everything needed to answer "what exactly produced this scorecard?".

    `timestamp` / `git_metadata` are injectable purely so unit tests can build
    a deterministic manifest without a clock or a git repo. `eval_set_strict`
    and `gates` are recorded because README promises permissive-mode numbers
    are not comparable against a strict run — the manifest has to say which
    one this was."""
    moment = timestamp or datetime.now(UTC)
    moment = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)

    retrieval = {
        "top_k": settings.RETRIEVAL_TOP_K,
        "rerank_top_n": settings.RERANK_TOP_N,
        "distance_strategy": settings.PGVECTOR_DISTANCE_STRATEGY,
        "collection_name": settings.PGVECTOR_COLLECTION_NAME,
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
    git = git_metadata if git_metadata is not None else globals()["git_metadata"]()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "timestamp_utc": moment.isoformat().replace("+00:00", "Z"),
        "git": {"commit_sha": git.get("commit_sha"), "dirty": git.get("dirty")},
        "eval_set_sha256": sha256_hex(eval_set_path.read_bytes()) if eval_set_path else None,
        "eval_set_name": eval_set_path.name if eval_set_path else None,
        "eval_set_strict": eval_set_strict,
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
        "gates": dict(gates) if gates else {},
        "result_content_mode": settings.RESULT_CONTENT_MODE,
        "python_version": platform.python_version(),
        "runtime": runtime_metadata if runtime_metadata is not None else collect_runtime_metadata(),
        "prompt_sha256": sha256_hex(RAG_PROMPT_TEMPLATE.encode("utf-8")),
        "config_sha256": sha256_hex(
            json.dumps(config_fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }
