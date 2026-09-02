# ============================================================================
# Centralized configuration — Pydantic Settings
# ============================================================================
# Single source of truth for every knob the harness reads. Everything has a
# sane localhost default so `Settings()` never raises on a bare checkout —
# only override what you need via `.env` (copy `.env.example`) or real env
# vars. Nothing here is a secret: this is a local eval harness, not a
# multi-tenant service, so there's no case for a required-with-no-default
# field.
# ============================================================================

from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlparse

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ROOT = Path(__file__).resolve().parent
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class Settings(BaseSettings):
    """Every setting the eval harness needs, in one place."""

    # --- Ollama (local inference — no cloud LLM provider) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # Generic placeholder — point this at whatever chat model you have pulled
    # in Ollama (`ollama pull <model>`), or override with --model / this env var.
    OLLAMA_CHAT_MODEL: str = "my-finetuned-model"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    # Near-greedy decoding. An eval is only evidence if a re-run gives roughly
    # the same numbers, so sampling stays low — and configurable here rather
    # than hardcoded inside the chain.
    OLLAMA_TEMPERATURE: float = 0.1
    # Deliberate, not left at Ollama's defaults: big enough for a reranked RAG
    # context; keep_alive avoids reloading the model between eval questions.
    OLLAMA_NUM_CTX: int = 4096
    OLLAMA_NUM_PREDICT: int = 512
    RAG_CONTEXT_RESERVE_TOKENS: int = 512
    OLLAMA_KEEP_ALIVE: str = "30m"
    # Per-request timeout (seconds) so a stalled model fails loud, not hangs.
    OLLAMA_REQUEST_TIMEOUT: float = 120.0
    # bench_ollama.py runs on the host, so Ollama is plain localhost there too.
    OLLAMA_BENCH_URL: str = "http://localhost:11434"
    # Ollama HOST tuning flags, recorded in the run manifest for provenance only
    # (the harness does not set them; export them before `ollama serve`).
    OLLAMA_FLASH_ATTENTION: str | None = None
    OLLAMA_KV_CACHE_TYPE: str | None = None
    OLLAMA_NUM_PARALLEL: str | None = None

    # --- Postgres + pgvector (wherever your ingested chunks live) ---
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_USER: str = "raguser"
    DATABASE_PASSWORD: str = "ragpassword"
    DATABASE_NAME: str = "rag_eval"
    DATABASE_SSLMODE: Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"] = "prefer"
    DATABASE_CONNECT_TIMEOUT: int = 10
    # Bounds every statement, not just the handshake: a stalled pgvector scan
    # must fail loud instead of hanging the whole eval run.
    DATABASE_STATEMENT_TIMEOUT_MS: int = 30_000

    # --- Retrieval / reranking ---
    # Retrieve a wider candidate set from pgvector, then rerank down to the
    # top N with a cross-encoder — cheap similarity search first, expensive
    # cross-encoder only on the shortlist.
    RETRIEVAL_TOP_K: int = 20
    RERANK_TOP_N: int = 5
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_MODEL_REVISION: str | None = None
    # Keep the 0.3B reranker out of VRAM by default on 8 GB laptops. Users can
    # opt into CUDA only after measuring generator + embedder headroom.
    RERANK_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"
    RERANK_MAX_LENGTH: int = 512
    RERANK_BATCH_SIZE: int = 8
    PGVECTOR_DISTANCE_STRATEGY: Literal["cosine"] = "cosine"
    # All chunks share one collection — simplest schema that works (KISS).
    PGVECTOR_COLLECTION_NAME: str = "rag_documents"
    # Context budget estimate: Ollama tokenization is model-specific and not
    # exposed through the chain, so chars/token is an explicit approximation.
    CHARS_PER_TOKEN_ESTIMATE: int = 4
    # User-managed corpus/ingestion version. Set this to a commit, dataset
    # digest, or immutable release ID before treating comparisons as citable.
    CORPUS_REVISION: str = "unversioned"

    # --- MLflow (experiment tracking for eval + bench runs) ---
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "rag-eval-harness"
    MLFLOW_BENCH_EXPERIMENT_NAME: str = "rag-eval-harness-bench"
    # Scorecards carry verbatim questions/answers. Shipping them to a non-local
    # tracking server is opt-in: loopback and file/sqlite stores always work,
    # anything else needs https AND this flag.
    MLFLOW_ALLOW_REMOTE: bool = False

    # Artifact privacy. `full` preserves the historical scorecard; `redacted`
    # stores hashes instead of question/answer text.
    RESULT_CONTENT_MODE: Literal["full", "redacted"] = "full"

    # --- Harness internals ---
    # Wall-clock bound for best-effort host probes (git, nvidia-smi).
    HOST_PROBE_TIMEOUT: float = 2.0
    # compare_runs.py ignores per-question latency deltas below this (noise).
    COMPARE_LATENCY_NOISE_S: float = 0.5

    @property
    def database_url(self) -> str:
        """SQLAlchemy-compatible connection string for pgvector (psycopg v3
        driver, which LangChain's PGVector integration expects)."""
        user = quote(self.DATABASE_USER, safe="")
        password = quote(self.DATABASE_PASSWORD, safe="")
        database = quote(self.DATABASE_NAME, safe="")
        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{database}"
            f"?sslmode={self.DATABASE_SSLMODE}&connect_timeout={self.DATABASE_CONNECT_TIMEOUT}"
        )

    @field_validator(
        "OLLAMA_NUM_CTX",
        "OLLAMA_NUM_PREDICT",
        "RAG_CONTEXT_RESERVE_TOKENS",
        "RETRIEVAL_TOP_K",
        "RERANK_TOP_N",
        "RERANK_MAX_LENGTH",
        "RERANK_BATCH_SIZE",
        "DATABASE_CONNECT_TIMEOUT",
        "DATABASE_STATEMENT_TIMEOUT_MS",
        "CHARS_PER_TOKEN_ESTIMATE",
    )
    @classmethod
    def positive_integers(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than zero")
        return value

    @field_validator("OLLAMA_REQUEST_TIMEOUT", "HOST_PROBE_TIMEOUT")
    @classmethod
    def positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be greater than zero")
        return value

    @field_validator("COMPARE_LATENCY_NOISE_S")
    @classmethod
    def non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must not be negative")
        return value

    @field_validator("RERANK_MODEL_REVISION", mode="before")
    @classmethod
    def blank_revision_is_unpinned(cls, value: object) -> object:
        # `.env` files cannot express null; `RERANK_MODEL_REVISION=` must mean
        # "unpinned", not the literal revision "".
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("OLLAMA_TEMPERATURE")
    @classmethod
    def valid_temperature(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("must be between 0 and 2")
        return value

    @model_validator(mode="after")
    def valid_retrieval_window(self) -> "Settings":
        if self.RERANK_TOP_N > self.RETRIEVAL_TOP_K:
            raise ValueError("RERANK_TOP_N cannot exceed RETRIEVAL_TOP_K")
        if self.OLLAMA_NUM_PREDICT + self.RAG_CONTEXT_RESERVE_TOKENS >= self.OLLAMA_NUM_CTX:
            raise ValueError("output and prompt reserves must leave room for retrieved context")
        if self.DATABASE_HOST.casefold() not in LOCAL_HOSTS:
            if self.DATABASE_PASSWORD == "ragpassword":
                raise ValueError("the default database password is only allowed for localhost")
            if self.DATABASE_SSLMODE in {"disable", "allow", "prefer"}:
                raise ValueError(
                    "a non-local database must enforce TLS: set DATABASE_SSLMODE to "
                    "require, verify-ca or verify-full (prefer/allow silently fall back to plaintext)"
                )
        self._check_mlflow_uri()
        return self

    def _check_mlflow_uri(self) -> None:
        """Loopback and local stores (file:, sqlite:, a bare path) always pass.
        Any other host must use https AND set MLFLOW_ALLOW_REMOTE=true, because
        `--mlflow` uploads the full scorecard artifact there."""
        parsed = urlparse(self.MLFLOW_TRACKING_URI)
        if parsed.scheme not in {"http", "https"}:
            return
        host = (parsed.hostname or "").casefold()
        if host in LOCAL_HOSTS:
            return
        if not self.MLFLOW_ALLOW_REMOTE:
            raise ValueError(
                f"MLFLOW_TRACKING_URI {self.MLFLOW_TRACKING_URI!r} is not local; scorecards contain "
                "questions and answers, so set MLFLOW_ALLOW_REMOTE=true to confirm the upload target"
            )
        if parsed.scheme != "https":
            raise ValueError("a remote MLFLOW_TRACKING_URI must use https")

    model_config = SettingsConfigDict(
        env_file=APP_ROOT / ".env",
        case_sensitive=True,
        extra="ignore",
    )


# Singleton — imported everywhere as `from config import settings`.
settings = Settings()
