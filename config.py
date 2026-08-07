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

from pydantic_settings import BaseSettings


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
    OLLAMA_KEEP_ALIVE: str = "30m"
    # Per-request timeout (seconds) so a stalled model fails loud, not hangs.
    OLLAMA_REQUEST_TIMEOUT: float = 120.0
    # bench_ollama.py runs on the host, so Ollama is plain localhost there too.
    OLLAMA_BENCH_URL: str = "http://localhost:11434"

    # --- Postgres + pgvector (wherever your ingested chunks live) ---
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_USER: str = "raguser"
    DATABASE_PASSWORD: str = "ragpassword"
    DATABASE_NAME: str = "rag_eval"

    # --- Retrieval / reranking ---
    # Retrieve a wider candidate set from pgvector, then rerank down to the
    # top N with a cross-encoder — cheap similarity search first, expensive
    # cross-encoder only on the shortlist.
    RETRIEVAL_TOP_K: int = 20
    RERANK_TOP_N: int = 5
    RERANK_MODEL: str = "BAAI/bge-reranker-base"

    # --- MLflow (experiment tracking for eval + bench runs) ---
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "rag-eval-harness"
    MLFLOW_BENCH_EXPERIMENT_NAME: str = "rag-eval-harness-bench"

    @property
    def database_url(self) -> str:
        """SQLAlchemy-compatible connection string for pgvector (psycopg v3
        driver, which LangChain's PGVector integration expects)."""
        return (
            f"postgresql+psycopg://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton — imported everywhere as `from config import settings`.
settings = Settings()
