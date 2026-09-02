"""
Vector store connection — LangChain PGVector over Postgres.

Single responsibility: build and own the embeddings client + PGVector store
the RAG chain retrieves from. Both are process-wide singletons, constructed
once and reused across every eval question — never rebuilt per request/query
(the AI-service rule this harness follows: init heavy objects once).

Out of scope for this repo: actually getting documents INTO pgvector. Bring
your own ingestion pipeline (chunk -> embed -> upsert), or point
`get_vector_store()` at a Postgres instance someone else already populated.
See README.md.
"""
from __future__ import annotations

import logging

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

from config import settings

logger = logging.getLogger(__name__)

# All chunks share one collection — simplest schema that works (KISS); split
# only if a second, genuinely different vector space shows up.
COLLECTION_NAME = settings.PGVECTOR_COLLECTION_NAME

_embeddings: OllamaEmbeddings | None = None
_vector_store: PGVector | None = None


def get_embeddings() -> OllamaEmbeddings:
    """Ollama embeddings client (singleton). Requires the model in
    `OLLAMA_EMBED_MODEL` to already be pulled on the target Ollama host
    (`ollama pull nomic-embed-text`)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBED_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            client_kwargs={"timeout": settings.OLLAMA_REQUEST_TIMEOUT},
        )
        logger.info(
            "OllamaEmbeddings ready (model=%s, base_url=%s)",
            settings.OLLAMA_EMBED_MODEL,
            settings.OLLAMA_BASE_URL,
        )
    return _embeddings


def engine_args() -> dict[str, object]:
    """SQLAlchemy engine settings for the strictly sequential eval: one pooled
    connection, pre-ping to survive an idle Postgres, and a server-side
    statement_timeout so a stalled vector scan fails loud (QueryCanceled)
    instead of hanging the run — DATABASE_CONNECT_TIMEOUT only bounds the
    handshake."""
    return {
        "pool_size": 1,
        "max_overflow": 1,
        "pool_pre_ping": True,
        "connect_args": {"options": f"-c statement_timeout={settings.DATABASE_STATEMENT_TIMEOUT_MS}"},
    }


def get_vector_store() -> PGVector:
    """PGVector store over Postgres (singleton — owns its own connection
    pool). Assumes a Postgres instance with the `pgvector` extension enabled
    and a `rag_documents` collection already populated by whatever ingestion
    pipeline you point this harness at."""
    global _vector_store
    if _vector_store is None:
        _vector_store = PGVector(
            embeddings=get_embeddings(),
            collection_name=COLLECTION_NAME,
            connection=settings.database_url,
            distance_strategy=DistanceStrategy.COSINE,
            engine_args=engine_args(),
            use_jsonb=True,
        )
        logger.info("PGVector store ready (collection=%s)", COLLECTION_NAME)
    return _vector_store
