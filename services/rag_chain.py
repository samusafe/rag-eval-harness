"""
RAG chain builder — retriever + prompt + LLM, composed with LangChain LCEL.

┌───────────────────────────────────────────────────────────────────────┐
│  THE PIPELINE                                                         │
│                                                                       │
│  question ──► OllamaEmbeddings ──► pgvector cosine search (top K)    │
│                                          │                            │
│                                          ▼                            │
│                          cross-encoder rerank (top N)                 │
│                                          │                            │
│                                          ▼                            │
│                 prompt(context + chat_history + question)             │
│                                          │                            │
│                                          ▼                            │
│                     ChatOllama (local, fine-tuned or base)             │
│                                          │                            │
│                                          ▼                            │
│                       AIMessage (answer + timings/tokens)             │
└───────────────────────────────────────────────────────────────────────┘

This module ONLY builds the chain; the prompt text, refusal sentence and
context formatting live in services/rag_prompt.py (dependency-free), and
eval/run_eval.py drives and scores the output. It intentionally does NOT
implement chat history persistence, streaming, semantic caching, or
access-scoped retrieval filtering — those belong to a full chat service, not
an eval harness, and adding them here would violate YAGNI (no 2nd caller
needs them).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

from config import settings
from services.rag_prompt import RAG_PROMPT_TEMPLATE
from services.vector_store import get_vector_store

__all__ = [
    "RAG_PROMPT_TEMPLATE",
    "LocalCrossEncoder",
    "TimedRerankRetriever",
    "get_rag_chain",
    "get_reranker_model",
]

logger = logging.getLogger(__name__)

_reranker_model: LocalCrossEncoder | None = None


class LocalCrossEncoder(BaseCrossEncoder):
    """Minimal maintained adapter around sentence-transformers CrossEncoder."""

    def __init__(self) -> None:
        device = None if settings.RERANK_DEVICE == "auto" else settings.RERANK_DEVICE
        self.client = CrossEncoder(
            settings.RERANK_MODEL,
            device=device,
            max_length=settings.RERANK_MAX_LENGTH,
            revision=settings.RERANK_MODEL_REVISION,
            trust_remote_code=False,
        )

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.client.predict(text_pairs, batch_size=settings.RERANK_BATCH_SIZE)
        return [float(score) for score in scores]


class TimedRerankRetriever:
    """Sequential base retrieval + reranking with observable stage timings."""

    def __init__(self, base_retriever: Any, compressor: CrossEncoderReranker) -> None:
        self.base_retriever = base_retriever
        self.compressor = compressor
        self.last_timings: dict[str, float] = {}

    def invoke(self, query: str) -> list[Document]:
        started = time.perf_counter()
        candidates = self.base_retriever.invoke(query)
        retrieved = time.perf_counter()
        documents = list(self.compressor.compress_documents(candidates, query))
        finished = time.perf_counter()
        self.last_timings = {
            "vector_retrieval_latency_s": retrieved - started,
            "rerank_latency_s": finished - retrieved,
        }
        return documents


def get_reranker_model() -> LocalCrossEncoder:
    """Cross-encoder reranker (singleton — loaded once, on first use)."""
    global _reranker_model
    if _reranker_model is None:
        logger.info("Loading cross-encoder reranker: %s", settings.RERANK_MODEL)
        _reranker_model = LocalCrossEncoder()
    return _reranker_model


def get_rag_chain(model: str | None = None):
    """
    Build the retriever + prompt + LLM for the RAG chain.

    `model` overrides which Ollama chat model answers; falls back to
    `settings.OLLAMA_CHAT_MODEL` when None — this is what lets
    `run_eval.py --model <name>` A/B two models without touching `.env`.

    Returns `(prompt, llm, get_retriever)`:
      - `prompt | llm` is the generation chain; callers retain AIMessage metadata.
      - `get_retriever(collection_id=None)` is a factory — call it to get a
        fresh, optionally collection-scoped retriever without rebuilding the
        vector store / reranker singletons.
    """
    vector_store = get_vector_store()

    def get_retriever(collection_id: str | None = None):
        search_kwargs: dict = {"k": settings.RETRIEVAL_TOP_K}
        if collection_id:
            search_kwargs["filter"] = {"collection_id": collection_id}

        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )
        compressor = CrossEncoderReranker(
            model=get_reranker_model(), top_n=settings.RERANK_TOP_N
        )
        return TimedRerankRetriever(base_retriever, compressor)

    llm = ChatOllama(
        model=model or settings.OLLAMA_CHAT_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.OLLAMA_TEMPERATURE,
        num_ctx=settings.OLLAMA_NUM_CTX,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        num_predict=settings.OLLAMA_NUM_PREDICT,
        client_kwargs={"timeout": settings.OLLAMA_REQUEST_TIMEOUT},
    )

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    return prompt, llm, get_retriever
