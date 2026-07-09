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
│                             StrOutputParser -> answer                 │
└───────────────────────────────────────────────────────────────────────┘

This module builds the chain; eval/run_eval.py and eval/compare_runs.py
drive it and score the output. It intentionally does NOT implement chat
history persistence, streaming, semantic caching, or access-scoped
retrieval filtering — those belong to a full chat service, not an eval
harness, and adding them here would violate YAGNI (no 2nd caller needs them).
"""
from __future__ import annotations

import logging

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import settings
from services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

_reranker_model: HuggingFaceCrossEncoder | None = None


def get_reranker_model() -> HuggingFaceCrossEncoder:
    """Cross-encoder reranker (singleton — loaded once, on first use)."""
    global _reranker_model
    if _reranker_model is None:
        logger.info("Loading cross-encoder reranker: %s", settings.RERANK_MODEL)
        _reranker_model = HuggingFaceCrossEncoder(model_name=settings.RERANK_MODEL)
    return _reranker_model


# ============================================================================
# RAG prompt template
# ============================================================================
# Constrains the LLM to: (1) only use the provided context — no hallucinating,
# (2) cite sources exactly as labeled, (3) refuse with one fixed sentence when
# the context doesn't cover the question. That fixed refusal sentence is what
# eval/eval_service.py's `refusal_ok` checks for verbatim.
# ============================================================================

RAG_PROMPT_TEMPLATE = """You are an internal AI assistant with access to the organization's knowledge base.
Answer the user's question based STRICTLY on the provided context below. Do not use any external knowledge.

**Rules:**
- If the user is just greeting you (e.g., "hi", "hello"), reply with a short, polite greeting and ask how you can help. Do not cite any sources.
- If the context contains the answer, give a clear, professional response. Only cite a source you actually used, written exactly as it appears in the "[Source N: <file>, Page <p>]" header. Never invent, guess, or paraphrase a source name.
- If the context does NOT contain enough information, respond with EXACTLY: "I don't have enough information in the knowledge base to answer this question." Add no extra explanation and cite no sources.
- Keep answers concise; do not narrate your internal reasoning.

---

**Previous conversation:**
{chat_history}

---

**Context from knowledge base:**
{context}

---

**User question:** {question}

**Answer:**"""


def _format_history(history: list[dict] | None) -> str:
    """Render chat history into the prompt's plain-text form."""
    if not history:
        return "No previous conversation."
    lines = [
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content')}"
        for m in history
    ]
    return "\n\n".join(lines)


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved chunks into one context string, each tagged with its
    source so the LLM can cite it — and so the eval can check whether the
    expected source actually made the reranked top-N (retrieval hit-rate)."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "N/A")
        parts.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def get_rag_chain(model: str | None = None):
    """
    Build the retriever + prompt + LLM for the RAG chain.

    `model` overrides which Ollama chat model answers; falls back to
    `settings.OLLAMA_CHAT_MODEL` when None — this is what lets
    `run_eval.py --model <name>` A/B two models without touching `.env`.

    Returns `(prompt, llm, get_retriever)`:
      - `prompt | llm | StrOutputParser()` is the generation chain.
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
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

    llm = ChatOllama(
        model=model or settings.OLLAMA_CHAT_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,
        num_ctx=settings.OLLAMA_NUM_CTX,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        client_kwargs={"timeout": settings.OLLAMA_REQUEST_TIMEOUT},
    )

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    return prompt, llm, get_retriever
