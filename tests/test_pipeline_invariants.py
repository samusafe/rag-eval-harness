"""Regression tests pinning two pipeline invariants that are easy to break
silently: the cross-encoder reranker is a process-wide singleton (never
reloaded per retriever/question), and the eval path never serves a cached
answer (no LLM cache is wired into the generation chain).
"""
import inspect

from services import rag_chain, vector_store
from services.eval_service import EvalCase, eval_one


class _Doc:
    def __init__(self, source_file: str, page_content: str):
        self.metadata = {"source_file": source_file, "page": "N/A"}
        self.page_content = page_content


def test_reranker_model_is_loaded_once_per_process(monkeypatch):
    constructions: list[object] = []
    monkeypatch.setattr(
        rag_chain.LocalCrossEncoder,
        "__init__",
        lambda self: constructions.append(self) or None,
    )
    monkeypatch.setattr(rag_chain, "_reranker_model", None)

    first = rag_chain.get_reranker_model()
    second = rag_chain.get_reranker_model()

    assert first is second
    assert len(constructions) == 1


def test_every_retriever_shares_the_one_reranker(monkeypatch):
    constructions: list[object] = []
    monkeypatch.setattr(
        rag_chain.LocalCrossEncoder,
        "__init__",
        lambda self: constructions.append(self) or None,
    )
    monkeypatch.setattr(rag_chain, "_reranker_model", None)

    class _FakeBaseRetriever:
        pass

    class _FakeStore:
        def __init__(self) -> None:
            self.as_retriever_calls = 0

        def as_retriever(self, **kwargs):
            self.as_retriever_calls += 1
            return _FakeBaseRetriever()

    fake_store = _FakeStore()
    monkeypatch.setattr(rag_chain, "get_vector_store", lambda: fake_store)
    monkeypatch.setattr(rag_chain, "ChatOllama", lambda **kwargs: object())

    _prompt, _llm, get_retriever = rag_chain.get_rag_chain("some-model")
    a = get_retriever()
    b = get_retriever()

    # Direct attribute path: both retrievers' compressors must wrap the exact
    # same reranker instance, and it must have been built exactly once.
    assert a.compressor.model is b.compressor.model
    assert len(constructions) == 1
    assert fake_store.as_retriever_calls == 2


def test_vector_store_and_embeddings_are_singletons(monkeypatch):
    from config import settings

    def _make_counting_stub():
        class _Stub:
            calls: list["_Stub"] = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                type(self).calls.append(self)

        return _Stub

    embeddings_stub = _make_counting_stub()
    pgvector_stub = _make_counting_stub()

    monkeypatch.setattr(vector_store, "OllamaEmbeddings", embeddings_stub)
    monkeypatch.setattr(vector_store, "PGVector", pgvector_stub)
    monkeypatch.setattr(vector_store, "_embeddings", None)
    monkeypatch.setattr(vector_store, "_vector_store", None)

    first_embeddings = vector_store.get_embeddings()
    second_embeddings = vector_store.get_embeddings()
    assert first_embeddings is second_embeddings
    assert len(embeddings_stub.calls) == 1

    first_store = vector_store.get_vector_store()
    second_store = vector_store.get_vector_store()
    assert first_store is second_store
    assert len(pgvector_stub.calls) == 1

    kwargs = first_store.kwargs
    assert kwargs["collection_name"] == settings.PGVECTOR_COLLECTION_NAME
    assert kwargs["engine_args"]["connect_args"] == {
        "options": f"-c statement_timeout={settings.DATABASE_STATEMENT_TIMEOUT_MS}"
    }


def test_eval_never_serves_a_cached_answer():
    class _RecordingRetriever:
        def __init__(self) -> None:
            self.questions: list[str] = []

        def invoke(self, question: str):
            self.questions.append(question)
            return [_Doc("policy.pdf", "The policy grants 20 days.")]

    class _VaryingChain:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _inputs):
            self.calls += 1
            return f"20 days [Source 1: policy.pdf, Page N/A] (call {self.calls})"

    retriever = _RecordingRetriever()
    chain = _VaryingChain()
    row = EvalCase(
        id="q1",
        question="How many vacation days?",
        expected_sources=["policy"],
        expected_keywords=["20 days"],
        must_refuse=False,
    )

    first = eval_one(row, retriever, chain)
    second = eval_one(row, retriever, chain)

    assert retriever.questions == [row.question, row.question]
    assert chain.calls == 2
    assert first["answer"] != second["answer"]
    assert first["keyword_recall"] == 1.0
    assert second["keyword_recall"] == 1.0
    assert first["citation_validity"] == 1.0
    assert second["citation_validity"] == 1.0
    # The retriever above carries no `last_timings` attribute, so the stage
    # timing must come back as unknown (None), never a fabricated 0.
    assert first["vector_retrieval_latency_s"] is None
    assert second["vector_retrieval_latency_s"] is None


def test_rag_chain_wires_no_cache_into_the_generation_path():
    source = inspect.getsource(rag_chain)
    for forbidden in ("set_llm_cache", "InMemoryCache", "SQLiteCache", "cache="):
        assert forbidden not in source

    from langchain_core.globals import get_llm_cache

    assert get_llm_cache() is None
