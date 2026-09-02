"""Unit tests for services/rag_prompt.py — the prompt template, refusal
sentence, and the context-formatting/truncation logic the scorer's citation
metric depends on. Dependency-free module, so these run with no ML/DB stack.
"""
import math

from services import rag_prompt
from services.eval_metrics import _CITATION_RE


class _Doc:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def _set_budget(monkeypatch, *, num_ctx, num_predict=0, reserve=0, chars_per_token=1):
    # Config validation is NOT re-run when monkeypatching attributes directly
    # on the already-constructed `settings` singleton, so any ints work here
    # even combinations `Settings()` itself would reject.
    monkeypatch.setattr(rag_prompt.settings, "OLLAMA_NUM_CTX", num_ctx)
    monkeypatch.setattr(rag_prompt.settings, "OLLAMA_NUM_PREDICT", num_predict)
    monkeypatch.setattr(rag_prompt.settings, "RAG_CONTEXT_RESERVE_TOKENS", reserve)
    monkeypatch.setattr(rag_prompt.settings, "CHARS_PER_TOKEN_ESTIMATE", chars_per_token)


def test_context_stays_within_budget(monkeypatch):
    _set_budget(monkeypatch, num_ctx=1000)
    docs = [
        _Doc("first chunk of text", {"source_file": "a.pdf", "page": "1"}),
        _Doc("second chunk of text", {"source_file": "b.pdf", "page": "2"}),
    ]
    context, stats = rag_prompt.format_context_with_stats(docs)
    assert stats["used_chars"] <= stats["budget_chars"]
    assert len(context) == stats["used_chars"]


def test_oversized_chunk_is_truncated_and_counted(monkeypatch):
    _set_budget(monkeypatch, num_ctx=40)  # budget_chars == 40
    doc = _Doc("A" * 1000, {"source_file": "policy.pdf", "page": "N/A"})

    context, stats = rag_prompt.format_context_with_stats([doc])

    header = "[Source 1: policy.pdf, Page N/A]\n"
    expected_content_len = 40 - len(header)
    assert stats["dropped_or_truncated_chunks"] == 1
    assert context == header + doc.page_content[:expected_content_len]
    assert doc.page_content.startswith(context[len(header):])


def test_chunks_past_the_budget_are_dropped_not_silently_lost(monkeypatch):
    docs = [
        _Doc("first content", {"source_file": "a.pdf", "page": "1"}),
        _Doc("second content", {"source_file": "b.pdf", "page": "2"}),
        _Doc("third content", {"source_file": "c.pdf", "page": "3"}),
    ]
    header1 = "[Source 1: a.pdf, Page 1]\n"
    # Budget fits exactly the first doc's header + full content and nothing else.
    _set_budget(monkeypatch, num_ctx=len(header1) + len(docs[0].page_content))

    context, stats = rag_prompt.format_context_with_stats(docs)

    assert context == header1 + docs[0].page_content
    assert stats["dropped_or_truncated_chunks"] == 2


def test_headers_are_exactly_what_the_citation_regex_extracts(monkeypatch):
    _set_budget(monkeypatch, num_ctx=10_000)
    docs = [
        _Doc("alpha content", {"source_file": "a.pdf", "page": "1"}),
        _Doc("beta content", {"source_file": "b.pdf", "page": "2"}),
    ]

    context, _stats = rag_prompt.format_context_with_stats(docs)

    assert _CITATION_RE.findall(context) == [
        "[Source 1: a.pdf, Page 1]",
        "[Source 2: b.pdf, Page 2]",
    ]


def test_estimated_context_tokens_is_ceiling_division(monkeypatch):
    _set_budget(monkeypatch, num_ctx=10_000, chars_per_token=4)
    docs = [_Doc("x" * 10, {"source_file": "a.pdf", "page": "1"})]

    _context, stats = rag_prompt.format_context_with_stats(docs)

    assert stats["estimated_context_tokens"] == math.ceil(stats["used_chars"] / 4)


def test_empty_docs_returns_empty_context_and_zero_used_chars(monkeypatch):
    _set_budget(monkeypatch, num_ctx=1000)

    context, stats = rag_prompt.format_context_with_stats([])

    assert context == ""
    assert stats["used_chars"] == 0


def test_safe_label_strips_brackets_control_chars_and_collapses_whitespace():
    assert rag_prompt.safe_label("  a\tb\n\nc  ") == "a b c"
    assert rag_prompt.safe_label("[bracketed]") == "bracketed"
    assert rag_prompt.safe_label("a\x00b") == "ab"


def test_safe_label_truncates_to_max_length():
    assert rag_prompt.safe_label("x" * 300, max_length=10) == "x" * 10


def test_safe_label_returns_unknown_for_blank():
    assert rag_prompt.safe_label("") == "Unknown"
    assert rag_prompt.safe_label("   ") == "Unknown"


def test_format_history_empty_is_no_previous_conversation():
    assert rag_prompt.format_history(None) == "No previous conversation."
    assert rag_prompt.format_history([]) == "No previous conversation."


def test_format_history_maps_user_and_assistant_roles():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert rag_prompt.format_history(history) == "User: Hi\n\nAssistant: Hello"


def test_rag_prompt_template_contains_required_placeholders_and_refusal():
    for token in ("{chat_history}", "{context}", "{question}"):
        assert token in rag_prompt.RAG_PROMPT_TEMPLATE
    assert f'EXACTLY: "{rag_prompt.REFUSAL}"' in rag_prompt.RAG_PROMPT_TEMPLATE
