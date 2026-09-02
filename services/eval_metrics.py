"""
Eval metrics — pure, deterministic scorers and the scorecard aggregate.

Standard library only (plus the refusal sentence from `services/rag_prompt`),
so every function here unit-tests with plain dicts and fake docs: no LangChain,
no torch, no fakes of a passing pipeline.

Every metric here is a LEXICAL regression signal, not a factuality judgement —
there is deliberately no LLM-as-judge. See README "What these metrics are not".
"""
from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from typing import Any, cast

from services.rag_prompt import REFUSAL

__all__ = [
    "REFUSAL",
    "aggregate",
    "answerability_ok",
    "citation_scores",
    "document_id",
    "exact_retrieval_hit",
    "keyword_recall",
    "normalize_refusal",
    "percentile",
    "reciprocal_rank",
    "refusal_ok",
    "retrieval_hit",
]

# Rates are rounded for the scorecard, but far past any plausible gate
# threshold: 3 places would let a true 0.7996 pass `--gate-hit-rate 0.8`.
_RATE_PLACES = 6
_SECONDS_PLACES = 4


def retrieval_hit(docs: Sequence[Any], expected_sources: list[str]) -> bool | None:
    """True if any expected source substring appears in any retrieved doc's
    source_file. None when the row lists no expected_sources (e.g. refusal
    rows) — excluded from the rate rather than counted as a miss.

    Substring matching over file names, not exact chunk identity: it tolerates
    the path/extension noise real ingestion pipelines produce. See
    `EvalCase.expected_source_ids` for the exact-match successor."""
    if not expected_sources:
        return None
    retrieved = [str(doc.metadata.get("source_file", "")).lower() for doc in docs]
    return any(expected.lower() in source for expected in expected_sources for source in retrieved)


def document_id(doc: Any) -> str | None:
    """Return a stable ingester-provided ID without fabricating one."""
    metadata = getattr(doc, "metadata", {})
    for key in ("source_id", "chunk_id", "document_id", "id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    value = getattr(doc, "id", None)
    return str(value).strip() if value is not None and str(value).strip() else None


def exact_retrieval_hit(docs: Sequence[Any], expected_ids: list[str] | None) -> bool | None:
    if not expected_ids:
        return None
    expected = {value.casefold() for value in expected_ids}
    return any((doc_id := document_id(doc)) is not None and doc_id.casefold() in expected for doc in docs)


def reciprocal_rank(docs: Sequence[Any], expected_ids: list[str] | None) -> float | None:
    if not expected_ids:
        return None
    expected = {value.casefold() for value in expected_ids}
    for rank, doc in enumerate(docs, 1):
        doc_id = document_id(doc)
        if doc_id is not None and doc_id.casefold() in expected:
            return round(1 / rank, 4)
    return 0.0


def keyword_recall(answer: str, expected_keywords: list[str]) -> float | None:
    """Fraction of expected keywords present in the answer (case-insensitive
    substring match). None if none specified for this row.

    LEXICAL, NOT FACTUAL: this measures whether the expected strings appear,
    not whether the answer is true. "The policy is not 20 days" scores 1.0
    against `["20 days"]`. Treat it as a cheap regression signal on answer
    content, never as a correctness or hallucination score — refusal accuracy
    is the anti-hallucination metric."""
    if not expected_keywords:
        return None
    normalized_answer = answer.lower()
    return sum(keyword.lower() in normalized_answer for keyword in expected_keywords) / len(
        expected_keywords
    )


# Typographic punctuation a chat model may substitute for the ASCII in REFUSAL.
_TYPOGRAPHIC_TO_ASCII = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
# Markdown emphasis a model may wrap a one-sentence answer in.
_EMPHASIS_CHARS = "*_"


def normalize_refusal(text: str) -> str:
    """Trim, collapse internal whitespace, casefold, map curly quotes to ASCII
    and strip surrounding markdown emphasis — and nothing else.

    Deliberately small: it forgives formatting the model can't control
    (a trailing newline, a double space, capitalization, a typographic
    apostrophe, `**bold**`) without forgiving any added words."""
    collapsed = " ".join(text.split()).translate(_TYPOGRAPHIC_TO_ASCII)
    return collapsed.strip(_EMPHASIS_CHARS).strip().casefold()


def refusal_ok(answer: str, must_refuse: bool) -> bool | None:
    """For out-of-KB questions, correct == the WHOLE answer is the refusal
    sentence. None for every other row.

    Full-string equality after `normalize_refusal`, not a substring test: a
    substring test would score "The capital is Canberra. I don't have enough
    information in the knowledge base to answer this question." as a correct
    refusal, i.e. it would pass a hallucination as anti-hallucination."""
    if not must_refuse:
        return None
    return normalize_refusal(answer) == normalize_refusal(REFUSAL)


def answerability_ok(answer: str, must_refuse: bool) -> bool | None:
    """For answerable rows, correct == the model did NOT emit the refusal
    sentence. None for refusal rows. Penalizes over-refusal, the mirror image
    of `refusal_ok`."""
    if must_refuse:
        return None
    return normalize_refusal(answer) != normalize_refusal(REFUSAL)


_CITATION_RE = re.compile(r"\[Source \d+: [^\]\r\n]+\]")


def citation_scores(answer: str, context: str, must_refuse: bool) -> tuple[bool | None, float | None]:
    """Deterministic citation presence and validity against emitted headers.

    Returns `(present, validity)`. `validity` is the fraction of the answer's
    citations that match a header actually supplied in the context — and
    `None` when the answer cites nothing, so that "didn't cite" (already
    measured by `citation_coverage`) is never scored as "fabricated a source".
    Refusal rows score neither."""
    if must_refuse:
        return None, None
    citations = _CITATION_RE.findall(answer)
    if not citations:
        return False, None
    valid = set(_CITATION_RE.findall(context))
    return True, sum(citation in valid for citation in citations) / len(citations)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation percentile over the sorted values (no numpy)."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate(results: list[dict[str, Any]]) -> dict[str, int | float | None]:
    """Roll per-question results into the scorecard summary. Each metric only
    averages over rows where it applies (None-filtered, and a missing key is
    treated as None) — a mixed eval set (some rows scoring retrieval, some
    refusal) never dilutes either metric."""

    def present(key: str) -> list[Any]:
        return [row.get(key) for row in results if row.get(key) is not None]

    def rate(values: list[Any]) -> float | None:
        return round(sum(values) / len(values), _RATE_PLACES) if values else None

    def median_seconds(values: list[Any]) -> float | None:
        return round(statistics.median(values), _SECONDS_PLACES) if values else None

    latency = cast(list[float], present("latency_s"))
    reciprocal_ranks = present("reciprocal_rank")
    return {
        "n": len(results),
        "retrieval_hit_rate": rate(present("retrieval_hit")),
        "answer_keyword_recall": rate(present("keyword_recall")),
        "refusal_accuracy": rate(present("refusal_ok")),
        "answerability_accuracy": rate(present("answerability_ok")),
        "citation_coverage": rate(present("citation_present")),
        "citation_validity": rate(present("citation_validity")),
        "exact_retrieval_hit_rate": rate(present("exact_retrieval_hit")),
        "mean_reciprocal_rank": (
            round(statistics.fmean(reciprocal_ranks), _RATE_PLACES) if reciprocal_ranks else None
        ),
        "median_latency_s": median_seconds(latency),
        "p95_latency_s": round(percentile(latency, 0.95), _SECONDS_PLACES) if latency else None,
        "median_retrieval_latency_s": median_seconds(present("retrieval_latency_s")),
        "median_generation_latency_s": median_seconds(present("generation_latency_s")),
        "median_vector_retrieval_latency_s": median_seconds(present("vector_retrieval_latency_s")),
        "median_rerank_latency_s": median_seconds(present("rerank_latency_s")),
    }
