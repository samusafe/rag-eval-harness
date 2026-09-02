"""
Scorecard comparability — can two runs' deltas be read as causal?

Pure dict logic, no rich/argparse, so `eval/compare_runs.py` (and any other
consumer) imports it from `services/` rather than the other way round.
"""
from __future__ import annotations

from typing import Any

# Provenance that defines what quality was measured. A difference here means
# the deltas are not attributable to the change under test.
BLOCKING_PATHS: tuple[tuple[str, ...], ...] = (
    ("eval_set_sha256",),
    ("prompt_sha256",),
    ("embedding_model",),
    ("reranker_model",),
    ("reranker", "revision"),
    ("retrieval",),
    ("runtime", "ollama_embedding_model", "digest"),
)

# Provenance worth a warning: it can move latency or generation, not what was
# retrieved or scored.
ADVISORY_PATHS: tuple[tuple[str, ...], ...] = (
    ("generation",),
    ("runtime", "ollama_chat_model", "digest"),
    ("runtime", "ollama_chat_model", "quantization_level"),
    ("runtime", "packages"),
    ("runtime", "gpus"),
)

UNVERSIONED_CORPUS = "unversioned"


def _value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: object = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def compatibility_issues(base: dict[str, Any], candidate: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (blocking, advisory) provenance differences between two scorecards."""
    left = base.get("run_manifest")
    right = candidate.get("run_manifest")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return ["one or both scorecards have no run_manifest"], []

    def differences(paths: tuple[tuple[str, ...], ...]) -> list[str]:
        issues = []
        for path in paths:
            left_value, right_value = _value(left, path), _value(right, path)
            if left_value != right_value:
                issues.append(f"{'.'.join(path)} differs: {left_value!r} vs {right_value!r}")
        return issues

    blocking = differences(BLOCKING_PATHS)
    advisory = differences(ADVISORY_PATHS)

    # A permissive run silently dropped rows, so its rates are not comparable
    # against a strict one. Pre-v3 manifests do not record the flag (None):
    # unknown is not a mismatch, only an explicit `false` is.
    strictness = (_value(left, ("eval_set_strict",)), _value(right, ("eval_set_strict",)))
    if False in strictness:
        blocking.append(
            f"eval_set_strict: {strictness[0]!r} vs {strictness[1]!r} (a permissive run is not comparable)"
        )

    # Two runs both at the default corpus revision compare as "equal" while
    # neither can prove the database contents matched — say so.
    if UNVERSIONED_CORPUS in (
        _value(left, ("retrieval", "corpus_revision")),
        _value(right, ("retrieval", "corpus_revision")),
    ):
        advisory.append(
            "retrieval.corpus_revision is 'unversioned' on at least one run; "
            "identical settings cannot prove the corpus was unchanged"
        )
    return blocking, advisory
