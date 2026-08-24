"""The run manifest is what makes a scorecard citable evidence rather than a
number in a file. These tests pin its shape and its determinism by injecting
the clock and the git lookup — no repo, no network, no live services."""
import hashlib
import platform
from datetime import UTC, datetime, timedelta, timezone

from services import eval_service
from services.eval_service import build_run_manifest
from services.rag_chain import RAG_PROMPT_TEMPLATE

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
FIXED_GIT = {"commit_sha": "0123456789abcdef0123456789abcdef01234567", "dirty": False}


def _manifest(**kwargs):
    kwargs.setdefault("timestamp", FIXED_TIME)
    kwargs.setdefault("git_metadata", FIXED_GIT)
    kwargs.setdefault("runtime_metadata", {"test": True})
    return build_run_manifest(kwargs.pop("eval_set_path", None), kwargs.pop("model", "m"), **kwargs)


def test_manifest_is_deterministic_for_fixed_inputs():
    assert _manifest() == _manifest()


def test_manifest_records_provenance_fields():
    manifest = _manifest(model="my-finetuned-model-v2")
    assert manifest["schema_version"] == 2
    assert manifest["timestamp_utc"] == "2026-01-02T03:04:05Z"
    assert manifest["git"] == FIXED_GIT
    assert manifest["chat_model"] == "my-finetuned-model-v2"
    assert manifest["embedding_model"] == eval_service.settings.OLLAMA_EMBED_MODEL
    assert manifest["reranker_model"] == eval_service.settings.RERANK_MODEL
    assert manifest["retrieval"] == {
        "top_k": eval_service.settings.RETRIEVAL_TOP_K,
        "rerank_top_n": eval_service.settings.RERANK_TOP_N,
        "distance_strategy": eval_service.settings.PGVECTOR_DISTANCE_STRATEGY,
        "collection_name": "rag_documents",
        "collection_id_filter": None,
        "corpus_revision": eval_service.settings.CORPUS_REVISION,
    }
    assert manifest["python_version"] == platform.python_version()
    assert manifest["runtime"] == {"test": True}


def test_manifest_hashes_the_prompt_actually_used():
    expected = hashlib.sha256(RAG_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    assert _manifest()["prompt_sha256"] == expected


def test_manifest_hashes_the_eval_set_bytes(tmp_path):
    eval_set = tmp_path / "eval_set.jsonl"
    eval_set.write_bytes(b"{}\n")
    assert _manifest(eval_set_path=eval_set)["eval_set_sha256"] == hashlib.sha256(b"{}\n").hexdigest()


def test_manifest_eval_set_hash_is_null_when_no_path_given():
    assert _manifest()["eval_set_sha256"] is None


def test_config_hash_changes_when_a_retrieval_knob_changes(monkeypatch):
    before = _manifest()["config_sha256"]
    monkeypatch.setattr(eval_service.settings, "RERANK_TOP_N", 99)
    assert _manifest()["config_sha256"] != before


def test_config_hash_changes_when_the_model_changes():
    assert _manifest(model="a")["config_sha256"] != _manifest(model="b")["config_sha256"]


def test_naive_timestamp_is_treated_as_utc():
    manifest = _manifest(timestamp=datetime(2026, 1, 2, 3, 4, 5))
    assert manifest["timestamp_utc"] == "2026-01-02T03:04:05Z"


def test_aware_timestamp_is_converted_to_utc():
    lisbon_summer = timezone(timedelta(hours=2))
    manifest = _manifest(timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=lisbon_summer))
    assert manifest["timestamp_utc"] == "2026-01-02T01:04:05Z"


def test_missing_git_metadata_is_explicit_null_not_a_guess():
    manifest = _manifest(git_metadata={"commit_sha": None, "dirty": None})
    assert manifest["git"] == {"commit_sha": None, "dirty": None}


def test_git_lookup_never_raises_and_always_answers_both_keys():
    # Runs the real lookup: whatever the environment (no git, no repo, a hang),
    # it must degrade to None rather than break the eval run.
    git = eval_service._git_metadata()
    assert set(git) == {"commit_sha", "dirty"}
    assert git["commit_sha"] is None or isinstance(git["commit_sha"], str)
    assert git["dirty"] is None or isinstance(git["dirty"], bool)
