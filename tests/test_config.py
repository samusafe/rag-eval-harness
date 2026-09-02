"""Config/env parsing must load with sane defaults and no .env present, and
must honor real env var overrides. No network/DB/Ollama involved.

Every test builds its own `Settings(_env_file=None, ...)` instead of reloading
`config`: a reload would replace `config.settings` while every service module
still holds the original object (pinned by tests/test_module_layout.py)."""
import os

import pytest

import config as config_module

ENV_PREFIXES = (
    "OLLAMA_",
    "DATABASE_",
    "MLFLOW_",
    "RETRIEVAL_",
    "RERANK_",
    "CORPUS_",
    "RESULT_",
    "PGVECTOR_",
    "RAG_",
    "CHARS_",
    "HOST_",
    "COMPARE_",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in list(os.environ):
        if var.startswith(ENV_PREFIXES):
            monkeypatch.delenv(var, raising=False)


def test_settings_load_with_no_env_vars():
    s = config_module.Settings(_env_file=None)

    assert s.OLLAMA_BASE_URL == "http://localhost:11434"
    assert s.OLLAMA_CHAT_MODEL == "my-finetuned-model"
    assert s.OLLAMA_EMBED_MODEL == "nomic-embed-text"
    assert s.MLFLOW_TRACKING_URI == "http://localhost:5000"
    assert s.RETRIEVAL_TOP_K == 20
    assert s.RERANK_TOP_N == 5


def test_database_url_is_well_formed():
    s = config_module.Settings(_env_file=None)

    assert s.database_url == (
        f"postgresql+psycopg://{s.DATABASE_USER}:{s.DATABASE_PASSWORD}"
        f"@{s.DATABASE_HOST}:{s.DATABASE_PORT}/{s.DATABASE_NAME}"
        f"?sslmode={s.DATABASE_SSLMODE}&connect_timeout={s.DATABASE_CONNECT_TIMEOUT}"
    )
    assert s.database_url.startswith("postgresql+psycopg://")


def test_database_url_escapes_credentials():
    s = config_module.Settings(_env_file=None, DATABASE_USER="user@example", DATABASE_PASSWORD="p@ss:/word%")
    assert "user%40example:p%40ss%3A%2Fword%25@" in s.database_url


def test_settings_honor_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "custom-eval-model")
    assert config_module.Settings(_env_file=None).OLLAMA_CHAT_MODEL == "custom-eval-model"


def test_remote_database_rejects_development_password():
    with pytest.raises(ValueError, match="default database password"):
        config_module.Settings(_env_file=None, DATABASE_HOST="db.example.com")


# --------------------------------------------------------------------------
# Hardening: every knob lives here, and remote endpoints are opt-in
# --------------------------------------------------------------------------


def _settings(**overrides):
    return config_module.Settings(_env_file=None, **overrides)


def test_new_knobs_have_localhost_defaults():
    s = _settings()
    assert s.PGVECTOR_COLLECTION_NAME == "rag_documents"
    assert s.CHARS_PER_TOKEN_ESTIMATE == 4
    assert s.HOST_PROBE_TIMEOUT == 2.0
    assert s.DATABASE_STATEMENT_TIMEOUT_MS == 30_000
    assert s.COMPARE_LATENCY_NOISE_S == 0.5
    assert s.MLFLOW_ALLOW_REMOTE is False
    assert s.OLLAMA_FLASH_ATTENTION is None
    assert s.OLLAMA_KV_CACHE_TYPE is None
    assert s.OLLAMA_NUM_PARALLEL is None


def test_blank_reranker_revision_means_unpinned_not_empty_string():
    assert _settings(RERANK_MODEL_REVISION="").RERANK_MODEL_REVISION is None
    assert _settings(RERANK_MODEL_REVISION=" abc ").RERANK_MODEL_REVISION == "abc"


def test_rerank_top_n_cannot_exceed_top_k():
    with pytest.raises(ValueError, match="cannot exceed"):
        _settings(RETRIEVAL_TOP_K=5, RERANK_TOP_N=6)


def test_context_reserves_must_leave_room():
    with pytest.raises(ValueError, match="leave room"):
        _settings(OLLAMA_NUM_CTX=1024, OLLAMA_NUM_PREDICT=512, RAG_CONTEXT_RESERVE_TOKENS=512)


@pytest.mark.parametrize("sslmode", ["disable", "allow", "prefer"])
def test_remote_database_rejects_downgradeable_sslmode(sslmode):
    with pytest.raises(ValueError, match="TLS"):
        _settings(DATABASE_HOST="db.example.com", DATABASE_PASSWORD="x", DATABASE_SSLMODE=sslmode)


def test_remote_database_accepts_enforced_tls():
    s = _settings(DATABASE_HOST="db.example.com", DATABASE_PASSWORD="x", DATABASE_SSLMODE="require")
    assert s.DATABASE_SSLMODE == "require"


@pytest.mark.parametrize(
    "uri",
    [
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://[::1]:5000",
        "file:./mlruns",
        "sqlite:///m.db",
        "./mlruns",
    ],
)
def test_local_mlflow_uris_need_no_opt_in(uri):
    assert uri == _settings(MLFLOW_TRACKING_URI=uri).MLFLOW_TRACKING_URI


def test_remote_mlflow_uri_requires_explicit_opt_in_and_https():
    with pytest.raises(ValueError, match="MLFLOW_ALLOW_REMOTE"):
        _settings(MLFLOW_TRACKING_URI="https://mlflow.example.com")
    with pytest.raises(ValueError, match="https"):
        _settings(MLFLOW_TRACKING_URI="http://mlflow.example.com", MLFLOW_ALLOW_REMOTE=True)
    ok = _settings(MLFLOW_TRACKING_URI="https://mlflow.example.com", MLFLOW_ALLOW_REMOTE=True)
    assert ok.MLFLOW_TRACKING_URI == "https://mlflow.example.com"
