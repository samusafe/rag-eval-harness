"""Config/env parsing must load with sane defaults and no .env present, and
must honor real env var overrides. No network/DB/Ollama involved."""
import importlib

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
)


def _clear_relevant_env(monkeypatch):
    import os

    for var in list(os.environ):
        if var.startswith(ENV_PREFIXES):
            monkeypatch.delenv(var, raising=False)


def test_settings_load_with_no_env_vars(monkeypatch):
    _clear_relevant_env(monkeypatch)
    importlib.reload(config_module)
    s = config_module.settings

    assert s.OLLAMA_BASE_URL == "http://localhost:11434"
    assert s.OLLAMA_CHAT_MODEL == "my-finetuned-model"
    assert s.OLLAMA_EMBED_MODEL == "nomic-embed-text"
    assert s.MLFLOW_TRACKING_URI == "http://localhost:5000"
    assert s.RETRIEVAL_TOP_K == 20
    assert s.RERANK_TOP_N == 5


def test_database_url_is_well_formed(monkeypatch):
    _clear_relevant_env(monkeypatch)
    importlib.reload(config_module)
    s = config_module.settings

    assert s.database_url == (
        f"postgresql+psycopg://{s.DATABASE_USER}:{s.DATABASE_PASSWORD}"
        f"@{s.DATABASE_HOST}:{s.DATABASE_PORT}/{s.DATABASE_NAME}"
        f"?sslmode={s.DATABASE_SSLMODE}&connect_timeout={s.DATABASE_CONNECT_TIMEOUT}"
    )
    assert s.database_url.startswith("postgresql+psycopg://")


def test_database_url_escapes_credentials(monkeypatch):
    _clear_relevant_env(monkeypatch)
    monkeypatch.setenv("DATABASE_USER", "user@example")
    monkeypatch.setenv("DATABASE_PASSWORD", "p@ss:/word%")
    importlib.reload(config_module)
    try:
        assert "user%40example:p%40ss%3A%2Fword%25@" in config_module.settings.database_url
    finally:
        _clear_relevant_env(monkeypatch)
        importlib.reload(config_module)


def test_settings_honor_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "custom-eval-model")
    importlib.reload(config_module)
    try:
        assert config_module.settings.OLLAMA_CHAT_MODEL == "custom-eval-model"
    finally:
        monkeypatch.delenv("OLLAMA_CHAT_MODEL", raising=False)
        importlib.reload(config_module)


def test_remote_database_rejects_development_password():
    with pytest.raises(ValueError, match="default database password"):
        config_module.Settings(_env_file=None, DATABASE_HOST="db.example.com")
