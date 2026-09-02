"""Every module must import cleanly with no Ollama/Postgres/MLflow running —
this is the CI safety net for a repo whose real pipeline needs live services.
Heavy singletons (vector store, reranker, LLM client) are built lazily inside
functions, never at import time, specifically so this test can pass."""
import importlib

import pytest

MODULES = [
    "config",
    "services.bench_core",
    "services.vector_store",
    "services.rag_prompt",
    "services.rag_chain",
    "services.eval_set",
    "services.eval_metrics",
    "services.eval_manifest",
    "services.eval_compat",
    "services.eval_service",
    "services.gates",
    "services.ollama_info",
    "eval.run_eval",
    "eval.compare_runs",
    "eval.bench_ollama",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_cleanly(module_name):
    importlib.import_module(module_name)
