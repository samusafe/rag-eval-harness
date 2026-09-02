"""One file, one responsibility -- and the pure eval core must not drag the ML
stack (torch / sentence-transformers) into every scorer test or into the
throughput benchmark. These tests pin the module boundaries."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

LIGHT_MODULES = [
    "services.rag_prompt",
    "services.eval_set",
    "services.eval_metrics",
    "services.eval_manifest",
    "services.eval_service",
    "services.eval_compat",
    "services.gates",
    "eval.compare_runs",
    "eval.bench_ollama",
]


@pytest.mark.parametrize("module_name", LIGHT_MODULES)
def test_module_does_not_import_the_ml_stack(module_name):
    # A fresh interpreter, so modules already imported by other tests can't mask a leak.
    code = (
        f"import importlib, sys; importlib.import_module({module_name!r}); "
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in {'torch', 'sentence_transformers', 'transformers'}); "
        "print(heavy)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]", f"{module_name} imported {completed.stdout.strip()}"


def test_prompt_template_embeds_the_exact_refusal_sentence():
    from services.rag_prompt import RAG_PROMPT_TEMPLATE, REFUSAL

    assert f'EXACTLY: "{REFUSAL}"' in RAG_PROMPT_TEMPLATE


def test_refusal_constant_is_re_exported_where_callers_expect_it():
    from services.eval_metrics import REFUSAL as metrics_refusal
    from services.eval_service import REFUSAL as service_refusal
    from services.rag_prompt import REFUSAL

    assert metrics_refusal is REFUSAL
    assert service_refusal is REFUSAL


def test_services_share_the_config_singleton():
    import config
    from services import eval_manifest, eval_service, rag_chain, rag_prompt

    assert eval_service.settings is config.settings
    assert eval_manifest.settings is config.settings
    assert rag_chain.settings is config.settings
    assert rag_prompt.settings is config.settings
