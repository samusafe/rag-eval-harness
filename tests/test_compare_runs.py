"""Scorecard compatibility checks are pure and require no live services."""

from eval.compare_runs import compatibility_issues


def _run(**manifest_overrides):
    manifest = {
        "eval_set_sha256": "set",
        "prompt_sha256": "prompt",
        "embedding_model": "embed",
        "reranker_model": "reranker",
        "retrieval": {"top_k": 20},
        "generation": {"num_ctx": 4096},
        "runtime": {
            "ollama_chat_model": {"digest": "abc", "quantization_level": "Q4_K_M"},
            "packages": {"langchain": "0.3.1"},
            "gpus": ["GPU, 8192"],
        },
    }
    manifest.update(manifest_overrides)
    return {"run_manifest": manifest}


def test_compatible_manifests_have_no_issues():
    assert compatibility_issues(_run(), _run()) == ([], [])


def test_quality_input_difference_is_blocking():
    blocking, advisory = compatibility_issues(_run(), _run(eval_set_sha256="other"))
    assert any("eval_set_sha256" in issue for issue in blocking)
    assert advisory == []


def test_hardware_difference_is_advisory():
    candidate = _run()
    candidate["run_manifest"]["runtime"]["gpus"] = ["Other GPU, 4096"]
    blocking, advisory = compatibility_issues(_run(), candidate)
    assert blocking == []
    assert any("runtime.gpus" in issue for issue in advisory)


def test_missing_manifest_blocks_comparison():
    blocking, _ = compatibility_issues({}, _run())
    assert blocking
