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


# --------------------------------------------------------------------------
# --latest N must pick the N newest RUNS, not the N lexicographically-last
# file names (names start with the model name, so a name sort is model-first).
# --------------------------------------------------------------------------

import json  # noqa: E402

from eval.compare_runs import pick_latest  # noqa: E402


def _scorecard(path, model: str, timestamp_utc: str) -> None:
    payload = {
        "model": model,
        "summary": {},
        "results": [],
        "run_manifest": {"timestamp_utc": timestamp_utc},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pick_latest_orders_by_run_timestamp_across_models(tmp_path):
    _scorecard(tmp_path / "eval_zephyr_aaaaaaaa_20250101T000000_000000Z.json", "zephyr", "2025-01-01T00:00:00Z")
    _scorecard(tmp_path / "eval_llama3_bbbbbbbb_20260901T000000_000000Z.json", "llama3", "2026-09-01T00:00:00Z")
    _scorecard(tmp_path / "eval_llama3_bbbbbbbb_20260902T000000_000000Z.json", "llama3", "2026-09-02T00:00:00Z")
    _scorecard(tmp_path / "eval_mistral_cccccccc_20260903T000000_000000Z.json", "mistral", "2026-09-03T00:00:00Z")

    picked = [path.name for path in pick_latest(tmp_path, 2)]

    assert picked == [
        "eval_llama3_bbbbbbbb_20260902T000000_000000Z.json",
        "eval_mistral_cccccccc_20260903T000000_000000Z.json",
    ]


def test_pick_latest_falls_back_to_mtime_without_a_manifest(tmp_path):
    import os

    older = tmp_path / "eval_zzz_00000000_x.json"
    newer = tmp_path / "eval_aaa_00000000_x.json"
    for path in (older, newer):
        path.write_text(json.dumps({"summary": {}, "results": []}), encoding="utf-8")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    assert pick_latest(tmp_path, 2) == [older, newer]



# --------------------------------------------------------------------------
# Comparability lives in services/ (pure), and knows about v3 manifests
# --------------------------------------------------------------------------

from services.eval_compat import compatibility_issues as service_compatibility_issues  # noqa: E402


def test_compare_runs_uses_the_service_implementation():
    assert compatibility_issues is service_compatibility_issues


def test_permissive_run_is_blocking_but_unknown_strictness_is_not():
    blocking, _ = service_compatibility_issues(_run(eval_set_strict=True), _run(eval_set_strict=False))
    assert any("eval_set_strict" in issue for issue in blocking)
    blocking, _ = service_compatibility_issues(_run(), _run(eval_set_strict=True))
    assert blocking == []


def test_unversioned_corpus_is_advisory():
    _, advisory = service_compatibility_issues(
        _run(retrieval={"corpus_revision": "unversioned"}),
        _run(retrieval={"corpus_revision": "unversioned"}),
    )
    assert any("corpus_revision" in issue for issue in advisory)
