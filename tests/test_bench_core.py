"""Unit tests for the dependency-free throughput math in services/bench_core.py
— no httpx, mlflow, or running Ollama involved."""
from services.bench_core import summarize_runs, tps


def test_tps_none_on_missing_count():
    assert tps(None, 1_000_000_000) is None


def test_tps_none_on_missing_duration():
    assert tps(10, None) is None


def test_tps_none_on_zero_duration():
    assert tps(10, 0) is None


def test_tps_basic_math():
    assert tps(100, 1_000_000_000) == 100.0


def test_summarize_runs_empty_samples():
    summary = summarize_runs([])
    assert summary["runs"] == 0
    assert summary["gen_tps_median"] is None
    assert summary["prompt_tps_median"] is None
    assert summary["load_ms_median"] is None


def test_summarize_runs_computes_medians():
    samples = [
        {
            "eval_count": 100,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 50,
            "prompt_eval_duration": 500_000_000,
            "load_duration": 200_000_000,
        },
        {
            "eval_count": 120,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 60,
            "prompt_eval_duration": 500_000_000,
            "load_duration": 300_000_000,
        },
    ]
    summary = summarize_runs(samples)
    assert summary["runs"] == 2
    assert summary["gen_tps_median"] == 110.0
    assert summary["prompt_tps_median"] == 110.0
    assert summary["load_ms_median"] == 250.0


def test_summarize_runs_skips_incomplete_samples():
    samples = [{"eval_count": None, "eval_duration": None}, {"eval_count": 50, "eval_duration": 1_000_000_000}]
    summary = summarize_runs(samples)
    assert summary["runs"] == 2
    assert summary["gen_tps_median"] == 50.0
