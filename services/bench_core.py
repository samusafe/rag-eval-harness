"""
Pure throughput math for the Ollama benchmark (efficiency tracking).

Dependency-free (statistics only) so it unit-tests without httpx, mlflow, or a
running Ollama. The I/O runner that sends prompts + logs to MLflow lives in
eval/bench_ollama.py and imports these. Ollama returns nanosecond timings on a
non-streamed /api/chat response (eval_count/eval_duration, prompt_eval_*,
load_duration); this module turns them into tokens/sec.
"""
from __future__ import annotations

import statistics
from typing import Optional

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000


def tps(count: Optional[int], duration_ns: Optional[int]) -> Optional[float]:
    """Tokens/sec from a token count + a nanosecond duration; None if either
    is missing/zero (a benchmark must report "no data", never divide-by-zero)."""
    if not count or not duration_ns:
        return None
    return round(count / (duration_ns / NS_PER_S), 2)


def summarize_runs(samples: list[dict]) -> dict:
    """
    Aggregate raw Ollama timing dicts into median throughput metrics. Defensive:
    a sample missing a field simply doesn't contribute to that metric (rather
    than raising), and medians are used so one cold-start outlier can't skew
    the result.
    """
    gen: list[float] = []
    prompt: list[float] = []
    load_ms: list[float] = []
    for s in samples:
        g = tps(s.get("eval_count"), s.get("eval_duration"))
        if g is not None:
            gen.append(g)
        p = tps(s.get("prompt_eval_count"), s.get("prompt_eval_duration"))
        if p is not None:
            prompt.append(p)
        ld = s.get("load_duration")
        if ld:
            load_ms.append(round(ld / NS_PER_MS, 1))
    return {
        "runs": len(samples),
        "gen_tps_median": statistics.median(gen) if gen else None,
        "prompt_tps_median": statistics.median(prompt) if prompt else None,
        "load_ms_median": statistics.median(load_ms) if load_ms else None,
    }
