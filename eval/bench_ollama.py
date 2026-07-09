"""
Ollama throughput benchmark -> MLflow (efficiency tracking on your own GPU).

Sends a fixed prompt to the host Ollama N times and reads Ollama's own timing
metrics to compute generation + prompt tokens/sec — NO llama.cpp build
required, it measures the engine you already run. Use it to A/B efficiency
knobs:

  # baseline
  python eval/bench_ollama.py --model my-finetuned-model --runs 5 --mlflow
  # then enable host flash-attn + KV-cache q8 and re-run to compare in MLflow:
  #   (Linux/macOS) export OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
  #   (Windows)     setx OLLAMA_FLASH_ATTENTION 1 & setx OLLAMA_KV_CACHE_TYPE q8_0
  #   restart `ollama serve`, then run the same command again.

Runs log to the MLFLOW_BENCH_EXPERIMENT_NAME experiment (separate from eval).
The throughput math is the pure, unit-tested services/bench_core; this file is
just the I/O runner.
"""
from __future__ import annotations

import argparse
import os
import sys

import httpx

# Allow `python eval/bench_ollama.py` from the repo root to import services.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings  # noqa: E402
from services.bench_core import summarize_runs, tps  # noqa: E402

DEFAULT_PROMPT = (
    "Summarize, in three sentences, the key responsibilities of a data "
    "protection officer under GDPR."
)


def _bench_once(client, base_url, model, prompt, num_ctx, num_predict) -> dict:
    resp = client.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": num_predict},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Ollama throughput.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--num-ctx", type=int, default=settings.OLLAMA_NUM_CTX)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    # The bench runs on the HOST, so Ollama is at localhost by default —
    # override with --base-url or OLLAMA_BENCH_URL if yours lives elsewhere.
    parser.add_argument("--base-url", default=settings.OLLAMA_BENCH_URL)
    parser.add_argument("--mlflow", action="store_true", help="Log summary to MLflow.")
    args = parser.parse_args()

    samples: list[dict] = []
    with httpx.Client() as client:
        print(f"Warming up {args.model} (excluded from stats)...")
        _bench_once(client, args.base_url, args.model, args.prompt, args.num_ctx, args.num_predict)
        for i in range(args.runs):
            s = _bench_once(client, args.base_url, args.model, args.prompt, args.num_ctx, args.num_predict)
            samples.append(s)
            print(f"  run {i + 1}/{args.runs}: gen {tps(s.get('eval_count'), s.get('eval_duration'))} tok/s")

    summary = summarize_runs(samples)
    print("\nSummary:", summary)

    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_BENCH_EXPERIMENT_NAME)
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "model": args.model,
                    "num_ctx": args.num_ctx,
                    "num_predict": args.num_predict,
                    "runs": args.runs,
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in summary.items() if isinstance(v, (int, float)) and v is not None}
            )
        print(f"Logged to MLflow experiment '{settings.MLFLOW_BENCH_EXPERIMENT_NAME}'.")


if __name__ == "__main__":
    main()
