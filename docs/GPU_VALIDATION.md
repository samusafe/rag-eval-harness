# RTX 4060 GPU validation

This runbook must be executed on the actual GPU laptop. No values below are pre-filled
or inferred from a CPU-only development machine.

## Preconditions

1. Install the reviewed project dependencies in a fresh virtual environment.
2. Pull the exact chat and embedding models and set an immutable `CORPUS_REVISION`.
3. Populate `rag_documents` and confirm stable source/chunk IDs are present.
4. Save `nvidia-smi`, `ollama --version`, `ollama list`, and `pip freeze` output with the run.

## Baseline

```powershell
python eval/run_eval.py --preflight-only
python eval/run_eval.py --out eval/results/baseline --gate-hit-rate 0.8 --gate-recall 0.7 --gate-refusal 0.9
python eval/bench_ollama.py --model <exact-model-tag> --runs 5 --warmups 1 --out eval/results/bench-baseline.json
ollama ps
nvidia-smi
```

Confirm that the scorecard manifest contains the Ollama digest/quantization, actual
GPU identity, corpus revision, package versions, context settings, and reranker device.

## Controlled matrix

Change one variable at a time and use the same eval set/corpus:

| Candidate | Setting | Status | Quality gates | Median/p95 | tok/s | VRAM/RAM | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | CPU reranker, current Ollama defaults | NOT RUN | — | — | — | — | — |
| CUDA reranker | `RERANK_DEVICE=cuda` | NOT RUN | — | — | — | — | Watch combined Python + Ollama VRAM |
| Flash attention | `OLLAMA_FLASH_ATTENTION=1` | NOT RUN | — | — | — | — | Restart Ollama first |
| Quantized KV | Flash attention + `OLLAMA_KV_CACHE_TYPE=q8_0` | NOT RUN | — | — | — | — | Compare quality, not only speed |
| Smaller context | Lower `OLLAMA_NUM_CTX` with valid reserves | NOT RUN | — | — | — | — | Check dropped context stats |
| Q4/Q5 model | Exact candidate digest | NOT RUN | — | — | — | — | Reject CPU spill or failed gates |

## Acceptance criteria

- No CUDA OOM and no unexplained CPU spill.
- Candidate passes the same retrieval, recall, refusal, answerability, and citation gates.
- No blocking provenance mismatch when comparing baseline and candidate; use
  `--allow-incompatible` only for diagnosis, never for a promotion decision.
- Record median and p95, cold/warm behavior, prompt/generation throughput, model VRAM,
  process RSS, and any context truncation.
- Prefer the lowest-memory configuration within measurement noise when quality and
  latency are equivalent.

## Rollback

Restore `RERANK_DEVICE=cpu`, unset Ollama performance environment variables, restart
`ollama serve`, and rerun preflight plus baseline. Keep failed artifacts for diagnosis;
do not overwrite or present them as successful runs.
