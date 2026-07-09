# RAG Eval Harness

![CI](https://github.com/samusafe/rag-eval-harness/actions/workflows/ci.yml/badge.svg)

A small, honest evaluation harness for a local Retrieval-Augmented Generation
(RAG) pipeline. It runs a fixed question set through the **real** chain —
pgvector similarity search, a cross-encoder reranker, prompt assembly, and a
local Ollama chat model — and scores the answers on the axes that actually
decide whether a RAG system is any good. Then it lets you diff two runs to
see whether a change (new model, LoRA adapter, prompt, chunking, embedder)
made things better or worse, instead of guessing from a handful of manual
spot-checks.

## Why evals matter

RAG has a lot of knobs that all sound plausible in isolation: swap the
generator, tweak the prompt, change chunk size, fine-tune an adapter, add a
reranker. Every one of them can silently make retrieval worse, recall worse,
or hallucination more likely — while looking fine on the two questions you
happened to try by hand.

**The rule this repo encodes: never ship a RAG change unmeasured.** Run the
eval before the change, run it after, and diff the scorecards
(`compare_runs.py`). If a change can't show its work in a scorecard, it
doesn't ship.

## Architecture

```
                     ┌─────────────────────┐
  eval_set.jsonl ──► │  eval/run_eval.py    │  (CLI: argparse, rich scorecard)
                     └──────────┬──────────┘
                                │ drives
                                ▼
                     ┌─────────────────────┐
                     │ services/eval_service│  scoring + persistence core
                     │  - load_eval_set      │  (dependency-light, unit-tested)
                     │  - retrieval_hit      │
                     │  - keyword_recall     │
                     │  - refusal_ok         │
                     │  - aggregate          │
                     │  - write_results      │
                     │  - log_mlflow ────────┼──► MLflow (:5000)
                     └──────────┬──────────┘
                                │ per question
                                ▼
                     ┌─────────────────────┐
                     │ services/rag_chain   │  the actual RAG chain (LCEL)
                     └──────────┬──────────┘
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
  services/vector_store   cross-encoder        ChatOllama
   (pgvector similarity)   reranker (top-N)    (local LLM, :11434)
   OllamaEmbeddings

  eval/results/*.json ──► eval/compare_runs.py ──► summary + per-question
                                                     regressions (rich table)

  eval/bench_ollama.py ──► services/bench_core (pure tok/s math) ──► MLflow
   (raw Ollama throughput, separate from RAG quality)
```

Everything is local: Ollama for both the chat model and the embeddings, no
cloud LLM provider anywhere in this repo.

## Quickstart

Prerequisites: Python 3.11+, a reachable [Ollama](https://ollama.com) with a
chat model and an embedding model pulled, and a Postgres instance with the
`pgvector` extension and a `rag_documents` collection already populated by
*some* ingestion pipeline (see "What this repo does NOT include" below).

```bash
# 1. Install
python -m venv .venv
. .venv/Scripts/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure (every value has a localhost default — only override what you need)
cp .env.example .env

# 3. Run the eval
python eval/run_eval.py
# or, for a specific model / with MLflow logging:
python eval/run_eval.py --model my-finetuned-model --mlflow

# 4. Compare against a previous run
python eval/compare_runs.py --latest 2

# 5. (optional) Benchmark raw Ollama throughput
python eval/bench_ollama.py --model my-finetuned-model --runs 5 --mlflow
```

Shell-agnostic wrappers ship for both platforms:
`scripts/run_eval.ps1` / `scripts/run_eval.sh` and
`scripts/compare.ps1` / `scripts/compare.sh`.

### What this repo does NOT include

This is the **evaluation harness**, not a full RAG service. There is no
document ingestion pipeline here (chunking, OCR, upload API) — you bring your
own already-populated `pgvector` collection, or point `services/vector_store.py`
at whatever vector store your own RAG stack already uses. The eval set
(`eval/eval_set.example.jsonl`) ships with synthetic questions about a
fictional company handbook so the schema is obvious; swap in your own
questions and source-document substrings once you've ingested real docs.

## Eval set schema (`eval/eval_set.example.jsonl`, one JSON object per line)

| Field | Meaning |
|---|---|
| `id` | short, unique label |
| `question` | the user question |
| `expected_sources` | substrings expected in a retrieved chunk's `source_file` metadata (retrieval hit). Empty list = not scored |
| `expected_keywords` | strings the answer should contain (keyword recall). Empty list = not scored |
| `must_refuse` | `true` for out-of-KB questions — the answer must be the exact refusal sentence |

Lines starting with `//` are treated as comments and skipped by the loader.
Include several `must_refuse: true` rows — those are what catch hallucination,
the highest-risk RAG failure mode.

## Metrics explained

Computed in `services/eval_service.py`, printed by `eval/run_eval.py`:

- **Retrieval hit-rate** (target ≥ 0.80) — did a retrieved chunk's
  `source_file` contain one of the row's `expected_sources` substrings, after
  reranking? Bad retrieval caps everything downstream — the generator can't
  answer from a document it never saw.
- **Answer keyword recall** (target ≥ 0.70) — fraction of `expected_keywords`
  present in the generated answer (case-insensitive substring match). A cheap
  but effective proxy for "did the answer contain the actual facts."
- **Refusal accuracy** (target ≥ 0.90) — for `must_refuse` rows, did the model
  produce the exact configured refusal sentence instead of inventing an
  answer? This is the anti-hallucination signal.
- **Median latency** — wall-clock seconds per question (retrieval + rerank +
  generation). Your speed baseline; watch it whenever you swap models or
  retrieval knobs.

Each metric only averages over the rows where it applies — a `must_refuse`
row doesn't get scored on retrieval/keywords, and an in-KB row doesn't get
scored on refusal accuracy — so a mixed eval set never dilutes either number.

## Comparing runs

Each run writes `eval/results/eval_<model>_<timestamp>.json`. Use
`compare_runs.py` to diff them:

```bash
python eval/compare_runs.py --latest 2                                # 2 newest runs
python eval/compare_runs.py eval/results/<old>.json eval/results/<new>.json
python eval/compare_runs.py --latest 3                                # summary across 3 runs
```

For exactly two runs it also prints a per-question breakdown, **regressions
first** — that's where the debugging value is. If the two runs used different
embedders, it warns that retrieval-hit isn't comparable between them (only
keyword recall / refusal accuracy / latency are, since those move with the
generator, not the embedder).

## MLflow

`--mlflow` on `run_eval.py` and `bench_ollama.py` logs params + metrics +
the scorecard JSON as an artifact to a local MLflow tracking server
(`MLFLOW_TRACKING_URI`, default `http://localhost:5000`), under the
`MLFLOW_EXPERIMENT_NAME` / `MLFLOW_BENCH_EXPERIMENT_NAME` experiments. This
is what lets several models/adapters/prompt versions sit side-by-side across
runs, not just the two most recent.

<!-- MLflow screenshots: drop PNGs of the experiment view / run comparison
     here once you have real eval runs against your own model + docs. -->

## Project layout

```
config.py                   # single source of truth for every env-driven setting
services/
  vector_store.py           # pgvector connection + Ollama embeddings (singletons)
  rag_chain.py               # retriever + reranker + prompt + LLM (LCEL)
  eval_service.py             # scoring, aggregation, persistence, MLflow logging
  bench_core.py                # pure tok/s math (no I/O — easy to unit-test)
eval/
  run_eval.py                # CLI: run the eval, print a rich scorecard
  compare_runs.py             # CLI: diff two or more scorecards
  bench_ollama.py             # CLI: raw Ollama throughput benchmark
  eval_set.example.jsonl      # synthetic example question set
  results/                    # eval run JSON output (gitignored)
scripts/                     # .ps1 / .sh twins for the above
tests/                       # schema/config/import/pure-function tests (no live services)
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
python -m pytest
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest on every push/PR against a
plain `ubuntu-latest` runner with **no Ollama, Postgres, or MLflow present** —
the test suite is scoped to exactly what that allows (schema validation,
config defaults, clean imports, pure scoring/bench math). Running the harness
against a live model is a manual step against your own local stack.

## License

MIT — see [LICENSE](LICENSE).
