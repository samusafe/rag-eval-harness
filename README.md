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
| `expected_source_ids` | *optional, reserved* — exact chunk/document IDs for a future exact-match retrieval metric. Accepted and stored today, not scored yet |

Lines starting with `//` are treated as comments and skipped by the loader.
Include several `must_refuse: true` rows — those are what catch hallucination,
the highest-risk RAG failure mode.

### Strict by default

An eval set is a measuring instrument, so the loader is **fail-closed**: a bad
row aborts the run instead of silently shrinking the question set (which would
move every rate in the scorecard while the run still looked green). A row is
rejected — naming file, line number and cause — when it:

- isn't valid JSON,
- is missing a required field, or has the wrong type (no coercion: `"true"` is
  not `true`, `1` is not `"1"`),
- has a blank `id` or `question`,
- carries an unknown key (a typo like `expected_keyword` would otherwise score
  nothing, silently),
- reuses an `id` already seen in the file.

```bash
python eval/run_eval.py --permissive-eval-set   # log + skip bad rows instead
```

`--permissive-eval-set` exists only to investigate a broken file. It still
raises if nothing valid is left, and its numbers are not comparable against a
strict run.

## Metrics explained

Computed in `services/eval_service.py`, printed by `eval/run_eval.py`:

- **Retrieval hit-rate** (target ≥ 0.80) — did a retrieved chunk's
  `source_file` contain one of the row's `expected_sources` substrings, after
  reranking? Bad retrieval caps everything downstream — the generator can't
  answer from a document it never saw.
- **Answer keyword recall** (target ≥ 0.70) — fraction of `expected_keywords`
  present in the generated answer (case-insensitive substring match). A cheap
  regression signal on answer content.
- **Refusal accuracy** (target ≥ 0.90) — for `must_refuse` rows, is the
  **whole answer** the configured refusal sentence? This is the
  anti-hallucination signal.
- **Median latency** — wall-clock seconds per question (retrieval + rerank +
  generation). Your speed baseline; watch it whenever you swap models or
  retrieval knobs.

Each metric only averages over the rows where it applies — a `must_refuse`
row doesn't get scored on retrieval/keywords, and an in-KB row doesn't get
scored on refusal accuracy — so a mixed eval set never dilutes either number.

### What these metrics are not

Being explicit about this matters more than the numbers themselves:

- **Keyword recall is lexical, not factual.** It checks that strings appear,
  not that the answer is true. *"The policy is **not** 20 days"* scores a full
  1.0 against `["20 days"]`. Read it as "did the answer keep talking about the
  right things", never as a correctness or hallucination score.
- **Retrieval hit-rate is substring matching over file names**, not exact
  chunk identity. It tolerates the path/extension noise real ingestion
  pipelines produce, at the cost of counting a coincidental name match as a
  hit. `expected_source_ids` is the reserved slot for the exact-match version.
- **Refusal accuracy is exact, after whitespace/case normalization only.** The
  entire answer must equal the refusal sentence; trailing newlines, doubled
  spaces and capitalization are forgiven, added words are not. A substring
  test would score *"The capital is Canberra. I don't have enough information
  in the knowledge base to answer this question."* as a correct refusal — i.e.
  it would grade a hallucination as anti-hallucination.
- **There is no LLM-as-judge here, on purpose.** Every metric is
  deterministic, so re-running the same set against the same model moves the
  numbers only when the system changed, not when the judge felt different.

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

## Run provenance (`run_manifest`)

A score is only evidence if you can say what produced it. Every scorecard
carries a `run_manifest` block answering exactly that:

```json
{
  "run_manifest": {
    "schema_version": 1,
    "timestamp_utc": "2026-01-02T03:04:05Z",
    "git": { "commit_sha": "0123456789abcdef0123456789abcdef01234567", "dirty": false },
    "eval_set_sha256": "b9a2...",
    "chat_model": "my-finetuned-model-v2",
    "embedding_model": "nomic-embed-text",
    "reranker_model": "BAAI/bge-reranker-base",
    "retrieval": { "top_k": 20, "rerank_top_n": 5 },
    "python_version": "3.11.9",
    "prompt_sha256": "7c31...",
    "config_sha256": "e0f4..."
  }
}
```

Why each field earns its place:

- `git.commit_sha` / `git.dirty` — which code ran, and whether it was a clean
  checkout. **`dirty: true` means the scorecard is not reproducible.**
- `eval_set_sha256` — the exact question-set bytes. Two runs are only
  comparable when this matches.
- `prompt_sha256` / `config_sha256` — catches the silent killers: an edited
  prompt or a changed `RETRIEVAL_TOP_K` that would otherwise look like the
  model got better.
- model names, `retrieval`, `python_version`, `timestamp_utc` — the rest of
  the environment, in one place.

Git metadata is **best-effort and never fatal**: running from a tarball, a
container, or a machine without git yields explicit `null`s (`dirty: null`
means *unknown*, not *clean*) rather than an error or a fabricated value.

The block is purely additive — every pre-existing scorecard key is unchanged,
so older result files still diff against newer ones in `compare_runs.py`.

## Pointing the harness at another RAG (HTTP target)

Nothing about the scoring core is Ollama-specific. `_eval_one` only needs two
things: a retriever with `.invoke(question) -> [docs with .metadata["source_file"]]`
and a chain with `.invoke({...}) -> str`. Adapt any RAG service behind an HTTP
API by adding one module (keep the HTTP client in `services/`, per the
dependencies-point-inward rule — never inline in a CLI script):

```python
# services/http_rag_target.py
"""Score somebody else's RAG service with this harness. One round-trip per
question, split into the retriever/chain pair `_eval_one` drives."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from config import settings


@dataclass
class _Chunk:
    page_content: str
    metadata: dict


class HttpRagTarget:
    def __init__(self, base_url: str) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=settings.OLLAMA_REQUEST_TIMEOUT)
        self._last: dict = {}

    def invoke(self, question: str) -> list[_Chunk]:
        """Retriever side — also caches the answer for the chain side."""
        response = self._client.post("/query", json={"question": question})
        response.raise_for_status()          # fail loud: a 500 is not a 0.0 score
        self._last = response.json()
        return [
            _Chunk(chunk.get("text", ""), {"source_file": chunk["source_file"]})
            for chunk in self._last.get("sources", [])
        ]

    def as_chain(self) -> "HttpRagTarget":
        return self                          # chain side, below

    def invoke_chain(self, _inputs: dict) -> str:
        return self._last.get("answer", "")
```

Then swap the three lines in `run_eval.py` that build the local chain:

```python
target = HttpRagTarget("http://localhost:8000")
retriever, chain = target, SimpleNamespace(invoke=target.invoke_chain)
```

Three caveats before you trust the output:

1. **Retrieval hit-rate needs the API to return its sources.** If yours only
   returns an answer, leave `expected_sources` empty and score keyword recall
   and refusal accuracy only — an unscored metric is honest, a fake one isn't.
2. **Latency becomes the full remote round trip**, including that service's
   own queueing — not comparable against local-chain runs.
3. **The refusal sentence must match.** Either make the remote service emit
   the sentence in `REFUSAL` verbatim, or change `REFUSAL` in
   `services/eval_service.py` to whatever contract that service promises.
   Refusal accuracy measures adherence to a contract; there has to be one.

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
ruff check .        # lint (ruff.toml)
python -m mypy      # types  (mypy.ini)
python -m pytest
```

Type checking is deliberately pragmatic, not strict: mypy's default only
checks annotated functions, which covers the typed core (eval-set loading,
metrics, run manifest) without demanding a codebase-wide annotation pass. The
one blanket exemption is `ignore_missing_imports` — the LangChain / Ollama /
pgvector / MLflow stack ships no usable type information and has no stub
packages.

CI (`.github/workflows/ci.yml`) runs ruff + mypy + pytest on every push/PR
against a plain `ubuntu-latest` runner with **no Ollama, Postgres, or MLflow
present** — the test suite is scoped to exactly what that allows (eval-set
validation and its rejection paths, refusal-contract scoring, run-manifest
construction, config defaults, clean imports, pure scoring/bench math). It
does **not** fake a passing end-to-end pipeline; running the harness against a
live model is a manual step against your own local stack.

## License

MIT — see [LICENSE](LICENSE).

---

*Extracted from LocalVault, a private on-premise AI platform I'm building.*
