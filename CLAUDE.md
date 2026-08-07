# RAG Eval Harness

A standalone evaluation harness for a local RAG (Retrieval-Augmented
Generation) pipeline: LangChain LCEL orchestration over a local Ollama LLM +
embeddings, pgvector retrieval, and a HuggingFace cross-encoder reranker.
It scores a fixed question set on retrieval hit-rate, answer keyword recall,
and refusal accuracy (anti-hallucination), logs scorecards to MLflow, and
diffs runs so a change (new model, adapter, prompt, chunking) can be judged
on evidence instead of vibes. Extracted and sanitized from a private project
(see README attribution).

## Philosophy

**KISS** (simplest design that solves today's problem) > **SOLID** (single
responsibility above all) > **YAGNI** (no abstraction without a 2nd real
duplication). This is a small, focused repo — resist adding layers.

## Rules

1. **1 file = 1 responsibility.** `services/vector_store.py` only connects to
   the vector DB; `services/rag_chain.py` only builds the chain;
   `services/eval_service.py` only scores/persists. Need "and" to describe a
   file's job → split it.
2. **No new abstraction until the 2nd real duplication.** Nothing built "for
   the future" — this harness has exactly the knobs it needs today.
3. **Fail loud.** No `catch`/`except` that swallows an error without either
   re-raising or an explicit, contextful log (see `log_mlflow`'s deliberate
   `except Exception` — it logs the cause and is commented as intentionally
   non-fatal; that's the only acceptable pattern, never a bare silent pass).
4. **Zero hardcoded magic.** Every config value (model names, URLs, timeouts,
   retrieval knobs) lives in `config.py` and nowhere else — imported via
   `from config import settings`.
5. **Dependencies point inward.** CLI scripts (`eval/*.py`) → `services/` →
   LangChain/Ollama/pgvector clients. No raw `httpx`/SQL calls inline in a
   CLI script — route through a service module (`bench_ollama.py` is the one
   exception: it IS the thin I/O runner for `services/bench_core.py`'s pure
   math, by design).

## Commands (run from the repo root)

```bash
# Install (use a venv)
pip install -r requirements.txt          # runtime only
pip install -r requirements-dev.txt      # + pytest, ruff

# Run the eval (needs Postgres/pgvector + Ollama reachable — see .env.example)
python eval/run_eval.py
python eval/run_eval.py --model my-finetuned-model-v2 --mlflow
python eval/run_eval.py --permissive-eval-set   # investigate a broken eval set

# Compare two (or more) prior runs
python eval/compare_runs.py --latest 2

# Benchmark raw Ollama throughput (tokens/sec)
python eval/bench_ollama.py --model my-finetuned-model --runs 5 --mlflow

# Tests (no live services required — schema/config/import only)
python -m pytest

# Lint + types
ruff check .
python -m mypy
```

Equivalent wrapper scripts ship for both shells: `scripts/run_eval.ps1` /
`scripts/run_eval.sh`, `scripts/compare.ps1` / `scripts/compare.sh`. Every ops
script ships both a `.ps1` and a `.sh` twin — dev may happen on Windows, but
treat Linux as the deployment target. Keep `.ps1` files ASCII-only.

## Env

Windows + PowerShell during development. Use `python`, never `python3` (on
Windows, `python3` is frequently an unconfigured Store stub, not a real
interpreter). All config is environment-driven with localhost defaults — see
`.env.example` and `config.py` (the single source of truth; nothing reads
`os.getenv` outside that file).

## Tests

`tests/` only covers what's true without Ollama/Postgres/MLflow running:
eval-set schema validation (including every strict-mode rejection path),
refusal-contract scoring, run-manifest construction, config defaults/overrides,
clean module imports, and the pure scoring/aggregation/bench-math functions. It does NOT fake a
passing end-to-end pipeline — there is no mock LLM pretending eval scores
are real. Running the harness against a live model is a manual/CI-external
step, by design (see README).

## Git

Never commit, branch, or push unless explicitly asked — the user manages git.
