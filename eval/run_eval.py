"""
RAG Evaluation Harness
======================================================================
Runs a fixed question set through the REAL RAG pipeline (the same retriever,
cross-encoder reranker, prompt, and local Ollama model the harness is
configured for) and scores it on the axes that actually decide RAG quality:

  1. Retrieval hit-rate    — did the expected source document make the
                             reranked top-N?
  2. Answer keyword recall — does the answer contain the expected facts?
  3. Refusal accuracy      — for out-of-KB questions, does it correctly say
                             "I don't have enough information..." instead of
                             hallucinating?
  + Median latency per query.

WHY THIS EXISTS
    It's the only way to know whether a change actually helped. Run it BEFORE
    and AFTER any change (new adapter/fine-tune, smaller generator, different
    embedder, new chunking, prompt edit) and diff the two scorecards.
    No eval = guessing.

ARCHITECTURE
    The reusable, observability-only core (scoring helpers, write_results,
    log_mlflow, the per-question runner) lives in services/eval_service.py.
    This CLI keeps only the presentation layer: the rich scorecard and argparse.

USAGE (run from the repo root, with the same env/.env the RAG pipeline needs —
       Postgres/pgvector + Ollama must be reachable):

    python eval/run_eval.py
    python eval/run_eval.py --set eval/eval_set.example.jsonl --out eval/results
    python eval/run_eval.py --collection <collection_id>   # scope to one collection
    python eval/run_eval.py --model my-finetuned-model-v2  # eval a different model
    python eval/run_eval.py --mlflow                       # log the scorecard to MLflow
    python eval/run_eval.py --permissive-eval-set          # skip invalid rows (debug only)
======================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when run as `python eval/run_eval.py`.
APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table  # noqa: E402

from config import settings  # noqa: E402
from services.eval_service import (  # noqa: E402
    _eval_one,
    aggregate,
    load_eval_set,
    log_mlflow,
    write_results,
)
from services.rag_chain import get_rag_chain  # noqa: E402

console = Console()


def _print_row(r: dict) -> None:
    def mark(v):
        if v is None:
            return "[dim]-[/dim]"
        if isinstance(v, bool):
            return "[green]OK[/green]" if v else "[red]X[/red]"
        return f"{v:.2f}"

    console.print(
        f"  [bold]{r['id']}[/bold]  "
        f"retr={mark(r['retrieval_hit'])} kw={mark(r['keyword_recall'])} "
        f"refuse={mark(r['refusal_ok'])} {r['latency_s']}s"
    )


def _print_summary(agg: dict) -> None:
    table = Table(title="RAG Eval Scorecard", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    table.add_column("Target", justify="right", style="dim")
    table.add_row("Retrieval hit-rate", _fmt(agg["retrieval_hit_rate"]), ">= 0.80")
    table.add_row("Answer keyword recall", _fmt(agg["answer_keyword_recall"]), ">= 0.70")
    table.add_row("Refusal accuracy", _fmt(agg["refusal_accuracy"]), ">= 0.90")
    table.add_row(
        "Median latency", f"{agg['median_latency_s']}s" if agg["median_latency_s"] else "-", "-"
    )
    table.add_row("Questions", str(agg["n"]), "-")
    console.print()
    console.print(table)


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.0%}" if v <= 1 else str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG eval harness.")
    parser.add_argument("--set", default=str(APP_ROOT / "eval" / "eval_set.example.jsonl"))
    parser.add_argument("--out", default=str(APP_ROOT / "eval" / "results"))
    parser.add_argument("--collection", default=None, help="Optional collection_id filter")
    parser.add_argument(
        "--model",
        default=None,
        help="Override OLLAMA_CHAT_MODEL for this run (e.g. my-finetuned-model-v2). "
        "Lets you eval several models in a row without touching .env.",
    )
    parser.add_argument("--mlflow", action="store_true", help="Also log this run to MLflow")
    parser.add_argument(
        "--permissive-eval-set",
        action="store_true",
        help="Skip invalid rows for investigation only (strict validation is the default).",
    )
    args = parser.parse_args()

    # --model overrides the configured chat model for THIS process only. Every
    # downstream read (chain build, scorecard filename, MLflow run name) goes
    # through settings.OLLAMA_CHAT_MODEL, so setting it here is enough.
    if args.model:
        settings.OLLAMA_CHAT_MODEL = args.model

    eval_set_path = Path(args.set)
    rows = load_eval_set(eval_set_path, strict=not args.permissive_eval_set)
    console.print(
        f"[bold]Loaded {len(rows)} eval questions[/bold] - "
        f"model=[cyan]{settings.OLLAMA_CHAT_MODEL}[/cyan] - "
        f"embed=[cyan]{settings.OLLAMA_EMBED_MODEL}[/cyan]\n"
    )

    # Reuse the REAL pipeline components. Building them is itself slow the first
    # time (vector-store handshake + cross-encoder load/download), so it gets its
    # own spinner — otherwise the run looks hung before question 1 even starts.
    with console.status("[cyan]Connecting to vector store and loading reranker...[/cyan]"):
        prompt, llm, get_retriever = get_rag_chain()
        retriever = get_retriever(args.collection)
        chain = prompt | llm | StrOutputParser()

    # A local LLM answers in tens of seconds, so a bare loop looks frozen for
    # minutes on end. Live progress (which question is running, how many are
    # done, elapsed, ETA) is the difference between "it's working" and "is it
    # stuck?" — the per-question result lines still scroll above the bar.
    results: list[dict] = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )
    try:
        with progress:
            task = progress.add_task("Evaluating", total=len(rows))
            for row in rows:
                progress.update(task, description=f"[cyan]{row.id}[/cyan]")
                r = _eval_one(row, retriever, chain)
                results.append(r)
                _print_row(r)
                progress.advance(task)
    except KeyboardInterrupt:
        # Deliberately no scorecard on a partial run: compare_runs.py treats
        # every result file as a complete run, and a half-finished one would
        # quietly produce a wrong-looking regression.
        raise SystemExit(
            f"Interrupted after {len(results)}/{len(rows)} questions - no scorecard "
            "written (a partial run isn't comparable against a full one)."
        ) from None

    agg = aggregate(results)
    _print_summary(agg)
    out_path = write_results(
        results,
        agg,
        Path(args.out),
        settings.OLLAMA_CHAT_MODEL,
        eval_set_path=eval_set_path,
    )
    console.print(f"\nSaved -> [cyan]{out_path}[/cyan]  (diff against a prior run to see if a change helped)")

    if args.mlflow:
        log_mlflow(agg, settings.OLLAMA_CHAT_MODEL, out_path)


if __name__ == "__main__":
    main()
