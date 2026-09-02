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
    The reusable core lives in services/: eval_set (loader), eval_metrics
    (pure scorers), eval_manifest (provenance), eval_service (per-question
    runner, persistence, MLflow), gates (thresholds). This CLI keeps only the
    presentation layer: the rich scorecard, argparse and gate wiring.

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

from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402
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
from rich.text import Text  # noqa: E402

from config import settings  # noqa: E402
from services.eval_manifest import collect_runtime_metadata  # noqa: E402
from services.eval_metrics import aggregate  # noqa: E402
from services.eval_service import (  # noqa: E402
    DEFAULT_EVAL_SET,
    DEFAULT_OUT_DIR,
    eval_one,
    log_mlflow,
    write_results,
)
from services.eval_set import EvalCase, load_eval_set, metrics_supported_by  # noqa: E402
from services.gates import check_gates, unsatisfiable_gates  # noqa: E402
from services.ollama_info import fetch_ollama_metadata  # noqa: E402
from services.rag_chain import get_rag_chain  # noqa: E402

console = Console()


def _print_row(r: dict) -> None:
    def mark(v):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "OK" if v else "X"
        return f"{v:.2f}"

    console.print(
        Text.assemble(
            "  ",
            (str(r["id"]), "bold"),
            f"  retr={mark(r['retrieval_hit'])} kw={mark(r['keyword_recall'])} ",
            f"refuse={mark(r['refusal_ok'])} {r['latency_s']:.2f}s",
        ),
        markup=False,
    )


def _print_summary(agg: dict) -> None:
    table = Table(title="RAG Eval Scorecard", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    table.add_column("Target", justify="right", style="dim")
    table.add_row("Retrieval hit-rate", _fmt(agg["retrieval_hit_rate"]), ">= 0.80")
    table.add_row("Answer keyword recall", _fmt(agg["answer_keyword_recall"]), ">= 0.70")
    table.add_row("Refusal accuracy", _fmt(agg["refusal_accuracy"]), ">= 0.90")
    table.add_row("Exact retrieval hit-rate", _fmt(agg["exact_retrieval_hit_rate"]), "-")
    table.add_row("Mean reciprocal rank", _fmt(agg["mean_reciprocal_rank"]), "-")
    table.add_row("Answerability accuracy", _fmt(agg["answerability_accuracy"]), "-")
    table.add_row("Citation coverage", _fmt(agg["citation_coverage"]), "-")
    table.add_row("Citation validity", _fmt(agg["citation_validity"]), "-")
    table.add_row("Median latency", _seconds(agg["median_latency_s"]), "-")
    table.add_row("P95 latency", _seconds(agg["p95_latency_s"]), "-")
    table.add_row("Questions", str(agg["n"]), "-")
    console.print()
    console.print(table)


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.0%}"


def _seconds(v) -> str:
    return "-" if v is None else f"{v:.2f}s"


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _non_negative(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RAG eval harness.")
    parser.add_argument("--set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--collection", default=None, help="Optional collection_id filter")
    parser.add_argument(
        "--model",
        default=None,
        help="Override OLLAMA_CHAT_MODEL for this run (e.g. my-finetuned-model-v2). "
        "Lets you eval several models in a row without touching .env.",
    )
    parser.add_argument("--mlflow", action="store_true", help="Also log this run to MLflow")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate Ollama, pgvector, embedding, and reranker with one retrieval, then exit.",
    )
    parser.add_argument(
        "--permissive-eval-set",
        action="store_true",
        help="Skip invalid rows for investigation only (strict validation is the default).",
    )
    # CI / promotion gates: exit 2 when the scorecard misses a threshold, so a
    # pipeline (GitHub Actions, or qlora-8gb-pipeline's post-export check) can
    # block a regressed model without parsing output.
    parser.add_argument("--gate-hit-rate", type=_probability, default=None, metavar="0.8",
                        help="Fail unless retrieval_hit_rate >= this")
    parser.add_argument("--gate-recall", type=_probability, default=None, metavar="0.6",
                        help="Fail unless answer_keyword_recall >= this")
    parser.add_argument("--gate-refusal", type=_probability, default=None, metavar="0.9",
                        help="Fail unless refusal_accuracy >= this")
    parser.add_argument("--gate-max-latency", type=_non_negative, default=None, metavar="8.0",
                        help="Fail if median_latency_s exceeds this (seconds)")
    parser.add_argument("--gate-max-p95-latency", type=_non_negative, default=None, metavar="12.0",
                        help="Fail if p95_latency_s exceeds this (seconds)")
    parser.add_argument("--gate-exact-hit-rate", type=_probability, default=None, metavar="0.8",
                        help="Fail unless exact_retrieval_hit_rate >= this (needs expected_source_ids)")
    parser.add_argument("--gate-mrr", type=_probability, default=None, metavar="0.6",
                        help="Fail unless mean_reciprocal_rank >= this (needs expected_source_ids)")
    parser.add_argument("--gate-answerability", type=_probability, default=None, metavar="0.9",
                        help="Fail unless answerability_accuracy >= this (no refusal on answerable rows)")
    parser.add_argument("--gate-citation-coverage", type=_probability, default=None, metavar="0.8",
                        help="Fail unless citation_coverage >= this (answers that cite at all)")
    parser.add_argument("--gate-citation-validity", type=_probability, default=None, metavar="1.0",
                        help="Fail unless citation_validity >= this (cited headers were really supplied)")
    return parser


def gate_thresholds(args: argparse.Namespace) -> tuple[dict[str, float], dict[str, float]]:
    """Translate gate flags into the (minimums, maximums) dicts check_gates reads."""
    minimums = {
        key: threshold
        for key, threshold in (
            ("retrieval_hit_rate", args.gate_hit_rate),
            ("answer_keyword_recall", args.gate_recall),
            ("refusal_accuracy", args.gate_refusal),
            ("exact_retrieval_hit_rate", args.gate_exact_hit_rate),
            ("mean_reciprocal_rank", args.gate_mrr),
            ("answerability_accuracy", args.gate_answerability),
            ("citation_coverage", args.gate_citation_coverage),
            ("citation_validity", args.gate_citation_validity),
        )
        if threshold is not None
    }
    maximums = {
        key: threshold
        for key, threshold in (
            ("median_latency_s", args.gate_max_latency),
            ("p95_latency_s", args.gate_max_p95_latency),
        )
        if threshold is not None
    }
    return minimums, maximums


def refuse_unsatisfiable_gates(
    parser: argparse.ArgumentParser,
    rows: list[EvalCase],
    minimums: dict[str, float],
    maximums: dict[str, float],
) -> None:
    """A gate the eval set cannot produce data for is a usage error, reported
    before the first LLM call (exit 2, argparse's usage-error code)."""
    problems = unsatisfiable_gates(minimums, maximums, metrics_supported_by(rows))
    if problems:
        parser.error("unsatisfiable gate(s):\n  " + "\n  ".join(problems))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    selected_model = args.model or settings.OLLAMA_CHAT_MODEL
    eval_set_path = Path(args.set)
    strict = not args.permissive_eval_set
    rows = load_eval_set(eval_set_path, strict=strict)
    minimums, maximums = gate_thresholds(args)
    refuse_unsatisfiable_gates(parser, rows, minimums, maximums)
    if settings.CORPUS_REVISION == "unversioned":
        console.print(
            "[yellow]Warning: CORPUS_REVISION is unversioned; this run cannot prove "
            "that the underlying corpus matches another run.[/yellow]"
        )
    if settings.RERANK_MODEL_REVISION is None:
        console.print(
            "[yellow]Warning: RERANK_MODEL_REVISION is unpinned; the reranker resolves to whatever "
            "the Hub's default branch points at, so two runs may not use the same weights.[/yellow]"
        )

    runtime_metadata = collect_runtime_metadata()
    with console.status("[cyan]Checking Ollama model metadata...[/cyan]"):
        runtime_metadata["ollama_chat_model"] = fetch_ollama_metadata(
            settings.OLLAMA_BASE_URL, selected_model, settings.OLLAMA_REQUEST_TIMEOUT
        )
        runtime_metadata["ollama_embedding_model"] = fetch_ollama_metadata(
            settings.OLLAMA_BASE_URL,
            settings.OLLAMA_EMBED_MODEL,
            settings.OLLAMA_REQUEST_TIMEOUT,
        )

    console.print(
        Text.assemble(
            (f"Loaded {len(rows)} eval questions - model=", "bold"),
            (selected_model, "cyan"),
            " - embed=",
            (settings.OLLAMA_EMBED_MODEL, "cyan"),
        )
    )

    # Reuse the REAL pipeline components. Building them is itself slow the first
    # time (vector-store handshake + cross-encoder load/download), so it gets its
    # own spinner — otherwise the run looks hung before question 1 even starts.
    with console.status("[cyan]Connecting to vector store and loading reranker...[/cyan]"):
        prompt, llm, get_retriever = get_rag_chain(selected_model)
        retriever = get_retriever(args.collection)
        # Keep AIMessage metadata (token counts and Ollama phase timings).
        chain = prompt | llm

    if args.preflight_only:
        with console.status("[cyan]Running one retrieval preflight...[/cyan]"):
            docs = retriever.invoke(rows[0].question)
        console.print(
            f"[green]Preflight passed:[/green] Ollama models available and retrieval returned "
            f"{len(docs)} reranked document(s)."
        )
        return

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
                progress.update(task, description=f"[cyan]{escape(str(row.id))}[/cyan]")
                r = eval_one(row, retriever, chain)
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
    try:
        runtime_metadata["ollama_chat_model"] = fetch_ollama_metadata(
            settings.OLLAMA_BASE_URL, selected_model, settings.OLLAMA_REQUEST_TIMEOUT
        )
    except RuntimeError as error:
        console.print(f"[yellow]Could not refresh post-run Ollama VRAM metadata: {escape(str(error))}[/yellow]")
    _print_summary(agg)
    out_path = write_results(
        results,
        agg,
        Path(args.out),
        selected_model,
        eval_set_path=eval_set_path,
        collection_id=args.collection,
        runtime_metadata=runtime_metadata,
        eval_set_strict=strict,
        gates={**minimums, **maximums},
    )
    console.print(
        f"\nSaved -> [cyan]{escape(str(out_path))}[/cyan]  (diff against a prior run to see if a change helped)"
    )

    if args.mlflow:
        log_mlflow(agg, selected_model, out_path)

    # Gates run LAST: the scorecard is already written and logged, so a failed
    # gate still leaves a full result file to diff against.
    if minimums or maximums:
        failures = check_gates(agg, minimums, maximums)
        if failures:
            console.print("\n[bold red]GATE FAILURES[/bold red]")
            for failure in failures:
                console.print(f"  [red]x[/red] {failure}")
            raise SystemExit(2)
        console.print("\n[bold green]All gates passed.[/bold green]")


if __name__ == "__main__":
    main()
