"""
Compare RAG eval runs
======================================================================
Diffs the scorecards written by run_eval.py (eval/results/*.json) so you can
answer:
  - Did fine-tuning a new adapter improve quality?  baseline vs candidate
  - Is a smaller generator as good as the bigger one?  run each, compare

Shows a summary table across all runs (one column each) and, for exactly two
runs, a per-question breakdown of what improved vs regressed (regressions
first — that's where the debugging value is).

USAGE (from the repo root):
    python eval/compare_runs.py eval/results/eval_A.json eval/results/eval_B.json
    python eval/compare_runs.py --latest 2      # auto-pick the 2 newest result files
    python eval/compare_runs.py --latest 3      # summary table across the 3 newest
======================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.table import Table  # noqa: E402

from config import settings  # noqa: E402
from services.eval_compat import compatibility_issues  # noqa: E402

__all__ = ["compatibility_issues", "main", "per_question_diff", "pick_latest"]

console = Console()

# (summary key, label, higher_is_better)
SUMMARY_METRICS = [
    ("retrieval_hit_rate", "Retrieval hit-rate", True),
    ("exact_retrieval_hit_rate", "Exact retrieval hit-rate", True),
    ("mean_reciprocal_rank", "Mean reciprocal rank", True),
    ("answer_keyword_recall", "Answer keyword recall", True),
    ("refusal_accuracy", "Refusal accuracy", True),
    ("answerability_accuracy", "Answerability accuracy", True),
    ("citation_coverage", "Citation coverage", True),
    ("citation_validity", "Citation validity", True),
    ("median_latency_s", "Median latency (s)", False),
    ("p95_latency_s", "P95 latency (s)", False),
]

# (per-result key, label, higher_is_better)
PQ_METRICS = [
    ("retrieval_hit", "retrieval", True),
    ("exact_retrieval_hit", "exact retrieval", True),
    ("reciprocal_rank", "reciprocal rank", True),
    ("keyword_recall", "keyword recall", True),
    ("refusal_ok", "refusal", True),
    ("answerability_ok", "answerability", True),
    ("citation_present", "citation present", True),
    ("citation_validity", "citation validity", True),
    ("latency_s", "latency", False),
]

# Rates print as percentages; these are not rates.
_SECONDS_KEYS = {"median_latency_s", "p95_latency_s", "latency_s"}
_PLAIN_KEYS = {"mean_reciprocal_rank", "reciprocal_rank"}


def load_run(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot load scorecard {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("summary"), dict) or not isinstance(data.get("results"), list):
        raise SystemExit(f"Invalid scorecard {path}: expected object with summary and results")
    data["_label"] = escape(f"{data.get('model', '?')} @ {data.get('timestamp', '?')}")
    data["_file"] = path.name
    return data


def _run_started_at(path: Path) -> str:
    """Sort key for a scorecard: the manifest's ISO-8601 UTC timestamp (lexically
    chronological), falling back to the file mtime for pre-manifest scorecards.
    Unreadable files sort first so they never displace a real run."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8")).get("run_manifest")
    except (OSError, json.JSONDecodeError, AttributeError):
        manifest = None
    if isinstance(manifest, dict) and isinstance(manifest.get("timestamp_utc"), str):
        return manifest["timestamp_utc"]
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat().replace("+00:00", "Z")


def pick_latest(out_dir: Path, n: int) -> list[Path]:
    """The N most recent RUNS, oldest first. Filenames start with the model
    name (eval_<model>_<hash>_<stamp>.json), so a name sort is model-first,
    not chronological -- `--latest 2` across two models would compare a model
    against itself and hide the newest run. Order by run time instead."""
    if n < 2:
        raise SystemExit("--latest must be at least 2.")
    files = sorted(out_dir.glob("eval_*.json"), key=_run_started_at)
    if len(files) < n:
        raise SystemExit(f"Need {n} result files in {out_dir}, found {len(files)}.")
    return files[-n:]


def _fmt_summary(key: str, v) -> str:
    if v is None:
        return "-"
    if key in _SECONDS_KEYS:
        return f"{v:.2f}s"
    if key in _PLAIN_KEYS:
        return f"{v:.3f}"
    return f"{v:.0%}"


def _delta(key: str, base, cand, higher_better: bool) -> str:
    if base is None or cand is None:
        return "[dim]n/a[/dim]"
    d = cand - base
    if abs(d) < 1e-9:
        return "[dim]=[/dim]"
    improved = (d > 0) == higher_better
    color = "green" if improved else "red"
    arrow = "^" if d > 0 else "v"
    if key in _SECONDS_KEYS:
        shown = f"{arrow}{abs(d):.2f}s"
    elif key in _PLAIN_KEYS:
        shown = f"{arrow}{abs(d):.3f}"
    else:
        shown = f"{arrow}{abs(d) * 100:.1f}pts"
    return f"[{color}]{shown}[/{color}]"


def summary_table(runs: list[dict]) -> None:
    table = Table(title="Summary across runs", header_style="bold cyan")
    table.add_column("Metric")
    for r in runs:
        table.add_column(r["_label"], justify="right")
    if len(runs) == 2:
        table.add_column("Delta cand vs base", justify="right")

    for key, label, higher in SUMMARY_METRICS:
        row = [label] + [_fmt_summary(key, r["summary"].get(key)) for r in runs]
        if len(runs) == 2:
            row.append(_delta(key, runs[0]["summary"].get(key), runs[1]["summary"].get(key), higher))
        table.add_row(*row)
    console.print(table)


def _pq_fmt(key: str, v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "OK" if v else "X"
    return f"{v:.2f}s" if key in _SECONDS_KEYS else f"{v:.2f}"


def per_question_diff(base: dict, cand: dict) -> tuple[list, list, list]:
    b = {r["id"]: r for r in base["results"]}
    c = {r["id"]: r for r in cand["results"]}
    rows = []  # (improved: bool, id, key, label, base_val, cand_val)
    for i in (i for i in c if i in b):
        for key, label, higher in PQ_METRICS:
            bv, cv = b[i].get(key), c[i].get(key)
            if bv is None or cv is None:
                continue
            if isinstance(bv, bool) or isinstance(cv, bool):
                if bv == cv:
                    continue
                improved = (not bv) and bool(cv)
            else:
                # Ignore per-question latency deltas under the configured noise floor.
                if key == "latency_s" and abs(cv - bv) < settings.COMPARE_LATENCY_NOISE_S:
                    continue
                if abs(cv - bv) < 1e-9:
                    continue
                improved = (cv > bv) == higher
            rows.append((improved, i, key, label, bv, cv))
    rows.sort(key=lambda x: x[0])  # False (regressions) first
    only_base = [i for i in b if i not in c]
    only_cand = [i for i in c if i not in b]
    return rows, only_base, only_cand


def print_per_question(rows: list, only_base: list, only_cand: list) -> None:
    if rows:
        t = Table(title="Per-question changes (regressions first)", header_style="bold cyan")
        t.add_column("")
        t.add_column("id")
        t.add_column("metric")
        t.add_column("base -> cand", justify="right")
        for improved, i, key, label, bv, cv in rows:
            mark = "[green]^[/green]" if improved else "[red]v[/red]"
            t.add_row(mark, escape(str(i)), label, f"{_pq_fmt(key, bv)} -> {_pq_fmt(key, cv)}")
        console.print(t)
    else:
        console.print("[dim]No per-question changes between these two runs.[/dim]")

    if only_base or only_cand:
        console.print(
            "[yellow]Note: question sets differ[/yellow] - "
            f"only-in-base: {escape(str(only_base or '(none)'))}, "
            f"only-in-candidate: {escape(str(only_cand or '(none)'))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG eval runs.")
    parser.add_argument("files", nargs="*", help="result JSON files (baseline first)")
    parser.add_argument("--latest", type=int, default=None, help="auto-pick the N newest result files")
    parser.add_argument("--out", default=str(APP_ROOT / "eval" / "results"))
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="Compare despite blocking provenance differences (results may be misleading).",
    )
    args = parser.parse_args()

    if args.latest:
        paths = pick_latest(Path(args.out), args.latest)
    elif len(args.files) >= 2:
        paths = [Path(f) for f in args.files]
    else:
        raise SystemExit("Pass at least 2 result files, or use --latest N.")

    runs = [load_run(p) for p in paths]
    for candidate in runs[1:]:
        blocking, advisory = compatibility_issues(runs[0], candidate)
        for issue in advisory:
            console.print(f"[yellow]Comparability note:[/yellow] {escape(issue)}")
        if blocking:
            for issue in blocking:
                console.print(f"[red]Incompatible scorecards:[/red] {escape(issue)}")
            if not args.allow_incompatible:
                raise SystemExit(
                    "Comparison stopped. Re-run with --allow-incompatible to inspect it anyway."
                )
    console.print(f"[bold]Comparing {len(runs)} runs[/bold] (baseline first):")
    for r in runs:
        console.print(f"  - {r['_label']}  ([dim]{escape(r['_file'])}[/dim])")

    # Retrieval is only comparable when the embedder is held constant.
    embedders = {r.get("embed_model") for r in runs}
    if len(embedders) > 1:
        console.print(
            f"[yellow]Warning: different embedders across runs ({escape(str(embedders))}) - "
            f"retrieval metrics are NOT comparable; trust keyword/refusal/latency only.[/yellow]"
        )
    console.print()

    summary_table(runs)
    if len(runs) == 2:
        console.print()
        rows, only_base, only_cand = per_question_diff(runs[0], runs[1])
        print_per_question(rows, only_base, only_cand)


if __name__ == "__main__":
    main()
