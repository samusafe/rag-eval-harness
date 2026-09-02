"""run_eval.py wiring that must hold without a live stack: gate flags map to
known gate keys, unsatisfiable gates are refused BEFORE the run, and the
scorecard printer reads only keys aggregate() produces."""
import argparse

import pytest

from eval import run_eval
from services.eval_metrics import aggregate
from services.gates import MAX_GATES, MIN_GATES


def _args(**overrides):
    parser = run_eval.build_parser()
    return parser.parse_args(overrides.pop("argv", [])), parser


def test_gate_flags_map_to_known_gate_keys():
    args, _ = _args(
        argv=[
            "--gate-hit-rate", "0.8", "--gate-recall", "0.6", "--gate-refusal", "0.9",
            "--gate-max-latency", "8", "--gate-exact-hit-rate", "0.5", "--gate-mrr", "0.4",
            "--gate-answerability", "0.7", "--gate-citation-coverage", "0.3",
            "--gate-citation-validity", "1.0", "--gate-max-p95-latency", "12",
        ]
    )
    minimums, maximums = run_eval.gate_thresholds(args)
    assert set(minimums) == set(MIN_GATES)
    assert set(maximums) == set(MAX_GATES)
    assert minimums["mean_reciprocal_rank"] == 0.4
    assert maximums["p95_latency_s"] == 12.0


def test_no_gate_flags_means_no_gates():
    args, _ = _args()
    assert run_eval.gate_thresholds(args) == ({}, {})


def test_unsatisfiable_gate_is_refused_before_the_run():
    from services.eval_set import EvalCase

    rows = [EvalCase(id="q", question="?", expected_sources=["a"], expected_keywords=[], must_refuse=False)]
    args, parser = _args(argv=["--gate-mrr", "0.5"])
    with pytest.raises(SystemExit) as excinfo:
        run_eval.refuse_unsatisfiable_gates(parser, rows, *run_eval.gate_thresholds(args))
    assert excinfo.value.code == 2


def test_every_gate_flag_has_help_text():
    parser = run_eval.build_parser()
    for action in parser._actions:
        if any(option.startswith("--gate-") for option in action.option_strings):
            assert action.help, f"{action.option_strings} has no help text"


def test_print_summary_reads_only_keys_aggregate_produces(capsys):
    run_eval._print_summary(aggregate([]))
    out = capsys.readouterr().out
    assert "P95 latency" in out


def test_fmt_renders_rates_as_percent():
    assert run_eval._fmt(None) == "-"
    assert run_eval._fmt(0.85) == "85%"
    assert run_eval._fmt(1.0) == "100%"


def test_namespace_shape_is_stable():
    # argparse.Namespace is the contract gate_thresholds reads; keep it explicit.
    args, _ = _args()
    assert isinstance(args, argparse.Namespace)
