<#
.SYNOPSIS
  Compare RAG eval runs (diff scorecards written by run_eval.py).

.DESCRIPTION
  Thin wrapper around eval/compare_runs.py. Exits with the Python exit code,
  so a blocked comparison (incompatible scorecards) is visible to a pipeline.
  Any extra arguments are forwarded verbatim, with or without -Latest.

.EXAMPLE
  ./scripts/compare.ps1 -Latest 2
.EXAMPLE
  ./scripts/compare.ps1 -Latest 2 --allow-incompatible
.EXAMPLE
  ./scripts/compare.ps1 eval/results/eval_a.json eval/results/eval_b.json
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Files = @(),
    [int]$Latest = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

$runArgs = @("eval/compare_runs.py")
if ($Latest -gt 0) {
    # With -Latest, the remaining arguments are flags (e.g. --allow-incompatible), not files.
    $runArgs += @("--latest", "$Latest")
    $runArgs += $Files
} elseif ($Files.Count -gt 0) {
    $runArgs += $Files
} else {
    $runArgs += @("--latest", "2")
}

Push-Location $repoRoot
try {
    Write-Host "==> python $($runArgs -join ' ')" -ForegroundColor Cyan
    & python @runArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($exitCode -ne 0) {
    Write-Host "compare_runs.py failed (exit $exitCode)" -ForegroundColor Red
}
exit $exitCode
