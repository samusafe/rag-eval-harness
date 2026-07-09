<#
.SYNOPSIS
  Compare RAG eval runs (diff scorecards written by run_eval.py).

.EXAMPLE
  ./scripts/compare.ps1 -Latest 2
.EXAMPLE
  ./scripts/compare.ps1 eval/results/eval_a.json eval/results/eval_b.json
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Files,
    [int]$Latest = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

$runArgs = @("eval/compare_runs.py")
if ($Latest -gt 0) {
    $runArgs += @("--latest", "$Latest")
} elseif ($Files.Count -gt 0) {
    $runArgs += $Files
} else {
    $runArgs += @("--latest", "2")
}

Push-Location $repoRoot
try {
    Write-Host "==> python $($runArgs -join ' ')" -ForegroundColor Cyan
    & python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "compare_runs.py failed (exit $LASTEXITCODE)" }
} finally {
    Pop-Location
}
