<#
.SYNOPSIS
  Run the RAG eval harness against one or more Ollama models.

.DESCRIPTION
  Thin wrapper around eval/run_eval.py. Requires Postgres+pgvector and Ollama
  reachable per your .env (see .env.example). Run with the project's venv
  activated (or `python` resolvable on PATH). Exits with the Python exit code,
  so a failed gate (exit 2) survives into a CI pipeline; with several -Models
  the first non-zero exit stops the loop.

.EXAMPLE
  ./scripts/run_eval.ps1
      # eval the default model (OLLAMA_CHAT_MODEL / .env)

.EXAMPLE
  ./scripts/run_eval.ps1 -Models my-finetuned-model,my-finetuned-model-v2 -Mlflow
      # eval two models in a row, logging both to MLflow

.EXAMPLE
  ./scripts/run_eval.ps1 --gate-refusal 0.9 --gate-max-p95-latency 12
      # any extra arguments are forwarded to run_eval.py
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string[]]$Models = @(),
    [switch]$Mlflow,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments = @()
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Eval {
    param([string]$Model)
    $runArgs = @("eval/run_eval.py")
    if ($Model) { $runArgs += @("--model", $Model) }
    if ($Mlflow) { $runArgs += "--mlflow" }
    $runArgs += $Arguments
    Write-Host "==> python $($runArgs -join ' ')" -ForegroundColor Cyan
    & python @runArgs
    return $LASTEXITCODE
}

$exitCode = 0
Push-Location $repoRoot
try {
    if ($Models.Count -eq 0) {
        $exitCode = Invoke-Eval -Model $null
    } else {
        foreach ($m in $Models) {
            $exitCode = Invoke-Eval -Model $m
            if ($exitCode -ne 0) { break }
        }
    }
} finally {
    Pop-Location
}
if ($exitCode -ne 0) {
    Write-Host "run_eval.py failed (exit $exitCode)" -ForegroundColor Red
}
exit $exitCode
