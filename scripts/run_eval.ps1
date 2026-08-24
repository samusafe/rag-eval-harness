<#
.SYNOPSIS
  Run the RAG eval harness against one or more Ollama models.

.DESCRIPTION
  Thin wrapper around eval/run_eval.py. Requires Postgres+pgvector and Ollama
  reachable per your .env (see .env.example). Run with the project's venv
  activated (or `python` resolvable on PATH).

.EXAMPLE
  ./scripts/run_eval.ps1
      # eval the default model (OLLAMA_CHAT_MODEL / .env)

.EXAMPLE
  ./scripts/run_eval.ps1 -Models my-finetuned-model,my-finetuned-model-v2 -Mlflow
      # eval two models in a row, logging both to MLflow
#>
[CmdletBinding()]
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
    if ($LASTEXITCODE -ne 0) { throw "run_eval.py failed (exit $LASTEXITCODE)" }
}

Push-Location $repoRoot
try {
    if ($Models.Count -eq 0) {
        Invoke-Eval -Model $null
    } else {
        foreach ($m in $Models) { Invoke-Eval -Model $m }
    }
} finally {
    Pop-Location
}
