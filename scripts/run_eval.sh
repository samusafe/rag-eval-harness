#!/usr/bin/env bash
# Run the RAG eval harness against one or more Ollama models.
#
# Usage:
#   ./scripts/run_eval.sh                              # default model
#   ./scripts/run_eval.sh --mlflow                      # + log to MLflow
#   MODELS="model-a model-b" ./scripts/run_eval.sh --mlflow
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mlflow_flag=()
for arg in "$@"; do
  if [ "$arg" = "--mlflow" ]; then
    mlflow_flag=(--mlflow)
  fi
done

if [ -z "${MODELS:-}" ]; then
  echo "==> python eval/run_eval.py ${mlflow_flag[*]:-}"
  python eval/run_eval.py "${mlflow_flag[@]}"
else
  for model in $MODELS; do
    echo "==> python eval/run_eval.py --model $model ${mlflow_flag[*]:-}"
    python eval/run_eval.py --model "$model" "${mlflow_flag[@]}"
  done
fi
