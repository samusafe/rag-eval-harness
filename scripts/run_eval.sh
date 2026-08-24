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

if [ -z "${MODELS:-}" ]; then
  echo "==> python eval/run_eval.py $*"
  python eval/run_eval.py "$@"
else
  for model in $MODELS; do
    echo "==> python eval/run_eval.py --model $model $*"
    python eval/run_eval.py --model "$model" "$@"
  done
fi
