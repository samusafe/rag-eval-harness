#!/usr/bin/env bash
# Compare RAG eval runs.
#
# Usage:
#   ./scripts/compare.sh --latest 2
#   ./scripts/compare.sh eval/results/eval_a.json eval/results/eval_b.json
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [ "$#" -eq 0 ]; then
  set -- --latest 2
fi

echo "==> python eval/compare_runs.py $*"
python eval/compare_runs.py "$@"
