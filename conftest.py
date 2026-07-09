"""Repo-root pytest conftest: guarantees the repo root is on sys.path so
`config`, `services.*`, and `eval.*` import the same way whether pytest is
invoked from the repo root or a subdirectory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
