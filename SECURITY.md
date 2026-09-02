# Security

Do not report vulnerabilities through public issues when they expose credentials or
sensitive evaluation data. Contact the repository owner privately.

Scorecards may contain user questions, model answers, and retrieved source names/IDs.
Keep results and MLflow private, use `RESULT_CONTENT_MODE=redacted` when full text is
unnecessary — it hashes the question, the answer, and the retrieved source
names/IDs (`retrieved_sources`, `retrieved_source_ids`), since those are also
corpus-identifying — and use authenticated TLS for remote services. Result files are
created world-unreadable (`0600`, results directory `0700`) on POSIX. The localhost
database password and HTTP URLs are development defaults, not production credentials
or transport settings; a non-local `DATABASE_HOST` is rejected unless
`DATABASE_SSLMODE` is `require`, `verify-ca` or `verify-full` (`prefer`/`allow` are
also rejected, not just `disable`), and a non-local `MLFLOW_TRACKING_URI` is rejected
unless it uses `https` and `MLFLOW_ALLOW_REMOTE=true` is set explicitly.

Retrieved documents and their metadata are untrusted model input. The harness normalizes
source labels and strengthens prompt boundaries, but prompt injection cannot be fully
eliminated by formatting. Include adversarial cases in the evaluation set and scope
database/service access independently of model behavior.

Pin model revisions/digests (including `RERANK_MODEL_REVISION` — left blank the
harness warns that the reranker is unpinned) and review dependency audit output
before promotion. Never put database passwords, MLflow credentials, or tokens in
result manifests. The `git`/`nvidia-smi` provenance probes resolve their executable
via `shutil.which` (never a stray binary dropped into the checkout) and are bounded
by `HOST_PROBE_TIMEOUT` so a hung probe cannot hang the whole eval run.
