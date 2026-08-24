# Security

Do not report vulnerabilities through public issues when they expose credentials or
sensitive evaluation data. Contact the repository owner privately.

Scorecards may contain user questions, model answers, and source names. Keep results and
MLflow private, use `RESULT_CONTENT_MODE=redacted` when full text is unnecessary, and
use authenticated TLS for remote services. The localhost database password and HTTP
URLs are development defaults, not production credentials or transport settings.

Retrieved documents and their metadata are untrusted model input. The harness normalizes
source labels and strengthens prompt boundaries, but prompt injection cannot be fully
eliminated by formatting. Include adversarial cases in the evaluation set and scope
database/service access independently of model behavior.

Pin model revisions/digests and review dependency audit output before promotion. Never
put database passwords, MLflow credentials, or tokens in result manifests.
