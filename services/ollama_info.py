"""Small Ollama preflight/provenance client; no model inference."""
from __future__ import annotations

from typing import Any

import httpx


def fetch_ollama_metadata(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    """Validate Ollama/model availability and return non-secret model/runtime metadata."""
    try:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=min(timeout, 10.0)) as client:
            tags = client.get("/api/tags")
            tags.raise_for_status()
            running = client.get("/api/ps")
            running.raise_for_status()
    except httpx.HTTPError as error:
        raise RuntimeError(f"Ollama preflight failed at {base_url}: {error}") from error

    available = tags.json().get("models", [])
    selected = next(
        (item for item in available if item.get("name") == model or item.get("model") == model),
        None,
    )
    if selected is None:
        names = sorted(str(item.get("name")) for item in available)
        raise RuntimeError(f"Ollama model {model!r} is not available; installed models: {names}")

    loaded = next(
        (
            item
            for item in running.json().get("models", [])
            if item.get("name") == model or item.get("model") == model
        ),
        None,
    )
    details = selected.get("details", {})
    return {
        "digest": selected.get("digest"),
        "size_bytes": selected.get("size"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "loaded": loaded is not None,
        "size_vram_bytes": loaded.get("size_vram") if loaded else None,
        "actual_context_length": loaded.get("context_length") if loaded else None,
    }
