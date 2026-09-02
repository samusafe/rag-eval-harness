"""Unit tests for services/ollama_info.py — the Ollama preflight/provenance
client. No live Ollama server: `httpx.Client` is monkeypatched to route
through `httpx.MockTransport` instead of the network.
"""
import httpx
import pytest

from services import ollama_info

# `services.ollama_info` does `import httpx`, so its `httpx` is the same
# module object as this file's — monkeypatching `ollama_info.httpx.Client`
# replaces `httpx.Client` everywhere. Capture the real class up front so the
# stub factory below can still build genuine (MockTransport-backed) clients
# instead of recursing into itself.
_RealClient = httpx.Client


def _client_stub(handler, captured_kwargs):
    """Stand in for `httpx.Client`: keeps every kwarg the real code passed
    (so the timeout cap can be asserted) and swaps in a MockTransport."""

    def factory(**kwargs):
        captured_kwargs.append(kwargs)
        return _RealClient(transport=httpx.MockTransport(handler), **kwargs)

    return factory


def _tags_response(models):
    return httpx.Response(200, json={"models": models})


def _ps_response(models):
    return httpx.Response(200, json={"models": models})


def test_fetch_metadata_for_a_loaded_model(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return _tags_response(
                [
                    {
                        "name": "my-model",
                        "digest": "sha256:abc123",
                        "size": 4_000_000_000,
                        "details": {"parameter_size": "7B", "quantization_level": "Q4_0"},
                    }
                ]
            )
        if request.url.path == "/api/ps":
            return _ps_response(
                [{"name": "my-model", "size_vram": 4_100_000_000, "context_length": 4096}]
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    monkeypatch.setattr(ollama_info.httpx, "Client", _client_stub(handler, []))

    result = ollama_info.fetch_ollama_metadata("http://localhost:11434", "my-model", 30.0)

    assert result == {
        "digest": "sha256:abc123",
        "size_bytes": 4_000_000_000,
        "parameter_size": "7B",
        "quantization_level": "Q4_0",
        "loaded": True,
        "size_vram_bytes": 4_100_000_000,
        "actual_context_length": 4096,
    }


def test_fetch_metadata_matches_on_either_name_or_model_key(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return _tags_response(
                [{"model": "my-model", "digest": "sha256:def", "size": 1, "details": {}}]
            )
        if request.url.path == "/api/ps":
            return _ps_response([{"model": "my-model", "size_vram": 1, "context_length": 1}])
        raise AssertionError(f"unexpected path: {request.url.path}")

    monkeypatch.setattr(ollama_info.httpx, "Client", _client_stub(handler, []))

    result = ollama_info.fetch_ollama_metadata("http://localhost:11434", "my-model", 5.0)

    assert result["loaded"] is True
    assert result["digest"] == "sha256:def"


def test_fetch_metadata_missing_model_lists_installed_names(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return _tags_response([{"name": "other-model-a"}, {"name": "other-model-b"}])
        if request.url.path == "/api/ps":
            return _ps_response([])
        raise AssertionError(f"unexpected path: {request.url.path}")

    monkeypatch.setattr(ollama_info.httpx, "Client", _client_stub(handler, []))

    with pytest.raises(RuntimeError) as exc_info:
        ollama_info.fetch_ollama_metadata("http://localhost:11434", "missing-model", 5.0)

    message = str(exc_info.value)
    assert "is not available" in message
    assert "other-model-a" in message
    assert "other-model-b" in message


def test_fetch_metadata_transport_error_becomes_runtime_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(ollama_info.httpx, "Client", _client_stub(handler, []))

    with pytest.raises(RuntimeError) as exc_info:
        ollama_info.fetch_ollama_metadata("http://localhost:11434", "my-model", 5.0)

    assert "Ollama preflight failed" in str(exc_info.value)


def test_fetch_metadata_non_2xx_status_becomes_runtime_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    monkeypatch.setattr(ollama_info.httpx, "Client", _client_stub(handler, []))

    with pytest.raises(RuntimeError) as exc_info:
        ollama_info.fetch_ollama_metadata("http://localhost:11434", "my-model", 5.0)

    assert "Ollama preflight failed" in str(exc_info.value)


def test_fetch_metadata_caps_client_timeout_at_ten_seconds(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return _tags_response([{"name": "my-model", "details": {}}])
        if request.url.path == "/api/ps":
            return _ps_response([])
        raise AssertionError(f"unexpected path: {request.url.path}")

    captured: list[dict] = []
    monkeypatch.setattr(ollama_info.httpx, "Client", _client_stub(handler, captured))

    ollama_info.fetch_ollama_metadata("http://localhost:11434", "my-model", 30.0)
    assert captured[-1]["timeout"] == 10.0

    ollama_info.fetch_ollama_metadata("http://localhost:11434", "my-model", 3.0)
    assert captured[-1]["timeout"] == 3.0
