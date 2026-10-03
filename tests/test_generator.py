import pytest

import src.generator as generator


class _BadResponse:
    text = "not-json"

    def raise_for_status(self):
        return None

    def json(self):
        raise ValueError("invalid JSON")


def test_llm_client_reports_malformed_response(monkeypatch):
    monkeypatch.setattr(generator.requests, "post", lambda *args, **kwargs: _BadResponse())
    client = generator.LLMClient(api_key="test-key", base_url="https://example.test/v1")

    with pytest.raises(RuntimeError, match="Unexpected LLM response"):
        client.generate("hello")
