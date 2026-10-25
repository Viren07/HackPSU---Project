import io
import json
import urllib.error

import pytest

from km import assistant


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ask_builds_request_and_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k123")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    seen = []

    def fake_urlopen(req, timeout=0):
        seen.append(req)
        if "gemini-3.8-flash" in req.full_url:   # pretend the first model isn't available for this key
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, io.BytesIO(b"model not found"))
        return FakeResp(json.dumps({"candidates": [{"content": {"parts": [
            {"text": "thinking", "thought": True}, {"text": "HR 0.74 means 26% lower risk."}]}}]}).encode())
    monkeypatch.setattr(assistant.urllib.request, "urlopen", fake_urlopen)
    out = assistant.ask([{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"},
                         {"role": "user", "text": "what is my HR?"}], {"rebuilt_result": {"hr": {"hr": 0.74}}})
    assert out == "HR 0.74 means 26% lower risk."
    body = json.loads(seen[-1].data)
    assert seen[-1].get_header("X-goog-api-key") == "k123" and "key=" not in seen[-1].full_url
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert "Guyot" in body["system_instruction"]["parts"][0]["text"]
    assert '"hr": 0.74' in body["system_instruction"]["parts"][0]["text"]


def test_ask_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(assistant.AssistantError, match=".env"):
        assistant.ask([{"role": "user", "text": "hi"}])
