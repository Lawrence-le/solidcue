import httpx
import pytest

from solidcue.providers.client import HTTPClient


def _build_http_status_error(status_code: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.com/chat/completions")
    response = httpx.Response(status_code, headers=headers or {}, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


def test_post_retries_on_429_and_succeeds(monkeypatch: pytest.MonkeyPatch):
    client = HTTPClient(max_retries=2, initial_retry_delay_seconds=0.1)

    class StubResponse:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise _build_http_status_error(429)
        return StubResponse({"ok": True})

    sleep_calls: list[float] = []
    monkeypatch.setattr(client.client, "post", fake_post)
    monkeypatch.setattr("solidcue.providers.client.time.sleep", sleep_calls.append)

    data = client.post("https://api.example.com", headers={}, json={})

    assert data == {"ok": True}
    assert calls["count"] == 2
    assert sleep_calls == [0.1]


def test_post_honors_retry_after_header(monkeypatch: pytest.MonkeyPatch):
    client = HTTPClient(max_retries=2, initial_retry_delay_seconds=0.1)

    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise _build_http_status_error(429, headers={"Retry-After": "2"})
        return StubResponse()

    sleep_calls: list[float] = []
    monkeypatch.setattr(client.client, "post", fake_post)
    monkeypatch.setattr("solidcue.providers.client.time.sleep", sleep_calls.append)

    client.post("https://api.example.com", headers={}, json={})

    assert sleep_calls == [2.0]


def test_post_does_not_retry_on_non_retryable_status(monkeypatch: pytest.MonkeyPatch):
    client = HTTPClient(max_retries=2)

    def fake_post(*args, **kwargs):
        raise _build_http_status_error(400)

    monkeypatch.setattr(client.client, "post", fake_post)

    with pytest.raises(RuntimeError) as exc_info:
        client.post("https://api.example.com", headers={}, json={})
    assert "HTTP 400" in str(exc_info.value)


def test_post_raises_after_max_retries(monkeypatch: pytest.MonkeyPatch):
    client = HTTPClient(max_retries=2, initial_retry_delay_seconds=0.1)

    def fake_post(*args, **kwargs):
        raise _build_http_status_error(503)

    sleep_calls: list[float] = []
    monkeypatch.setattr(client.client, "post", fake_post)
    monkeypatch.setattr("solidcue.providers.client.time.sleep", sleep_calls.append)

    with pytest.raises(RuntimeError) as exc_info:
        client.post("https://api.example.com", headers={}, json={})

    assert sleep_calls == [0.1, 0.2]
    assert "HTTP 503" in str(exc_info.value)
