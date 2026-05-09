from solidcue.providers.openai_compatible import OpenAICompatibleProvider


class _StubClient:
    def __init__(self):
        self.captured = None

    def post(self, url: str, headers: dict, json: dict):
        self.captured = {"url": url, "headers": headers, "json": json}
        return {"choices": [{"message": {"content": "ok"}}]}


def test_openai_provider_normalizes_null_message_content() -> None:
    provider = OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        api_key="test",
        model="gpt-4.1",
    )
    stub = _StubClient()
    provider.client = stub

    provider.generate(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_1", "content": None},
        ]
    )

    sent = stub.captured["json"]["messages"]
    assert sent[1]["content"] == ""
    assert sent[2]["content"] == ""
