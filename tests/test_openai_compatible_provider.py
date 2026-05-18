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


def test_openai_provider_forwards_tools_payload() -> None:
    provider = OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        api_key="test",
        model="gpt-4.1",
    )
    stub = _StubClient()
    provider.client = stub

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]

    provider.generate([{"role": "user", "content": "hi"}], tools=tools, tool_choice="auto")

    payload = stub.captured["json"]
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"


def test_openai_provider_forwards_temperature() -> None:
    provider = OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        api_key="test",
        model="gpt-4.1",
        temperature=0.4,
    )
    stub = _StubClient()
    provider.client = stub

    provider.generate([{"role": "user", "content": "hi"}])

    payload = stub.captured["json"]
    assert payload["temperature"] == 0.4


def test_openai_provider_forwards_max_tokens() -> None:
    provider = OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        api_key="test",
        model="gpt-4.1",
    )
    stub = _StubClient()
    provider.client = stub

    provider.generate([{"role": "user", "content": "hi"}], max_tokens=300)

    payload = stub.captured["json"]
    assert payload["max_tokens"] == 300


def test_openai_provider_captures_usage_tokens() -> None:
    provider = OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        api_key="test",
        model="gpt-4.1",
    )

    class _UsageClient:
        def post(self, url: str, headers: dict, json: dict):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "total_tokens": 125,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            }

    provider.client = _UsageClient()
    provider.generate([{"role": "user", "content": "hi"}])
    usage = provider.get_last_usage()
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 25
    assert usage["total_tokens"] == 125
    assert usage["cached_tokens"] == 40
