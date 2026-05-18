from solidcue.providers.anthropic import AnthropicProvider


def test_anthropic_provider_captures_usage_tokens() -> None:
    provider = AnthropicProvider(
        api_key="test",
        model="claude-sonnet",
    )

    class _UsageClient:
        def post(self, url: str, headers: dict, json: dict):
            return {
                "content": [{"text": "ok"}],
                "usage": {
                    "input_tokens": 300,
                    "output_tokens": 120,
                    "cache_read_input_tokens": 80,
                    "cache_creation_input_tokens": 20,
                },
            }

    provider.client = _UsageClient()
    provider.generate([{"role": "user", "content": "hello"}])
    usage = provider.get_last_usage()
    assert usage["prompt_tokens"] == 300
    assert usage["completion_tokens"] == 120
    assert usage["total_tokens"] == 420
    assert usage["cached_tokens"] == 100
