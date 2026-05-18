from solidcue.core.utils.metrics import timed_generate


class _ProviderWithUsage:
    model = "test-model"

    def generate(self, messages, *, max_tokens=None):
        return "hello"

    def get_last_usage(self):
        return {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
            "cached_tokens": 5,
            "method": "provider_reported",
        }


def test_timed_generate_prefers_provider_reported_usage() -> None:
    output, metric = timed_generate(_ProviderWithUsage(), [{"role": "user", "content": "hi"}])
    assert output == "hello"
    assert metric["tokens"]["prompt_tokens"] == 50
    assert metric["tokens"]["completion_tokens"] == 10
    assert metric["tokens"]["total_tokens"] == 60
    assert metric["tokens"]["cached_tokens"] == 5
