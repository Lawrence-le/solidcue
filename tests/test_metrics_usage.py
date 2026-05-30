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


def test_timed_generate_emits_langfuse_generation_usage(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_generation = object()

    def _start(**kwargs):
        captured["start"] = kwargs
        return fake_generation

    def _end(generation, **kwargs):
        captured["generation"] = generation
        captured["end"] = kwargs

    monkeypatch.setattr("solidcue.core.utils.metrics.start_langfuse_generation", _start)
    monkeypatch.setattr("solidcue.core.utils.metrics.end_langfuse_generation", _end)

    _output, _metric = timed_generate(
        _ProviderWithUsage(),
        [{"role": "user", "content": "hi"}],
        node_name="decision",
    )

    assert captured["generation"] is fake_generation
    start_kwargs = captured["start"]
    assert isinstance(start_kwargs, dict)
    assert start_kwargs["name"] == "decision"

    end_kwargs = captured["end"]
    assert isinstance(end_kwargs, dict)
    usage = end_kwargs["usage_details"]
    assert usage == {"input": 50, "output": 10, "total": 60, "cached": 5}
