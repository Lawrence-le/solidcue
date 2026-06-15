from solidcue.core.utils.metrics import (
    _generation_observation_name,
    _langfuse_usage_details,
    timed_generate,
)
from solidcue.observability import langfuse as langfuse_observability


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
    assert start_kwargs["name"] == "decision.llm"

    end_kwargs = captured["end"]
    assert isinstance(end_kwargs, dict)
    usage = end_kwargs["usage_details"]
    assert usage == {"input": 50, "output": 10, "total": 60, "cached": 5}


def test_langfuse_generation_uses_current_observation_context(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class _FakeObservation:
        def update(self, **kwargs):
            events.append(("update", kwargs))

        def end(self):
            events.append(("end", None))

    class _FakeContext:
        def __enter__(self):
            events.append(("enter", None))
            return _FakeObservation()

        def __exit__(self, exc_type, exc, tb):
            events.append(("exit", exc_type))

    class _FakeClient:
        def start_as_current_observation(self, **kwargs):
            events.append(("start_as_current_observation", kwargs))
            return _FakeContext()

    monkeypatch.setenv("SOLIDCUE_LANGFUSE_ENABLED", "true")
    monkeypatch.setattr(langfuse_observability, "_LANGFUSE_CLIENT", _FakeClient())
    monkeypatch.setattr(langfuse_observability, "_LANGFUSE_HANDLER", object())

    generation = langfuse_observability.start_langfuse_generation(
        name="decision",
        input_payload=[{"role": "user", "content": "hi"}],
        model="test-model",
        model_parameters={"temperature": 0.0},
    )
    langfuse_observability.end_langfuse_generation(
        generation,
        output_payload="hello",
        usage_details={"input": 1, "output": 1, "total": 2},
    )

    assert events[0] == (
        "start_as_current_observation",
        {
            "name": "decision",
            "as_type": "generation",
            "input": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "model_parameters": {"temperature": 0.0},
            "end_on_exit": False,
        },
    )
    assert [event for event, _payload in events] == [
        "start_as_current_observation",
        "enter",
        "update",
        "end",
        "exit",
    ]


def test_generation_observation_name_marks_llm_calls() -> None:
    assert _generation_observation_name("decision") == "decision.llm"
    assert _generation_observation_name("decision.llm") == "decision.llm"
    assert _generation_observation_name("") == "llm.llm"


def test_langfuse_usage_details_falls_back_to_estimated_tokens() -> None:
    usage = _langfuse_usage_details(
        {
            "estimated_total": 120,
            "estimated_assistant": 30,
        }
    )

    assert usage == {"input": 90, "output": 30, "total": 120}


def test_langfuse_usage_details_prefers_provider_tokens() -> None:
    usage = _langfuse_usage_details(
        {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
            "cached_tokens": 5,
            "estimated_total": 120,
            "estimated_assistant": 30,
        }
    )

    assert usage == {"input": 50, "output": 10, "total": 60, "cached": 5}


def test_langgraph_run_id_becomes_langfuse_trace_context() -> None:
    trace_attributes = langfuse_observability._langgraph_trace_attributes(
        {
            "run_id": "019ecc37-8726-7f33-9909-e7b9b86b7909",
            "thread_id": "019ecc37-8715-7313-b69f-a65731a58faf",
            "langfuse_trace_name": "solidcue:router",
        }
    )

    assert trace_attributes == {
        "trace_context": {"trace_id": "019ecc3787267f339909e7b9b86b7909"},
        "trace_name": "solidcue:router",
        "session_id": "019ecc37-8715-7313-b69f-a65731a58faf",
    }


def test_langgraph_trace_context_rejects_invalid_run_id() -> None:
    assert langfuse_observability._langgraph_trace_context({"run_id": "not-a-run"}) is None


def test_langgraph_graph_id_defaults_trace_name() -> None:
    trace_attributes = langfuse_observability._langgraph_trace_attributes(
        {
            "run_id": "019ecc37-8726-7f33-9909-e7b9b86b7909",
            "thread_id": "019ecc37-8715-7313-b69f-a65731a58faf",
            "graph_id": "router",
        }
    )

    assert trace_attributes is not None
    assert trace_attributes["trace_name"] == "solidcue:router"
