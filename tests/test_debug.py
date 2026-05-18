from solidcue.core.utils.debug import (
    _truncate_text_content_fields,
    format_debug_value,
    sanitize_debug_text,
)


def test_format_debug_value_pretty_prints_nested_json_strings() -> None:
    formatted = format_debug_value(
        [
            {
                "function": {
                    "name": "search_web",
                    "arguments": '{"query":"Arsenal fixtures","num_results":5}',
                }
            }
        ]
    )

    assert '"arguments": {' in formatted
    assert '"query": "Arsenal fixtures"' in formatted
    assert '"num_results": 5' in formatted


def test_sanitize_debug_text_redacts_sensitive_values() -> None:
    text = (
        'url=https://serpapi.com/search?q=Arsenal&api_key=secret&hl=en '
        '{"token":"also-secret"}'
    )

    sanitized = sanitize_debug_text(text)

    assert "secret" not in sanitized
    assert "api_key=[redacted]" in sanitized
    assert '"token":"[redacted]"' in sanitized


def test_truncate_text_content_fields_only_truncates_text_and_content_keys() -> None:
    source = [
        {
            "tool_name": "browser_get_text",
            "content": "c" * 140,
            "content_base64": "b" * 160,
            "result": {
                "text": "t" * 150,
                "title": "x" * 150,
                "nested": [{"Text": "z" * 130}],
            },
        }
    ]

    truncated = _truncate_text_content_fields(source, max_len=100)
    entry = truncated[0]

    assert entry["content"].endswith("...")
    assert len(entry["content"]) == 103
    assert entry["content_base64"].endswith("...")
    assert len(entry["content_base64"]) == 103
    assert entry["result"]["text"].endswith("...")
    assert len(entry["result"]["text"]) == 103
    assert entry["result"]["nested"][0]["Text"].endswith("...")
    assert entry["result"]["title"] == "x" * 150
