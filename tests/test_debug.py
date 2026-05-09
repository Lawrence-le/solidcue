from solidcue.core.utils.debug import format_debug_value, sanitize_debug_text


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
