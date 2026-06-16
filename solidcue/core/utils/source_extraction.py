from __future__ import annotations

import hashlib
import re
from typing import Any


def _item_key_from_url(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"u_{digest}"


def _extract_urls_from_input(user_input: str) -> list[str]:
    if not isinstance(user_input, str) or not user_input.strip():
        return []
    pattern = re.compile(r"https?://[^\s)>\"']+")
    raw_urls = pattern.findall(user_input)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in raw_urls:
        normalized = url.strip().rstrip(".,;:!?)]")
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _extract_urls_from_chat_history(chat_history: list[dict[str, Any]] | None) -> list[str]:
    if not isinstance(chat_history, list) or not chat_history:
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for entry in chat_history:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("role") or "").strip() != "user":
            continue
        for url in _extract_urls_from_input(str(entry.get("content") or "")):
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def build_target_artifacts_source(
    user_input: str,
    chat_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    urls = _extract_urls_from_chat_history(chat_history)
    for url in _extract_urls_from_input(user_input):
        if url not in urls:
            urls.append(url)
    return [
        {
            "index": idx,
            "source_type": "url",
            "source_ref": url,
            "item_key": _item_key_from_url(url),
        }
        for idx, url in enumerate(urls, start=1)
    ]
