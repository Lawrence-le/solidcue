import time
from collections.abc import AsyncIterator, Iterator

import httpx


class AsyncHTTPClient:
    def __init__(self, timeout: int = 120) -> None:
        self.timeout = timeout

    async def post(self, url: str, headers: dict, json: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=json)
            response.raise_for_status()
            return response.json()

    async def stream_post(self, url: str, headers: dict, json: dict) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=headers, json=json) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line


class HTTPClient:
    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 3,
        initial_retry_delay_seconds: float = 1.0,
        max_retry_delay_seconds: float = 8.0,
    ):
        self.client = httpx.Client(timeout=timeout)
        self.max_retries = max_retries
        self.initial_retry_delay_seconds = initial_retry_delay_seconds
        self.max_retry_delay_seconds = max_retry_delay_seconds

    def post(self, url: str, headers: dict, json: dict):
        attempt = 0

        while True:
            try:
                response = self.client.post(url, headers=headers, json=json)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code

                if (
                    not self._is_retryable_status(status_code)
                    or attempt >= self.max_retries
                ):
                    raise RuntimeError(self._build_error_message(exc)) from exc

                retry_after_seconds = self._get_retry_after_seconds(exc.response)
                delay_seconds = (
                    retry_after_seconds
                    if retry_after_seconds is not None
                    else self._compute_backoff_seconds(attempt)
                )
                time.sleep(delay_seconds)
                attempt += 1

    def stream_post(self, url: str, headers: dict, json: dict) -> Iterator[str]:
        try:
            with self.client.stream("POST", url, headers=headers, json=json) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    yield line
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(self._build_error_message(exc)) from exc

    def _is_retryable_status(self, status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code < 600

    def _compute_backoff_seconds(self, attempt: int) -> float:
        delay_seconds = self.initial_retry_delay_seconds * (2**attempt)
        return min(delay_seconds, self.max_retry_delay_seconds)

    def _get_retry_after_seconds(self, response: httpx.Response) -> float | None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is None:
            return None

        try:
            seconds = float(retry_after)
        except (TypeError, ValueError):
            return None

        return max(0.0, min(seconds, self.max_retry_delay_seconds))

    def _build_error_message(self, exc: httpx.HTTPStatusError) -> str:
        response = exc.response
        status_line = (
            f"HTTP {response.status_code} for {response.request.method} "
            f"{response.request.url}"
        )

        details: str | None = None
        try:
            payload = response.json()
            if isinstance(payload, dict):
                error_obj = payload.get("error")
                if isinstance(error_obj, dict):
                    message = error_obj.get("message")
                    code = error_obj.get("code")
                    error_type = error_obj.get("type")
                    parts = [str(message).strip()] if message else []
                    if error_type:
                        parts.append(f"type={error_type}")
                    if code:
                        parts.append(f"code={code}")
                    details = "; ".join(part for part in parts if part)
                if not details:
                    details = str(payload)
        except Exception:
            details = response.text.strip() or None

        if details:
            return f"{status_line}. Details: {details}"
        return status_line
