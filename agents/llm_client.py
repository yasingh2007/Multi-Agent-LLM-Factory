from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    pass


class RateLimiter:
    """Serialize LLM calls and enforce spacing / batch pauses (thread-safe)."""

    def __init__(
        self,
        *,
        requests_per_minute: float | None = None,
        min_interval_sec: float = 0.0,
        batch_size: int = 0,
        batch_sleep_sec: float = 0.0,
    ) -> None:
        interval = 0.0
        if requests_per_minute and requests_per_minute > 0:
            interval = 60.0 / float(requests_per_minute)
        if min_interval_sec > 0:
            interval = max(interval, float(min_interval_sec))

        self._min_interval = interval
        self._batch_size = max(0, int(batch_size))
        self._batch_sleep = max(0.0, float(batch_sleep_sec))
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._call_count = 0
        self.total_wait_sec = 0.0

    @property
    def enabled(self) -> bool:
        return self._min_interval > 0 or (self._batch_size > 0 and self._batch_sleep > 0)

    def wait(self) -> None:
        if not self.enabled:
            return

        with self._lock:
            slept = 0.0
            now = time.monotonic()
            if self._min_interval > 0 and self._last_call > 0:
                gap = self._min_interval - (now - self._last_call)
                if gap > 0:
                    time.sleep(gap)
                    slept += gap

            self._call_count += 1
            if (
                self._batch_size > 0
                and self._batch_sleep > 0
                and self._call_count % self._batch_size == 0
            ):
                time.sleep(self._batch_sleep)
                slept += self._batch_sleep

            self._last_call = time.monotonic()
            self.total_wait_sec += slept


def build_rate_limiter(llm_cfg: dict[str, Any]) -> RateLimiter | None:
    rl = llm_cfg.get("rate_limit") or {}
    rpm = rl.get("requests_per_minute", llm_cfg.get("requests_per_minute"))
    limiter = RateLimiter(
        requests_per_minute=float(rpm) if rpm is not None else None,
        min_interval_sec=float(rl.get("min_interval_sec", llm_cfg.get("min_interval_sec", 0))),
        batch_size=int(rl.get("batch_size", llm_cfg.get("batch_size", 0))),
        batch_sleep_sec=float(rl.get("batch_sleep_sec", llm_cfg.get("batch_sleep_sec", 0))),
    )
    return limiter if limiter.enabled else None


def resolve_llm_settings(llm_cfg: dict[str, Any]) -> dict[str, Any]:
    key_env = str(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(key_env, "").strip()
    base_url = (
        os.environ.get(str(llm_cfg.get("base_url_env", "OPENAI_BASE_URL")), "").strip()
        or str(llm_cfg.get("base_url", "https://api.openai.com/v1")).rstrip("/")
    )
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": str(llm_cfg.get("model", "gpt-4o-mini")),
        "temperature": float(llm_cfg.get("temperature", 0.2)),
        "max_tokens": int(llm_cfg.get("max_tokens", 256)),
        "timeout_sec": float(llm_cfg.get("timeout_sec", 60)),
    }


def chat_completion(
    *,
    settings: dict[str, Any],
    system: str,
    user: str,
    rate_limiter: RateLimiter | None = None,
) -> str:
    if not settings.get("api_key"):
        raise LLMError("API key missing (set env var from labeling.llm.api_key_env)")

    if rate_limiter is not None:
        rate_limiter.wait()

    url = f"{settings['base_url']}/chat/completions"
    body = {
        "model": settings["model"],
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings['api_key']}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=settings["timeout_sec"]) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise LLMError(f"HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(str(e)) from e

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response shape: {payload!r}") from e

    return str(content).strip()
