from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    pass


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
) -> str:
    if not settings.get("api_key"):
        raise LLMError("API key missing (set env var from labeling.llm.api_key_env)")

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
