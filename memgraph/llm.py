"""LLM provider abstraction.

Real providers talk plain REST through urllib, so the only runtime
dependency besides FalkorDB is the standard library. Tests and offline
demos use FakeLLM.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class LLMError(Exception):
    """Raised when the model cannot be reached or returns garbage."""


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "", json_mode: bool = False) -> str:
        """Return the model's raw text response."""


def _post_json(
    url: str, payload: dict, headers: dict | None = None, timeout: int = 60
) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        raise LLMError(f"LLM HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"LLM connection failed: {e.reason}") from e


def strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers models love to add."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


class GeminiLLM(LLM):
    """Google Gemini via REST. The free tier works with a GEMINI_API_KEY."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system: str = "", json_mode: bool = False) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload: dict = {"contents": [{"parts": [{"text": prompt}]}]}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            payload["generationConfig"] = {"responseMimeType": "application/json"}
        res = _post_json(url, payload)
        try:
            return res["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected Gemini response: {str(res)[:300]}") from e


class OpenAICompatLLM(LLM):
    """Any OpenAI-compatible chat API (OpenAI, OpenRouter, local servers)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system: str = "", json_mode: bool = False) -> str:
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict = {"model": self.model, "messages": messages}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        res = _post_json(
            url, payload, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        try:
            return res["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected API response: {str(res)[:300]}") from e


class FakeLLM(LLM):
    """Deterministic stand-in for tests and offline demos."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, prompt: str, system: str = "", json_mode: bool = False) -> str:
        self.calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})
        if not self.responses:
            raise LLMError("FakeLLM ran out of canned responses")
        return self.responses.pop(0)


def llm_from_env() -> LLM:
    """Pick a provider from the environment.

    GEMINI_API_KEY -> Gemini (free tier). OPENAI_API_KEY -> any
    OpenAI-compatible endpoint (override with OPENAI_BASE_URL / OPENAI_MODEL).
    """
    if key := os.environ.get("GEMINI_API_KEY"):
        return GeminiLLM(key, model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    if key := os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatLLM(
            key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        )
    raise LLMError(
        "No LLM configured. Set GEMINI_API_KEY (free tier) or OPENAI_API_KEY."
    )
