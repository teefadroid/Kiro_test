"""Ollama provider for local vision models (qwen2.5vl, llama3.2-vision, etc.)."""

from __future__ import annotations

import base64
import os

import requests

from .base import ProviderError, VisionProvider

# Vision models are slow on CPU; a 7B on a cold start can easily chew through
# 5+ minutes on one page. 15 minutes is a safer default than the usual
# "request timeout" value. Callers can still override via env or CLI.
_DEFAULT_TIMEOUT_S = 900.0


class OllamaProvider(VisionProvider):
    name = "ollama"
    default_model = "qwen2.5vl:7b"

    def __init__(
        self,
        *,
        model: str | None = None,
        host: str | None = None,
        timeout: float | None = None,
    ) -> None:
        resolved_timeout = (
            timeout
            if timeout is not None
            else float(os.environ.get("OLLAMA_TIMEOUT", _DEFAULT_TIMEOUT_S))
        )
        super().__init__(model=model, timeout=resolved_timeout)
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def transcribe(
        self,
        *,
        image_png: bytes,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        b64 = base64.b64encode(image_png).decode("ascii")
        # Ollama's /api/chat accepts images as a list of base64 strings attached to the user message.
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": [b64]},
            ],
        }
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectTimeout as e:
            raise ProviderError(
                f"Could not reach Ollama at {self.host} (connect timeout). "
                f"Is the server running? Try: 'ollama serve'"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise ProviderError(
                f"Could not reach Ollama at {self.host}. "
                f"Is the server running? Try: 'ollama serve'. Details: {e}"
            ) from e
        except requests.exceptions.ReadTimeout as e:
            raise ProviderError(
                f"Ollama read timeout after {self.timeout:.0f}s -- the server was "
                f"reached but inference did not finish in time. "
                f"Try a smaller model (e.g. 'qwen2.5vl:3b'), a lower --dpi (e.g. 150), "
                f"or a longer --timeout. On CPU, a 7B vision model can take "
                f"many minutes per page on first load."
            ) from e
        except requests.RequestException as e:
            raise ProviderError(f"Ollama request failed: {e}") from e

        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama error {resp.status_code}: {resp.text[:500]}"
            )
        try:
            data = resp.json()
            return data["message"]["content"].strip()
        except (KeyError, ValueError) as e:
            raise ProviderError(
                f"Unexpected Ollama response shape: {resp.text[:500]}"
            ) from e
