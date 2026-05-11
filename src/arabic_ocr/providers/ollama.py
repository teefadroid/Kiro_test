"""Ollama provider for local vision models (qwen2.5vl, llama3.2-vision, etc.)."""

from __future__ import annotations

import base64
import os

import requests

from .base import ProviderError, VisionProvider


class OllamaProvider(VisionProvider):
    name = "ollama"
    default_model = "qwen2.5vl:7b"

    def __init__(
        self,
        *,
        model: str | None = None,
        host: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(model=model, timeout=timeout)
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
        except requests.RequestException as e:
            raise ProviderError(
                f"Ollama request failed (is the server running at {self.host}?): {e}"
            ) from e

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
