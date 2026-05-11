"""Anthropic Claude messages provider."""

from __future__ import annotations

import base64
import os

import requests

from .base import ProviderError, VisionProvider

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(VisionProvider):
    name = "anthropic"
    default_model = "claude-3-5-sonnet-latest"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model=model, timeout=timeout)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (base_url or "https://api.anthropic.com/v1").rstrip("/")
        self.max_tokens = max_tokens
        if not self.api_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Export it or pass --api-key."
            )

    def transcribe(
        self,
        *,
        image_png: bytes,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        b64 = base64.b64encode(image_png).decode("ascii")
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        try:
            resp = requests.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise ProviderError(f"Anthropic request failed: {e}") from e

        if resp.status_code >= 400:
            raise ProviderError(
                f"Anthropic API error {resp.status_code}: {resp.text[:500]}"
            )
        try:
            data = resp.json()
            # content is a list of blocks; collect all text blocks.
            parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            text = "".join(parts).strip()
            if not text:
                raise ProviderError(f"Anthropic returned no text: {data}")
            return text
        except (KeyError, ValueError) as e:
            raise ProviderError(
                f"Unexpected Anthropic response shape: {resp.text[:500]}"
            ) from e
