"""OpenAI chat-completions provider (GPT-4o family)."""

from __future__ import annotations

import base64
import os

import requests

from .base import ProviderError, VisionProvider


class OpenAIProvider(VisionProvider):
    name = "openai"
    default_model = "gpt-4o"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model=model, timeout=timeout)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        if not self.api_key:
            raise ProviderError(
                "OPENAI_API_KEY is not set. Export it or pass --api-key."
            )

    def transcribe(
        self,
        *,
        image_png: bytes,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        b64 = base64.b64encode(image_png).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise ProviderError(f"OpenAI request failed: {e}") from e

        if resp.status_code >= 400:
            raise ProviderError(
                f"OpenAI API error {resp.status_code}: {resp.text[:500]}"
            )
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, ValueError, IndexError) as e:
            raise ProviderError(
                f"Unexpected OpenAI response shape: {resp.text[:500]}"
            ) from e
