"""Google Gemini provider (Gemini Developer API, not Vertex AI).

Uses the public REST endpoint at generativelanguage.googleapis.com. Any
Gemini model that supports image input will work; we default to
``gemini-2.5-flash`` for a good speed/quality/price trade-off and recommend
``gemini-2.5-pro`` when quality matters more than cost.

Auth is a single API key from https://aistudio.google.com/app/apikey,
exported as ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY`` as a fallback).
"""

from __future__ import annotations

import base64
import os

import requests

from .base import ProviderError, VisionProvider


class GeminiProvider(VisionProvider):
    name = "gemini"
    default_model = "gemini-2.5-flash"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model=model, timeout=timeout)
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.base_url = (
            base_url
            or os.environ.get("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        if not self.api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set. Get one at "
                "https://aistudio.google.com/app/apikey and export it, "
                "or pass --api-key."
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
            # System instructions are a first-class field in generateContent.
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "image/png", "data": b64}},
                        {"text": user_prompt},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                # Arabic pages with tashkeel can be long; give the model headroom
                # rather than letting it truncate silently.
                "maxOutputTokens": 8192,
            },
        }
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/models/{self.model}:generateContent"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.exceptions.ReadTimeout as e:
            raise ProviderError(
                f"Gemini read timeout after {self.timeout:.0f}s. "
                f"Try --model gemini-2.5-flash (faster) or a longer --timeout."
            ) from e
        except requests.RequestException as e:
            raise ProviderError(f"Gemini request failed: {e}") from e

        if resp.status_code >= 400:
            # Gemini returns structured error messages; surface them verbatim so
            # quota / invalid-key / safety-block causes are obvious.
            raise ProviderError(
                f"Gemini API error {resp.status_code}: {resp.text[:800]}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderError(f"Gemini returned non-JSON: {resp.text[:500]}") from e

        candidates = data.get("candidates") or []
        if not candidates:
            # Prompt-level block (e.g. safety filter on the whole request).
            feedback = data.get("promptFeedback")
            raise ProviderError(
                f"Gemini returned no candidates. promptFeedback={feedback}"
            )

        cand = candidates[0]
        finish_reason = cand.get("finishReason")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()

        if not text:
            # Candidate exists but had no text -- typically SAFETY or RECITATION
            # block on the response itself, or MAX_TOKENS hit before any token.
            raise ProviderError(
                f"Gemini returned an empty response "
                f"(finishReason={finish_reason!r}). Raw: {resp.text[:500]}"
            )
        return text
