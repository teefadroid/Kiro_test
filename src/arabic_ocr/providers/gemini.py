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
        timeout: float | None = None,
    ) -> None:
        # Vision OCR on a dense Arabic page can generate thousands of output
        # tokens; 120s is too tight even on Flash. Default to 10 minutes and
        # let the user override via GEMINI_TIMEOUT or --timeout.
        resolved_timeout = (
            timeout
            if timeout is not None
            else float(os.environ.get("GEMINI_TIMEOUT", 600.0))
        )
        super().__init__(model=model, timeout=resolved_timeout)
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
                # rather than letting it truncate silently. 32k is well within
                # the 2.5 family's 65k output ceiling.
                "maxOutputTokens": 32768,
                # Critical for OCR: gemini-2.5-flash defaults to "thinking" mode,
                # which silently consumes most of the maxOutputTokens budget on
                # invisible reasoning tokens before any text is emitted. For a
                # verbatim transcription task we don't want any reasoning -- we
                # want every output token spent on the transcription itself.
                # Setting thinkingBudget to 0 disables thinking entirely.
                "thinkingConfig": {"thinkingBudget": 0},
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
                f"Try a longer timeout via --timeout 1200 or "
                f"set GEMINI_TIMEOUT=1200 in your environment. "
                f"Dense pages with maxOutputTokens=32768 can take several minutes."
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
            if finish_reason == "MAX_TOKENS":
                raise ProviderError(
                    "Gemini hit MAX_TOKENS before producing any visible text. "
                    "This usually means thinking-mode consumed the entire budget. "
                    "Confirm thinkingConfig.thinkingBudget=0 is being honored, "
                    "or pick a non-thinking model variant."
                )
            raise ProviderError(
                f"Gemini returned an empty response "
                f"(finishReason={finish_reason!r}). Raw: {resp.text[:500]}"
            )

        if finish_reason == "MAX_TOKENS":
            # Response is non-empty but was cut off mid-transcription -- surface
            # this loudly so the caller knows the page is incomplete instead of
            # silently writing a truncated transcript.
            import logging
            logging.getLogger(__name__).warning(
                "Gemini response was truncated (MAX_TOKENS hit at %d output "
                "tokens). The page transcript is incomplete -- consider raising "
                "maxOutputTokens or using --pages to split the document.",
                len(text),
            )
            text = text + "\n\n[output truncated by Gemini MAX_TOKENS]"

        return text
