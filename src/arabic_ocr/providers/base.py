"""Abstract base for vision-LLM providers.

Every provider takes a system prompt, a user prompt, and a single page image
(as PNG bytes) and returns the model's text response. Keeping the interface
this narrow means the OCR orchestrator doesn't care which backend is in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderError(RuntimeError):
    """Raised for any provider-level failure (HTTP, auth, malformed response)."""


class VisionProvider(ABC):
    name: str = "base"
    default_model: str = ""

    def __init__(self, *, model: str | None = None, timeout: float = 120.0) -> None:
        self.model = model or self.default_model
        self.timeout = timeout

    @abstractmethod
    def transcribe(
        self,
        *,
        image_png: bytes,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return the model's text response for one page image."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}(model={self.model!r})"
