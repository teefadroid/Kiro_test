"""Vision-LLM provider backends."""

from __future__ import annotations

from .base import VisionProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider

_REGISTRY: dict[str, type[VisionProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


def get_provider(name: str, *, model: str | None = None, timeout: float | None = None, **kwargs) -> VisionProvider:
    """Construct a provider by name. ``model=None`` uses the provider's default.

    ``timeout`` is forwarded to the provider only when explicitly set so each
    provider keeps its own sensible default.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown provider {name!r}. Valid: {valid}.")
    if timeout is not None:
        kwargs["timeout"] = timeout
    return _REGISTRY[key](model=model, **kwargs)


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "VisionProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "OllamaProvider",
    "get_provider",
    "available_providers",
]
