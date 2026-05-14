"""OCR orchestrator — loops pages through a provider and assembles results."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .loader import PageImage, load_pages
from .prompts import NO_ARABIC_SENTINEL, SYSTEM_PROMPT, build_user_prompt
from .providers import VisionProvider, get_provider
from .providers.base import ProviderError

log = logging.getLogger(__name__)

PAGE_SEPARATOR = "\n\n===== PAGE {page} =====\n\n"


@dataclass
class PageResult:
    page_number: int
    text: str
    error: str | None = None
    skipped: bool = False  # True when page had no Arabic content
    latency_s: float = 0.0


@dataclass
class OCRResult:
    source: str
    provider: str
    model: str
    pages: list[PageResult] = field(default_factory=list)

    def as_text(self, *, include_separators: bool = True) -> str:
        # Filter out skipped pages (no Arabic content)
        active = [p for p in self.pages if not p.skipped]
        if not active:
            return ""
        if not include_separators or len(active) == 1:
            return "\n\n".join(p.text for p in active).strip()
        parts = []
        for p in active:
            parts.append(PAGE_SEPARATOR.format(page=p.page_number).strip())
            parts.append(p.text)
        return "\n\n".join(parts).strip() + "\n"

    def as_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def as_markdown(self) -> str:
        """Render as a Markdown document with proper RTL handling.

        Each page is wrapped in ``<div dir="rtl">`` so renderers like GitHub,
        VS Code preview, and Obsidian display the Arabic with the correct
        direction even when surrounded by Latin text or punctuation.
        Single-page documents skip the per-page header for cleanliness.
        """
        src_name = Path(self.source).name
        lines: list[str] = [
            f"# {src_name}",
            "",
            f"*OCR via `{self.provider}` (`{self.model}`)*",
            "",
        ]

        def _page_block(p: PageResult) -> list[str]:
            if p.error:
                return [f"> **Error on page {p.page_number}:** {p.error}", ""]
            if p.skipped:
                return []  # omit from markdown entirely
            # Blank lines around the raw HTML so the markdown parser doesn't
            # treat the Arabic content as inside an HTML block.
            return ['<div dir="rtl" markdown="1">', "", p.text, "", "</div>", ""]

        active = [p for p in self.pages if not p.skipped]
        if not active:
            lines.append("*No Arabic content found in this document.*\n")
            return "\n".join(lines).rstrip() + "\n"

        if len(active) == 1:
            lines.extend(_page_block(active[0]))
        else:
            for p in active:
                lines.append(f"## Page {p.page_number}")
                lines.append("")
                lines.extend(_page_block(p))

        return "\n".join(lines).rstrip() + "\n"


def _ocr_one_page(
    provider: VisionProvider,
    page: PageImage,
    *,
    preserve_diacritics: bool,
) -> PageResult:
    user_prompt = build_user_prompt(
        preserve_diacritics=preserve_diacritics,
        page_number=page.page_number,
    )
    start = time.monotonic()
    try:
        text = provider.transcribe(
            image_png=page.to_png_bytes(),
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        elapsed = round(time.monotonic() - start, 3)
        # If the model returned the sentinel, the page had no Arabic content.
        stripped = text.strip()
        if stripped == NO_ARABIC_SENTINEL or stripped == NO_ARABIC_SENTINEL.strip("[]"):
            log.info("Page %d: no Arabic content, skipping.", page.page_number)
            return PageResult(
                page_number=page.page_number,
                text="",
                skipped=True,
                latency_s=elapsed,
            )
        return PageResult(
            page_number=page.page_number,
            text=text,
            latency_s=elapsed,
        )
    except ProviderError as e:
        log.error("Page %d failed: %s", page.page_number, e)
        return PageResult(
            page_number=page.page_number,
            text="",
            error=str(e),
            latency_s=round(time.monotonic() - start, 3),
        )


def ocr_file(
    input_path: str | Path,
    *,
    provider: str | VisionProvider = "ollama",
    model: str | None = None,
    dpi: int = 200,
    page_spec: str | None = None,
    preserve_diacritics: bool = True,
    timeout: float | None = None,
    progress: Callable[[int, int, PageResult], None] | None = None,
) -> OCRResult:
    """Run OCR on a PDF or image and return an ``OCRResult``.

    Parameters
    ----------
    input_path:
        Path to the PDF or image.
    provider:
        Either a provider name (``"openai"``, ``"anthropic"``, ``"ollama"``) or
        an already-constructed ``VisionProvider`` instance.
    model:
        Optional model override. Ignored if ``provider`` is already an instance.
    dpi:
        PDF rendering DPI (ignored for image inputs).
    page_spec:
        Optional page range like ``"1,3-5"`` (ignored for image inputs).
    preserve_diacritics:
        If ``True``, ask the model to keep Arabic diacritics.
    timeout:
        Per-page HTTP timeout in seconds. ``None`` keeps the provider default
        (15 minutes for Ollama, 2 minutes for hosted providers).
    progress:
        Optional callback ``(current, total, page_result) -> None`` invoked after
        each page.
    """
    prov = (
        provider
        if isinstance(provider, VisionProvider)
        else get_provider(provider, model=model, timeout=timeout)
    )

    pages = load_pages(input_path, dpi=dpi, page_spec=page_spec)
    results: list[PageResult] = []
    for idx, page in enumerate(pages, start=1):
        result = _ocr_one_page(prov, page, preserve_diacritics=preserve_diacritics)
        results.append(result)
        if progress is not None:
            progress(idx, len(pages), result)

    return OCRResult(
        source=str(Path(input_path).resolve()),
        provider=prov.name,
        model=prov.model,
        pages=results,
    )
