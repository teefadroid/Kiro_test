"""Arabic-specific prompts for the OCR task.

These prompts were tuned with three concerns in mind:

1. **Verbatim transcription.** Models love to "help" by translating, explaining,
   or summarizing. We have to say "no" loudly and repeatedly.
2. **Right-to-left layout.** Arabic reads RTL, but lines and paragraphs still
   flow top-to-bottom. The prompt calls this out so the model doesn't reverse
   line order on multi-column scans.
3. **Diacritics (tashkeel).** The caller decides whether to keep them. Many
   printed texts include them selectively; forcing the model one way or the
   other matches the user's expectation.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a meticulous OCR engine specialized in Arabic script. "
    "Your sole job is to transcribe the Arabic (and any embedded Latin/digits) "
    "text visible in the provided image, verbatim. "
    "Do not translate. Do not summarize. Do not add commentary, headers, or "
    "markdown. Return only the transcribed text."
)


def build_user_prompt(*, preserve_diacritics: bool, page_number: int | None = None) -> str:
    """Build the per-page user prompt."""
    diacritics_rule = (
        "Preserve all Arabic diacritics (tashkeel: fatha, kasra, damma, shadda, "
        "sukun, tanwin) exactly as they appear."
        if preserve_diacritics
        else "Strip Arabic diacritics (tashkeel) from the output; keep only the base letters."
    )

    page_hint = f" (page {page_number})" if page_number is not None else ""

    return (
        f"Transcribe every piece of text visible in this image{page_hint}.\n\n"
        "Rules:\n"
        "- Reading order: right-to-left within each line; lines and paragraphs flow top-to-bottom.\n"
        "- Preserve line breaks and paragraph breaks as you see them.\n"
        "- Keep punctuation, numbers, and any Latin words exactly as written.\n"
        f"- {diacritics_rule}\n"
        "- If the image contains no legible text, output an empty string.\n"
        "- If part of the text is illegible, replace only that part with the "
        "token [illegible]; do not guess.\n"
        "- Do not wrap the output in code fences or quotes. Output raw text only."
    )
