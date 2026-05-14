"""Arabic-specific prompts for the OCR task.

These prompts were tuned with four concerns in mind:

1. **Arabic-only extraction.** The document is mixed English/Arabic. Pages that
   are entirely in English should be skipped (empty output). On mixed pages,
   only Arabic text (plus any English/Latin words that appear *within* Arabic
   sentences) should be transcribed.
2. **Verbatim transcription.** Models love to "help" by translating, explaining,
   or summarizing. We have to say "no" loudly and repeatedly.
3. **Right-to-left layout.** Arabic reads RTL, but lines and paragraphs still
   flow top-to-bottom. The prompt calls this out so the model doesn't reverse
   line order on multi-column scans.
4. **Diacritics (tashkeel).** The caller decides whether to keep them. Many
   printed texts include them selectively; forcing the model one way or the
   other matches the user's expectation.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a meticulous OCR engine specialized in extracting Arabic text. "
    "Your job is to transcribe ONLY the Arabic-script text visible in the "
    "provided image. If a page is entirely in English (or any non-Arabic "
    "language), return exactly the string: [NO_ARABIC]\n\n"
    "On mixed pages that contain both Arabic and English:\n"
    "- Transcribe all Arabic paragraphs/sentences verbatim.\n"
    "- Include English/Latin words or numbers ONLY if they appear inline "
    "within an Arabic sentence (e.g. brand names, technical terms, dates).\n"
    "- SKIP standalone English paragraphs, headings, captions, headers, and "
    "footers that are not part of an Arabic sentence.\n\n"
    "Do not translate. Do not summarize. Do not add commentary, headers, or "
    "markdown. Return only the transcribed Arabic text (with inline English "
    "where applicable)."
)

# Sentinel the model returns when a page has no Arabic content at all.
NO_ARABIC_SENTINEL = "[NO_ARABIC]"


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
        f"Extract only the Arabic text from this image{page_hint}.\n\n"
        "Rules:\n"
        "- If this page contains NO Arabic text at all (entirely English or "
        "another non-Arabic language), return exactly: [NO_ARABIC]\n"
        "- Otherwise, transcribe all Arabic-script text verbatim.\n"
        "- Include English/Latin words or numbers ONLY when they are embedded "
        "inside an Arabic sentence. Skip standalone English paragraphs.\n"
        "- Reading order: right-to-left within each line; lines and paragraphs flow top-to-bottom.\n"
        "- Preserve line breaks and paragraph breaks as you see them in the Arabic portions.\n"
        f"- {diacritics_rule}\n"
        "- If part of the Arabic text is illegible, replace only that part with the "
        "token [illegible]; do not guess.\n"
        "- Do not wrap the output in code fences or quotes. Output raw text only."
    )
