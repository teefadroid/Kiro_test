"""Command-line interface for arabic-llm-ocr."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .ocr import OCRResult, PageResult, ocr_file
from .providers import available_providers
from .providers.base import ProviderError


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arabic-ocr",
        description="OCR Arabic text from PDFs and images using vision-capable LLMs.",
    )
    p.add_argument("input", help="Path to a PDF or image file.")
    p.add_argument(
        "-o", "--output",
        help="Write result to this file. If omitted, prints to stdout.",
    )
    p.add_argument(
        "--provider",
        choices=available_providers(),
        default="ollama",
        help="Vision LLM backend (default: ollama).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model override (e.g. gpt-4o-mini, claude-3-5-haiku-latest, qwen2.5vl:7b).",
    )
    p.add_argument(
        "--dpi", type=int, default=200,
        help="PDF rendering DPI (default: 200). Higher = sharper but slower.",
    )
    p.add_argument(
        "--pages", default=None,
        help='Page selection like "1,3-5,8" (1-indexed). PDFs only.',
    )
    p.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Output format (default: text).",
    )

    diacritics = p.add_mutually_exclusive_group()
    diacritics.add_argument(
        "--preserve-diacritics", dest="preserve_diacritics",
        action="store_true", default=True,
        help="Keep Arabic tashkeel in the output (default).",
    )
    diacritics.add_argument(
        "--no-diacritics", dest="preserve_diacritics",
        action="store_false",
        help="Strip Arabic tashkeel from the output.",
    )

    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def _progress(current: int, total: int, page: PageResult) -> None:
    status = "ok" if page.error is None else f"ERROR: {page.error}"
    print(
        f"[{current}/{total}] page {page.page_number} "
        f"({page.latency_s}s) — {status}",
        file=sys.stderr,
    )


def _render(result: OCRResult, fmt: str) -> str:
    if fmt == "json":
        return result.as_json()
    return result.as_text(include_separators=len(result.pages) > 1)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        result = ocr_file(
            args.input,
            provider=args.provider,
            model=args.model,
            dpi=args.dpi,
            page_spec=args.pages,
            preserve_diacritics=args.preserve_diacritics,
            progress=_progress,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ProviderError as e:
        print(f"provider error: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    rendered = _render(result, args.format)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")

    # Non-zero exit if any page failed, so scripts can detect it.
    if any(p.error for p in result.pages):
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
